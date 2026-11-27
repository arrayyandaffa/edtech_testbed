from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from modules.data_loader import get_column_info, load_data
from modules.modeling import get_default_model
from modules.preprocessing import build_preprocessor

# Konfigurasi eksperimen (samakan dengan setelan di sidebar aplikasi)
CSV_PATH = Path("edtech_students.csv")
TARGET_COL = "motivation_level"
NUM_IMPUTER = "mean"
CAT_IMPUTER = "most_frequent"
SCALER_TYPE = "StandardScaler"
ENCODER_TYPE = "OneHotEncoder"
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

MODEL_LIST = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "K-Nearest Neighbors",
    "Naive Bayes",
]

OUTPUT_DIR = Path("paper_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    with CSV_PATH.open("rb") as f:
        df = load_data(f)

    if df is None or df.empty:
        raise RuntimeError("Dataset gagal dimuat atau kosong.")

    id_cols, num_cols, cat_cols = get_column_info(df)
    features = [c for c in df.columns if c not in id_cols and c != TARGET_COL]
    active_num = [c for c in num_cols if c in features]
    active_cat = [c for c in cat_cols if c in features]

    X = df[features]
    y = df[TARGET_COL]

    # Split dilakukan terlebih dahulu sebelum preprocessing & CV
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(
        active_num, active_cat, NUM_IMPUTER, CAT_IMPUTER, SCALER_TYPE, ENCODER_TYPE
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    summary_rows = []
    cm_lines = [
        f"# Confusion Matrix Hasil Benchmark (`random_state={RANDOM_STATE}`, `test_size={TEST_SIZE}`)\n"
    ]

    for model_name in MODEL_LIST:
        model = get_default_model(model_name, "Classification")
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

        # Cross-validation murni pada data training menggunakan Pipeline
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_weighted")

        # Fit pada X_train, evaluasi pada X_test
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        summary_rows.append({
            "Model": model_name,
            f"CV_{CV_FOLDS}Fold_F1_Mean": round(float(cv_scores.mean()), 4),
            f"CV_{CV_FOLDS}Fold_F1_Std": round(float(cv_scores.std()), 4),
            "Test_Accuracy": round(float(acc), 4),
            "Test_Precision_Weighted": round(float(prec), 4),
            "Test_Recall_Weighted": round(float(rec), 4),
            "Test_F1_Weighted": round(float(f1), 4),
        })

        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual_{lbl}" for lbl in labels],
            columns=[f"Pred_{lbl}" for lbl in labels],
        )
        cm_lines.append(f"## {model_name}\n")
        cm_lines.append(cm_df.to_markdown() + "\n")

    summary_df = pd.DataFrame(summary_rows)
    csv_out = OUTPUT_DIR / "metrics_summary.csv"
    md_out = OUTPUT_DIR / "paper_tables.md"

    summary_df.to_csv(csv_out, index=False)
    with md_out.open("w", encoding="utf-8") as out:
        out.write("# Ringkasan Metrik Evaluasi Model\n\n")
        out.write(summary_df.to_markdown(index=False) + "\n\n")
        out.write("\n".join(cm_lines))

    print("=== RINGKASAN METRIK ===")
    print(summary_df.to_string(index=False))
    print(f"\nTersimpan di: {csv_out} dan {md_out}")


if __name__ == "__main__":
    main()