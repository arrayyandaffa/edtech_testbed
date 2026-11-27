import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from modules.evaluation import evaluate_classification, evaluate_regression
from modules.preprocessing import build_preprocessor


def get_model(model_name, task_type, random_state=42):
    st.write(f"**Hyperparameter Config:** `{model_name}`")

    if model_name == "Logistic Regression":
        c_val = st.slider("C (Inverse Regularization)", 0.01, 10.0, 1.0, step=0.05)
        return LogisticRegression(C=c_val, max_iter=1000, random_state=random_state)
    if model_name == "Linear Regression":
        st.caption("Model regresi linear standar tanpa hyperparameter tuning utama.")
        return LinearRegression()
    if model_name == "Naive Bayes":
        var_smooth = st.select_slider(
            "var_smoothing",
            options=[1e-11, 1e-9, 1e-7, 1e-5],
            value=1e-9,
        )
        return GaussianNB(var_smoothing=var_smooth)
    if "Decision Tree" in model_name:
        max_depth = st.slider("Max Depth", 1, 30, 10)
        min_samples = st.slider("Min Samples Split", 2, 20, 2)
        if task_type == "Classification":
            return DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples,
                random_state=random_state,
            )
        return DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples,
            random_state=random_state,
        )
    if "Random Forest" in model_name:
        n_est = st.slider("N Estimators", 10, 200, 100, step=10)
        max_depth = st.slider("Max Depth", 1, 30, 10)
        if task_type == "Classification":
            return RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_depth,
                random_state=random_state,
            )
        return RandomForestRegressor(
            n_estimators=n_est,
            max_depth=max_depth,
            random_state=random_state,
        )
    if "KNN" in model_name or "K-Nearest" in model_name:
        n_neighbors = st.slider("N Neighbors", 1, 25, 5)
        if task_type == "Classification":
            return KNeighborsClassifier(n_neighbors=n_neighbors)
        return KNeighborsRegressor(n_neighbors=n_neighbors)
    if "Support Vector" in model_name:
        c_val = st.slider("C (Regularization)", 0.1, 10.0, 1.0, step=0.1)
        if task_type == "Classification":
            return SVC(C=c_val, probability=True, random_state=random_state)
        return SVR(C=c_val)

    return DecisionTreeClassifier(random_state=random_state)


def get_default_model(m_name, task_type, random_state=42):
    if m_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if m_name == "Linear Regression":
        return LinearRegression()
    if m_name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=10, min_samples_split=2, random_state=random_state)
    if m_name == "Decision Tree Regressor":
        return DecisionTreeRegressor(max_depth=10, min_samples_split=2, random_state=random_state)
    if m_name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state)
    if m_name == "Random Forest Regressor":
        return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state)
    if m_name in ["K-Nearest Neighbors", "KNN Regressor"]:
        return KNeighborsClassifier(n_neighbors=5) if task_type == "Classification" else KNeighborsRegressor(n_neighbors=5)
    if m_name == "Naive Bayes":
        return GaussianNB()
    if m_name == "Support Vector Machine":
        return SVC(C=1.0, probability=True, random_state=random_state)
    if m_name == "Support Vector Regression":
        return SVR(C=1.0)
    return DecisionTreeClassifier(random_state=random_state)


def _split_data(X, y, task_type, test_size, random_state):
    stratify = y if task_type == "Classification" else None
    try:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )


def train_evaluate_model(
    X,
    y,
    features,
    num_cols,
    cat_cols,
    num_imputer,
    cat_imputer,
    scaler_type,
    encoder_type,
    model,
    task_type,
    test_size,
    random_state,
    use_cv,
    cv_folds,
):
    active_num = [c for c in num_cols if c in features]
    active_cat = [c for c in cat_cols if c in features]

    preprocessor = build_preprocessor(
        active_num, active_cat, num_imputer, cat_imputer, scaler_type, encoder_type
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = _split_data(
        X, y, task_type, test_size, random_state
    )
    st.success(
        f"✅ Train-Test Split selesai: **{X_train.shape[0]}** baris training | "
        f"**{X_test.shape[0]}** baris testing."
    )

    if use_cv:
        cv = (
            StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            if task_type == "Classification"
            else KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        )
        scoring = "f1_weighted" if task_type == "Classification" else "r2"
        try:
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring=scoring
            )
            st.info(
                f"**{cv_folds}-Fold Cross-Validation ({scoring}) pada Training Set**: "
                f"**{cv_scores.mean():.4f}** (± {cv_scores.std():.4f})"
            )
        except Exception as exc:
            st.warning(f"Cross-validation tidak dapat dijalankan: {exc}")

    try:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
    except Exception as exc:
        st.error(f"Gagal melatih atau mengevaluasi model: {exc}")
        return

    if task_type == "Classification":
        evaluate_classification(y_test, y_pred, model, pipeline, active_num, active_cat)
    else:
        evaluate_regression(y_test, y_pred)


def compare_models(
    X,
    y,
    model_names,
    task_type,
    features,
    num_cols,
    cat_cols,
    num_imputer,
    cat_imputer,
    scaler_type,
    encoder_type,
    test_size,
    random_state,
    use_cv=True,
    cv_folds=5,
):
    active_num = [c for c in num_cols if c in features]
    active_cat = [c for c in cat_cols if c in features]
    preprocessor = build_preprocessor(
        active_num, active_cat, num_imputer, cat_imputer, scaler_type, encoder_type
    )

    X_train, X_test, y_train, y_test = _split_data(
        X, y, task_type, test_size, random_state
    )
    cv = (
        StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        if task_type == "Classification"
        else KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    )

    results = []
    for m_name in model_names:
        est = get_default_model(m_name, task_type, random_state=random_state)
        pipe = Pipeline([("preprocessor", preprocessor), ("model", est)])

        try:
            cv_mean = np.nan
            cv_std = np.nan
            if use_cv:
                scoring = "f1_weighted" if task_type == "Classification" else "r2"
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            if task_type == "Classification":
                results.append(
                    {
                        "Model": m_name,
                        "CV F1 (Mean)": round(cv_mean, 4) if use_cv else None,
                        "CV F1 (Std)": round(cv_std, 4) if use_cv else None,
                        "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                        "Precision (Weighted)": round(
                            float(
                                precision_score(
                                    y_test, y_pred, average="weighted", zero_division=0
                                )
                            ),
                            4,
                        ),
                        "Recall (Weighted)": round(
                            float(
                                recall_score(
                                    y_test, y_pred, average="weighted", zero_division=0
                                )
                            ),
                            4,
                        ),
                        "F1 (Weighted)": round(
                            float(
                                f1_score(
                                    y_test, y_pred, average="weighted", zero_division=0
                                )
                            ),
                            4,
                        ),
                        "Macro F1": round(
                            float(
                                f1_score(
                                    y_test, y_pred, average="macro", zero_division=0
                                )
                            ),
                            4,
                        ),
                    }
                )
            else:
                results.append(
                    {
                        "Model": m_name,
                        "CV R² (Mean)": round(cv_mean, 4) if use_cv else None,
                        "CV R² (Std)": round(cv_std, 4) if use_cv else None,
                        "R² Score": round(float(r2_score(y_test, y_pred)), 4),
                        "RMSE": round(
                            float(np.sqrt(mean_squared_error(y_test, y_pred))), 4
                        ),
                    }
                )
        except Exception as exc:
            st.error(f"Error pada model `{m_name}`: {exc}")

    if not results:
        st.warning("Tidak ada hasil model yang berhasil dihitung.")
        return

    sort_col = "F1 (Weighted)" if task_type == "Classification" else "R² Score"
    res_df = pd.DataFrame(results).sort_values(by=sort_col, ascending=False).reset_index(drop=True)

    highlight_cols = (
        ["Accuracy", "F1 (Weighted)", "Macro F1"]
        if task_type == "Classification"
        else ["R² Score"]
    )
    st.dataframe(
        res_df.style.highlight_max(axis=0, color="#c8e6c9", subset=highlight_cols),
        use_container_width=True,
    )

    top_model = res_df.iloc[0]["Model"]
    top_score = res_df.iloc[0][sort_col]
    st.info(
        f"Peringkat teratas berdasarkan **{sort_col}**: **{top_model}** ({top_score:.4f}). "
        "Bandingkan juga **Macro F1** dan **Accuracy** secara proporsional—jika selisih antar-model "
        "tipis atau akurasi tinggi disertai Macro F1 rendah, model berpotensi didominasi kelas mayoritas."
    )