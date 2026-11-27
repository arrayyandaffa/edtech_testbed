import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def _extract_feature_names(pipeline, active_num, active_cat):
    names = []
    if "preprocessor" not in pipeline.named_steps:
        return names

    prep = pipeline.named_steps["preprocessor"]
    if not hasattr(prep, "named_transformers_"):
        return names

    if active_num and "num" in prep.named_transformers_:
        names.extend(active_num)

    if active_cat and "cat" in prep.named_transformers_:
        cat_pipe = prep.named_transformers_["cat"]
        if hasattr(cat_pipe, "named_steps") and "encoder" in cat_pipe.named_steps:
            encoder = cat_pipe.named_steps["encoder"]
            if hasattr(encoder, "get_feature_names_out"):
                names.extend(encoder.get_feature_names_out(active_cat).tolist())
            else:
                names.extend(active_cat)
        else:
            names.extend(active_cat)

    return names


def evaluate_classification(y_true, y_pred, model, pipeline, active_num, active_cat):
    st.markdown("---")
    st.subheader("📊 Hasil Evaluasi Testing Set (Unseen Data)")

    acc = float(accuracy_score(y_true, y_pred))
    prec_w = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_w = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("Precision (Weighted)", f"{prec_w:.4f}")
    c3.metric("Recall (Weighted)", f"{rec_w:.4f}")
    c4.metric("F1 (Weighted)", f"{f1_w:.4f}")
    c5.metric("Macro F1", f"{macro_f1:.4f}")

    pred_counts = pd.Series(y_pred).value_counts()
    if len(pred_counts) == 1:
        st.warning(
            f"⚠️ Seluruh sampel uji diprediksi ke satu kelas saja (`{pred_counts.index[0]}`). "
            "Perhatikan Macro F1 dan Confusion Matrix untuk menilai apakah terjadi majority-class collapse."
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Classification Report (Per Kelas & Rata-Rata)**")
        report_dict = classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df, use_container_width=True)

    with col_b:
        st.write("**Confusion Matrix**")
        labels = sorted(pd.Series(y_true).unique(), key=str)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.2))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[str(lbl) for lbl in labels],
            yticklabels=[str(lbl) for lbl in labels],
            ax=ax_cm,
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title("Confusion Matrix")
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    st.markdown("---")
    st.write("**Model Interpretability / Feature Importance**")
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        feat_names = _extract_feature_names(pipeline, active_num, active_cat)
        if len(feat_names) == len(importances):
            feat_df = (
                pd.DataFrame({"Feature": feat_names, "Importance": importances})
                .sort_values(by="Importance", ascending=False)
                .head(10)
            )
            fig_fi, ax_fi = plt.subplots(figsize=(7.5, 4))
            sns.barplot(data=feat_df, x="Importance", y="Feature", ax=ax_fi)
            ax_fi.set_title("Top 10 Feature Importances")
            plt.tight_layout()
            st.pyplot(fig_fi)
            plt.close(fig_fi)
        else:
            st.info(
                "Jumlah nama fitur hasil transformasi tidak sesuai dengan panjang "
                "array `feature_importances_` pada konfigurasi saat ini."
            )
    elif hasattr(model, "coef_"):
        coef_arr = np.asarray(model.coef_)
        feat_names = _extract_feature_names(pipeline, active_num, active_cat)
        mean_abs_coef = np.mean(np.abs(coef_arr), axis=0) if coef_arr.ndim > 1 else np.abs(coef_arr.ravel())
        if len(feat_names) == len(mean_abs_coef):
            coef_df = (
                pd.DataFrame(
                    {"Feature": feat_names, "Mean |Coefficient|": mean_abs_coef}
                )
                .sort_values(by="Mean |Coefficient|", ascending=False)
                .head(10)
            )
            st.dataframe(coef_df, use_container_width=True)
        else:
            st.info("Koefisien model tersedia, tetapi dimensi nama fitur tidak selaras untuk ditampilkan.")
    else:
        st.info("Feature importance is not directly available for this model.")


def evaluate_regression(y_true, y_pred):
    st.markdown("---")
    st.subheader("📊 Hasil Evaluasi Regresi (Unseen Test Set)")

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{mae:.4f}")
    c2.metric("MSE", f"{mse:.4f}")
    c3.metric("RMSE", f"{rmse:.4f}")
    c4.metric("R² Score", f"{r2:.4f}")

    col_a, col_b = st.columns(2)
    with col_a:
        fig_sc, ax_sc = plt.subplots(figsize=(5.5, 4))
        ax_sc.scatter(y_true, y_pred, alpha=0.5)
        min_v = min(float(np.min(y_true)), float(np.min(y_pred)))
        max_v = max(float(np.max(y_true)), float(np.max(y_pred)))
        ax_sc.plot([min_v, max_v], [min_v, max_v], "r--")
        ax_sc.set_xlabel("Actual Values")
        ax_sc.set_ylabel("Predicted Values")
        ax_sc.set_title("Actual vs Predicted")
        plt.tight_layout()
        st.pyplot(fig_sc)
        plt.close(fig_sc)

    with col_b:
        residuals = np.asarray(y_true) - np.asarray(y_pred)
        fig_res, ax_res = plt.subplots(figsize=(5.5, 4))
        sns.histplot(residuals, kde=True, ax=ax_res)
        ax_res.set_xlabel("Residual (Actual - Predicted)")
        ax_res.set_title("Distribusi Residual")
        plt.tight_layout()
        st.pyplot(fig_res)
        plt.close(fig_res)