import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_classification(y_true, y_pred, model, pipeline, active_num, active_cat):
    st.markdown("---")
    st.subheader("📊 Hasil Evaluasi Testing Set (Unseen Data)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", round(accuracy_score(y_true, y_pred), 4))
    col2.metric("Precision (Weighted)", round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4))
    col3.metric("Recall (Weighted)", round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4))
    col4.metric("F1 Score (Weighted)", round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4))
    
    colA, colB = st.columns(2)
    with colA:
        st.write("**Classification Report**")
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        
    with colB:
        st.write("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        st.pyplot(fig)
        
    # Feature Importance jika didukung model (Tree-based)
    if hasattr(model, 'feature_importances_'):
        st.write("**Feature Importances (Top 10)**")
        try:
            importances = model.feature_importances_
            # Mendapatkan nama fitur hasil preprocessor OHE
            feature_names = []
            if 'num' in pipeline.named_steps['preprocessor'].named_transformers_:
                feature_names.extend(active_num)
            if 'cat' in pipeline.named_steps['preprocessor'].named_transformers_:
                # Ambil fitur dari encoder
                encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
                if hasattr(encoder, 'get_feature_names_out'):
                    cat_names = encoder.get_feature_names_out(active_cat)
                    feature_names.extend(cat_names)
                else:
                    feature_names.extend(active_cat)
            
            if len(feature_names) == len(importances):
                feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
                
                fig2, ax2 = plt.subplots(figsize=(8,4))
                sns.barplot(data=feat_df, x='Importance', y='Feature', ax=ax2, palette="viridis")
                st.pyplot(fig2)
            else:
                st.info("Dimensi fitur tidak selaras untuk ditampilkan, karena jumlah transformasi encoding.")
        except Exception as e:
            st.info("Feature importance belum dapat divisualisasikan dengan konfigurasi preprocessor saat ini.")
    else:
        st.info("Model yang dipilih tidak menyediakan Feature Importances (bukan Tree-based / Algoritma koefisien).")

def evaluate_regression(y_true, y_pred):
    st.markdown("---")
    st.subheader("📊 Hasil Evaluasi Regresi")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", round(mean_absolute_error(y_true, y_pred), 4))
    col2.metric("MSE", round(mean_squared_error(y_true, y_pred), 4))
    col3.metric("RMSE", round(np.sqrt(mean_squared_error(y_true, y_pred)), 4))
    col4.metric("R² Score", round(r2_score(y_true, y_pred), 4))
    
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(y_true, y_pred, alpha=0.5, color='teal')
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)