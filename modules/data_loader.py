import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        # Handle tipe boolean menjadi string agar lebih mudah di-encode oleh pipeline
        for col in df.select_dtypes(include=['bool']).columns:
            df[col] = df[col].astype(str)
        return df
    except Exception as e:
        return None

def get_column_info(df):
    # Deteksi identifier (kolom dimana semua baris bernilai unik)
    id_cols = [col for col in df.columns if df[col].nunique() == len(df) or 'id' in col.lower()]
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(id_cols, errors='ignore').tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.drop(id_cols, errors='ignore').tolist()
    return id_cols, num_cols, cat_cols

def data_quality_check(df, id_cols, target_col):
    st.subheader("🛠️ Automatic Data Quality Check")
    issues = 0
    
    if df.isnull().sum().sum() > 0:
        st.warning(f"⚠️ Ditemukan {df.isnull().sum().sum()} missing values di dataset. Sistem akan melakukan imputasi.")
        issues += 1
        
    if df.duplicated().sum() > 0:
        st.warning(f"⚠️ Ditemukan {df.duplicated().sum()} baris duplikat.")
        issues += 1
        
    if len(id_cols) > 0:
        st.info(f"ℹ️ Kolom identifier terdeteksi dan akan diabaikan dari fitur ML: {', '.join(id_cols)}")
        
    class_counts = df[target_col].value_counts(normalize=True)
    if class_counts.min() < 0.05 and df[target_col].dtype == 'object':
        st.warning(f"⚠️ Imbalance Data: Class '{class_counts.idxmin()}' di target '{target_col}' hanya memiliki {class_counts.min()*100:.1f}% data.")
        issues += 1

    if issues == 0:
        st.success("✅ Kualitas data tampak sangat baik! Tidak ditemukan missing values atau masalah kritis.")