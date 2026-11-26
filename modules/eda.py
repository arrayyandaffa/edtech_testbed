import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def show_overview(df):
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Duplicate Rows", df.duplicated().sum())
    
    st.write("### Raw Data Sample")
    st.dataframe(df.head())
    
    colA, colB = st.columns(2)
    with colA:
        st.write("### Data Types Summary")
        type_df = pd.DataFrame(df.dtypes, columns=['Data Type']).astype(str)
        st.dataframe(type_df, use_container_width=True)
    with colB:
        st.write("### Missing & Unique Summary")
        summary_df = pd.DataFrame({'Missing': df.isnull().sum(), 'Unique': df.nunique()})
        st.dataframe(summary_df, use_container_width=True)

def show_eda(df, num_cols, cat_cols, target_col):
    st.subheader("Visualisasi & Distribusi Data")
    
    st.write(f"**Distribusi Target Variabel: `{target_col}`**")
    fig, ax = plt.subplots(figsize=(8, 4))
    if df[target_col].dtype in ['int64', 'float64']:
        sns.histplot(df[target_col], kde=True, ax=ax, color='teal')
    else:
        sns.countplot(x=df[target_col], ax=ax, palette='viridis')
    st.pyplot(fig)
    
    st.markdown("---")
    st.write("**Eksplorasi Fitur Prediktor**")
    feat = st.selectbox("Pilih Fitur yang ingin dianalisis:", [c for c in df.columns if c != target_col])
    
    col1, col2 = st.columns([2, 1])
    fig, ax = plt.subplots(figsize=(6,4))
    
    with col1:
        if feat in num_cols:
            sns.histplot(df[feat], kde=True, ax=ax, color='coral')
            st.pyplot(fig)
        elif feat in cat_cols or df[feat].dtype == 'object':
            sns.countplot(y=df[feat], ax=ax, palette='Set2')
            st.pyplot(fig)
            
    with col2:
        st.write("**Statistik Deskriptif:**")
        st.write(df[feat].describe() if feat in num_cols else df[feat].value_counts())

    st.markdown("---")
    if len(num_cols) > 1:
        st.write("**Correlation Matrix (Fitur Numerik)**")
        fig2, ax2 = plt.subplots(figsize=(8,5))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
        st.pyplot(fig2)