import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


def show_overview(df, id_cols=None, num_cols=None, cat_cols=None, target_col=None, task_type=None):
    id_cols = id_cols or []
    num_cols = num_cols or []
    cat_cols = cat_cols or []

    st.subheader("Dataset Overview")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Rows", int(df.shape[0]))
    c2.metric("Columns", int(df.shape[1]))
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicate Rows", int(df.duplicated().sum()))
    c5.metric("Numerical Features", len(num_cols))
    c6.metric("Categorical Features", len(cat_cols))

    if target_col and task_type:
        st.caption(
            f"Target aktif: `{target_col}` | Tipe task terdeteksi: **{task_type}** | "
            f"Identifier dikecualikan: `{', '.join(id_cols) if id_cols else 'Tidak ada'}`"
        )

    st.write("### Raw Data Sample")
    st.dataframe(df.head(10), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("### Data Types Summary")
        type_df = pd.DataFrame(df.dtypes, columns=["Data Type"]).astype(str)
        st.dataframe(type_df, use_container_width=True)

    with col_b:
        st.write("### Missing & Unique Summary")
        summary_df = pd.DataFrame(
            {
                "Missing": df.isnull().sum(),
                "Unique": df.nunique(dropna=True),
            }
        )
        st.dataframe(summary_df, use_container_width=True)


def show_eda(df, num_cols, cat_cols, target_col):
    st.subheader("Exploratory Data Analysis (EDA)")

    st.write(f"### 1. Distribusi Target / Atribut Kelas: `{target_col}`")
    fig_target, ax_target = plt.subplots(figsize=(8, 3.8))
    if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 15:
        sns.histplot(df[target_col].dropna(), kde=True, ax=ax_target)
        ax_target.set_ylabel("Frequency")
    else:
        order = df[target_col].value_counts(dropna=False).index
        sns.countplot(x=df[target_col].astype(str), order=order.astype(str), ax=ax_target)
        ax_target.set_ylabel("Count")
        for container in ax_target.containers:
            ax_target.bar_label(container, padding=2)
    ax_target.set_xlabel(target_col)
    ax_target.set_title(f"Distribusi {target_col}")
    plt.tight_layout()
    st.pyplot(fig_target)
    plt.close(fig_target)

    if not pd.api.types.is_numeric_dtype(df[target_col]) or df[target_col].nunique() <= 15:
        dist_df = pd.DataFrame(
            {
                "Count": df[target_col].value_counts(dropna=False),
                "Proportion (%)": (
                    df[target_col].value_counts(normalize=True, dropna=False) * 100
                ).round(2),
            }
        )
        st.dataframe(dist_df, use_container_width=True)

    st.markdown("---")
    st.write("### 2. Eksplorasi Fitur Individual")
    candidate_feats = [c for c in df.columns if c != target_col]
    feat = st.selectbox("Pilih fitur untuk divisualisasikan:", candidate_feats)

    col_plot, col_stat = st.columns([2, 1])
    with col_plot:
        fig_feat, ax_feat = plt.subplots(figsize=(7, 4))
        if feat in num_cols:
            chart_mode = st.radio(
                "Jenis plot fitur numerik:", ["Histogram", "Boxplot"], horizontal=True
            )
            if chart_mode == "Histogram":
                sns.histplot(df[feat].dropna(), kde=True, ax=ax_feat)
            else:
                sns.boxplot(x=df[feat].dropna(), ax=ax_feat)
            ax_feat.set_xlabel(feat)
            ax_feat.set_title(f"{chart_mode} - {feat}")
        else:
            order_f = df[feat].value_counts(dropna=False).index.astype(str)[:15]
            sns.countplot(y=df[feat].astype(str), order=order_f, ax=ax_feat)
            ax_feat.set_xlabel("Count")
            ax_feat.set_ylabel(feat)
            ax_feat.set_title(f"Distribusi Kategori - {feat}")
        plt.tight_layout()
        st.pyplot(fig_feat)
        plt.close(fig_feat)

    with col_stat:
        st.write("**Ringkasan Fitur:**")
        if feat in num_cols:
            st.dataframe(df[feat].describe(), use_container_width=True)
        else:
            vc_df = pd.DataFrame(
                {
                    "Count": df[feat].astype(str).value_counts(dropna=False),
                    "Share (%)": (
                        df[feat].astype(str).value_counts(normalize=True, dropna=False) * 100
                    ).round(2),
                }
            )
            st.dataframe(vc_df, use_container_width=True)

    if len(num_cols) >= 2:
        st.markdown("---")
        st.write("### 3. Correlation Matrix (Fitur Numerik)")
        fig_corr, ax_corr = plt.subplots(figsize=(6.5, 4.5))
        sns.heatmap(
            df[num_cols].corr(numeric_only=True),
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            fmt=".2f",
            ax=ax_corr,
        )
        ax_corr.set_title("Matriks Korelasi Pearson (Fitur Numerik)")
        plt.tight_layout()
        st.pyplot(fig_corr)
        plt.close(fig_corr)