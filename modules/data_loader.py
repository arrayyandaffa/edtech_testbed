import pandas as pd
import streamlit as st


@st.cache_data
def load_data(file):
    if file is None:
        return None
    try:
        df = pd.read_csv(file)
        if df.empty:
            return None
        for col in df.select_dtypes(include=["bool"]).columns:
            df[col] = df[col].astype(str)
        return df
    except UnicodeDecodeError:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding="latin-1")
            if df.empty:
                return None
            for col in df.select_dtypes(include=["bool"]).columns:
                df[col] = df[col].astype(str)
            return df
        except Exception:
            return None
    except Exception:
        return None


def get_column_info(df):
    id_cols = [
        col
        for col in df.columns
        if df[col].nunique(dropna=True) == len(df)
        or col.lower() == "id"
        or col.lower().endswith("_id")
    ]
    num_cols = (
        df.select_dtypes(include=["int64", "float64", "int32", "float32"])
        .columns.drop(id_cols, errors="ignore")
        .tolist()
    )
    cat_cols = (
        df.select_dtypes(include=["object", "category", "string"])
        .columns.drop(id_cols, errors="ignore")
        .tolist()
    )
    return id_cols, num_cols, cat_cols


def data_quality_check(df, id_cols, target_col, task_type="Classification"):
    st.subheader("🛠️ Automatic Data Quality & Provenance Check")
    issues = 0

    total_missing = int(df.isnull().sum().sum())
    if total_missing > 0:
        st.warning(
            f"⚠️ Ditemukan {total_missing} missing values pada dataset. "
            "Imputasi akan dijalankan di dalam Pipeline setelah train-test split."
        )
        issues += 1

    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        st.warning(f"⚠️ Ditemukan {dup_count} baris duplikat pada dataset.")
        issues += 1

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        st.warning(
            f"⚠️ Kolom konstan terdeteksi (hanya 1 nilai): {', '.join(constant_cols)}."
        )
        issues += 1

    if id_cols:
        st.info(
            f"ℹ️ Kolom identifier terdeteksi dan dikecualikan dari fitur prediktor: "
            f"{', '.join(id_cols)}."
        )

    if task_type == "Classification" and target_col in df.columns:
        prop = df[target_col].value_counts(normalize=True, dropna=True)
        if not prop.empty:
            min_class = prop.idxmin()
            min_share = prop.min() * 100
            max_class = prop.idxmax()
            max_share = prop.max() * 100
            if min_share < 10.0:
                st.warning(
                    f"⚠️ Ketimpangan kelas pada target `{target_col}`: kelas `{min_class}` "
                    f"hanya {min_share:.2f}%, sedangkan kelas mayoritas `{max_class}` mencapai "
                    f"{max_share:.2f}%. Perhatikan metrik Macro F1 selain Accuracy."
                )
                issues += 1
            else:
                st.info(
                    f"ℹ️ Distribusi kelas target `{target_col}`: mayoritas `{max_class}` "
                    f"({max_share:.2f}%) dan minoritas `{min_class}` ({min_share:.2f}%)."
                )

    st.caption(
        "Catatan Audit: Selalu verifikasi asal pengumpulan data (*provenance*) dan relevansi "
        "fitur terhadap domain sebelum menarik kesimpulan kausal atau intervensi nyata."
    )

    if issues == 0:
        st.success(
            "✅ Tidak ditemukan missing value, baris duplikat, kolom konstan, maupun "
            "ketimpangan kelas ekstrem pada pemeriksaan awal."
        )