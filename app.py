import warnings
import streamlit as st

warnings.filterwarnings("ignore")

from modules.data_loader import data_quality_check, get_column_info, load_data
from modules.eda import show_eda, show_overview
from modules.modeling import compare_models, get_model, train_evaluate_model

st.set_page_config(
    page_title="EdTech Learning Analytics Testbed",
    page_icon="🎓",
    layout="wide",
)
st.title("🎓 Educational Technology Learning Analytics Testbed")
st.markdown(
    "Mini data-mining testbed end-to-end untuk eksplorasi data pendidikan, "
    "pra-pemrosesan bebas *data leakage*, pelatihan model, dan evaluasi komparatif."
)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("1. Upload & Setup")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info(
        "👋 Selamat datang! Silakan unggah file CSV (contoh: `edtech_students.csv`) "
        "pada sidebar di sebelah kiri untuk memulai analisis dan pemodelan."
    )
    st.markdown(
        """
        ### Alur Kerja Testbed
        1. **Upload Dataset & Overview**: Periksa struktur data, konteks/cerita dataset, dan kualitas data.
        2. **Exploratory Data Analysis (EDA)**: Amati distribusi atribut kelas, fitur prediktor, dan korelasi numerik.
        3. **Preprocessing dalam Pipeline**: *Train-test split* dilakukan terlebih dahulu, lalu *imputation*, *scaling*, dan *encoding* dipelajari (`fit`) hanya pada data latih.
        4. **Modeling, Evaluation & Comparison**: Uji satu model dengan *cross-validation* atau bandingkan beberapa model sekaligus secara objektif.
        """
    )
    st.stop()

df = load_data(uploaded_file)
if df is None or df.empty:
    st.error("Gagal membaca file CSV atau dataset kosong. Pastikan format file valid.")
    st.stop()

if df.shape[1] < 2:
    st.error("Dataset harus memiliki minimal 2 kolom (fitur prediktor dan atribut kelas/target).")
    st.stop()

id_cols, num_cols, cat_cols = get_column_info(df)

st.sidebar.header("2. Target / Atribut Kelas & Task")
valid_targets = [c for c in df.columns if c not in id_cols]
if not valid_targets:
    st.error("Tidak ada kolom target valid setelah mengecualikan kolom identifier.")
    st.stop()

default_target_idx = (
    valid_targets.index("motivation_level") if "motivation_level" in valid_targets else 0
)
target_col = st.sidebar.selectbox(
    "Select Target Variable (Atribut Kelas)",
    valid_targets,
    index=default_target_idx,
)

unique_target_count = df[target_col].nunique(dropna=True)
if unique_target_count < 2:
    st.sidebar.error("Kolom target yang dipilih hanya memiliki kurang dari 2 nilai unik.")
    st.stop()

if target_col in cat_cols or unique_target_count < 20:
    default_task = "Classification"
else:
    default_task = "Regression"

task_type = st.sidebar.radio(
    "Task Type",
    ["Classification", "Regression"],
    index=0 if default_task == "Classification" else 1,
)

st.sidebar.header("3. Preprocessing Config")
num_imputer = st.sidebar.selectbox(
    "Missing Value (Numerical)", ["mean", "median", "most_frequent"]
)
cat_imputer = st.sidebar.selectbox(
    "Missing Value (Categorical)", ["most_frequent", "constant"]
)
scaler_type = st.sidebar.selectbox(
    "Numerical Scaling", ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"]
)
encoder_type = st.sidebar.selectbox(
    "Categorical Encoding", ["OneHotEncoder", "OrdinalEncoder"]
)

st.sidebar.header("4. Train-Test Split")
test_size = st.sidebar.slider("Test Size (%)", min_value=20, max_value=40, value=20) / 100.0
random_state = int(st.sidebar.number_input("Random State", value=42, step=1))

st.sidebar.header("5. Cross Validation")
use_cv = st.sidebar.checkbox("Enable Cross Validation", value=True)
cv_folds = (
    st.sidebar.slider("Number of Folds", min_value=3, max_value=10, value=5)
    if use_cv
    else 5
)

# --- MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Overview & Quality",
        "📈 Exploratory Data Analysis",
        "⚙️ Modeling & Evaluation",
        "🏆 Compare Models",
    ]
)

features = [c for c in df.columns if c not in id_cols and c != target_col]
if not features:
    st.error("Tidak ada fitur prediktor tersisa setelah mengecualikan identifier dan target.")
    st.stop()

X = df[features]
y = df[target_col]

with tab1:
    st.subheader("📖 Dataset Context & Provenance (Business & Data Understanding)")

    if uploaded_file.name == "edtech_students.csv":
        default_story = (
            "Dataset `edtech_students.csv` memuat 8.000 baris dan 10 kolom yang merepresentasikan "
            "karakteristik profil dan preferensi belajar siswa pada konteks Educational Technology "
            "(usia, tingkat pendidikan, bidang studi, gaya belajar, preferensi perangkat, zona waktu, "
            "jumlah kursus daring sebelumnya, status disabilitas belajar, serta `motivation_level`).\n\n"
            "Tujuan Analisis: Mengklasifikasikan atribut kelas `motivation_level` (Low, Medium, High) "
            "melalui alur pipeline bebas data leakage dan mengevaluasi daya pisah fitur secara objektif.\n\n"
            "Catatan Provenance: Berkas tidak memuat metadata asal pengumpulan data primer dan "
            "menunjukkan indikasi data sintetis/simulasi pada atribut profil statis (tanpa log aktivitas "
            "belajar harian LMS), sehingga hasil model perlu ditafsirkan secara kritis dan tidak "
            "digunakan sebagai satu-satunya dasar keputusan intervensi siswa."
        )
    else:
        default_story = (
            f"Tuliskan latar belakang, sumber/asal data, tujuan analisis, dan keterbatasan dataset "
            f"`{uploaded_file.name}` di sini..."
        )

    st.text_area(
        "Cerita / Konteks Dataset (Dapat diedit langsung saat presentasi):",
        value=default_story,
        height=170,
    )

    st.markdown("---")
    show_overview(df, id_cols, num_cols, cat_cols, target_col, task_type)
    st.markdown("---")
    data_quality_check(df, id_cols, target_col, task_type)

with tab2:
    show_eda(df, num_cols, cat_cols, target_col)

with tab3:
    st.header("Pipeline & Single-Model Evaluation")
    st.info(
        "🛡️ **Data Leakage Protection Active**: *Train-test split* dilakukan terlebih dahulu, "
        "lalu `ColumnTransformer` (`SimpleImputer`, `Scaling`, `Encoding`) digabungkan dengan model "
        "di dalam `scikit-learn Pipeline`. Seluruh parameter transformasi hanya di-`fit` pada "
        "data training (termasuk di tiap lipatan *Cross-Validation*)."
    )

    col1, col2 = st.columns(2)
    with col1:
        if task_type == "Classification":
            model_name = st.selectbox(
                "Pilih Model Klasifikasi",
                [
                    "Logistic Regression",
                    "Decision Tree",
                    "Random Forest",
                    "K-Nearest Neighbors",
                    "Naive Bayes",
                    "Support Vector Machine",
                ],
            )
        else:
            model_name = st.selectbox(
                "Pilih Model Regresi",
                [
                    "Linear Regression",
                    "Decision Tree Regressor",
                    "Random Forest Regressor",
                    "KNN Regressor",
                    "Support Vector Regression",
                ],
            )

    with col2:
        model = get_model(model_name, task_type, random_state)

    if st.button("🚀 Train & Evaluate Model", type="primary"):
        with st.spinner("Melatih pipeline dan mengevaluasi model..."):
            train_evaluate_model(
                X=X,
                y=y,
                features=features,
                num_cols=num_cols,
                cat_cols=cat_cols,
                num_imputer=num_imputer,
                cat_imputer=cat_imputer,
                scaler_type=scaler_type,
                encoder_type=encoder_type,
                model=model,
                task_type=task_type,
                test_size=test_size,
                random_state=random_state,
                use_cv=use_cv,
                cv_folds=cv_folds,
            )

with tab4:
    st.header("Compare Multiple Models")
    st.caption(
        "Seluruh model diuji pada pembagian train-test dan konfigurasi preprocessing yang identik."
    )

    if task_type == "Classification":
        models_to_compare = st.multiselect(
            "Pilih Model untuk Dibandingkan",
            [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "K-Nearest Neighbors",
                "Naive Bayes",
                "Support Vector Machine",
            ],
            default=[
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "K-Nearest Neighbors",
                "Naive Bayes",
            ],
        )
    else:
        models_to_compare = st.multiselect(
            "Pilih Model untuk Dibandingkan",
            [
                "Linear Regression",
                "Decision Tree Regressor",
                "Random Forest Regressor",
                "KNN Regressor",
                "Support Vector Regression",
            ],
            default=[
                "Linear Regression",
                "Decision Tree Regressor",
                "Random Forest Regressor",
            ],
        )

    if st.button("🏁 Run Model Comparison", type="primary"):
        if not models_to_compare:
            st.warning("Pilih minimal satu model untuk dibandingkan.")
        else:
            with st.spinner("Menjalankan benchmark antar-model..."):
                compare_models(
                    X=X,
                    y=y,
                    model_names=models_to_compare,
                    task_type=task_type,
                    features=features,
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                    num_imputer=num_imputer,
                    cat_imputer=cat_imputer,
                    scaler_type=scaler_type,
                    encoder_type=encoder_type,
                    test_size=test_size,
                    random_state=random_state,
                    use_cv=use_cv,
                    cv_folds=cv_folds,
                )