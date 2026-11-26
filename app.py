import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from modules.data_loader import load_data, get_column_info, data_quality_check
from modules.eda import show_overview, show_eda
from modules.modeling import train_evaluate_model, compare_models, get_model

st.set_page_config(page_title="EdTech ML Testbed", page_icon="🎓", layout="wide")
st.title("🎓 Educational Technology Learning Analytics Testbed")
st.markdown("Sebuah Data Mining Testbed interaktif untuk mengeksplorasi, memproses, dan memodelkan data pendidikan.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("1. Upload & Setup")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("👋 Selamat datang! Silakan upload dataset (contoh: edtech_students.csv) di sidebar untuk memulai.")
    st.stop()

df = load_data(uploaded_file)
if df is None or df.empty:
    st.error("Gagal membaca dataset atau dataset kosong.")
    st.stop()

id_cols, num_cols, cat_cols = get_column_info(df)

st.sidebar.header("2. Target & Task")
valid_targets = [c for c in df.columns if c not in id_cols]
target_col = st.sidebar.selectbox("Select Target Variable", valid_targets, index=valid_targets.index('motivation_level') if 'motivation_level' in valid_targets else 0)

# Deteksi tipe task
if target_col in cat_cols or df[target_col].nunique() < 20:
    default_task = "Classification"
else:
    default_task = "Regression"
task_type = st.sidebar.radio("Task Type", ["Classification", "Regression"], index=0 if default_task == "Classification" else 1)

st.sidebar.header("3. Preprocessing Config")
num_imputer = st.sidebar.selectbox("Missing Value (Num)", ["mean", "median", "most_frequent"])
cat_imputer = st.sidebar.selectbox("Missing Value (Cat)", ["most_frequent", "constant"])
scaler_type = st.sidebar.selectbox("Scaling", ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"])
encoder_type = st.sidebar.selectbox("Encoding", ["OneHotEncoder", "OrdinalEncoder"])

st.sidebar.header("4. Train-Test Split")
test_size = st.sidebar.slider("Test Size (%)", 20, 40, 20) / 100
random_state = st.sidebar.number_input("Random State", value=42)

st.sidebar.header("5. Cross Validation")
use_cv = st.sidebar.checkbox("Enable Cross Validation", value=True)
cv_folds = st.sidebar.slider("Number of Folds", 3, 10, 5) if use_cv else 5

# --- MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview & Quality", "📈 Exploratory Data Analysis", "⚙️ Modeling & Evaluation", "🏆 Compare Models"])

features = [c for c in df.columns if c not in id_cols and c != target_col]
X = df[features]
y = df[target_col]

with tab1:
    st.subheader("📖 Dataset Context (Business Understanding)")
    
    # Deteksi otomatis cerita berdasarkan nama file yang di-upload
    if uploaded_file.name == 'edtech_students.csv':
        default_story = """Dataset ini merekam data historis dari 8.000 siswa di sebuah platform Educational Technology (EdTech). Data yang dikumpulkan mencakup profil demografis (usia, tingkat pendidikan), preferensi teknis (penggunaan perangkat, zona waktu), gaya belajar, hingga riwayat kursus online sebelumnya. 

Tujuan Analisis (Data Mining Goal): 
Mengekstraksi pola dari karakteristik siswa untuk memprediksi tingkat motivasi belajar mereka (Atribut Kelas: motivation_level). Insight dari model klasifikasi ini akan digunakan oleh pihak akademik/platform untuk memberikan intervensi dini (seperti notifikasi penyemangat atau konseling) kepada siswa yang diprediksi memiliki motivasi "Low", guna mencegah risiko putus belajar (dropout)."""
    else:
        default_story = "Tuliskan cerita, latar belakang, atau tujuan analisis (Business Understanding) dari dataset yang Anda gunakan di sini..."
        
    # Text area agar cerita bisa diedit langsung saat presentasi
    dataset_story = st.text_area("Deskripsi/Cerita Dataset (Dapat diedit):", value=default_story, height=180)
    
    st.markdown("---")
    show_overview(df)
    data_quality_check(df, id_cols, target_col)

with tab2:
    show_eda(df, num_cols, cat_cols, target_col)

with tab3:
    st.header("Pipeline & Evaluation")
    st.info("🛡️ **Data Leakage Protection Active**: Preprocessing (Scaling, Encoding, Imputation) dibangun menggunakan `scikit-learn Pipeline`. Seluruh transformasi hanya mempelajari (fit) dari data training untuk mencegah kebocoran informasi dari data testing.")
    
    col1, col2 = st.columns(2)
    with col1:
        if task_type == "Classification":
            model_name = st.selectbox("Pilih Model", ["Logistic Regression", "Decision Tree", "Random Forest", "K-Nearest Neighbors", "Naive Bayes", "Support Vector Machine"])
        else:
            model_name = st.selectbox("Pilih Model", ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "KNN Regressor", "Support Vector Regression"])
    
    with col2:
        model = get_model(model_name, task_type)
        
    if st.button("🚀 Train & Evaluate Model", type="primary"):
        with st.spinner("Training model, please wait..."):
            train_evaluate_model(X, y, features, num_cols, cat_cols, num_imputer, cat_imputer, scaler_type, encoder_type, model, task_type, test_size, random_state, use_cv, cv_folds)

with tab4:
    st.header("Compare Multiple Models")
    if task_type == "Classification":
        models_to_compare = st.multiselect("Pilih Model untuk Dibandingkan", 
                                           ["Logistic Regression", "Decision Tree", "Random Forest", "K-Nearest Neighbors", "Naive Bayes"], 
                                           default=["Decision Tree", "Random Forest", "Logistic Regression"])
    else:
        models_to_compare = st.multiselect("Pilih Model untuk Dibandingkan", 
                                           ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"], 
                                           default=["Linear Regression", "Random Forest Regressor"])
    
    if st.button("🏁 Run Model Comparison", type="primary"):
        with st.spinner("Running benchmarks..."):
            compare_models(X, y, models_to_compare, task_type, features, num_cols, cat_cols, num_imputer, cat_imputer, scaler_type, encoder_type, test_size, random_state)