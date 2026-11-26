# Educational Technology Learning Analytics Testbed

## 1. Deskripsi & Tujuan
Aplikasi ini adalah sebuah **Mini Data-Mining Testbed End-to-End**. Sistem ini didesain secara khusus untuk mengeksplorasi, menganalisis, dan memodelkan dataset edukasi seperti interaksi pembelajaran siswa, tingkat motivasi, dan perilaku menggunakan pendekatan Machine Learning.

## 2. Dataset
Sistem dirancang adaptif terhadap dataset CSV apa pun, namun dioptimasi untuk dataset dengan studi kasus pendidikan (seperti `edtech_students.csv`). Fitur seperti identifier (`student_id`) secara otomatis diabaikan untuk mencegah bias prediktif.

## 3. Teknologi yang Digunakan
- **Python 3.9+**
- **Streamlit** (UI / Frontend)
- **Pandas & Numpy** (Data Manipulation)
- **Matplotlib & Seaborn** (Data Visualization)
- **Scikit-Learn** (Machine Learning & Pipelines)

## 4. Struktur Aplikasi
- **Overview & Quality:** Deteksi Missing Value, Duplicate, Imbalance Data.
- **EDA (Exploratory Data Analysis):** Distribusi target, Boxplot, Barchart, dan Korelasi.
- **Preprocessing:** Pipeline Scikit-Learn (Imputasi, Scaling, Encoding).
- **Modeling:** Hyperparameter tuning interaktif (Classification & Regression).
- **Evaluation:** Evaluasi metrik mutakhir tanpa Data Leakage.

## 5. Instalasi & Menjalankan Aplikasi
1. Install requirements:
   ```bash
   pip install -r requirements.txt