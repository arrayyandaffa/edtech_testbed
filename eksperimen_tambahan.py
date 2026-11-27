"""
eksperimen_tambahan.py
Tiga eksperimen tambahan untuk paper klasifikasi motivasi belajar siswa.

  E1. Baseline + validasi silang berulang   -> apakah model benar-benar lebih baik dari menebak?
  E2. Hubungan fitur dengan target          -> apakah fitur punya hubungan dengan motivasi?
  E3. Demo data leakage                     -> seberapa besar skor naik kalau pipeline dilanggar?

Cara pakai:
    python eksperimen_tambahan.py --data edtech_students.csv
Hasil disimpan di folder hasil_tambahan/ (file .csv) dan dicetak di layar.

Catatan: samakan fungsi buat_model() dengan get_default_model() di aplikasi Anda.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

TARGET = "motivation_level"
NUMERIK = ["age", "previous_online_courses"]
KATEGORI = ["education_level", "study_field", "learning_style",
            "device_preference", "timezone", "has_learning_disability"]
SEED = 42


def buat_preprocessor():
    """Pra-pemrosesan yang sama dengan di paper (Bagian 3.2)."""
    num = Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler())])
    kat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, NUMERIK), ("kat", kat, KATEGORI)])


def buat_model():
    """GANTI agar sama dengan get_default_model() di aplikasi Anda."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(random_state=SEED),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
    }


def pipeline(model):
    return Pipeline([("prep", buat_preprocessor()), ("model", model)])


# ---------------------------------------------------------------- E1
def eksperimen_1(X_latih, y_latih, ulang):
    print("\n=== E1: Baseline + validasi silang berulang ===")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=ulang, random_state=SEED)
    semua = {"Baseline: ZeroR": DummyClassifier(strategy="most_frequent"),
             "Baseline: tebakan acak": DummyClassifier(strategy="stratified", random_state=SEED)}
    semua.update(buat_model())
    baris = []
    for nama, model in semua.items():
        hasil = cross_validate(pipeline(model), X_latih, y_latih, cv=cv,
                               scoring=["accuracy", "balanced_accuracy", "f1_macro"])
        baris.append({
            "model": nama,
            "akurasi": f"{hasil['test_accuracy'].mean():.4f} ± {hasil['test_accuracy'].std():.4f}",
            "balanced_accuracy": f"{hasil['test_balanced_accuracy'].mean():.4f} ± {hasil['test_balanced_accuracy'].std():.4f}",
            "macro_f1": f"{hasil['test_f1_macro'].mean():.4f} ± {hasil['test_f1_macro'].std():.4f}",
        })
    tabel = pd.DataFrame(baris)
    print(tabel.to_string(index=False))
    return tabel


# ---------------------------------------------------------------- E2
def cramers_v(a, b):
    tab = pd.crosstab(a, b)
    chi2, p, _, _ = stats.chi2_contingency(tab, correction=False)
    n = tab.values.sum()
    v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
    return chi2, p, v


def eksperimen_2(df):
    print("\n=== E2: Hubungan fitur dengan motivation_level ===")
    y = df[TARGET].astype(str)
    baris = []
    for kol in KATEGORI:
        chi2, p, v = cramers_v(df[kol].astype(str), y)
        baris.append({"fitur": kol, "uji": "chi-square", "p_value": p, "ukuran_efek": v})
    for kol in NUMERIK:
        kelompok = [df.loc[y == k, kol] for k in y.unique()]
        h, p = stats.kruskal(*kelompok)
        baris.append({"fitur": kol, "uji": "Kruskal-Wallis", "p_value": p, "ukuran_efek": np.nan})
    tabel = pd.DataFrame(baris)
    tabel["p_bonferroni"] = (tabel["p_value"] * len(tabel)).clip(upper=1.0)  # koreksi sederhana
    print(tabel.round(4).to_string(index=False))

    # cek kombinasi usia - pendidikan yang tidak lazim
    n = len(df)
    edu = df["education_level"].astype(str)
    cek = []
    for nama, m_usia, m_edu in [
        ("usia <= 17 & Graduate/Professional", df["age"] <= 17, edu.isin(["Graduate", "Professional"])),
        ("usia >= 60 & High School", df["age"] >= 60, edu.eq("High School")),
    ]:
        cek.append({"kombinasi": nama, "teramati": int((m_usia & m_edu).sum()),
                    "harapan_jika_independen": round(float(m_usia.mean() * m_edu.mean() * n), 1)})
    cek = pd.DataFrame(cek)
    print("\nKombinasi tidak lazim (teramati vs harapan):")
    print(cek.to_string(index=False))
    dup = int(df.drop(columns=["student_id"]).duplicated().sum())
    print(f"Baris duplikat (tanpa kolom student_id): {dup}")
    return tabel, cek


# ---------------------------------------------------------------- E3
def duplikasi_minoritas(X, y, rng):
    """Oversampling sederhana: gandakan baris kelas kecil sampai sama dengan kelas terbesar."""
    y = np.asarray(y)
    kelas, jumlah = np.unique(y, return_counts=True)
    idx = []
    for k, j in zip(kelas, jumlah):
        i = np.where(y == k)[0]
        tambahan = rng.choice(i, size=jumlah.max() - j, replace=True)
        idx.append(np.concatenate([i, tambahan]))
    idx = np.concatenate(idx)
    return X[idx], y[idx]


def eksperimen_3(X_latih, y_latih, ulang):
    print("\n=== E3: Demo data leakage (oversampling sebelum vs sesudah pemisahan) ===")
    rng = np.random.default_rng(SEED)
    y_arr = np.asarray(y_latih)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=ulang, random_state=SEED)
    baris = []
    for nama in ["KNN", "Random Forest"]:
        bersih, bocor = [], []
        # BOCOR: data digandakan dulu, baru dibagi ke lipatan -> salinan sama ada di latih dan validasi
        prep_semua = buat_preprocessor().fit(X_latih)
        X_all, y_all = duplikasi_minoritas(prep_semua.transform(X_latih), y_arr, rng)
        for tr, va in cv.split(X_all, y_all):
            m = buat_model()[nama].fit(X_all[tr], y_all[tr])
            bocor.append(balanced_accuracy_score(y_all[va], m.predict(X_all[va])))
        # BERSIH: bagi dulu, gandakan hanya data latih tiap lipatan
        for tr, va in cv.split(X_latih, y_latih):
            prep = buat_preprocessor().fit(X_latih.iloc[tr])
            Xt, yt = duplikasi_minoritas(prep.transform(X_latih.iloc[tr]), y_arr[tr], rng)
            m = buat_model()[nama].fit(Xt, yt)
            bersih.append(balanced_accuracy_score(y_arr[va], m.predict(prep.transform(X_latih.iloc[va]))))
        baris.append({"model": nama,
                      "bersih (benar)": f"{np.mean(bersih):.4f} ± {np.std(bersih):.4f}",
                      "bocor (salah)": f"{np.mean(bocor):.4f} ± {np.std(bocor):.4f}"})
    tabel = pd.DataFrame(baris)
    print(tabel.to_string(index=False))
    print("Nilai tebakan acak untuk balanced accuracy 3 kelas = 0,3333")
    return tabel


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--keluar", default="hasil_tambahan")
    ap.add_argument("--ulang", type=int, default=5, help="jumlah pengulangan 5-fold CV")
    args = ap.parse_args()
    os.makedirs(args.keluar, exist_ok=True)

    df = pd.read_csv(args.data)
    df["has_learning_disability"] = df["has_learning_disability"].astype(str)
    X, y = df[NUMERIK + KATEGORI], df[TARGET].astype(str)
    X_latih, X_uji, y_latih, y_uji = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    print(f"Data latih: {len(X_latih)} | Data uji: {len(X_uji)} (data uji tidak dipakai di sini)")

    eksperimen_1(X_latih, y_latih, args.ulang).to_csv(f"{args.keluar}/e1_baseline_cv.csv", index=False)
    t2, cek = eksperimen_2(df)
    t2.to_csv(f"{args.keluar}/e2_fitur_target.csv", index=False)
    cek.to_csv(f"{args.keluar}/e2_kombinasi_tak_lazim.csv", index=False)
    eksperimen_3(X_latih, y_latih, args.ulang).to_csv(f"{args.keluar}/e3_leakage.csv", index=False)
    print(f"\nSelesai. File hasil ada di folder: {args.keluar}/")


if __name__ == "__main__":
    main()
