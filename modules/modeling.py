import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR

from modules.preprocessing import build_preprocessor
from modules.evaluation import evaluate_classification, evaluate_regression
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

def get_model(model_name, task_type):
    st.write(f"**Hyperparameter Config:** `{model_name}`")
    
    if model_name in ["Logistic Regression", "Linear Regression", "Naive Bayes"]:
        st.write("Model ini menggunakan default parameters.")
        if model_name == "Logistic Regression": return LogisticRegression(max_iter=1000)
        elif model_name == "Linear Regression": return LinearRegression()
        elif model_name == "Naive Bayes": return GaussianNB()
        
    elif "Decision Tree" in model_name:
        max_depth = st.slider("Max Depth", 1, 30, 10)
        min_samples = st.slider("Min Samples Split", 2, 20, 2)
        if task_type == "Classification": return DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples, random_state=42)
        else: return DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples, random_state=42)
        
    elif "Random Forest" in model_name:
        n_est = st.slider("N Estimators", 10, 200, 100)
        max_depth = st.slider("Max Depth", 1, 30, 10)
        if task_type == "Classification": return RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
        else: return RandomForestRegressor(n_estimators=n_est, max_depth=max_depth, random_state=42)
        
    elif "KNN" in model_name or "K-Nearest" in model_name:
        n_neighbors = st.slider("N Neighbors", 1, 20, 5)
        if task_type == "Classification": return KNeighborsClassifier(n_neighbors=n_neighbors)
        else: return KNeighborsRegressor(n_neighbors=n_neighbors)
        
    elif "Support Vector" in model_name:
        C = st.slider("C parameter (Regularization)", 0.1, 10.0, 1.0)
        if task_type == "Classification": return SVC(C=C, probability=True)
        else: return SVR(C=C)

def train_evaluate_model(X, y, features, num_cols, cat_cols, num_imputer, cat_imputer, scaler_type, encoder_type, model, task_type, test_size, random_state, use_cv, cv_folds):
    active_num = [c for c in num_cols if c in features]
    active_cat = [c for c in cat_cols if c in features]
    
    preprocessor = build_preprocessor(active_num, active_cat, num_imputer, cat_imputer, scaler_type, encoder_type)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    
    # Proteksi Leakage: Train test split terlebih dahulu
    stratify = y if task_type == "Classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    except:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
    st.success(f"✅ Data berhasil di-split! Training: {X_train.shape[0]} baris | Testing: {X_test.shape[0]} baris")
    
    if use_cv:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state) if task_type == "Classification" else KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scoring = 'f1_weighted' if task_type == "Classification" else 'r2'
        try:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring)
            st.info(f"**Cross Validation Score ({cv_folds}-Fold)**: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")
        except Exception as e:
            st.warning(f"Cross validation gagal dijalankan (mungkin karena parameter yang tidak kompatibel): {e}")

    # Final Training
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    if task_type == "Classification":
        evaluate_classification(y_test, y_pred, model, pipeline, active_num, active_cat)
    else:
        evaluate_regression(y_test, y_pred)

def get_default_model(m_name, task_type):
    if m_name == "Logistic Regression": return LogisticRegression(max_iter=500)
    if m_name == "Linear Regression": return LinearRegression()
    if m_name == "Decision Tree": return DecisionTreeClassifier(random_state=42)
    if m_name == "Decision Tree Regressor": return DecisionTreeRegressor(random_state=42)
    if m_name == "Random Forest": return RandomForestClassifier(random_state=42)
    if m_name == "Random Forest Regressor": return RandomForestRegressor(random_state=42)
    if m_name == "K-Nearest Neighbors": return KNeighborsClassifier()
    if m_name == "Naive Bayes": return GaussianNB()
    return DecisionTreeClassifier()

def compare_models(X, y, model_names, task_type, features, num_cols, cat_cols, num_imputer, cat_imputer, scaler_type, encoder_type, test_size, random_state):
    active_num = [c for c in num_cols if c in features]
    active_cat = [c for c in cat_cols if c in features]
    preprocessor = build_preprocessor(active_num, active_cat, num_imputer, cat_imputer, scaler_type, encoder_type)
    
    stratify = y if task_type == "Classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    except:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    results = []
    for m_name in model_names:
        model = get_default_model(m_name, task_type)
        pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
        
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            
            if task_type == "Classification":
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                results.append({"Model": m_name, "Accuracy": round(acc,4), "F1 Score": round(f1,4)})
            else:
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                results.append({"Model": m_name, "R² Score": round(r2,4), "RMSE": round(rmse,4)})
        except Exception as e:
            st.error(f"Error pada model {m_name}: {e}")
            
    if results:
        res_df = pd.DataFrame(results).sort_values(by="F1 Score" if task_type=="Classification" else "R² Score", ascending=False)
        
        # PERBAIKAN: Pisahkan subset kolom untuk di-highlight agar tidak memanggil kolom yang tidak ada
        subset_cols = ["Accuracy", "F1 Score"] if task_type == "Classification" else ["R² Score"]
        
        st.dataframe(res_df.style.highlight_max(axis=0, color='lightgreen', subset=subset_cols))
        st.success(f"🏆 Model Terbaik: **{res_df.iloc[0]['Model']}**")