from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder

def build_preprocessor(active_num, active_cat, num_strategy, cat_strategy, scaler_type, encoder_type):
    # Preprocessing Numerik
    num_transformers = [('imputer', SimpleImputer(strategy=num_strategy))]
    if scaler_type == "StandardScaler":
        num_transformers.append(('scaler', StandardScaler()))
    elif scaler_type == "MinMaxScaler":
        num_transformers.append(('scaler', MinMaxScaler()))
    elif scaler_type == "RobustScaler":
        num_transformers.append(('scaler', RobustScaler()))
        
    num_pipeline = Pipeline(steps=num_transformers)
    
    # Preprocessing Kategorikal
    cat_transformers = [('imputer', SimpleImputer(strategy=cat_strategy))]
    if encoder_type == "OneHotEncoder":
        cat_transformers.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
    else:
        cat_transformers.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
        
    cat_pipeline = Pipeline(steps=cat_transformers)
    
    # Bundle dengan ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, active_num),
            ('cat', cat_pipeline, active_cat)
        ], remainder='drop'
    )
    return preprocessor