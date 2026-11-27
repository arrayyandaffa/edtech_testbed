from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


def build_preprocessor(
    active_num, active_cat, num_strategy, cat_strategy, scaler_type, encoder_type
):
    transformers = []

    if active_num:
        num_steps = [("imputer", SimpleImputer(strategy=num_strategy))]
        if scaler_type == "StandardScaler":
            num_steps.append(("scaler", StandardScaler()))
        elif scaler_type == "MinMaxScaler":
            num_steps.append(("scaler", MinMaxScaler()))
        elif scaler_type == "RobustScaler":
            num_steps.append(("scaler", RobustScaler()))
        transformers.append(("num", Pipeline(steps=num_steps), active_num))

    if active_cat:
        fill_val = "Missing" if cat_strategy == "constant" else None
        cat_imputer = SimpleImputer(strategy=cat_strategy, fill_value=fill_val)
        if encoder_type == "OneHotEncoder":
            cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            cat_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
        cat_pipeline = Pipeline(
            steps=[("imputer", cat_imputer), ("encoder", cat_encoder)]
        )
        transformers.append(("cat", cat_pipeline, active_cat))

    return ColumnTransformer(transformers=transformers, remainder="drop")