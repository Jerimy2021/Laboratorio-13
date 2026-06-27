"""
Inference module — carga modelo/preprocessor y expone predict().
"""

import logging
import os
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

NUMERICAL_FEATURES   = ["sqft", "bedrooms", "bathrooms", "house_age", "price_per_sqft", "bed_bath_ratio"]
CATEGORICAL_FEATURES = ["location", "condition"]
SELECTED_FEATURES    = [
    "sqft", "house_age", "bedrooms", "bathrooms",
    "price_per_sqft", "bed_bath_ratio",
    "location_Suburbs", "location_Downtown",
    "condition_Good", "condition_Excellent",
]

_model        = None
_preprocessor = None


def load_artifacts(model_path: str, preprocessor_path: str):
    global _model, _preprocessor
    _model        = joblib.load(model_path)
    _preprocessor = joblib.load(preprocessor_path)
    logger.info(f"Model loaded: {type(_model).__name__}")
    logger.info("Preprocessor loaded successfully")


def is_loaded() -> bool:
    return _model is not None and _preprocessor is not None


def predict(req) -> dict:
    house_age      = datetime.now().year - req.year_built
    bed_bath_ratio = req.bedrooms / req.bathrooms if req.bathrooms > 0 else 0.0

    raw = pd.DataFrame([{
        "sqft":           req.sqft,
        "bedrooms":       req.bedrooms,
        "bathrooms":      req.bathrooms,
        "house_age":      house_age,
        "price_per_sqft": req.price_per_sqft,
        "bed_bath_ratio": bed_bath_ratio,
        "location":       req.location,
        "condition":      req.condition,
    }])

    X_transformed = _preprocessor.transform(raw)
    cat_names = list(_preprocessor.named_transformers_["cat"]["onehot"]
                     .get_feature_names_out(CATEGORICAL_FEATURES))
    col_names = NUMERICAL_FEATURES + cat_names
    X_df = pd.DataFrame(X_transformed, columns=col_names)

    for col in SELECTED_FEATURES:
        if col not in X_df.columns:
            X_df[col] = 0.0
    X_model = X_df[SELECTED_FEATURES]

    price  = float(_model.predict(X_model)[0])
    margin = price * 0.10
    return {
        "predicted_price":     round(price, 2),
        "confidence_interval": [round(price - margin, 2), round(price + margin, 2)],
        "features_importance": {},
        "prediction_time":     datetime.utcnow().isoformat(),
    }
