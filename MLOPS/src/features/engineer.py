"""
Feature Engineering Script — MLOps Pipeline
"""

import argparse
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("feature-engineering")

NUMERICAL_FEATURES   = ["sqft", "bedrooms", "bathrooms", "house_age", "price_per_sqft", "bed_bath_ratio"]
CATEGORICAL_FEATURES = ["location", "condition"]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating new features")
    df = df.copy()
    df["house_age"]      = datetime.now().year - df["year_built"]
    df["price_per_sqft"] = df["price"] / df["sqft"]
    df["bed_bath_ratio"] = (df["bedrooms"] / df["bathrooms"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    logger.info(f"Created featured dataset with shape: {df.shape}")
    return df


def create_preprocessor() -> ColumnTransformer:
    logger.info("Creating preprocessor pipeline")
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="mean"))]),  NUMERICAL_FEATURES),
        ("cat", Pipeline([("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_FEATURES),
    ])


def run_feature_engineering(input_file: str, output_file: str, preprocessor_file: str):
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    df_featured   = create_features(df)
    preprocessor  = create_preprocessor()

    X = df_featured.drop(columns=["price"], errors="ignore")
    y = df_featured["price"] if "price" in df_featured.columns else None

    X_transformed = preprocessor.fit_transform(X)
    logger.info("Fitted the preprocessor and transformed the features")

    cat_names = list(preprocessor.named_transformers_["cat"]["onehot"]
                     .get_feature_names_out(CATEGORICAL_FEATURES))
    col_names = NUMERICAL_FEATURES + cat_names
    logger.info(f"Output columns ({len(col_names)}): {col_names}")

    os.makedirs(os.path.dirname(preprocessor_file), exist_ok=True)
    joblib.dump(preprocessor, preprocessor_file)
    logger.info(f"Saved preprocessor to {preprocessor_file}")

    df_out = pd.DataFrame(X_transformed, columns=col_names)
    if y is not None:
        df_out["price"] = y.values
    df_out.to_csv(output_file, index=False)
    logger.info(f"Saved featured data to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Engineering — MLOps")
    parser.add_argument("--input",        required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--preprocessor", required=True)
    args = parser.parse_args()
    run_feature_engineering(args.input, args.output, args.preprocessor)
