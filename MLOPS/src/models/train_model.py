"""
Model Training Script — MLOps Pipeline
"""

import argparse
import logging
import os
import numpy as np
import pandas as pd
import joblib
import yaml

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and register final model from config.")
    parser.add_argument("--config",              type=str, required=True)
    parser.add_argument("--data",                type=str, required=True)
    parser.add_argument("--models-dir",          type=str, required=True)
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    return parser.parse_args()


def get_model_instance(name, params):
    model_map = {
        "LinearRegression":     LinearRegression,
        "RandomForest":         RandomForestRegressor,
        "GradientBoosting":     GradientBoostingRegressor,
        "HistGradientBoosting": HistGradientBoostingRegressor,
    }
    if name not in model_map:
        raise ValueError(f"Unsupported model: {name}")
    return model_map[name](**params)


def load_and_split_data(data_path, config):
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    selected = config["model"]["feature_sets"]["rfe"]
    logger.info(f"Using {len(selected)} selected features")
    X = data[selected]
    y = data[config["model"]["target_variable"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Data split: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    logger.info("Training model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "mae":  float(mean_absolute_error(y_test, y_pred)),
        "r2":   float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_test - y_pred) ** 2))),
    }
    logger.info(f"R²: {metrics['r2']:.4f} | MAE: {metrics['mae']:.2f} | RMSE: {metrics['rmse']:.2f}")
    return model, metrics


def save_artifacts(model, config, metrics, models_dir):
    os.makedirs(os.path.join(models_dir, "trained"), exist_ok=True)
    model_path  = os.path.join(models_dir, "trained", "house_price_model.pkl")
    config_path = os.path.join(models_dir, "trained", "house_price_model.yaml")
    joblib.dump(model, model_path)
    config["model"]["final_metrics"] = metrics
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Config saved to {config_path}")


def register_in_mlflow(model, config, metrics, tracking_uri):
    if not tracking_uri:
        logger.info("No MLflow URI provided — skipping MLflow logging")
        return None
    import mlflow, mlflow.sklearn
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("House Price Prediction - Production")
    model_name = config["model"]["name"]
    with mlflow.start_run(run_name="production_training"):
        mlflow.log_params(config["model"]["parameters"])
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model",
                                 registered_model_name=model_name)
        if metrics["r2"] > 0.7:
            client   = MlflowClient()
            versions = client.get_latest_versions(model_name, stages=["None"])
            if versions:
                client.transition_model_version_stage(
                    name=model_name, version=versions[0].version, stage="Production")
                logger.info("Model promoted to Production in MLflow")
        logger.info(f"MLflow run_id: {mlflow.active_run().info.run_id}")


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    X_train, X_test, y_train, y_test = load_and_split_data(args.data, config)
    model = get_model_instance(config["model"]["best_model"], config["model"]["parameters"])
    logger.info(f"Model: {config['model']['best_model']}")
    trained_model, metrics = train_and_evaluate(model, X_train, y_train, X_test, y_test)
    save_artifacts(trained_model, config, metrics, args.models_dir)
    register_in_mlflow(trained_model, config, metrics, args.mlflow_tracking_uri)
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
