import argparse
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils.logger import get_logger
from src.utils.config import load_config

def train_model(config_path: str = "params.yaml") -> tuple:
    """Trains XGBoost model and logs metrics & artifacts to MLflow."""
    config = load_config(config_path)
    logger = get_logger("model_training", level=config.get("base", {}).get("log_level", "INFO"))
    
    train_cfg = config.get("train", {})
    train_path = config.get("feature_engineering", {}).get("train_path", "data/processed/train.csv")
    model_dir = train_cfg.get("model_dir", "models")
    model_path = train_cfg.get("model_path", os.path.join(model_dir, "model.joblib"))
    preprocessor_path = train_cfg.get("preprocessor_path", os.path.join(model_dir, "preprocessor.joblib"))
    
    mlflow_uri = train_cfg.get("mlflow_tracking_uri", "mlruns")
    exp_name = train_cfg.get("experiment_name", "retail_churn_prediction")
    xgb_params = train_cfg.get("xgboost_params", {})
    
    logger.info(f"Loading training data from {train_path}...")
    df_train = pd.read_csv(train_path)
    
    feature_cols = [
        "recency_days", "frequency", "monetary_total", "avg_order_value",
        "total_items_purchased", "unique_products", "cancellation_rate",
        "tenure_days", "is_uk"
    ]
    
    X_train = df_train[feature_cols].copy()
    y_train = df_train["churn"].copy()
    
    logger.info(f"Training features shape: {X_train.shape}, target balance: {y_train.value_counts().to_dict()}")
    
    # Preprocessor
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # MLflow Setup
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(exp_name)
    
    logger.info("Initializing and training XGBoost Classifier...")
    model = XGBClassifier(**xgb_params)
    
    with mlflow.start_run(run_name="xgboost_churn_baseline") as run:
        # Log parameters
        mlflow.log_params(xgb_params)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("features", feature_cols)
        
        # Fit model
        model.fit(X_train_scaled, y_train)
        
        # Save artifacts locally
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, model_path)
        joblib.dump({
            "scaler": scaler,
            "feature_cols": feature_cols
        }, preprocessor_path)
        
        logger.info(f"Saved trained model to {model_path}")
        logger.info(f"Saved preprocessor to {preprocessor_path}")
        
        # Log artifacts to MLflow
        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")
        
        run_id = run.info.run_id
        logger.info(f"MLflow Run completed successfully with Run ID: {run_id}")
        
    return model_path, preprocessor_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost Churn Model")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    train_model(args.config)
