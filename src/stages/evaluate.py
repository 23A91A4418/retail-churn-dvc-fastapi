import argparse
import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve
)

from src.utils.logger import get_logger
from src.utils.config import load_config

def evaluate_model(config_path: str = "params.yaml") -> dict:
    """Evaluates the trained model on the test dataset and generates metrics & plots."""
    config = load_config(config_path)
    logger = get_logger("model_evaluation", level=config.get("base", {}).get("log_level", "INFO"))
    
    eval_cfg = config.get("evaluate", {})
    test_path = config.get("feature_engineering", {}).get("test_path", "data/processed/test.csv")
    model_path = config.get("train", {}).get("model_path", "models/model.joblib")
    preprocessor_path = config.get("train", {}).get("preprocessor_path", "models/preprocessor.joblib")
    metrics_path = eval_cfg.get("metrics_path", "results/metrics.json")
    plots_dir = eval_cfg.get("plots_dir", "results/plots")
    threshold = eval_cfg.get("classification_threshold", 0.5)
    
    logger.info(f"Loading test data from {test_path}...")
    df_test = pd.read_csv(test_path)
    
    logger.info(f"Loading model from {model_path} and preprocessor from {preprocessor_path}...")
    model = joblib.load(model_path)
    preproc = joblib.load(preprocessor_path)
    
    scaler = preproc["scaler"]
    feature_cols = preproc["feature_cols"]
    
    X_test = df_test[feature_cols].copy()
    y_test = df_test["churn"].copy()
    
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    y_preds = (y_probs >= threshold).astype(int)
    
    # Calculate Metrics
    acc = accuracy_score(y_test, y_preds)
    prec = precision_score(y_test, y_preds, zero_division=0)
    rec = recall_score(y_test, y_preds, zero_division=0)
    f1 = f1_score(y_test, y_preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    cm = confusion_matrix(y_test, y_preds)
    
    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "test_samples": int(len(y_test)),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1])
        }
    }
    
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved evaluation metrics to {metrics_path}: {metrics}")
    
    # Plots
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Retained (0)", "Churned (1)"],
        yticklabels=["Retained (0)", "Churned (1)"]
    )
    plt.title(f"Confusion Matrix (Test Set, Threshold={threshold})")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_plot_path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    
    # 2. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2563EB", lw=2, label=f"XGBoost (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Customer Churn Prediction")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_plot_path = os.path.join(plots_dir, "roc_curve.png")
    plt.savefig(roc_plot_path, dpi=200)
    plt.close()
    
    # 3. Feature Importance Plot
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values(by="importance", ascending=False)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(data=feat_imp_df, x="importance", y="feature", palette="viridis")
    plt.title("XGBoost Feature Importances (Churn Prediction)")
    plt.xlabel("Relative Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    feat_plot_path = os.path.join(plots_dir, "feature_importance.png")
    plt.savefig(feat_plot_path, dpi=200)
    plt.close()
    
    logger.info(f"Saved evaluation plots to {plots_dir}")
    
    # Log to MLflow if available
    try:
        mlflow_uri = config.get("train", {}).get("mlflow_tracking_uri", "mlruns")
        exp_name = config.get("train", {}).get("experiment_name", "retail_churn_prediction")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(exp_name)
        with mlflow.start_run(run_name="xgboost_evaluation") as run:
            mlflow.log_metrics({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc
            })
            mlflow.log_artifact(metrics_path)
            mlflow.log_artifact(cm_plot_path)
            mlflow.log_artifact(roc_plot_path)
            mlflow.log_artifact(feat_plot_path)
    except Exception as e:
        logger.warning(f"Could not log evaluation to MLflow: {e}")
        
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate XGBoost Churn Model")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    evaluate_model(args.config)
