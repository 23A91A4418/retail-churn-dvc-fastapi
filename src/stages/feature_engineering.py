import argparse
import datetime
import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger
from src.utils.config import load_config

def calculate_customer_features(df_obs: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """Computes RFM and behavioral metrics for customers up to cutoff_date."""
    # Ensure TotalPrice is available
    df_obs = df_obs.copy()
    df_obs["TotalPrice"] = df_obs["Quantity"] * df_obs["UnitPrice"]
    
    # Identify cancelled invoices
    df_obs["IsCancelled"] = df_obs["InvoiceNo"].astype(str).str.startswith("C")
    
    # Filter purchases (non-cancelled and positive quantity)
    purchases = df_obs[~df_obs["IsCancelled"] & (df_obs["Quantity"] > 0)].copy()
    
    # Aggregations per customer
    features_list = []
    
    for customer_id, group in df_obs.groupby("CustomerID"):
        cust_purchases = purchases[purchases["CustomerID"] == customer_id]
        
        # Tenure: days between first transaction and cutoff
        first_date = group["InvoiceDate"].min()
        tenure_days = max(0.0, (cutoff_date - first_date).total_seconds() / 86400.0)
        
        # Cancellations
        total_invoices = group["InvoiceNo"].nunique()
        cancelled_invoices = group[group["IsCancelled"]]["InvoiceNo"].nunique()
        cancellation_rate = (cancelled_invoices / total_invoices) if total_invoices > 0 else 0.0
        
        # Country
        country = group["Country"].iloc[0] if len(group) > 0 else "United Kingdom"
        is_uk = 1 if country == "United Kingdom" else 0
        
        if len(cust_purchases) == 0:
            # Customer only had cancellations or invalid purchases
            features_list.append({
                "CustomerID": int(customer_id),
                "recency_days": tenure_days,
                "frequency": 0,
                "monetary_total": 0.0,
                "avg_order_value": 0.0,
                "total_items_purchased": 0,
                "unique_products": 0,
                "cancellation_rate": float(cancellation_rate),
                "tenure_days": float(tenure_days),
                "is_uk": is_uk
            })
            continue
            
        # Recency: days between last purchase and cutoff
        last_date = cust_purchases["InvoiceDate"].max()
        recency_days = max(0.0, (cutoff_date - last_date).total_seconds() / 86400.0)
        
        # Frequency: distinct purchase invoices
        frequency = cust_purchases["InvoiceNo"].nunique()
        
        # Monetary total
        monetary_total = max(0.0, float(cust_purchases["TotalPrice"].sum()))
        
        # Average order value
        avg_order_value = monetary_total / frequency if frequency > 0 else 0.0
        
        # Total items
        total_items = int(cust_purchases["Quantity"].sum())
        
        # Unique products
        unique_products = int(cust_purchases["StockCode"].nunique())
        
        features_list.append({
            "CustomerID": int(customer_id),
            "recency_days": float(round(recency_days, 2)),
            "frequency": int(frequency),
            "monetary_total": float(round(monetary_total, 2)),
            "avg_order_value": float(round(avg_order_value, 2)),
            "total_items_purchased": int(total_items),
            "unique_products": int(unique_products),
            "cancellation_rate": float(round(cancellation_rate, 4)),
            "tenure_days": float(round(tenure_days, 2)),
            "is_uk": int(is_uk)
        })
        
    return pd.DataFrame(features_list)

def run_feature_engineering(config_path: str = "params.yaml") -> tuple:
    """Executes feature engineering and creates train/test datasets."""
    config = load_config(config_path)
    logger = get_logger("feature_engineering", level=config.get("base", {}).get("log_level", "INFO"))
    
    raw_path = config.get("data_ingestion", {}).get("raw_data_path", "data/raw/online_retail.csv")
    fe_cfg = config.get("feature_engineering", {})
    cutoff_days = fe_cfg.get("cutoff_days_before_end", 90)
    test_size = fe_cfg.get("test_size", 0.25)
    random_state = config.get("base", {}).get("random_state", 42)
    processed_dir = fe_cfg.get("processed_data_dir", "data/processed")
    train_path = fe_cfg.get("train_path", "data/processed/train.csv")
    test_path = fe_cfg.get("test_path", "data/processed/test.csv")
    info_path = fe_cfg.get("features_info_path", "data/processed/features_info.json")
    
    logger.info(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    # 1. Cleaning
    df = df.dropna(subset=["CustomerID"]).copy()
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    
    max_date = df["InvoiceDate"].max()
    cutoff_date = max_date - datetime.timedelta(days=cutoff_days)
    logger.info(f"Dataset date range: {df['InvoiceDate'].min()} to {max_date}")
    logger.info(f"Cutoff date for observation window: {cutoff_date} ({cutoff_days} days before end)")
    
    # 2. Split into observation and prediction periods
    df_obs = df[df["InvoiceDate"] <= cutoff_date].copy()
    df_future = df[df["InvoiceDate"] > cutoff_date].copy()
    
    obs_customers = set(df_obs["CustomerID"].unique())
    logger.info(f"Customers active in observation window: {len(obs_customers)}")
    
    # 3. Target labeling: Did customer make any valid purchases in prediction window?
    future_purchases = df_future[(~df_future["InvoiceNo"].astype(str).str.startswith("C")) & (df_future["Quantity"] > 0)]
    active_future_customers = set(future_purchases["CustomerID"].unique())
    
    # 4. Generate features on observation window
    logger.info("Computing customer RFM and behavioral features...")
    features_df = calculate_customer_features(df_obs, cutoff_date)
    
    # Target: churn = 1 if NOT in active_future_customers, else 0
    features_df["churn"] = features_df["CustomerID"].apply(lambda cid: 1 if cid not in active_future_customers else 0)
    
    churn_count = int(features_df["churn"].sum())
    total_cust = len(features_df)
    churn_rate = float(churn_count / total_cust) if total_cust > 0 else 0.0
    logger.info(f"Feature dataset created with {total_cust} customers. Churn rate: {churn_rate:.2%} ({churn_count}/{total_cust})")
    
    # 5. Stratified Train / Test Split
    feature_cols = [
        "recency_days", "frequency", "monetary_total", "avg_order_value",
        "total_items_purchased", "unique_products", "cancellation_rate",
        "tenure_days", "is_uk"
    ]
    
    train_df, test_df = train_test_split(
        features_df,
        test_size=test_size,
        random_state=random_state,
        stratify=features_df["churn"]
    )
    
    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    features_info = {
        "cutoff_date": str(cutoff_date),
        "cutoff_days_before_end": cutoff_days,
        "total_customers": total_cust,
        "churn_rate": round(churn_rate, 4),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "feature_columns": feature_cols,
        "target_column": "churn",
        "feature_summary": {
            col: {
                "mean": float(round(features_df[col].mean(), 2)),
                "std": float(round(features_df[col].std(), 2)),
                "min": float(round(features_df[col].min(), 2)),
                "max": float(round(features_df[col].max(), 2)),
            } for col in feature_cols
        }
    }
    
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(features_info, f, indent=2)
        
    logger.info(f"Saved processed train data to {train_path} ({len(train_df)} rows).")
    logger.info(f"Saved processed test data to {test_path} ({len(test_df)} rows).")
    logger.info(f"Saved features metadata to {info_path}.")
    
    return train_path, test_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RFM feature engineering for customer churn")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    run_feature_engineering(args.config)
