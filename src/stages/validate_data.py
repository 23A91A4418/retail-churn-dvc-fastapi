import argparse
import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import load_config

def validate_data(config_path: str = "params.yaml") -> dict:
    """Validates raw data schema, missing values, and data integrity."""
    config = load_config(config_path)
    logger = get_logger("data_validation", level=config.get("base", {}).get("log_level", "INFO"))
    
    val_cfg = config.get("data_validation", {})
    raw_csv_path = config.get("data_ingestion", {}).get("raw_data_path", "data/raw/online_retail.csv")
    report_path = val_cfg.get("report_path", "results/validation_report.json")
    required_cols = val_cfg.get("required_columns", [])
    
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_csv_path}")
        
    logger.info(f"Loading raw data from {raw_csv_path} for validation...")
    df = pd.read_csv(raw_csv_path)
    
    # 1. Check required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")
        
    # 2. Check shapes and nulls
    total_rows = len(df)
    null_customer_ids = int(df["CustomerID"].isna().sum())
    valid_customer_rows = total_rows - null_customer_ids
    
    # 3. Check dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    min_date = str(df["InvoiceDate"].min())
    max_date = str(df["InvoiceDate"].max())
    
    # 4. Check cancellations & negative quantities
    cancellations = int((df["InvoiceNo"].astype(str).str.startswith("C")).sum())
    negative_quantities = int((df["Quantity"] < 0).sum())
    zero_or_neg_price = int((df["UnitPrice"] <= 0).sum())
    
    unique_customers = int(df["CustomerID"].dropna().nunique())
    unique_countries = int(df["Country"].nunique())
    unique_products = int(df["StockCode"].nunique())
    
    report = {
        "dataset_path": raw_csv_path,
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_customer_ids": null_customer_ids,
        "valid_customer_rows": valid_customer_rows,
        "unique_customers": unique_customers,
        "unique_countries": unique_countries,
        "unique_products": unique_products,
        "date_range": {
            "start": min_date,
            "end": max_date
        },
        "cancellation_count": cancellations,
        "negative_quantities": negative_quantities,
        "zero_or_negative_unit_price": zero_or_neg_price,
        "validation_passed": True
    }
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Data validation passed successfully. Report saved to {report_path}.")
    logger.info(f"Total rows: {total_rows:,}, Unique customers: {unique_customers:,}, Date range: {min_date} to {max_date}")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Online Retail dataset")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    validate_data(args.config)
