import argparse
import io
import os
import sys
import urllib.request
import zipfile

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import load_config

def ingest_data(config_path: str = "params.yaml") -> str:
    """Downloads and extracts the UCI Online Retail dataset."""
    config = load_config(config_path)
    logger = get_logger("data_ingestion", level=config.get("base", {}).get("log_level", "INFO"))
    
    ingest_cfg = config.get("data_ingestion", {})
    dataset_url = ingest_cfg.get("dataset_url", "https://archive.ics.uci.edu/static/public/352/online+retail.zip")
    raw_dir = ingest_cfg.get("raw_data_dir", "data/raw")
    raw_csv_path = ingest_cfg.get("raw_data_path", os.path.join(raw_dir, "online_retail.csv"))
    
    os.makedirs(raw_dir, exist_ok=True)
    
    if os.path.exists(raw_csv_path) and os.path.getsize(raw_csv_path) > 1024 * 1024:
        logger.info(f"Raw data already exists at {raw_csv_path} ({os.path.getsize(raw_csv_path):,} bytes). Skipping download.")
        return raw_csv_path
        
    logger.info(f"Downloading dataset from: {dataset_url}")
    req = urllib.request.Request(dataset_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    with urllib.request.urlopen(req, timeout=60) as response:
        zip_content = response.read()
        logger.info(f"Downloaded zip archive ({len(zip_content):,} bytes).")
        
    with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
        excel_files = [f for f in z.namelist() if f.endswith(".xlsx") or f.endswith(".xls")]
        if not excel_files:
            raise FileNotFoundError(f"No Excel file found inside archive: {z.namelist()}")
        excel_filename = excel_files[0]
        logger.info(f"Extracting '{excel_filename}' from zip...")
        excel_bytes = z.read(excel_filename)
        
    logger.info(f"Parsing Excel file into DataFrame (this may take a few seconds)...")
    df = pd.read_excel(io.BytesIO(excel_bytes), engine="openpyxl")
    logger.info(f"Loaded DataFrame with shape: {df.shape}")
    
    logger.info(f"Saving raw dataset to {raw_csv_path}...")
    df.to_csv(raw_csv_path, index=False)
    logger.info(f"Successfully saved {raw_csv_path} ({os.path.getsize(raw_csv_path):,} bytes).")
    
    return raw_csv_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Online Retail dataset")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    ingest_data(args.config)
