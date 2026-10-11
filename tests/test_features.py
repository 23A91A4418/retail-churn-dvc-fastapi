import datetime
import pandas as pd
import pytest
from src.stages.feature_engineering import calculate_customer_features

def test_calculate_customer_features_standard_customer():
    cutoff = pd.Timestamp("2011-09-10 12:00:00")
    data = {
        "CustomerID": [101, 101, 101],
        "InvoiceNo": ["5001", "5002", "C5003"],
        "InvoiceDate": [
            pd.Timestamp("2011-01-01 10:00:00"),
            pd.Timestamp("2011-08-10 12:00:00"),
            pd.Timestamp("2011-08-15 14:00:00")
        ],
        "Quantity": [10, 5, -2],
        "UnitPrice": [20.0, 10.0, 10.0],
        "StockCode": ["A1", "A2", "A2"],
        "Country": ["United Kingdom", "United Kingdom", "United Kingdom"]
    }
    df = pd.DataFrame(data)
    features_df = calculate_customer_features(df, cutoff)
    
    assert len(features_df) == 1
    row = features_df.iloc[0]
    assert row["CustomerID"] == 101
    assert row["frequency"] == 2
    assert row["monetary_total"] == 250.0  # 10*20 + 5*10
    assert row["avg_order_value"] == 125.0
    assert row["total_items_purchased"] == 15
    assert row["unique_products"] == 2
    assert row["is_uk"] == 1
    assert row["cancellation_rate"] == pytest.approx(1 / 3, 0.01)
    assert row["recency_days"] > 0
    assert row["tenure_days"] > row["recency_days"]

def test_calculate_customer_features_only_cancellations():
    cutoff = pd.Timestamp("2011-09-10 12:00:00")
    data = {
        "CustomerID": [102],
        "InvoiceNo": ["C9999"],
        "InvoiceDate": [pd.Timestamp("2011-05-01 10:00:00")],
        "Quantity": [-5],
        "UnitPrice": [15.0],
        "StockCode": ["B1"],
        "Country": ["France"]
    }
    df = pd.DataFrame(data)
    features_df = calculate_customer_features(df, cutoff)
    
    assert len(features_df) == 1
    row = features_df.iloc[0]
    assert row["CustomerID"] == 102
    assert row["frequency"] == 0
    assert row["monetary_total"] == 0.0
    assert row["total_items_purchased"] == 0
    assert row["is_uk"] == 0
    assert row["cancellation_rate"] == 1.0
