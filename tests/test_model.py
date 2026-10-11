import os
import pytest
from src.api.schemas import CustomerFeatures
from src.api.service import ModelService

def test_model_service_loading():
    service = ModelService(
        model_path="models/model.joblib",
        preprocessor_path="models/preprocessor.joblib",
        metrics_path="results/metrics.json"
    )
    assert service.is_ready is True
    assert service.model is not None
    assert service.scaler is not None
    assert len(service.feature_cols) == 9

def test_model_service_predict_single():
    service = ModelService()
    cust = CustomerFeatures(
        customer_id=12345,
        recency_days=45.0,
        frequency=4,
        monetary_total=520.0,
        avg_order_value=130.0,
        total_items_purchased=60,
        unique_products=15,
        cancellation_rate=0.02,
        tenure_days=200.0,
        is_uk=1
    )
    res = service.predict_single(cust)
    assert res.customer_id == 12345
    assert res.churn_prediction in [0, 1]
    assert 0.0 <= res.churn_probability <= 1.0
    assert res.risk_level in ["Low", "Medium", "High"]
    assert len(res.recommendations) >= 1

def test_model_service_predict_batch():
    service = ModelService()
    c1 = CustomerFeatures(
        customer_id=1,
        recency_days=150.0,
        frequency=1,
        monetary_total=30.0,
        avg_order_value=30.0,
        total_items_purchased=2,
        unique_products=1,
        cancellation_rate=0.0,
        tenure_days=150.0,
        is_uk=1
    )
    c2 = CustomerFeatures(
        customer_id=2,
        recency_days=5.0,
        frequency=12,
        monetary_total=2500.0,
        avg_order_value=208.33,
        total_items_purchased=350,
        unique_products=45,
        cancellation_rate=0.01,
        tenure_days=300.0,
        is_uk=1
    )
    batch_res = service.predict_batch([c1, c2])
    assert batch_res.total_evaluated == 2
    assert len(batch_res.predictions) == 2
    assert 0.0 <= batch_res.churn_rate <= 1.0
