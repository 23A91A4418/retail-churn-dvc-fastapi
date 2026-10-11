import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "version" in data

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["algorithm"] == "XGBClassifier"
    assert "feature_names" in data
    assert len(data["feature_names"]) == 9

def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "roc_auc" in data
    assert "f1_score" in data
    assert "accuracy" in data

def test_predict_single_success(client):
    payload = {
        "customer_id": 99901,
        "recency_days": 120.0,
        "frequency": 1,
        "monetary_total": 45.0,
        "avg_order_value": 45.0,
        "total_items_purchased": 5,
        "unique_products": 2,
        "cancellation_rate": 0.1,
        "tenure_days": 130.0,
        "is_uk": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == 99901
    assert data["churn_prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ["Low", "Medium", "High"]
    assert isinstance(data["recommendations"], list)

def test_predict_single_validation_error(client):
    # recency_days cannot be negative
    payload = {
        "customer_id": 99902,
        "recency_days": -10.0,
        "frequency": 2,
        "monetary_total": 100.0,
        "avg_order_value": 50.0,
        "total_items_purchased": 10,
        "unique_products": 3,
        "cancellation_rate": 1.5,  # must be <= 1.0
        "tenure_days": 20.0,
        "is_uk": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_batch_success(client):
    payload = {
        "customers": [
            {
                "customer_id": 101,
                "recency_days": 10.0,
                "frequency": 8,
                "monetary_total": 1200.0,
                "avg_order_value": 150.0,
                "total_items_purchased": 180,
                "unique_products": 30,
                "cancellation_rate": 0.0,
                "tenure_days": 240.0,
                "is_uk": 1
            },
            {
                "customer_id": 102,
                "recency_days": 180.0,
                "frequency": 1,
                "monetary_total": 25.0,
                "avg_order_value": 25.0,
                "total_items_purchased": 2,
                "unique_products": 1,
                "cancellation_rate": 0.5,
                "tenure_days": 185.0,
                "is_uk": 0
            }
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_evaluated"] == 2
    assert len(data["predictions"]) == 2
    assert 0.0 <= data["churn_rate"] <= 1.0
