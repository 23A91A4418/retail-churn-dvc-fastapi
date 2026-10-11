import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    CustomerFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from src.api.service import ModelService

# Global service instance
model_service: ModelService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_service
    model_service = ModelService(
        model_path="models/model.joblib",
        preprocessor_path="models/preprocessor.joblib",
        metrics_path="results/metrics.json"
    )
    yield

app = FastAPI(
    title="Retail Customer Churn Prediction API",
    description="Production-grade Machine Learning API serving XGBoost customer churn predictions on UCI Online Retail transaction data, orchestrated with DVC.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", summary="Root Status & Metadata")
def root() -> Dict[str, Any]:
    return {
        "service": "Retail Customer Churn Prediction API",
        "status": "operational",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health",
        "prediction_endpoint": "/predict",
        "batch_prediction_endpoint": "/predict/batch",
        "metrics_endpoint": "/metrics"
    }

@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health_check() -> HealthResponse:
    is_ready = model_service is not None and model_service.is_ready
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        version="1.0.0",
        service="retail-churn-fastapi"
    )

@app.get("/model-info", response_model=ModelInfoResponse, summary="Model Metadata & Info")
def get_model_info() -> ModelInfoResponse:
    if not model_service or not model_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded."
        )
    return ModelInfoResponse(
        algorithm="XGBClassifier",
        feature_names=model_service.feature_cols,
        metrics=model_service.metrics,
        threshold=model_service.threshold
    )

@app.get("/metrics", summary="Model Evaluation Metrics")
def get_metrics() -> Dict[str, Any]:
    if not model_service or not model_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded."
        )
    return model_service.metrics

@app.post("/predict", response_model=PredictionResponse, summary="Predict Single Customer Churn")
def predict_churn(customer: CustomerFeatures) -> PredictionResponse:
    if not model_service or not model_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is currently unavailable."
        )
    try:
        return model_service.predict_single(customer)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse, summary="Predict Batch Customer Churn")
def predict_batch_churn(batch: BatchPredictionRequest) -> BatchPredictionResponse:
    if not model_service or not model_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is currently unavailable."
        )
    try:
        return model_service.predict_batch(batch.customers)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
