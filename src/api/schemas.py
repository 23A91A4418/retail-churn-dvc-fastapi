from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

class CustomerFeatures(BaseModel):
    customer_id: Optional[int] = Field(None, description="Unique identifier for the customer", examples=[12345])
    recency_days: float = Field(..., ge=0, description="Days since last purchase", examples=[45.0])
    frequency: int = Field(..., ge=0, description="Number of historical purchase orders", examples=[3])
    monetary_total: float = Field(..., ge=0, description="Total monetary spend in GBP", examples=[450.50])
    avg_order_value: float = Field(..., ge=0, description="Average spend per order", examples=[150.17])
    total_items_purchased: int = Field(..., ge=0, description="Total quantity of items purchased", examples=[58])
    unique_products: int = Field(..., ge=0, description="Number of unique products bought", examples=[14])
    cancellation_rate: float = Field(..., ge=0.0, le=1.0, description="Proportion of cancelled orders", examples=[0.05])
    tenure_days: float = Field(..., ge=0, description="Days since customer's first purchase", examples=[180.0])
    is_uk: int = Field(1, ge=0, le=1, description="Binary flag indicating if customer is UK-based (1: UK, 0: Non-UK)", examples=[1])

    @field_validator("recency_days", "tenure_days")
    @classmethod
    def check_temporal_consistency(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Days cannot be negative")
        return v

class PredictionResponse(BaseModel):
    customer_id: Optional[int] = Field(None, description="Customer ID")
    churn_prediction: int = Field(..., description="Binary churn prediction (1: Churned, 0: Retained)")
    churn_probability: float = Field(..., description="Predicted probability of churning [0.0 - 1.0]")
    risk_level: str = Field(..., description="Churn risk category: High, Medium, or Low")
    recommendations: List[str] = Field(..., description="Actionable retention recommendations")

class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., min_length=1, description="List of customer profiles for evaluation")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="List of individual churn predictions")
    total_evaluated: int = Field(..., description="Total count of evaluated customers")
    churn_count: int = Field(..., description="Total count of predicted churners")
    churn_rate: float = Field(..., description="Predicted churn proportion across the batch")

class HealthResponse(BaseModel):
    status: str = Field(..., description="API operational status")
    model_loaded: bool = Field(..., description="Whether model weights are loaded in memory")
    version: str = Field(..., description="API semantic version")
    service: str = Field(..., description="Service identifier")

class ModelInfoResponse(BaseModel):
    algorithm: str = Field(..., description="Underlying ML algorithm")
    feature_names: List[str] = Field(..., description="Ordered input feature names")
    metrics: Dict[str, Any] = Field(..., description="Model evaluation metrics on holdout test set")
    threshold: float = Field(..., description="Probability threshold for positive churn label")
