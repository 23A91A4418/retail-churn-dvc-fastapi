import json
import os
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from src.api.schemas import CustomerFeatures, PredictionResponse, BatchPredictionResponse

class ModelService:
    def __init__(
        self,
        model_path: str = "models/model.joblib",
        preprocessor_path: str = "models/preprocessor.joblib",
        metrics_path: str = "results/metrics.json",
        threshold: float = 0.5
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.metrics_path = metrics_path
        self.threshold = threshold
        
        self.model = None
        self.scaler = None
        self.feature_cols = [
            "recency_days", "frequency", "monetary_total", "avg_order_value",
            "total_items_purchased", "unique_products", "cancellation_rate",
            "tenure_days", "is_uk"
        ]
        self.metrics: Dict[str, Any] = {}
        self.load_artifacts()
        
    def load_artifacts(self) -> None:
        """Loads trained model, scaler, and metrics from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        if os.path.exists(self.preprocessor_path):
            preproc = joblib.load(self.preprocessor_path)
            self.scaler = preproc.get("scaler")
            self.feature_cols = preproc.get("feature_cols", self.feature_cols)
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)
                
    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.scaler is not None

    def _features_to_df(self, features: CustomerFeatures) -> pd.DataFrame:
        data = {
            "recency_days": [features.recency_days],
            "frequency": [features.frequency],
            "monetary_total": [features.monetary_total],
            "avg_order_value": [features.avg_order_value],
            "total_items_purchased": [features.total_items_purchased],
            "unique_products": [features.unique_products],
            "cancellation_rate": [features.cancellation_rate],
            "tenure_days": [features.tenure_days],
            "is_uk": [features.is_uk]
        }
        return pd.DataFrame(data)[self.feature_cols]

    def _batch_to_df(self, features_list: List[CustomerFeatures]) -> pd.DataFrame:
        data = [
            {
                "recency_days": f.recency_days,
                "frequency": f.frequency,
                "monetary_total": f.monetary_total,
                "avg_order_value": f.avg_order_value,
                "total_items_purchased": f.total_items_purchased,
                "unique_products": f.unique_products,
                "cancellation_rate": f.cancellation_rate,
                "tenure_days": f.tenure_days,
                "is_uk": f.is_uk
            }
            for f in features_list
        ]
        return pd.DataFrame(data)[self.feature_cols]

    def _generate_recommendations(self, features: CustomerFeatures, prob: float, risk_level: str) -> List[str]:
        recommendations = []
        if risk_level == "High":
            recommendations.append("Immediate Win-Back Campaign: Offer a limited-time 20% discount on their top category.")
            if features.recency_days > 60:
                recommendations.append(f"High Inactivity ({features.recency_days:.0f} days): Send personalized 'We Miss You' re-engagement email.")
            if features.monetary_total > 500:
                recommendations.append("High-Value VIP Alert: Assign proactive support outreach to resolve pain points.")
        elif risk_level == "Medium":
            recommendations.append("Re-engagement Workflow: Send curated recommendations matching previous browsing and purchase history.")
            if features.cancellation_rate > 0.15:
                recommendations.append("Check Cancellation Cause: Inquire about product satisfaction or delivery delays.")
            else:
                recommendations.append("Loyalty Incentive: Provide free shipping on their next order to nudge conversion.")
        else:
            recommendations.append("Healthy Customer: Maintain standard nurture sequence and early access to product launches.")
            if features.monetary_total > 300:
                recommendations.append("Upsell Opportunity: Invite customer to join the Tier 1 VIP Loyalty Rewards Program.")
                
        return recommendations

    def predict_single(self, features: CustomerFeatures) -> PredictionResponse:
        if not self.is_ready:
            raise RuntimeError("Model or preprocessor artifacts are not loaded.")
            
        df = self._features_to_df(features)
        X_scaled = self.scaler.transform(df)
        prob = float(self.model.predict_proba(X_scaled)[0, 1])
        pred = 1 if prob >= self.threshold else 0
        
        if prob >= 0.65:
            risk_level = "High"
        elif prob >= 0.40:
            risk_level = "Medium"
        else:
            risk_level = "Low"
            
        recommendations = self._generate_recommendations(features, prob, risk_level)
        
        return PredictionResponse(
            customer_id=features.customer_id,
            churn_prediction=pred,
            churn_probability=round(prob, 4),
            risk_level=risk_level,
            recommendations=recommendations
        )

    def predict_batch(self, features_list: List[CustomerFeatures]) -> BatchPredictionResponse:
        if not self.is_ready:
            raise RuntimeError("Model or preprocessor artifacts are not loaded.")
            
        df = self._batch_to_df(features_list)
        X_scaled = self.scaler.transform(df)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        
        predictions: List[PredictionResponse] = []
        churn_count = 0
        
        for idx, (features, prob) in enumerate(zip(features_list, probs)):
            p = float(prob)
            pred = 1 if p >= self.threshold else 0
            if pred == 1:
                churn_count += 1
                
            if p >= 0.65:
                risk_level = "High"
            elif p >= 0.40:
                risk_level = "Medium"
            else:
                risk_level = "Low"
                
            recommendations = self._generate_recommendations(features, p, risk_level)
            predictions.append(PredictionResponse(
                customer_id=features.customer_id,
                churn_prediction=pred,
                churn_probability=round(p, 4),
                risk_level=risk_level,
                recommendations=recommendations
            ))
            
        total = len(features_list)
        return BatchPredictionResponse(
            predictions=predictions,
            total_evaluated=total,
            churn_count=churn_count,
            churn_rate=round(churn_count / total, 4) if total > 0 else 0.0
        )
