from pydantic import BaseModel, Field
from typing import Dict, List


class HousePredictionRequest(BaseModel):
    sqft:           float = Field(..., gt=0,  example=1527)
    bedrooms:       int   = Field(..., ge=1,  example=3)
    bathrooms:      float = Field(..., ge=1,  example=2.0)
    location:       str   = Field(...,        example="Suburbs")
    year_built:     int   = Field(...,        example=1990)
    condition:      str   = Field(...,        example="Good")
    price_per_sqft: float = Field(0.0, ge=0, example=200.0)


class PredictionResponse(BaseModel):
    predicted_price:     float
    confidence_interval: List[float]
    features_importance: Dict[str, float]
    prediction_time:     str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
