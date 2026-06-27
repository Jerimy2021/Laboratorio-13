"""
FastAPI — House Price Prediction API
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import HousePredictionRequest, PredictionResponse, BatchPredictionResponse, HealthResponse
from src.api import inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH        = os.getenv("MODEL_PATH",        "models/trained/house_price_model.pkl")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "models/trained/preprocessor.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    inference.load_artifacts(MODEL_PATH, PREPROCESSOR_PATH)
    yield


app = FastAPI(
    title="House Price Prediction API",
    description="MLOps pipeline — RandomForest model",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "healthy", "model_loaded": inference.is_loaded()}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: HousePredictionRequest):
    if not inference.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return inference.predict(request)


@app.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(requests: List[HousePredictionRequest]):
    if not inference.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"predictions": [inference.predict(r) for r in requests]}
