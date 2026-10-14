"""FastAPI application for Product Return Prediction."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model import get_bundle, predict
from schemas import (
    FeatureImportanceItem,
    MetricItem,
    ModelInfoResponse,
    OrderInput,
    PredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly load the model on startup so the first request isn't slow."""
    try:
        get_bundle()
        print("[OK] Model loaded successfully")
    except Exception as exc:
        print(f"[WARN] Model load failed: {exc}")
    yield


app = FastAPI(
    title="Product Return Prediction API",
    description="Predict whether an e-commerce order will be returned before it ships.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Return model metadata, metrics, and feature importances for the dashboard."""
    try:
        bundle = get_bundle()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model not available: {exc}")

    metrics = [
        MetricItem(name=k, value=round(v, 4))
        for k, v in bundle["metrics"].items()
    ]

    feature_importance = [
        FeatureImportanceItem(feature=fi["feature"], importance=fi["importance"])
        for fi in bundle.get("feature_importance", [])
    ]

    return ModelInfoResponse(
        model_name=bundle["model_name"],
        metrics=metrics,
        feature_importance=feature_importance,
        threshold=bundle.get("threshold", 0.5),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_return(order: OrderInput):
    """Predict whether a single order will be returned."""
    try:
        result = predict(order.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return PredictionResponse(**result)
