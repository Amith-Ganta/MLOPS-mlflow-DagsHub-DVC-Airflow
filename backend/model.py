"""Model loading, preprocessing, and prediction logic."""

from __future__ import annotations

import pathlib

import joblib
import numpy as np
import pandas as pd


ARTIFACT_PATH = pathlib.Path(__file__).resolve().parent.parent / "artifacts" / "model_data.joblib"

# Module-level cache — loaded once on first call
_bundle: dict | None = None


def _load_bundle() -> dict:
    """Load the model artifact from disk (cached after first call)."""
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(ARTIFACT_PATH)
    return _bundle


def get_bundle() -> dict:
    """Public accessor for the cached model bundle."""
    return _load_bundle()


# ---------------------------------------------------------------------------
# Feature engineering — must match the notebook's add_features()
# ---------------------------------------------------------------------------

def add_features(row: dict) -> dict:
    """Compute derived features for a single order dict.

    Must match the notebook's add_features(): only two engineered features are
    kept, and raw product_price is replaced by log_price. The removed features
    (net_price, discount_amount, order_value, avg_return_rate, log_review_count,
    log_shipping_distance) had ~0 signal and/or were collinear.
    """
    d = dict(row)
    d["total_support_contacts"] = d["customer_support_calls"] + d["chat_interactions"]
    d.pop("customer_support_calls", None)  # replaced by total_support_contacts
    d.pop("chat_interactions", None)       # (dropping avoids exact multicollinearity)
    d["log_price"] = np.log1p(d["product_price"])
    d.pop("product_price", None)           # replaced by log_price
    return d


# ---------------------------------------------------------------------------
# Preprocessing — impute, scale, encode to match training columns
# ---------------------------------------------------------------------------

def preprocess(order: dict) -> pd.DataFrame:
    """
    Take a raw order dict, apply feature engineering, impute, scale, and
    one-hot encode to produce a single-row DataFrame matching model_columns.
    """
    bundle = _load_bundle()
    scaler = bundle["scaler"]
    model_columns = bundle["model_columns"]
    numeric_features = bundle["numeric_features"]
    binary_flags = bundle["binary_flags"]
    categorical_features = bundle["categorical_features"]
    train_medians = bundle["train_medians"]

    # Feature engineering
    enriched = add_features(order)

    # Build a single-row DataFrame
    df = pd.DataFrame([enriched])

    # Clean fulfillment_type to match training
    if "fulfillment_type" in df.columns:
        df["fulfillment_type"] = df["fulfillment_type"].astype("string").str.strip().str.title()

    all_features = numeric_features + binary_flags

    # Impute with training medians
    for col in all_features:
        if col in df.columns and pd.isna(df[col].iloc[0]):
            df[col] = train_medians.get(col, 0)

    # Scale numeric features
    df[numeric_features] = scaler.transform(df[numeric_features])

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=categorical_features, dtype=int)

    # Align to model_columns (add missing columns, drop extras)
    df = df.reindex(columns=model_columns, fill_value=0)

    return df


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(order: dict) -> dict:
    """
    Run the full inference pipeline on a raw order dict.

    Returns:
        {
            "return_probability": float,
            "predicted_returned": bool,
            "risk_level": "Low" | "Medium" | "High",
        }
    """
    bundle = _load_bundle()
    model = bundle["model"]
    threshold = bundle.get("threshold", 0.5)

    X = preprocess(order)
    proba = float(model.predict_proba(X)[:, 1][0])
    predicted = proba >= threshold

    if proba < 0.3:
        risk = "Low"
    elif proba < 0.6:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "return_probability": round(proba, 4),
        "predicted_returned": predicted,
        "risk_level": risk,
    }
