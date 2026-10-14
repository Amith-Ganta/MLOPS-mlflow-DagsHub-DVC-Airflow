"""Evaluation stage: load trained model, compute metrics, gate the CI pipeline.

Logs metrics to MLflow/DagsHub. Writes artifacts/eval_report.json.
Exits with code 1 if F1 drops below params.yaml evaluate.min_f1_threshold.

Run from project root:
    python src/evaluate.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load(open(ROOT / "params.yaml"))

BINARY_FLAGS = ["fragile_item", "warranty_available", "delayed_delivery",
                "wishlist_before_purchase"]
CATEGORICAL_FEATURES = ["product_category", "sub_category", "brand",
                        "fulfillment_type", "payment_method"]

DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "")
MLFLOW_TRACKING_URI = "https://dagshub.com/Amith-Ganta/MLOPS-mlflow-DagsHub-DVC-Airflow.mlflow"


def setup_mlflow() -> bool:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    local_uri = f"sqlite:///{ROOT / 'mlruns.db'}"
    try:
        os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
        os.environ["MLFLOW_TRACKING_USERNAME"] = "Amith-Ganta"
        if DAGSHUB_TOKEN:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("product-return-prediction")
        return True
    except Exception as e:
        print(f"[WARN] MLflow remote unreachable, logging locally: {str(e)[:100]}")
        os.environ["MLFLOW_TRACKING_URI"] = local_uri
        mlflow.set_tracking_uri(local_uri)
        mlflow.set_experiment("product-return-prediction")
        return False


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d["total_support_contacts"] = d["customer_support_calls"] + d["chat_interactions"]
    d = d.drop(columns=["customer_support_calls", "chat_interactions"])
    d["log_price"] = np.log1p(d["product_price"])
    d = d.drop(columns=["product_price"])
    return d


def main():
    p_eval = PARAMS["evaluate"]
    p_train = PARAMS["train"]
    setup_mlflow()

    bundle = joblib.load(ROOT / p_eval["model"])
    model = bundle["model"]
    scaler = bundle["scaler"]
    model_columns = bundle["model_columns"]
    numeric_features = bundle["numeric_features"]
    train_medians = bundle["train_medians"]
    threshold = bundle.get("threshold", 0.5)

    df = pd.read_csv(ROOT / p_eval["data"])
    df_feat = add_features(df)
    X = df_feat.drop(columns=["returned"])
    y = df_feat["returned"].astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=p_train["random_state"])

    all_features = numeric_features + BINARY_FLAGS
    X_test[all_features] = X_test[all_features].fillna(train_medians)
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    X_test = pd.get_dummies(X_test, columns=CATEGORICAL_FEATURES, dtype=int)
    X_test = X_test.reindex(columns=model_columns, fill_value=0)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "threshold": float(threshold),
    }

    print(f"[evaluate] Model: {bundle['model_name']}, threshold: {threshold:.2f}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    report_path = ROOT / "artifacts" / "eval_report.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] Report saved -> {report_path}")

    try:
        with mlflow.start_run(run_name="evaluate"):
            mlflow.log_metrics({k: v for k, v in metrics.items() if k != "threshold"})
            mlflow.log_param("threshold", threshold)
            mlflow.log_artifact(str(report_path))
            print("[OK] Evaluation logged to MLflow DagsHub")
    except Exception as e:
        print(f"[WARN] MLflow logging skipped: {str(e)[:120]}")

    # CI gate: fail if F1 is below minimum
    min_f1 = p_eval["min_f1_threshold"]
    if metrics["f1"] < min_f1:
        print(f"[FAIL] F1 {metrics['f1']:.4f} < threshold {min_f1} — pipeline gate failed")
        sys.exit(1)
    else:
        print(f"[PASS] F1 {metrics['f1']:.4f} >= threshold {min_f1} — pipeline gate passed")


if __name__ == "__main__":
    main()
