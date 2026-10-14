"""Reproducible training pipeline for the return-prediction model.

Reads all config from params.yaml. Logs metrics, params, and model to
MLflow (DagsHub tracking server). Saves the model bundle to artifacts/.

Run from project root:
    python src/train.py
"""
from __future__ import annotations

import json
import os
import pathlib

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load(open(ROOT / "params.yaml"))

DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "")
MLFLOW_TRACKING_URI = "https://dagshub.com/Amith-Ganta/MLOPS-mlflow-DagsHub-DVC-Airflow.mlflow"

BINARY_FLAGS = ["fragile_item", "warranty_available", "delayed_delivery",
                "wishlist_before_purchase"]
CATEGORICAL_FEATURES = ["product_category", "sub_category", "brand",
                        "fulfillment_type", "payment_method"]


def setup_mlflow() -> bool:
    """Returns True if MLflow remote is reachable, False otherwise."""
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


def load_and_prepare(p: dict):
    df = pd.read_csv(ROOT / p["data"])
    df_feat = add_features(df)
    X = df_feat.drop(columns=["returned"])
    y = df_feat["returned"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=p["random_state"])

    q1, q3 = X_train["log_price"].quantile(0.25), X_train["log_price"].quantile(0.75)
    upper = q3 + 3 * (q3 - q1)
    mask = X_train["log_price"] <= upper
    X_train, y_train = X_train[mask], y_train[mask]

    numeric_features = [c for c in X_train.columns
                        if c not in BINARY_FLAGS + CATEGORICAL_FEATURES]
    all_features = numeric_features + BINARY_FLAGS

    train_medians = X_train[all_features].median().to_dict()
    X_train[all_features] = X_train[all_features].fillna(train_medians)
    X_test[all_features] = X_test[all_features].fillna(train_medians)

    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])

    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_FEATURES, dtype=int)
    X_test = pd.get_dummies(X_test, columns=CATEGORICAL_FEATURES, dtype=int)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return (X_train, X_test, y_train, y_test, scaler,
            list(X_train.columns), numeric_features, train_medians)


def train_models(X_train, y_train, p: dict):
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=p["random_state"]),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=p["random_state"]),
        "Random Forest": RandomForestClassifier(
            n_estimators=p["n_estimators"], class_weight="balanced",
            n_jobs=-1, random_state=p["random_state"]),
        "XGBoost": XGBClassifier(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
            max_depth=p["max_depth"], subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=p["random_state"], n_jobs=-1),
    }


def metrics_at(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def tune_threshold(y_true, proba, cost_fn, cost_fp):
    grid = np.linspace(0.05, 0.95, 91)
    best_cost, best_t = float("inf"), 0.5
    rows = []
    for t in grid:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        cost = cost_fn * fn + cost_fp * fp
        rows.append((round(float(t), 3), int(fp), int(fn), float(cost)))
        if cost < best_cost:
            best_cost, best_t = cost, float(t)
    curve = pd.DataFrame(rows, columns=["threshold", "false_pos", "false_neg", "cost"])
    return best_t, curve


def main():
    p = PARAMS["train"]
    mlflow_remote = setup_mlflow()

    (X_train, X_test, y_train, y_test, scaler, model_columns,
     numeric_features, train_medians) = load_and_prepare(p)

    fitted = train_models(X_train, y_train, p)
    for est in fitted.values():
        est.fit(X_train, y_train)

    comparison = {}
    for name, est in fitted.items():
        proba = est.predict_proba(X_test)[:, 1]
        comparison[name] = metrics_at(y_test, proba, 0.5)
    comp_df = pd.DataFrame(comparison).T.sort_values("f1", ascending=False)
    best_name = comp_df.index[0]
    best_model = fitted[best_name]

    print("Model comparison at threshold 0.50 (test set):")
    print(comp_df.round(4).to_string())
    print(f"\nBest by F1: {best_name}")

    proba = best_model.predict_proba(X_test)[:, 1]
    best_t, curve = tune_threshold(y_test, proba, p["cost_fn"], p["cost_fp"])
    default_metrics = metrics_at(y_test, proba, 0.5)
    tuned_metrics = metrics_at(y_test, proba, best_t)

    print(f"\nThreshold tuning (FN={p['cost_fn']}x FP={p['cost_fp']}):")
    print(f"  default 0.50 -> {default_metrics}")
    print(f"  tuned  {best_t:.2f} -> {tuned_metrics}")

    if hasattr(best_model, "coef_"):
        fi = pd.Series(np.abs(best_model.coef_[0]), index=model_columns)
    else:
        fi = pd.Series(getattr(best_model, "feature_importances_", np.zeros(len(model_columns))),
                       index=model_columns)
    fi = fi.sort_values(ascending=False).head(20)
    feature_importance = [{"feature": k, "importance": round(float(v), 6)}
                          for k, v in fi.items()]

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "scaler": scaler,
        "model_columns": model_columns,
        "numeric_features": numeric_features,
        "binary_flags": BINARY_FLAGS,
        "categorical_features": CATEGORICAL_FEATURES,
        "train_medians": train_medians,
        "metrics": tuned_metrics,
        "metrics_default_threshold": default_metrics,
        "threshold": best_t,
        "cost_fn": p["cost_fn"],
        "cost_fp": p["cost_fp"],
        "feature_importance": feature_importance,
    }

    artifact_path = ROOT / p["model"]
    artifact_path.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, artifact_path)
    print(f"\nSaved artifact -> {artifact_path}")

    curve_path = ROOT / "artifacts" / "threshold_cost_curve.csv"
    curve.to_csv(curve_path, index=False)

    # Write DVC metrics file
    metrics_path = ROOT / "artifacts" / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({**tuned_metrics, "threshold": best_t, "model_name": best_name}, f, indent=2)

    # Log to MLflow/DagsHub
    try:
        with mlflow.start_run(run_name=f"train-{best_name.lower().replace(' ', '-')}"):
            mlflow.log_params({
                "model_name": best_name,
                "n_estimators": p["n_estimators"],
                "max_depth": p["max_depth"],
                "learning_rate": p["learning_rate"],
                "random_state": p["random_state"],
                "cost_fn": p["cost_fn"],
                "cost_fp": p["cost_fp"],
                "threshold": best_t,
            })
            mlflow.log_metrics(tuned_metrics)
            mlflow.log_metrics({f"default_{k}": v for k, v in default_metrics.items()})
            mlflow.log_artifact(str(artifact_path))
            mlflow.log_artifact(str(curve_path))
            mlflow.sklearn.log_model(best_model, "model")
            print("[OK] Logged run to MLflow DagsHub")
    except Exception as e:
        print(f"[WARN] MLflow logging skipped: {str(e)[:120]}")


if __name__ == "__main__":
    main()
