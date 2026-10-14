# Product Return Prediction — MLOps Stack

End-to-end MLOps implementation for an e-commerce product-return classifier.
Demonstrates CI, CD, and CT (Continuous Training) targeting Accenture MLOps Engineer requirements.

---

## Architecture

```
GitHub Push
    │
    ▼
GitHub Actions CI
  ├── dvc pull (data from DagsHub)
  ├── dvc repro (preprocess → train → evaluate)
  ├── F1 gate (≥ 0.70 required to pass)
  ├── dvc push (artifacts back to DagsHub)
  └── upload model artifact
    │
    ▼ (on main branch, CI passes)
GitHub Actions CD
  └── docker build & push → Docker Hub

Apache Airflow (CT)
  └── Weekly DAG (Monday 02:00 UTC)
        pull_data → preprocess → train → evaluate → push_artifacts
        (DAG fails and halts at evaluate if F1 < 0.70)
```

---

## Model Results

| Model               | F1 (default) | Threshold | F1 (tuned) |
|---------------------|-------------|-----------|------------|
| Logistic Regression | 0.784       | 0.30      | **0.739**  |
| XGBoost             | 0.761       | —         | —          |
| Random Forest       | 0.748       | —         | —          |
| Decision Tree       | 0.712       | —         | —          |

Threshold tuned for a 5:1 FN:FP cost model. Logistic Regression selected as champion.

---

## Stack

| Layer              | Tool                    |
|--------------------|-------------------------|
| Pipeline versioning | DVC 3.x               |
| Experiment tracking | MLflow + DagsHub       |
| CI/CD              | GitHub Actions          |
| Continuous Training | Apache Airflow 2.9.3   |
| Serving API        | FastAPI + Uvicorn       |
| Containerisation   | Docker + Docker Compose |
| Remote storage     | DagsHub (Git + DVC + MLflow) |

---

## Quick Start

### 1 — Local pipeline run

```bash
# Install dependencies
pip install dvc[http] mlflow scikit-learn xgboost pandas numpy joblib fastapi uvicorn pydantic

# Configure DVC credentials (gitignored)
dvc remote modify dagshub_remote --local auth basic
dvc remote modify dagshub_remote --local user Amith-Ganta
dvc remote modify dagshub_remote --local password <DAGSHUB_TOKEN>

# Pull data and run pipeline
dvc pull dataset/returns_dataset.csv.dvc --no-run-cache
dvc repro

# Show metrics
dvc metrics show
```

### 2 — Run the FastAPI serving container

```bash
docker build -t product-return-api:local .
docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts product-return-api:local
```

API endpoints:
- `GET /health` — health check
- `GET /model-info` — model metrics and feature importances
- `POST /predict` — return prediction for a single order

### 3 — Start the Airflow CT stack

```bash
# Set token in environment
export DAGSHUB_TOKEN=<your_token>

# Start postgres
docker compose -f docker-compose.airflow.yml up -d postgres

# Initialise Airflow DB and create admin user (run once)
docker compose -f docker-compose.airflow.yml up airflow-init

# Start webserver and scheduler
docker compose -f docker-compose.airflow.yml up -d airflow-webserver airflow-scheduler
```

Open http://localhost:8080 (user: `airflow`, password: `airflow`).

Enable and trigger the `product_return_ct_pipeline` DAG manually to verify the CT pipeline end-to-end.

---

## GitHub Actions Secrets Required

Add these in **Settings > Secrets and variables > Actions** for the repo:

| Secret           | Value                            |
|------------------|----------------------------------|
| `DAGSHUB_TOKEN`  | DagsHub access token             |
| `DOCKER_USERNAME`| Docker Hub username              |
| `DOCKER_TOKEN`   | Docker Hub access token (not password) |

---

## Project Structure

```
├── .github/workflows/ci-cd.yml   # CI: dvc repro + F1 gate; CD: Docker build/push
├── airflow/dags/ct_pipeline.py   # Weekly CT DAG (5 tasks)
├── backend/
│   ├── main.py                   # FastAPI app
│   ├── model.py                  # Inference pipeline (load, preprocess, predict)
│   └── schemas.py                # Pydantic request/response models
├── src/
│   ├── preprocess.py             # DVC stage 1 — clean raw CSV
│   ├── train.py                  # DVC stage 2 — train 4 models, pick best
│   └── evaluate.py               # DVC stage 3 — evaluate + F1 gate
├── dataset/
│   └── returns_dataset.csv.dvc   # DVC pointer (data lives on DagsHub)
├── artifacts/
│   ├── metrics.json              # DVC metrics (tracked in git)
│   └── eval_report.json          # Full evaluation report
├── dvc.yaml                      # Pipeline definition (3 stages)
├── params.yaml                   # Hyperparameters and thresholds
├── Dockerfile                    # FastAPI serving image
└── docker-compose.airflow.yml    # Airflow + postgres + API stack
```

---

## DagsHub Integration

**Repository**: https://dagshub.com/Amith-Ganta/MLOPS-mlflow-DagsHub-DVC-Airflow

- DVC remote stores raw data and model artifacts
- MLflow tracking server at `https://dagshub.com/Amith-Ganta/MLOPS-mlflow-DagsHub-DVC-Airflow.mlflow`
- Initialised via `dagshub.init()` — both endpoints active
