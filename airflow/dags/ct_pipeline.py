"""Continuous Training (CT) DAG for product return prediction.

Runs weekly. Orchestrates the full DVC pipeline:
  preprocess -> train -> evaluate -> push artifacts to DagsHub

If the evaluate gate fails (F1 < 0.70), the DAG fails and alerts — the
model in production is NOT replaced. Only a passing run triggers a push.

Set these Airflow Variables before activating (Admin > Variables):
  PROJECT_PATH  : absolute path to the project root on the host
  DAGSHUB_TOKEN : your DagsHub access token
"""
from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

PROJECT_PATH = Variable.get("PROJECT_PATH",
    default_var="/opt/airflow/project")
PYTHON_BIN = Variable.get("PYTHON_BIN",
    default_var="python")
DAGSHUB_TOKEN = Variable.get("DAGSHUB_TOKEN", default_var="")

DEFAULT_ARGS = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _run(cmd: list[str], env_extra: dict | None = None):
    env = os.environ.copy()
    if DAGSHUB_TOKEN:
        env["DAGSHUB_TOKEN"] = DAGSHUB_TOKEN
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(cmd, cwd=PROJECT_PATH, env=env,
                            capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")


def pull_data(**ctx):
    _run(["dvc", "pull", "dataset/returns_dataset.csv.dvc", "--no-run-cache"])


def run_preprocess(**ctx):
    _run([PYTHON_BIN, "src/preprocess.py"])


def run_train(**ctx):
    _run([PYTHON_BIN, "src/train.py"])


def run_evaluate(**ctx):
    """Raises on F1 < 0.70 — stops the DAG here, no push happens."""
    _run([PYTHON_BIN, "src/evaluate.py"])


def push_artifacts(**ctx):
    _run(["dvc", "push"])
    print("[CT] Artifacts pushed to DagsHub DVC remote.")


with DAG(
    dag_id="product_return_ct_pipeline",
    description="Weekly Continuous Training for product-return-prediction",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 10, 14),
    schedule="0 2 * * 1",  # every Monday at 02:00 UTC
    catchup=False,
    tags=["mlops", "ct", "product-return"],
) as dag:

    t_pull = PythonOperator(task_id="pull_data", python_callable=pull_data)
    t_preprocess = PythonOperator(task_id="preprocess", python_callable=run_preprocess)
    t_train = PythonOperator(task_id="train", python_callable=run_train)
    t_evaluate = PythonOperator(task_id="evaluate", python_callable=run_evaluate)
    t_push = PythonOperator(task_id="push_artifacts", python_callable=push_artifacts)

    t_pull >> t_preprocess >> t_train >> t_evaluate >> t_push
