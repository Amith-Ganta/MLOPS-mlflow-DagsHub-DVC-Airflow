"""Preprocessing stage: clean and split the raw dataset.

Reads params.yaml for all config. Outputs dataset/processed_data.csv.
Run: python src/preprocess.py
"""
from __future__ import annotations

import pathlib
import yaml
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load(open(ROOT / "params.yaml"))


def preprocess():
    p = PARAMS["preprocess"]
    df = pd.read_csv(ROOT / p["input"])

    df["fulfillment_type"] = df["fulfillment_type"].astype("string").str.strip().str.title()
    df = df.drop(columns=["order_id"]).drop_duplicates().reset_index(drop=True)

    out_path = ROOT / p["output"]
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[preprocess] Saved {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    preprocess()
