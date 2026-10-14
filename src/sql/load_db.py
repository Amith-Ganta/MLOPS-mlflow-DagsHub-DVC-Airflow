"""Load the returns dataset into a SQLite database.

Why SQLite: it ships with Python (no extra dependency), yet runs standard SQL
(GROUP BY, JOIN, window functions), so the analytics layer is real SQL rather
than pandas dressed up as SQL. The database file is rebuilt from the CSV on every
run, so it is reproducible and safe to delete.

Run from the project root:
    python src/sql/load_db.py
"""

from __future__ import annotations

import pathlib
import sqlite3

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "dataset" / "returns_dataset.csv"
DB_PATH = ROOT / "dataset" / "returns.db"
TABLE = "orders"


def build_database(csv_path: pathlib.Path = CSV_PATH, db_path: pathlib.Path = DB_PATH) -> pathlib.Path:
    """Read the CSV and write it to a fresh SQLite table with useful indexes."""
    df = pd.read_csv(csv_path)

    # Normalise the known dirty categorical value so SQL groupings are correct.
    # (The notebook found a stray 'marketplace fulfilled ' with trailing space.)
    df["fulfillment_type"] = (
        df["fulfillment_type"].astype("string").str.strip().str.title()
    )

    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        df.to_sql(TABLE, conn, index=False, if_exists="replace")
        # Indexes on the columns we group and filter by most.
        for col in ("product_category", "brand", "payment_method", "returned"):
            conn.execute(f"CREATE INDEX idx_{col} ON {TABLE}({col});")
        conn.commit()
        n = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]

    print(f"Loaded {n} rows into {db_path} (table: {TABLE})")
    return db_path


if __name__ == "__main__":
    build_database()
