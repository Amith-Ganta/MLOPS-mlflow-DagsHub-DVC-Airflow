"""Analytics queries expressed as real SQL against the SQLite orders table.

These reproduce the notebook's exploratory findings, but as SQL a data team would
actually run: aggregate return rates by category, and a window function that ranks
sellers within each category by return rate. The point is to show the analysis was
done in SQL, not just in pandas.

Run from the project root (after load_db.py):
    python src/sql/queries.py
"""

from __future__ import annotations

import pathlib
import sqlite3

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "dataset" / "returns.db"


# Return rate per product category, plus a lift vs the overall average.
RETURN_RATE_BY_CATEGORY = """
WITH overall AS (
    SELECT AVG(returned) AS base_rate FROM orders
)
SELECT
    o.product_category,
    COUNT(*)                              AS orders,
    ROUND(AVG(o.returned), 4)             AS return_rate,
    ROUND(AVG(o.returned) - overall.base_rate, 4) AS lift_vs_avg
FROM orders o
CROSS JOIN overall
GROUP BY o.product_category, overall.base_rate
ORDER BY return_rate DESC;
"""

# Return rate by payment method (the notebook flagged this as a strong splitter).
RETURN_RATE_BY_PAYMENT = """
SELECT
    payment_method,
    COUNT(*)                  AS orders,
    ROUND(AVG(returned), 4)   AS return_rate
FROM orders
GROUP BY payment_method
ORDER BY return_rate DESC;
"""

# Window function: within each category, rank brands by their return rate.
# Shows the riskiest brand per category without a self-join.
TOP_RISK_BRAND_PER_CATEGORY = """
WITH brand_rates AS (
    SELECT
        product_category,
        brand,
        COUNT(*)                AS orders,
        AVG(returned)           AS return_rate
    FROM orders
    GROUP BY product_category, brand
    HAVING COUNT(*) >= 30
)
SELECT product_category, brand, orders, ROUND(return_rate, 4) AS return_rate, rnk
FROM (
    SELECT
        product_category,
        brand,
        orders,
        return_rate,
        RANK() OVER (
            PARTITION BY product_category
            ORDER BY return_rate DESC
        ) AS rnk
    FROM brand_rates
)
WHERE rnk = 1
ORDER BY return_rate DESC;
"""


def run(sql: str, db_path: pathlib.Path = DB_PATH) -> pd.DataFrame:
    """Execute a query and return the result as a DataFrame."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn)


if __name__ == "__main__":
    print("\nReturn rate by category (with lift vs overall average):")
    print(run(RETURN_RATE_BY_CATEGORY).to_string(index=False))

    print("\nReturn rate by payment method:")
    print(run(RETURN_RATE_BY_PAYMENT).to_string(index=False))

    print("\nRiskiest brand per category (window function, RANK):")
    print(run(TOP_RISK_BRAND_PER_CATEGORY).to_string(index=False))
