# Product Return Prediction

An end-to-end machine learning service that predicts which e-commerce orders are likely to be returned. Returns are one of the largest operational costs in retail: every returned order carries reverse logistics, inspection, repackaging, and refund costs, so flagging high-risk orders early is worth real money. This is an imbalanced binary classification problem (about 30% of orders are returned).

## Provenance and honesty note

The base modelling exercise (the dataset and the original notebook skeleton) comes from a Codebasics ML course capstone. On top of that I built the parts that make it a service rather than a coursework notebook:

- a SQL analytics layer (`src/sql/`) over the data, including a window-function query;
- a reproducible training script (`src/train.py`) with decision-threshold tuning against an explicit business cost;
- a leakage sanity check that measures how much the model leans on historical-prior features;
- a FastAPI backend (`backend/`) that serves the model, plus a small frontend.

The dataset (`dataset/returns_dataset.csv`, about 12,000 orders) is programmatically generated, not real customer data. It has a realistic return-rate imbalance and genuine signal in the features, which is enough to practise and demonstrate the full workflow, but the specific numbers below describe this synthetic dataset and would not transfer to a real one without retraining.

## What the model does

The pipeline engineers features (a combined support-contact count, a log-transformed price to tame a heavy right skew), imputes missing values with training-set medians, scales numeric features, and one-hot encodes categoricals. It compares four classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost), all with balanced class weights. Logistic Regression wins on F1.

Test-set results at the default 0.50 threshold:

| Metric    | Value |
|-----------|-------|
| Accuracy  | 0.856 |
| Precision | 0.719 |
| Recall    | 0.861 |
| F1        | 0.784 |

## Decision-threshold tuning

A fixed 0.50 threshold is rarely the right operating point for an imbalanced, cost-asymmetric problem. A missed return (false negative) is more expensive than a false alarm (false positive), because a missed return lands the full reverse-logistics bill with no chance to intervene. Treating a false negative as five times the cost of a false positive, `src/train.py` sweeps the threshold and picks the one that minimises total expected cost. The cost curve is written to `artifacts/threshold_cost_curve.csv`.

At the chosen threshold of 0.30, recall rises from 0.861 to 0.930 (the model now catches 93% of real returns) while precision falls to 0.614. That is the trade a business would actually make under this cost assumption. The serving code reads the tuned threshold from the model bundle, so the operating point is a stored decision, not a magic number in the code.

## Leakage sanity check

Two features, `product_return_rate` and `brand_return_rate`, are historical return-rate averages and the two strongest single signals in the data. A reviewer should ask whether the model is just parroting them. `src/train.py` retrains without those two features and reports the gap: F1 drops only from 0.784 to 0.773. The historical priors are additive, not load-bearing. Most of the model's signal comes from legitimate order-level features (delayed delivery, payment method, product rating, support contacts).

## SQL analytics layer

`src/sql/` loads the CSV into a SQLite database and runs the analysis as real SQL a data team would run, rather than pandas dressed up as SQL:

- return rate per product category, with lift versus the overall average (CTE plus a cross join);
- return rate by payment method;
- the riskiest brand within each category, using a `RANK() OVER (PARTITION BY ...)` window function.

The payment-method query surfaces the clearest operational signal in the data: cash on delivery returns at about 50%, versus 16 to 24% for prepaid methods.

## Repository layout

```
src/
  train.py            reproducible training + threshold tuning + leakage check
  sql/
    load_db.py        build the SQLite database from the CSV
    queries.py        analytics queries incl. the window function
backend/
  main.py             FastAPI app (/health, /model-info, /predict)
  model.py            load, preprocess, predict
  schemas.py          request/response models
artifacts/
  model_data.joblib   trained bundle (model, scaler, columns, threshold, metrics)
  threshold_cost_curve.csv
dataset/
  returns_dataset.csv
product_return_prediction.ipynb   the exploratory walkthrough
```

## How to run

Retrain the model and rebuild the analytics database from the project root:

```bash
pip install numpy pandas scikit-learn xgboost joblib
python src/train.py
python src/sql/load_db.py
python src/sql/queries.py
```

Serve the model:

```bash
pip install fastapi uvicorn
uvicorn backend.main:app --reload
```

Then POST an order to `/predict`. The `/model-info` endpoint returns the model name, metrics, feature importances, and the tuned threshold.
