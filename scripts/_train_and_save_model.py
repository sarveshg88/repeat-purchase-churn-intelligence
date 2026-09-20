"""One-time training run that persists the deployed model + reference
artifacts for scripts/03_scoring_pipeline.py. Not a deliverable itself —
run once after Phase 3, whenever the model is retrained.

Mirrors the training in notebooks/02_features_model.ipynb exactly (same
random_state, same feature engineering, same hyperparameters) so the
persisted model matches what's reported there.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sklearn.model_selection import train_test_split
import lightgbm as lgb

load_dotenv()

engine = create_engine(URL.create(
    "mysql+mysqlconnector",
    username=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    host=os.getenv("MYSQL_HOST"),
    port=int(os.getenv("MYSQL_PORT")),
    database=os.getenv("MYSQL_DATABASE"),
))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_sql("SELECT * FROM user_features", engine)

df["basket_trend_known"] = df["basket_size_trend"].notna().astype(int)
df["gap_trend_known"] = df["gap_trend"].notna().astype(int)
df["basket_size_trend"] = df["basket_size_trend"].fillna(0)
df["gap_trend"] = df["gap_trend"].fillna(0)

FEATURE_COLS = [
    "recency_at_cutoff", "tenure_at_cutoff", "lifetime_orders_pre_cutoff",
    "orders_last_30d", "orders_last_60d", "orders_last_90d",
    "avg_basket_size", "basket_size_trend", "basket_trend_known",
    "distinct_departments", "distinct_aisles", "distinct_products",
    "repeat_product_ratio", "weekend_order_ratio", "avg_order_hour",
    "gap_trend", "gap_trend_known",
]
X = df[FEATURE_COLS].astype(float)
y = df["churned"].astype(int)

# Same split as the notebook (random_state=42) — the persisted model is
# trained on exactly the same 80% the notebook reports metrics for.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

gbm = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    class_weight="balanced", random_state=42, verbose=-1,
)
gbm.fit(X_train, y_train)
joblib.dump(gbm, os.path.join(MODEL_DIR, "churn_model.joblib"))

with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)

# Cost-optimal threshold from Phase 4 (FN:FP = 5:1 sweep on the held-out
# test set). Persisted as a constant rather than re-derived here — in a
# real deployment this would be revisited periodically as part of the same
# monitoring cadence as drift, not fixed forever.
with open(os.path.join(MODEL_DIR, "threshold.json"), "w") as f:
    json.dump({"cost_optimal_threshold": 0.31, "fn_fp_ratio": "5:1",
               "source": "notebooks/02_features_model.ipynb, Phase 4"}, f, indent=2)

# ---------------------------------------------------------------------------
# Drift reference: 10 quantile-bin edges per feature, computed on the
# TRAINING split, plus the training set's own proportion of rows in each
# bin. scripts/03_scoring_pipeline.py bins each new scoring run's feature
# values into these SAME edges and compares proportions via PSI — the
# standard technique for detecting feature drift against a training
# baseline without needing labels on the new data.
# ---------------------------------------------------------------------------
reference = {}
for col in FEATURE_COLS:
    values = X_train[col].values
    try:
        edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
    except Exception:
        edges = np.array([values.min(), values.max()])
    if len(edges) < 3:
        # Degenerate (near-constant) feature — one wide bin.
        edges = np.array([values.min() - 1, values.max() + 1])
    counts, _ = np.histogram(values, bins=edges)
    proportions = (counts / counts.sum()).tolist()
    reference[col] = {"bin_edges": edges.tolist(), "train_proportions": proportions}

with open(os.path.join(MODEL_DIR, "training_reference.json"), "w") as f:
    json.dump(reference, f, indent=2)

print(f"Saved model + reference artifacts to {MODEL_DIR}")
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")
