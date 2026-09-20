"""Scratch build script for Phase 2 (feature EDA) + Phase 3 (baselines + models).
Not a deliverable itself — prototypes numbers before they go into
notebooks/02_features_model.ipynb, same pattern as _phase1_build.py.
"""
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, average_precision_score,
    precision_recall_curve, confusion_matrix,
)
import lightgbm as lgb

CHART_DIR = os.path.join(os.path.dirname(__file__), "..", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# Palette (validated default, from the dataviz skill) — same as _phase1_build.py
BLUE = "#2a78d6"
ORANGE = "#eb6834"
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
STATUS_GOOD = "#0ca30c"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial"],
    "axes.edgecolor": GRID,
    "axes.labelcolor": SECONDARY_INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
})


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x")
    ax.tick_params(length=0)

load_dotenv()

# URL.create() percent-encodes credentials safely — the password contains an
# '@' which broke naive f-string interpolation (sqlalchemy misparsed the
# host as everything after the last '@').
engine = create_engine(URL.create(
    "mysql+mysqlconnector",
    username=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    host=os.getenv("MYSQL_HOST"),
    port=int(os.getenv("MYSQL_PORT")),
    database=os.getenv("MYSQL_DATABASE"),
))

print("Loading user_features...")
df = pd.read_sql("SELECT * FROM user_features", engine)
print(f"Loaded {len(df):,} rows, churn rate {df['churned'].mean():.4f}")

# ---------------------------------------------------------------------------
# EDA: nulls + basic distribution sanity, before anything gets imputed.
# ---------------------------------------------------------------------------
print("\n--- Null counts ---")
print(df.isnull().sum())

print("\n--- Describe ---")
print(df.describe().T)

# ---------------------------------------------------------------------------
# Null handling. basket_size_trend / gap_trend are null only for users with
# thin pre-cutoff history (near the dataset's min of 4 lifetime orders) —
# not missing-at-random noise, but a real "not enough history to tell"
# signal. Add explicit has_trend flags and impute the trend value to 0
# (neutral: "no evidence of speeding up or slowing down") rather than
# dropping ~11-21% of users.
# ---------------------------------------------------------------------------
df["basket_trend_known"] = df["basket_size_trend"].notna().astype(int)
df["gap_trend_known"] = df["gap_trend"].notna().astype(int)
df["basket_size_trend"] = df["basket_size_trend"].fillna(0)
df["gap_trend"] = df["gap_trend"].fillna(0)

feature_cols = [
    "recency_at_cutoff", "tenure_at_cutoff", "lifetime_orders_pre_cutoff",
    "orders_last_30d", "orders_last_60d", "orders_last_90d",
    "avg_basket_size", "basket_size_trend", "basket_trend_known",
    "distinct_departments", "distinct_aisles", "distinct_products",
    "repeat_product_ratio", "weekend_order_ratio", "avg_order_hour",
    "gap_trend", "gap_trend_known",
]
X = df[feature_cols].astype(float)
y = df["churned"].astype(int)

X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train):,}  Test: {len(X_test):,}")
print(f"Train churn rate: {y_train.mean():.4f}  Test churn rate: {y_test.mean():.4f}")

results = {}


def evaluate(name, y_true, y_score, y_pred):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    ap = average_precision_score(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred)
    results[name] = dict(precision=p, recall=r, pr_auc=ap, confusion_matrix=cm.tolist())
    print(f"\n=== {name} ===")
    print(f"Precision: {p:.4f}  Recall: {r:.4f}  PR-AUC: {ap:.4f}")
    print(f"Confusion matrix [[TN,FP],[FN,TP]]:\n{cm}")


# ---------------------------------------------------------------------------
# Baseline 1 — rule: "churned if pre-cutoff recency already >= 30 days."
# NOTE on why this isn't the same as the label (no leakage): the label is
# built from the HELD-OUT gap after the cutoff (last_gap_days >= 30), which
# this baseline cannot see. This baseline instead assumes "past cadence
# predicts future cadence" using only recency_at_cutoff, a pre-cutoff
# feature — the naive persistence assumption the model needs to beat.
# ---------------------------------------------------------------------------
pred1 = (df_test["recency_at_cutoff"] >= 30).astype(int)
evaluate("Baseline 1 (rule: recency_at_cutoff >= 30)", y_test, pred1, pred1)

# ---------------------------------------------------------------------------
# Baseline 2 — RFM heuristic: bottom-recency quartile (highest recency_at_cutoff)
# AND bottom-frequency quartile (lowest orders_last_90d), computed on TRAIN
# thresholds and applied to TEST (avoid leaking test distribution into the
# threshold itself).
# ---------------------------------------------------------------------------
recency_q75 = df_train["recency_at_cutoff"].quantile(0.75)
freq_q25 = df_train["orders_last_90d"].quantile(0.25)
pred2 = ((df_test["recency_at_cutoff"] >= recency_q75) & (df_test["orders_last_90d"] <= freq_q25)).astype(int)
evaluate("Baseline 2 (RFM heuristic: bottom recency+freq quartiles)", y_test, pred2, pred2)

# ---------------------------------------------------------------------------
# Model A — logistic regression (interpretable, coefficient directions)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
logreg.fit(X_train_s, y_train)
score_lr = logreg.predict_proba(X_test_s)[:, 1]
pred_lr = (score_lr >= 0.5).astype(int)
evaluate("Model A (logistic regression)", y_test, score_lr, pred_lr)

print("\n--- Logistic regression coefficients (standardized) ---")
coef_df = pd.DataFrame({"feature": feature_cols, "coef": logreg.coef_[0]}).sort_values("coef")
print(coef_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Model B — LightGBM (performance model)
# ---------------------------------------------------------------------------
t0 = time.time()
gbm = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    class_weight="balanced", random_state=42, verbose=-1,
)
gbm.fit(X_train, y_train)
score_gbm = gbm.predict_proba(X_test)[:, 1]
pred_gbm = (score_gbm >= 0.5).astype(int)
evaluate("Model B (LightGBM)", y_test, score_gbm, pred_gbm)
print(f"LightGBM fit time: {time.time()-t0:.1f}s")

print("\n--- LightGBM feature importances ---")
fi_df = pd.DataFrame({"feature": feature_cols, "importance": gbm.feature_importances_}).sort_values("importance", ascending=False)
print(fi_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Headline comparison: GBM vs RFM baseline
# ---------------------------------------------------------------------------
print("\n=== Summary: PR-AUC by model ===")
for name, r in results.items():
    print(f"{name}: PR-AUC={r['pr_auc']:.4f}  Precision={r['precision']:.4f}  Recall={r['recall']:.4f}")

baseline2_ap = results["Baseline 2 (RFM heuristic: bottom recency+freq quartiles)"]["pr_auc"]
gbm_ap = results["Model B (LightGBM)"]["pr_auc"]
print(f"\nGBM vs RFM baseline PR-AUC lift: {gbm_ap - baseline2_ap:+.4f} ({(gbm_ap/baseline2_ap - 1)*100:+.1f}% relative)")

# ---------------------------------------------------------------------------
# Chart 5: PR curves — GBM vs logistic regression vs no-skill baseline,
# with the two rule/RFM baselines marked as single points (they're fixed
# thresholds, not score-ranked curves).
# ---------------------------------------------------------------------------
prec_lr, rec_lr, _ = precision_recall_curve(y_test, score_lr)
prec_gbm, rec_gbm, _ = precision_recall_curve(y_test, score_gbm)
no_skill = y_test.mean()

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(rec_gbm, prec_gbm, color=BLUE, linewidth=2, label=f"Model B: LightGBM (PR-AUC={results['Model B (LightGBM)']['pr_auc']:.3f})")
ax.plot(rec_lr, prec_lr, color=ORANGE, linewidth=2, label=f"Model A: Logistic regression (PR-AUC={results['Model A (logistic regression)']['pr_auc']:.3f})")
ax.axhline(no_skill, color=MUTED, linestyle=":", linewidth=1.2, label=f"No-skill baseline (churn rate={no_skill:.3f})")
r1, p1 = results["Baseline 1 (rule: recency_at_cutoff >= 30)"]["recall"], results["Baseline 1 (rule: recency_at_cutoff >= 30)"]["precision"]
r2, p2 = results["Baseline 2 (RFM heuristic: bottom recency+freq quartiles)"]["recall"], results["Baseline 2 (RFM heuristic: bottom recency+freq quartiles)"]["precision"]
ax.scatter([r1], [p1], color=STATUS_GOOD, zorder=5, s=60, marker="D", label="Baseline 1: recency rule")
ax.scatter([r2], [p2], color="#8a5cf0", zorder=5, s=60, marker="s", label="Baseline 2: RFM heuristic")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-recall: models vs. baselines (test set, n=41,242)", fontsize=11, loc="left")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.02)
style_axes(ax)
ax.grid(axis="y", color=GRID)
ax.legend(loc="upper right", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "05_pr_curve_comparison.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Chart 6: LightGBM feature importance
# ---------------------------------------------------------------------------
fi_sorted = fi_df.sort_values("importance", ascending=True)
fig, ax = plt.subplots(figsize=(7, 6.5))
y_pos = range(len(fi_sorted))
ax.barh(y_pos, fi_sorted["importance"], color=BLUE)
ax.set_yticks(y_pos)
ax.set_yticklabels(fi_sorted["feature"])
ax.set_xlabel("Split-count importance")
ax.set_title("LightGBM feature importance (Model B)", fontsize=11, loc="left")
style_axes(ax)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "06_feature_importance.png"), dpi=150)
plt.close(fig)

print("\nCharts saved to", CHART_DIR)
print("\nDONE")
