"""Phase 6 — Scheduled re-scoring pipeline.

Re-scores the current user population with the persisted model, assigns
intervention tiers (same logic as Phase 5), checks feature drift against the
training-time reference distribution, and refreshes a small dashboard.

Usage:
    .venv/Scripts/python.exe scripts/03_scoring_pipeline.py

Scheduling (not set up by this script — a deployment/ops decision, not a
code one): on Windows, register this command in Task Scheduler; on a
Linux host, a cron entry. This script is idempotent per run and safe to
call repeatedly — each run overwrites output/latest_scores.csv and
charts/dashboard_latest.png, and appends one entry to output/drift_log.jsonl.

Drift-check caveat: this dataset is a frozen historical snapshot, so there
is no genuinely new incoming batch to compare against the training window.
As a stand-in that still exercises the real mechanism end-to-end, this run
scores the FULL current population and compares its feature distribution
against the training-time reference (computed from an 80% split of that
same population) — so near-zero drift is the *expected* result here, not
evidence the drift check doesn't work. In production this would compare
against a genuinely new day's/week's users.
"""
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

load_dotenv()

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(ROOT, "models")
OUTPUT_DIR = os.path.join(ROOT, "output")
CHART_DIR = os.path.join(ROOT, "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

PSI_WARN = 0.1   # standard rule-of-thumb thresholds for population stability index
PSI_ALERT = 0.25

BLUE, ORANGE, MUTED, GRID, INK, SECONDARY_INK = (
    "#2a78d6", "#eb6834", "#898781", "#e1e0d9", "#0b0b0b", "#52514e",
)
STATUS_GOOD, STATUS_WARNING, STATUS_CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial"],
    "axes.edgecolor": GRID, "axes.labelcolor": SECONDARY_INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)


def load_artifacts():
    required = ["churn_model.joblib", "feature_cols.json", "threshold.json", "training_reference.json"]
    missing = [f for f in required if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        print(f"Missing model artifacts: {missing}. Run scripts/_train_and_save_model.py first.", file=sys.stderr)
        sys.exit(1)

    model = joblib.load(os.path.join(MODEL_DIR, "churn_model.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(MODEL_DIR, "threshold.json")) as f:
        threshold = json.load(f)["cost_optimal_threshold"]
    with open(os.path.join(MODEL_DIR, "training_reference.json")) as f:
        reference = json.load(f)
    return model, feature_cols, threshold, reference


def connect_db():
    return create_engine(URL.create(
        "mysql+mysqlconnector",
        username=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT")),
        database=os.getenv("MYSQL_DATABASE"),
    ))


def load_current_batch(engine, feature_cols):
    df = pd.read_sql("""
        SELECT uf.*, d.department AS top_department
        FROM user_features uf
        LEFT JOIN user_top_department ut ON ut.user_id = uf.user_id
        LEFT JOIN departments d ON d.department_id = ut.department_id
    """, engine)
    df["basket_trend_known"] = df["basket_size_trend"].notna().astype(int)
    df["gap_trend_known"] = df["gap_trend"].notna().astype(int)
    df["basket_size_trend"] = df["basket_size_trend"].fillna(0)
    df["gap_trend"] = df["gap_trend"].fillna(0)
    X = df[feature_cols].astype(float)
    return df, X


def compute_psi(current_values, bin_edges, train_proportions):
    edges = np.array(bin_edges)
    counts, _ = np.histogram(current_values, bins=edges)
    current_proportions = counts / max(counts.sum(), 1)
    eps = 1e-4  # avoid log(0) / div-by-0 for empty bins
    psi = 0.0
    for cur_p, train_p in zip(current_proportions, train_proportions):
        cur_p = max(cur_p, eps)
        train_p = max(train_p, eps)
        psi += (cur_p - train_p) * np.log(cur_p / train_p)
    return psi, current_proportions.tolist()


def assign_tiers(df, score, threshold, tenure_q25):
    flagged = score >= threshold
    gently_slowing = flagged & (df["recency_at_cutoff"] < 25) & (df["gap_trend"] <= 5)
    likely_lost = flagged & ~gently_slowing & (df["tenure_at_cutoff"] <= tenure_q25)
    at_risk = flagged & ~gently_slowing & ~likely_lost

    tier = pd.Series("Not flagged", index=df.index)
    tier[gently_slowing] = "1. Gently slowing"
    tier[at_risk] = "2. At risk"
    tier[likely_lost] = "3. Likely lost"
    return tier


def main():
    run_ts = datetime.now(timezone.utc).isoformat()
    print(f"=== Scoring pipeline run: {run_ts} ===")

    model, feature_cols, threshold, reference = load_artifacts()
    engine = connect_db()
    df, X = load_current_batch(engine, feature_cols)
    print(f"Loaded {len(df):,} users to score")

    score = model.predict_proba(X)[:, 1]
    df["score"] = score

    # tenure_q25 recomputed each run from the current batch — the tier
    # boundary is meant to track "bottom quartile of *this run's*
    # population," not a value frozen at training time.
    tenure_q25 = df["tenure_at_cutoff"].quantile(0.25)
    df["tier"] = assign_tiers(df, score, threshold, tenure_q25)

    out_cols = ["user_id", "score", "tier", "top_department"]
    df[out_cols].to_csv(os.path.join(OUTPUT_DIR, "latest_scores.csv"), index=False)
    print(f"Wrote scores for {len(df):,} users to output/latest_scores.csv")

    tier_counts = df["tier"].value_counts()
    print("\nTier population:")
    print(tier_counts.to_string())

    # -----------------------------------------------------------------
    # Drift check: PSI per feature, current full population vs. the
    # training-time reference bins.
    # -----------------------------------------------------------------
    print("\n=== Feature drift (PSI vs. training reference) ===")
    drift_rows = []
    for col in feature_cols:
        ref = reference[col]
        psi, current_props = compute_psi(df[col].astype(float).values, ref["bin_edges"], ref["train_proportions"])
        if psi >= PSI_ALERT:
            status = "ALERT"
        elif psi >= PSI_WARN:
            status = "WARN"
        else:
            status = "ok"
        drift_rows.append({"feature": col, "psi": psi, "status": status})
        flag = "  <-- " + status if status != "ok" else ""
        print(f"{col:30s} PSI={psi:.4f} {status}{flag}")

    drift_df = pd.DataFrame(drift_rows).sort_values("psi", ascending=False)
    n_alerts = (drift_df["status"] == "ALERT").sum()
    n_warns = (drift_df["status"] == "WARN").sum()
    print(f"\n{n_alerts} feature(s) at ALERT, {n_warns} at WARN (thresholds: WARN>={PSI_WARN}, ALERT>={PSI_ALERT})")

    # Append this run to the drift log (one line per run — a real
    # monitoring history builds up across scheduled runs).
    log_entry = {
        "run_ts": run_ts,
        "n_users_scored": int(len(df)),
        "n_alerts": int(n_alerts),
        "n_warns": int(n_warns),
        "max_psi_feature": drift_df.iloc[0]["feature"],
        "max_psi": float(drift_df.iloc[0]["psi"]),
        "tier_counts": tier_counts.to_dict(),
    }
    log_path = os.path.join(OUTPUT_DIR, "drift_log.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"Appended run summary to {log_path}")

    # -----------------------------------------------------------------
    # Refreshed dashboard: tier counts + top drift features, one PNG.
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    tier_order = ["1. Gently slowing", "2. At risk", "3. Likely lost", "Not flagged"]
    tier_colors = {"1. Gently slowing": STATUS_GOOD, "2. At risk": STATUS_WARNING,
                   "3. Likely lost": STATUS_CRITICAL, "Not flagged": MUTED}
    counts_ordered = [tier_counts.get(t, 0) for t in tier_order]
    axes[0].bar(range(len(tier_order)), counts_ordered, color=[tier_colors[t] for t in tier_order])
    axes[0].set_xticks(range(len(tier_order)))
    axes[0].set_xticklabels([t[3:] if t[0].isdigit() else t for t in tier_order], rotation=15, ha="right")
    axes[0].set_ylabel("Users")
    axes[0].set_title(f"Current at-risk tiers (n={len(df):,})", fontsize=11, loc="left")
    style_axes(axes[0])
    axes[0].grid(axis="x", visible=False)

    top_drift = drift_df.head(8).sort_values("psi")
    bar_colors = [STATUS_CRITICAL if s == "ALERT" else STATUS_WARNING if s == "WARN" else BLUE for s in top_drift["status"]]
    axes[1].barh(range(len(top_drift)), top_drift["psi"], color=bar_colors)
    axes[1].set_yticks(range(len(top_drift)))
    axes[1].set_yticklabels(top_drift["feature"])
    axes[1].axvline(PSI_WARN, color=STATUS_WARNING, linestyle="--", linewidth=1)
    axes[1].axvline(PSI_ALERT, color=STATUS_CRITICAL, linestyle="--", linewidth=1)
    axes[1].set_xlabel("PSI vs. training reference")
    axes[1].set_title("Top feature drift (highest 8)", fontsize=11, loc="left")
    style_axes(axes[1])
    axes[1].grid(axis="y", visible=False)

    fig.suptitle(f"Churn scoring pipeline — run {run_ts[:19]}Z", fontsize=10, color=SECONDARY_INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(CHART_DIR, "dashboard_latest.png"), dpi=150)
    plt.close(fig)
    print(f"\nDashboard refreshed: charts/dashboard_latest.png")


if __name__ == "__main__":
    main()
