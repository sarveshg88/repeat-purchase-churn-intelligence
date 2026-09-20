# Repeat Purchase & Churn Intelligence

*A retention-scoring project built on grocery reorder data, framed as a product decision rather than a leaderboard metric.*

**Finding:** A LightGBM model beats an RFM-heuristic baseline by 43% relative PR-AUC (0.55 vs. 0.38); tuning the decision threshold against an explicit 5:1 cost matrix (a missed churner costs ~5x a wasted retention nudge) cuts total intervention cost 14% versus a default 0.5 threshold.

**Recommendation:** Route flagged users into 3 intervention tiers — *Gently slowing* (in-app reminder only, no spend), *At risk* (targeted offer on their top category), *Likely lost* (leave alone or one low-cost touch) — and hold out ~10% of each tier as a no-contact control group before scaling spend, since a before/after comparison alone can't separate users the nudge actually saved from users who would have returned anyway.

---

## Why this project

Most companies with a repeat-purchase product face the same underlying question: *which users are about to stop transacting, and which of them can we actually change?* This project answers it once, on public grocery-reorder data (Instacart), with the product reasoning — not just the model — as the deliverable.

See [`00_problem_framing.md`](./00_problem_framing.md) for the full framing: business question, North Star metric, guardrail, and the decision this project enables.

## Repo structure

| File | Purpose |
|---|---|
| `00_problem_framing.md` | Business framing, written before any code |
| `notebooks/01_cohorts_retention.ipynb` | Cohort/retention analysis, RFM, churn definition |
| `notebooks/02_features_model.ipynb` | Feature engineering, baselines, models, cost-matrix threshold tuning |
| `scripts/00_download_data.py` | Pulls the Instacart dataset via Kaggle API |
| `scripts/03_scoring_pipeline.py` | Re-scoring pipeline + drift checks (Phase 6) |
| `sql/` | Cohort/retention SQL, run against MySQL |
| `charts/` | Exported chart images used in the README/summary |

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

1. Get a Kaggle API token (kaggle.com/settings) → place at `~/.kaggle/kaggle.json`
2. Accept the competition rules at kaggle.com/c/instacart-market-basket-analysis/rules
3. `python scripts/00_download_data.py`
4. Copy `.env.example` to `.env` and fill in your local MySQL credentials
5. Load the raw CSVs into MySQL (see `sql/00_load_data.sql`)

## Dataset

[Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis) — ~3.4M orders, ~200K users, ~50K products, 32M order-product rows. Contains a `reordered` flag natively.

## Method summary

1. **Cohort & retention analysis** (SQL) — weekly signup cohorts, retention curves, RFM tiers, order-gap distribution, category-level retention
2. **Churn definition** — derived from the observed inter-order gap distribution, not an arbitrary cutoff
3. **Feature engineering** — recency/frequency/monetary/breadth/temporal/trajectory features, strict time cutoff to avoid leakage
4. **Baselines before models** — rule-based, then RFM heuristic — headline result is lift over the RFM baseline, not raw model accuracy
5. **Models** — logistic regression (interpretable) and gradient boosting (XGBoost/LightGBM)
6. **Evaluation as a product trade-off** — precision/recall/PR-AUC, an explicit FN:FP cost matrix, threshold tuned to that cost — not to F1
7. **Segmentation** — at-risk users split into intervention tiers with different actions and an incrementality/holdout proposal
8. **Automation** — scheduled re-scoring, feature-drift logging, refreshed summary output

## One-page summary

See [`summary.md`](./summary.md) — written for a PM, not a professor.
