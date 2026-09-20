"""
Scratch script to validate Phase 1 analysis + chart generation before assembling
into the notebook deliverable. Not part of the final repo deliverables list.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

load_dotenv()
engine = create_engine(URL.create(
    "mysql+mysqlconnector",
    username=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    host=os.getenv("MYSQL_HOST"),
    port=int(os.getenv("MYSQL_PORT")),
    database=os.getenv("MYSQL_DATABASE"),
))

CHART_DIR = os.path.join(os.path.dirname(__file__), "..", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# Palette (validated default, from the dataviz skill)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"

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


# ---------------------------------------------------------------------------
# Chart 1: Retention / survival curve
# ---------------------------------------------------------------------------
retention = pd.read_sql("""
    SELECT
        (SELECT COUNT(*) FROM user_tenure) AS cohort_size,
        SUM(tenure_days >= 7)  / (SELECT COUNT(*) FROM user_tenure) AS d7,
        SUM(tenure_days >= 14) / (SELECT COUNT(*) FROM user_tenure) AS d14,
        SUM(tenure_days >= 30) / (SELECT COUNT(*) FROM user_tenure) AS d30,
        SUM(tenure_days >= 60) / (SELECT COUNT(*) FROM user_tenure) AS d60,
        SUM(tenure_days >= 90) / (SELECT COUNT(*) FROM user_tenure) AS d90
    FROM user_tenure
""", engine)

days = [7, 14, 30, 60, 90]
values = [retention.loc[0, f"d{d}"] * 100 for d in days]

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(days, values, color=BLUE, linewidth=2, marker="o", markersize=7, markerfacecolor=BLUE, markeredgecolor="#fcfcfb", markeredgewidth=1.5, zorder=3)
for d, v in zip(days, values):
    ax.annotate(f"{v:.1f}%", (d, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=INK)
ax.set_ylim(60, 102)
ax.set_xticks(days)
ax.set_xlabel("Days since first order")
ax.set_ylabel("% of cohort still active (tenure ≥ day)")
ax.set_title(f"Retention curve — n={retention.loc[0,'cohort_size']:,} users\n(dataset pre-filtered to users with ≥4 historical orders — see caveat)", fontsize=10, loc="left", color=SECONDARY_INK)
style_axes(ax)
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "01_retention_curve.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Chart 2: Order-gap distribution + churn threshold
# ---------------------------------------------------------------------------
gaps = pd.read_sql("""
    SELECT days_since_prior_order AS gap_days, COUNT(*) AS n
    FROM order_timeline WHERE days_since_prior_order IS NOT NULL
    GROUP BY gap_days ORDER BY gap_days
""", engine)

fig, ax = plt.subplots(figsize=(8, 4.2))
colors = [ORANGE if g == 30 else BLUE for g in gaps["gap_days"]]
ax.bar(gaps["gap_days"], gaps["n"], color=colors, width=0.85)
ax.axvline(x=30, color=STATUS_CRITICAL, linestyle="--", linewidth=1.2, zorder=0)
ax.text(29.5, gaps["n"].max() * 0.92, "churn threshold: 30 days\n(also Instacart's data cap)", ha="right", fontsize=9, color=STATUS_CRITICAL)
ax.set_xlabel("Days since previous order")
ax.set_ylabel("Number of orders")
ax.set_title("Order-gap distribution  (75% of gaps resolve within 15 days)", fontsize=11, loc="left")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K" if x >= 1000 else int(x)))
style_axes(ax)
ax.grid(axis="x", visible=False)
ax.grid(axis="y", color=GRID)
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "02_gap_distribution.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Chart 3: Risk tiers, based on each user's own last observed gap (self-
# relative — no shared "today" exists in this dataset). See sql comment for
# why this replaced an earlier flawed global-max-tenure recency definition.
# ---------------------------------------------------------------------------
tiers = pd.read_sql("""
    SELECT
        CASE
            WHEN last_gap_days <= 15 THEN '1. Active (normal cadence)'
            WHEN last_gap_days < 30 THEN '2. Gently slowing'
            ELSE '3. At/past churn threshold'
        END AS tier,
        COUNT(*) AS n_users
    FROM user_rfm
    GROUP BY tier ORDER BY tier
""", engine)

tier_colors = {"1. Active (normal cadence)": STATUS_GOOD, "2. Gently slowing": STATUS_WARNING, "3. At/past churn threshold": STATUS_CRITICAL}
fig, ax = plt.subplots(figsize=(7, 4.2))
x_pos = range(len(tiers))
bars = ax.bar(x_pos, tiers["n_users"], color=[tier_colors[t] for t in tiers["tier"]])
for b, n in zip(bars, tiers["n_users"]):
    ax.annotate(f"{n:,}\n({n/tiers['n_users'].sum()*100:.1f}%)", (b.get_x() + b.get_width()/2, b.get_height()), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
ax.set_ylabel("Number of users")
ax.set_title("Users by last-observed-gap risk tier", fontsize=11, loc="left")
ax.set_xticks(list(x_pos))
ax.set_xticklabels([t[3:] for t in tiers["tier"]])
style_axes(ax)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "03_risk_tiers.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Chart 4: Category-level retention
# ---------------------------------------------------------------------------
cat = pd.read_sql("""
    SELECT department, n_users, avg_tenure_days
    FROM (
        SELECT
            d.department,
            COUNT(DISTINCT ot.user_id) AS n_users,
            AVG(t.tenure_days) AS avg_tenure_days
        FROM order_products op
        JOIN order_timeline ot ON ot.order_id = op.order_id AND ot.eval_set IN ('prior','train')
        JOIN products p ON p.product_id = op.product_id
        JOIN departments d ON d.department_id = p.department_id
        JOIN user_tenure t ON t.user_id = ot.user_id
        GROUP BY d.department
    ) x
    ORDER BY avg_tenure_days DESC
""", engine)

fig, ax = plt.subplots(figsize=(7, 7))
y_pos = range(len(cat))
ax.barh(y_pos, cat["avg_tenure_days"], color=BLUE)
ax.set_yticks(y_pos)
ax.set_yticklabels(cat["department"])
ax.invert_yaxis()
ax.set_xlabel("Average tenure (days)")
ax.set_title("Avg. user tenure by department purchased from", fontsize=11, loc="left")
style_axes(ax)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "04_category_retention.png"), dpi=150)
plt.close(fig)

print("Charts saved to", CHART_DIR)
print(retention.to_string())
print(tiers.to_string())
