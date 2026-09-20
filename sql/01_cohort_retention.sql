-- Phase 1 analytical SQL: cohort/retention, RFM, order-gap distribution, category retention.
--
-- Data constraint driving the design: Instacart's `orders` table has no absolute calendar
-- date, only order_number (sequence) and days_since_prior_order (relative gap in days).
-- So "signup cohort by week/month" isn't possible. Instead we build each user's order
-- timeline in RELATIVE time (days since their own first order) via a running sum of
-- days_since_prior_order. This is the standard workaround for this dataset, and it has a
-- real advantage: it isolates lifecycle stage from seasonality.
--
-- eval_set note: 'prior' + 'train' orders have product-level data (order_products);
-- 'test' orders (75,000 of them, one per held-out user) do not — that's fine for anything
-- based on order occurrence/timing (retention, RFM, gap distribution), but category-level
-- analysis must restrict to 'prior'/'train' orders since only those have basket contents.

USE repeat_purchase_churn;

-- ============================================================================
-- 1. order_timeline: per-order cumulative days since the user's first order,
--    plus each user's lifetime order count. Materialized because it's reused
--    by every query below and expensive to recompute (window fn over 3.4M rows).
-- ============================================================================
DROP TABLE IF EXISTS order_timeline;
CREATE TABLE order_timeline AS
SELECT
    o.order_id,
    o.user_id,
    o.eval_set,
    o.order_number,
    o.order_dow,
    o.order_hour_of_day,
    o.days_since_prior_order,
    SUM(COALESCE(o.days_since_prior_order, 0)) OVER (
        PARTITION BY o.user_id ORDER BY o.order_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS days_since_first_order,
    MAX(o.order_number) OVER (PARTITION BY o.user_id) AS lifetime_orders
FROM orders o;

ALTER TABLE order_timeline ADD PRIMARY KEY (order_id);
ALTER TABLE order_timeline ADD INDEX (user_id);

-- ============================================================================
-- 2. user_tenure: one row per user — their tenure span (days between first and
--    last known order) and lifetime order count. tenure_days is the basis for
--    the retention/survival curve: retention(T) = P(tenure_days >= T).
-- ============================================================================
DROP TABLE IF EXISTS user_tenure;
CREATE TABLE user_tenure AS
SELECT
    user_id,
    MAX(days_since_first_order) AS tenure_days,
    MAX(lifetime_orders) AS lifetime_orders
FROM order_timeline
GROUP BY user_id;

ALTER TABLE user_tenure ADD PRIMARY KEY (user_id);

-- ============================================================================
-- 3. Retention / survival curve at fixed day checkpoints.
--    "Retained at day T" = the user's order history spans at least T days,
--    i.e. they were still placing orders that far out from acquisition.
-- ============================================================================
SELECT
    (SELECT COUNT(*) FROM user_tenure) AS cohort_size,
    SUM(tenure_days >= 7)  / (SELECT COUNT(*) FROM user_tenure) AS retained_day_7,
    SUM(tenure_days >= 14) / (SELECT COUNT(*) FROM user_tenure) AS retained_day_14,
    SUM(tenure_days >= 30) / (SELECT COUNT(*) FROM user_tenure) AS retained_day_30,
    SUM(tenure_days >= 60) / (SELECT COUNT(*) FROM user_tenure) AS retained_day_60,
    SUM(tenure_days >= 90) / (SELECT COUNT(*) FROM user_tenure) AS retained_day_90
FROM user_tenure;

-- ============================================================================
-- 4. Order-gap distribution — defines what a "normal" gap between orders is,
--    which in turn defines the churn threshold.
-- ============================================================================
SELECT
    days_since_prior_order AS gap_days,
    COUNT(*) AS n
FROM order_timeline
WHERE days_since_prior_order IS NOT NULL
GROUP BY days_since_prior_order
ORDER BY gap_days;

-- ============================================================================
-- 5. RFM segmentation. No price data exists in Instacart, so "Monetary" is
--    proxied by average basket size (items per order), not spend.
--
--    Recency cannot be measured against a real "today" — there is no shared
--    calendar clock, only per-user relative time starting at each user's own
--    first order. (A first attempt used `global_max_tenure - user_tenure`,
--    which is wrong: it's just an inverted relabeling of tenure itself, and
--    produced a nonsensical "79% likely lost" result. See PROGRESS_LOG.md.)
--
--    Fix: use `last_gap_days` — the gap observed immediately before each
--    user's final known order. This is self-relative (needs no shared clock)
--    and answers a defensible question: by the time we stop observing this
--    user, had their cadence already started widening? This is also the
--    same self-relative "gap widening" signal the spec calls the strongest
--    churn indicator, and the definition Phase 2/3's label will build on.
-- ============================================================================
DROP TABLE IF EXISTS user_rfm;
CREATE TABLE user_rfm AS
SELECT
    t.user_id,
    lg.last_gap_days,
    t.lifetime_orders AS frequency,
    b.avg_basket_size AS monetary_proxy
FROM user_tenure t
JOIN (
    SELECT ot.user_id, ot.days_since_prior_order AS last_gap_days
    FROM order_timeline ot
    JOIN user_tenure ut ON ut.user_id = ot.user_id AND ot.order_number = ut.lifetime_orders
) lg ON lg.user_id = t.user_id
JOIN (
    SELECT ot.user_id, AVG(basket.n_items) AS avg_basket_size
    FROM order_timeline ot
    JOIN (
        SELECT order_id, COUNT(*) AS n_items
        FROM order_products
        GROUP BY order_id
    ) basket ON basket.order_id = ot.order_id
    WHERE ot.eval_set IN ('prior', 'train')
    GROUP BY ot.user_id
) b ON b.user_id = t.user_id;

ALTER TABLE user_rfm ADD PRIMARY KEY (user_id);

-- ============================================================================
-- 6. Category-level retention — do "staple" buyers (produce/dairy) retain
--    better than one-off/discretionary category buyers? Restricted to
--    prior/train orders since only those have basket contents.
-- ============================================================================
SELECT
    d.department,
    COUNT(DISTINCT ot.user_id) AS n_users,
    AVG(t.tenure_days) AS avg_tenure_days,
    AVG(t.lifetime_orders) AS avg_lifetime_orders
FROM order_products op
JOIN order_timeline ot ON ot.order_id = op.order_id AND ot.eval_set IN ('prior', 'train')
JOIN products p ON p.product_id = op.product_id
JOIN departments d ON d.department_id = p.department_id
JOIN user_tenure t ON t.user_id = ot.user_id
GROUP BY d.department
ORDER BY avg_tenure_days DESC;
