-- Phase 2: per-user features + churn label, with a strict cutoff to avoid leakage.
--
-- LEAKAGE DESIGN — the central problem of this phase. Instacart has no calendar
-- date, so a normal "features before date X, label after date X" split is not
-- directly available. Adaptation, consistent with Phase 1's relative-time
-- approach: hold out each user's LAST observed order as "the future," and use
-- every order before it as "history." Concretely, per user:
--
--   cutoff_order_number = lifetime_orders - 1        (their second-to-last order)
--   history             = orders 1 .. cutoff_order_number   (features built only from these)
--   label event          = the transition from the cutoff order to the final order
--
-- This reuses Phase 1's `last_gap_days` (sql/01_cohort_retention.sql, section 5)
-- almost exactly, EXCEPT last_gap_days there was framed as a snapshot signal;
-- here it becomes the literal supervision target:
--
--   churned = 1 if last_gap_days >= 30 else 0        (same 30-day threshold as Phase 1)
--
-- Every feature below is computed with order_number <= cutoff_order_number (or,
-- equivalently, days_since_first_order <= cutoff_day) — never touching the held-out
-- final order. This is the leakage boundary; it must never be crossed when adding
-- a feature later.
--
-- Coverage note: minimum lifetime_orders across all users is 4 (Phase 1 finding),
-- so every user has >= 3 orders of history pre-cutoff — thin for some trend
-- features (e.g. "last 3 vs prior 3" baskets/gaps) but never empty.

USE repeat_purchase_churn;

-- ============================================================================
-- 1. user_cutoff: the leakage boundary itself. One row per user: which order
--    is the cutoff, what "day" that cutoff falls on (their own relative
--    clock), and the label.
-- ============================================================================
DROP TABLE IF EXISTS user_cutoff;
CREATE TABLE user_cutoff AS
SELECT
    t.user_id,
    t.lifetime_orders - 1 AS cutoff_order_number,
    cutoff.days_since_first_order AS cutoff_day,
    lg.last_gap_days,
    CASE WHEN lg.last_gap_days >= 30 THEN 1 ELSE 0 END AS churned
FROM user_tenure t
JOIN order_timeline cutoff
    ON cutoff.user_id = t.user_id AND cutoff.order_number = t.lifetime_orders - 1
JOIN (
    SELECT ot.user_id, ot.days_since_prior_order AS last_gap_days
    FROM order_timeline ot
    JOIN user_tenure ut ON ut.user_id = ot.user_id AND ot.order_number = ut.lifetime_orders
) lg ON lg.user_id = t.user_id;

ALTER TABLE user_cutoff ADD PRIMARY KEY (user_id);

-- ============================================================================
-- 2. user_features_history: base history rows, restricted to order_number <=
--    cutoff_order_number. Materialized because every feature block below
--    scans it.
-- ============================================================================
DROP TABLE IF EXISTS user_features_history;
CREATE TABLE user_features_history AS
SELECT ot.*
FROM order_timeline ot
JOIN user_cutoff uc
    ON uc.user_id = ot.user_id AND ot.order_number <= uc.cutoff_order_number;

ALTER TABLE user_features_history ADD INDEX (user_id);
ALTER TABLE user_features_history ADD INDEX (order_id);

-- ============================================================================
-- 3. Recency + tenure + frequency windows, all measured relative to
--    cutoff_day (each user's own clock — see Phase 1 note on why there is no
--    shared "today").
-- ============================================================================
DROP TABLE IF EXISTS user_features_rfm;
CREATE TABLE user_features_rfm AS
SELECT
    h.user_id,
    uc.cutoff_day,
    -- Recency: gap immediately before the cutoff order (last gap fully inside history).
    MAX(CASE WHEN h.order_number = uc.cutoff_order_number THEN h.days_since_prior_order END) AS recency_at_cutoff,
    -- Tenure: span of history observed, as of cutoff.
    uc.cutoff_day AS tenure_at_cutoff,
    -- Frequency: lifetime + rolling windows, all pre-cutoff.
    COUNT(*) AS lifetime_orders_pre_cutoff,
    SUM(CASE WHEN uc.cutoff_day - h.days_since_first_order <= 30 THEN 1 ELSE 0 END) AS orders_last_30d,
    SUM(CASE WHEN uc.cutoff_day - h.days_since_first_order <= 60 THEN 1 ELSE 0 END) AS orders_last_60d,
    SUM(CASE WHEN uc.cutoff_day - h.days_since_first_order <= 90 THEN 1 ELSE 0 END) AS orders_last_90d
FROM user_features_history h
JOIN user_cutoff uc ON uc.user_id = h.user_id
GROUP BY h.user_id, uc.cutoff_day, uc.cutoff_order_number;

ALTER TABLE user_features_rfm ADD PRIMARY KEY (user_id);

-- ============================================================================
-- 4. Basket size (avg + trend) and breadth (distinct categories/products,
--    repeat-product ratio). Restricted to eval_set IN ('prior','train')
--    within the pre-cutoff history, since only those orders have basket
--    contents (same constraint noted in Phase 1, sql/01 section 6).
--
--    Split into two steps rather than one query with three simultaneous
--    COUNT(DISTINCT ...) columns. MySQL builds a separate temp-table dedup
--    structure per distinct aggregate in a single GROUP BY, so three at once
--    multiplies the cost rather than adding it (measured: 48+ min and still
--    running on the 34M-row exploded join, killed and rewritten). Fix:
--    collapse to distinct (user_id, product_id) pairs once first — a much
--    smaller set — then count off of that.
-- ============================================================================
DROP TABLE IF EXISTS user_product_pairs;
CREATE TABLE user_product_pairs AS
SELECT DISTINCT h.user_id, op.product_id, p.department_id, p.aisle_id
FROM user_features_history h
JOIN order_products op ON op.order_id = h.order_id
JOIN products p ON p.product_id = op.product_id
WHERE h.eval_set IN ('prior', 'train');

ALTER TABLE user_product_pairs ADD INDEX (user_id);

DROP TABLE IF EXISTS user_features_breadth;
CREATE TABLE user_features_breadth AS
SELECT
    user_id,
    COUNT(DISTINCT department_id) AS distinct_departments,
    COUNT(DISTINCT aisle_id) AS distinct_aisles,
    COUNT(*) AS distinct_products
FROM user_product_pairs
GROUP BY user_id;

ALTER TABLE user_features_breadth ADD PRIMARY KEY (user_id);

DROP TABLE IF EXISTS user_features_basket;
CREATE TABLE user_features_basket AS
SELECT
    h.user_id,
    AVG(basket.n_items) AS avg_basket_size,
    -- Trend: avg basket size of the most recent <=3 pre-cutoff orders minus
    -- the avg of everything earlier. NULL when fewer than 4 basket-bearing
    -- orders exist pre-cutoff (thin history; left null rather than forced).
    AVG(CASE WHEN h.order_number > uc.cutoff_order_number - 3 THEN basket.n_items END)
        - AVG(CASE WHEN h.order_number <= uc.cutoff_order_number - 3 THEN basket.n_items END)
        AS basket_size_trend,
    SUM(op.reordered) / COUNT(*) AS repeat_product_ratio
FROM user_features_history h
JOIN user_cutoff uc ON uc.user_id = h.user_id
JOIN order_products op ON op.order_id = h.order_id
JOIN (
    SELECT order_id, COUNT(*) AS n_items
    FROM order_products
    GROUP BY order_id
) basket ON basket.order_id = h.order_id
WHERE h.eval_set IN ('prior', 'train')
GROUP BY h.user_id;

ALTER TABLE user_features_basket ADD PRIMARY KEY (user_id);

-- ============================================================================
-- 5. Temporal preferences: modal day-of-week / hour, weekday vs weekend mix.
--    Pre-cutoff only.
-- ============================================================================
DROP TABLE IF EXISTS user_features_temporal;
CREATE TABLE user_features_temporal AS
SELECT
    h.user_id,
    AVG(CASE WHEN h.order_dow IN (0, 6) THEN 1.0 ELSE 0.0 END) AS weekend_order_ratio,
    AVG(h.order_hour_of_day) AS avg_order_hour
FROM user_features_history h
GROUP BY h.user_id;

ALTER TABLE user_features_temporal ADD PRIMARY KEY (user_id);

-- ============================================================================
-- 6. Trajectory: is the inter-order gap widening? Compares the most recent
--    pre-cutoff gap(s) to earlier pre-cutoff gaps. Spec calls this "usually
--    the strongest signal" — kept as its own block so it's easy to isolate
--    in the notebook's feature-importance discussion.
-- ============================================================================
DROP TABLE IF EXISTS user_features_trajectory;
CREATE TABLE user_features_trajectory AS
SELECT
    h.user_id,
    AVG(CASE WHEN h.order_number > uc.cutoff_order_number - 3 THEN h.days_since_prior_order END)
        - AVG(CASE WHEN h.order_number <= uc.cutoff_order_number - 3 THEN h.days_since_prior_order END)
        AS gap_trend
FROM user_features_history h
JOIN user_cutoff uc ON uc.user_id = h.user_id
WHERE h.days_since_prior_order IS NOT NULL
GROUP BY h.user_id;

ALTER TABLE user_features_trajectory ADD PRIMARY KEY (user_id);

-- ============================================================================
-- 7. Final assembly: one row per user, all feature blocks + label joined.
--    This is the table the Phase 2 notebook loads into pandas.
-- ============================================================================
DROP TABLE IF EXISTS user_features;
CREATE TABLE user_features AS
SELECT
    uc.user_id,
    uc.churned,
    uc.last_gap_days,
    r.recency_at_cutoff,
    r.tenure_at_cutoff,
    r.lifetime_orders_pre_cutoff,
    r.orders_last_30d,
    r.orders_last_60d,
    r.orders_last_90d,
    b.avg_basket_size,
    b.basket_size_trend,
    br.distinct_departments,
    br.distinct_aisles,
    br.distinct_products,
    b.repeat_product_ratio,
    tm.weekend_order_ratio,
    tm.avg_order_hour,
    tr.gap_trend
FROM user_cutoff uc
JOIN user_features_rfm r ON r.user_id = uc.user_id
LEFT JOIN user_features_basket b ON b.user_id = uc.user_id
LEFT JOIN user_features_breadth br ON br.user_id = uc.user_id
LEFT JOIN user_features_temporal tm ON tm.user_id = uc.user_id
LEFT JOIN user_features_trajectory tr ON tr.user_id = uc.user_id;

ALTER TABLE user_features ADD PRIMARY KEY (user_id);

-- Sanity check: row count should equal user_tenure (206,209) and churn rate
-- should be roughly consistent with Phase 1's day-30 retention finding
-- (~74.67% retained at day 90 there was a different measure; expect this
-- label's churn rate to land in a plausible range, not 0% or 100%).
SELECT COUNT(*) AS n_users, AVG(churned) AS churn_rate FROM user_features;
