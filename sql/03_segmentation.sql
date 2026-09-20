-- Phase 5 support query: each user's top department by pre-cutoff product
-- breadth (distinct products bought from that department), for "targeted
-- offer on their top category" tier actions. Reuses user_product_pairs
-- (materialized in sql/02_feature_engineering.sql, already deduplicated to
-- (user_id, product_id, department_id, aisle_id) pairs) rather than
-- rescanning order_products.

USE repeat_purchase_churn;

DROP TABLE IF EXISTS user_top_department;
CREATE TABLE user_top_department AS
SELECT user_id, department_id, n_products
FROM (
    SELECT
        user_id,
        department_id,
        COUNT(*) AS n_products,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY COUNT(*) DESC, department_id) AS rn
    FROM user_product_pairs
    GROUP BY user_id, department_id
) ranked
WHERE rn = 1;

ALTER TABLE user_top_department ADD PRIMARY KEY (user_id);

SELECT COUNT(*) FROM user_top_department;
