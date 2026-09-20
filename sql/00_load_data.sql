-- Schema + load script for the Instacart Market Basket Analysis dataset.
-- Run after scripts/00_download_data.py has populated data/raw/, AND after copying
-- those same CSVs into MySQL's server-side upload directory (SHOW VARIABLES LIKE
-- 'secure_file_priv' to find it) — server-side LOAD DATA INFILE reads straight off
-- local disk instead of streaming row-by-row through the client connection, which
-- is dramatically faster for the 32M-row order_products table.
--
-- order_products' secondary index is added AFTER the bulk load (not in the CREATE
-- TABLE) because InnoDB builds a fresh index in one fast sorted pass, whereas
-- maintaining it incrementally during a 32M-row insert is much slower.
--
--   mysql -u root -p < sql/00_load_data.sql

CREATE DATABASE IF NOT EXISTS repeat_purchase_churn;
USE repeat_purchase_churn;

DROP TABLE IF EXISTS order_products;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS aisles;
DROP TABLE IF EXISTS departments;

CREATE TABLE aisles (
    aisle_id INT PRIMARY KEY,
    aisle VARCHAR(255)
);

CREATE TABLE departments (
    department_id INT PRIMARY KEY,
    department VARCHAR(255)
);

CREATE TABLE products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(500),
    aisle_id INT,
    department_id INT,
    INDEX (aisle_id),
    INDEX (department_id)
);

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    user_id INT,
    eval_set VARCHAR(10),
    order_number INT,
    order_dow TINYINT,
    order_hour_of_day TINYINT,
    days_since_prior_order FLOAT NULL,
    INDEX (user_id),
    INDEX (eval_set)
);

CREATE TABLE order_products (
    order_id INT,
    product_id INT,
    add_to_cart_order INT,
    reordered TINYINT,
    PRIMARY KEY (order_id, product_id)
);

LOAD DATA INFILE 'aisles.csv'
INTO TABLE aisles FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 ROWS;

LOAD DATA INFILE 'departments.csv'
INTO TABLE departments FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 ROWS;

LOAD DATA INFILE 'products.csv'
INTO TABLE products FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 ROWS;

LOAD DATA INFILE 'orders.csv'
INTO TABLE orders FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 ROWS
(order_id, user_id, eval_set, order_number, order_dow, order_hour_of_day, @days_since_prior_order)
SET days_since_prior_order = NULLIF(@days_since_prior_order, '');

LOAD DATA INFILE 'order_products__prior.csv'
INTO TABLE order_products FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 ROWS;

LOAD DATA INFILE 'order_products__train.csv'
INTO TABLE order_products FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 ROWS;

ALTER TABLE order_products ADD INDEX (product_id);
