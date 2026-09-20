"""Scratch helper: run a .sql file statement-by-statement against the project DB,
printing timing for each statement. Not a deliverable itself.
Usage: .venv/Scripts/python.exe scripts/_run_sql_file.py sql/02_feature_engineering.sql
"""
import os
import sys
import time
import re

from dotenv import load_dotenv
import mysql.connector

load_dotenv()

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    raw = f.read()

# Strip -- line comments, then split on ';'
lines = []
for line in raw.splitlines():
    stripped = line.strip()
    if stripped.startswith("--"):
        continue
    lines.append(line)
cleaned = "\n".join(lines)
statements = [s.strip() for s in cleaned.split(";") if s.strip()]

conn = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST"),
    port=int(os.getenv("MYSQL_PORT")),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DATABASE"),
)
cur = conn.cursor()

for i, stmt in enumerate(statements, 1):
    label = stmt.strip().split("\n")[0][:80]
    t0 = time.time()
    cur.execute(stmt)
    if cur.with_rows:
        rows = cur.fetchall()
        cols = cur.column_names
        print(f"[{i}/{len(statements)}] ({time.time()-t0:.1f}s) {label}")
        print("  cols:", cols)
        for row in rows[:10]:
            print("  ", row)
    else:
        conn.commit()
        print(f"[{i}/{len(statements)}] ({time.time()-t0:.1f}s) {label}  -> {cur.rowcount} rows affected")

conn.close()
print("DONE")
