import csv
import sys

import pywarpdb

# WarpDB requires a CUDA device at runtime. Skip cleanly when none is
# available (e.g. GPU-less CI) instead of failing during construction.
if pywarpdb.cuda_device_count() == 0:
    print("[SKIP] test_python.py: no CUDA device available")
    sys.exit(0)

# Constructor with explicit schema and parse policy
schema = [pywarpdb.DataType.Float32, pywarpdb.DataType.Int32]
db = pywarpdb.WarpDB("data/test.csv", schema, pywarpdb.ParsePolicy.Strict)

# Simple expression query
result = db.query("price + 1")
with open("data/test.csv", newline="") as f:
    reader = csv.DictReader(f)
    expected = [float(row["price"]) + 1 for row in reader]
assert len(result) == len(expected)
assert result == expected

# SQL query
sql_res = db.query_sql("SELECT price FROM test ORDER BY price ASC")
with open("data/test.csv", newline="") as f:
    reader = csv.DictReader(f)
    expected_sql = sorted(float(row["price"]) for row in reader)
assert sql_res == expected_sql

# Permissive parse policy handles malformed rows
malformed_schema = [pywarpdb.DataType.Float32, pywarpdb.DataType.Int32]
db_perm = pywarpdb.WarpDB("data/malformed.csv", malformed_schema, pywarpdb.ParsePolicy.Permissive)
perm_res = db_perm.query("price + quantity")
assert perm_res == [11.5, 20.0, 2.5]

