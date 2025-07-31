import csv
import pywarpdb

db = pywarpdb.WarpDB("data/test.csv")
result = db.query("price + 1")

with open("data/test.csv", newline="") as f:
    reader = csv.DictReader(f)
    expected = [float(row["price"]) + 1 for row in reader]

assert len(result) == len(expected)
assert result == expected

