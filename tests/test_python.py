import pywarpdb

db = pywarpdb.WarpDB("data/test.csv")
result = db.query("price + 1")
print("rows", len(result))
