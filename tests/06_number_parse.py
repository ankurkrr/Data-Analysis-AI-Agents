# tests/06_number_parse.py
import tests.bootstrap_path
from app.utils.number_parsing import parse_inr_number
tests = [
    "Total Revenue ₹ 61,437 Cr",
    "Net Profit: ₹ 12,345 Cr",
    "Revenue 61,437",
    "EPS 12.34"
]
for t in tests:
    print(t, "->", parse_inr_number(t))
