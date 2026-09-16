#!/usr/bin/env python3
"""Independent comparison of an organiser-style run against a Kaggle-submitted CSV.
Shares no code with the package. Usage: independent_compare.py <run.csv> <submitted.csv>"""
import csv, hashlib, sys
a, b = sys.argv[1], sys.argv[2]
ra = list(csv.reader(open(a, newline=""))); rb = list(csv.reader(open(b, newline="")))
ha = hashlib.sha256(open(a, "rb").read()).hexdigest(); hb = hashlib.sha256(open(b, "rb").read()).hexdigest()
print(f"sha256      {ha}\n            {hb}  -> {'IDENTICAL' if ha == hb else 'different'}")
print(f"rows        {len(ra) - 1} vs {len(rb) - 1}")
print(f"header      {ra[0]} vs {rb[0]} -> {'same' if ra[0] == rb[0] else 'DIFFERENT'}")
order = [r[0] for r in ra[1:]] == [r[0] for r in rb[1:]]
print(f"row order   {'same' if order else 'DIFFERENT'}")
if order:
    diff = [(r[0], r[1], q[1]) for r, q in zip(ra[1:], rb[1:]) if r[1] != q[1]]
    print(f"predictions {len(ra) - 1 - len(diff)}/{len(ra) - 1} equal; differing rows: {diff}")
try:
    import pandas as p
    print(f"pandas.equals {p.read_csv(a).equals(p.read_csv(b))}")
except Exception as e:
    print(f"pandas check skipped ({e.__class__.__name__})")
