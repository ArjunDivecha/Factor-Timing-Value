"""
tests/test_us_band.py — Value only: the U.S. weight must stay within [25%, 50%] (Arjun, 2026-10-14).
INPUTS (repo root): Step Eight Point Five Write Country Weights US Adjustment.py, T2_Final_Country_Weights.xlsx
(the output check is skipped if the data file is absent). OUTPUTS: none.
USAGE: python -m pytest tests/test_us_band.py -q
"""
import os, pandas as pd, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_band_constants_and_clamp():
    s = open(os.path.join(ROOT, "Step Eight Point Five Write Country Weights US Adjustment.py"), encoding="utf-8").read()
    assert "US_WEIGHT_MIN = 0.25" in s and "US_WEIGHT_MAX = 0.50" in s
    assert "max(US_WEIGHT_MIN, min(US_WEIGHT_MAX, new_us_weight))" in s and "max(0.0, min(1.0, new_us_weight))" not in s

def test_output_us_weight_in_band():
    p = os.path.join(ROOT, "T2_Final_Country_Weights.xlsx")
    if not os.path.exists(p): pytest.skip("weights file not present")
    w = pd.read_excel(p, sheet_name="All Periods", index_col=0).fillna(0)
    live = w.sum(axis=1) > 0.99
    us = w.loc[live, "U.S."]
    assert us.min() >= 0.25 - 1e-9 and us.max() <= 0.50 + 1e-9
