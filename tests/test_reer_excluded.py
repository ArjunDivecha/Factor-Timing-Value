"""
tests/test_reer_excluded.py — Guard: REER_CS/REER_TS must not be used anywhere in production (Arjun, 2026-10-14).
Checks the code (Steps Three, Four, Six exclude or reject REER; Step Five eligibility Max=0) and, when the data
files are present, that no production output carries a REER column.
INPUTS (repo root, read-only): Step Three Top20 Portfolios Fast.py, Step Four Create Monthly Top20 Returns FAST.py,
  Step Six Create Country alphas from Factor alphas.py, Step Factor Categories.xlsx, T2_Optimizer.xlsx, T60.xlsx,
  T2_rolling_window_weights.xlsx (data files skipped if absent, e.g. on a fresh clone)
OUTPUTS: none.   USAGE: venv/bin/python -m pytest tests/test_reer_excluded.py -q
"""
import os, re
import pandas as pd, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def src(name): return open(os.path.join(ROOT, name), encoding="utf-8").read()

def test_step_three_skips_reer():
    s = src("Step Three Top20 Portfolios Fast.py"); block = s[s.index("skip_variables = ["):s.index("]", s.index("skip_variables = ["))]
    assert '"REER_CS"' in block and '"REER_TS"' in block

def test_step_four_excludes_reer():
    s = src("Step Four Create Monthly Top20 Returns FAST.py")
    assert 'EXCLUDED_FACTORS = {"REER_CS", "REER_TS"}' in s and "- EXCLUDED_FACTORS" in s

def test_step_six_rejects_reer():
    assert "REER must not contribute to country alphas" in src("Step Six Create Country alphas from Factor alphas.py")

def test_eligibility_reer_max_zero():
    c = pd.read_excel(os.path.join(ROOT, "Step Factor Categories.xlsx"))
    r = c[c["Factor Name"].str.startswith("REER")]
    assert set(r["Factor Name"]) == {"REER_CS", "REER_TS"} and (r["Max"] == 0).all()

@pytest.mark.parametrize("fname", ["T2_Optimizer.xlsx", "T60.xlsx", "T2_rolling_window_weights.xlsx"])
def test_outputs_have_no_reer_column(fname):
    p = os.path.join(ROOT, fname)
    if not os.path.exists(p): pytest.skip(f"{fname} not present")
    cols = pd.read_excel(p, nrows=1).columns
    assert not [c for c in cols if "REER" in str(c).upper()]


def test_step_four_gap_fill_uses_eligible_factors_only():
    s = src("Step Four Create Monthly Top20 Returns FAST.py")
    assert "row.fillna(row.mean())" not in s
    assert s.count("row.fillna(row[elig_cols].mean())") == 2 and "def _eligible_factor_columns" in s
