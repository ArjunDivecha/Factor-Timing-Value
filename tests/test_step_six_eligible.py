"""
tests/test_step_six_eligible.py — Value: country alphas (Step Six) use eligible factors only (Arjun, 2026-10-16).

What it checks: Step Eight Point Five sizes the U.S. weight from Step Six's country alphas, so a switched-off
(Max=0) factor must not contribute to them. Three tests: (1) the eligible_factors() rule keeps Max > 0 factors and
drops Max=0 and unlisted ones, on a temporary categories file; (2) Step Six's main() applies that filter; (3) when
the production outputs are newer than the Step Six change, the latest U.S. alpha in T2_Country_Alphas.xlsx equals
the eligible-only sum recomputed independently from T60.xlsx and T2_Top_20_Exposure.csv.

INPUTS (read-only; data checks skipped if absent or stale):
  /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value/Step Six Create Country alphas from Factor alphas.py
  /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value/Step Factor Categories.xlsx
  /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value/T60.xlsx
  /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value/T2_Top_20_Exposure.csv
  /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value/T2_Country_Alphas.xlsx
OUTPUTS: none (test 1 writes a throwaway cats.xlsx in pytest's tmp_path).
USAGE: python -m pytest tests/test_step_six_eligible.py -q
"""
import importlib.util, os
import numpy as np, pandas as pd, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP_SIX = os.path.join(ROOT, "Step Six Create Country alphas from Factor alphas.py")


def load_step_six():
    spec = importlib.util.spec_from_file_location("step_six", STEP_SIX)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_eligible_factors_rule(tmp_path):
    p = tmp_path / "cats.xlsx"
    pd.DataFrame({"Factor Name": ["A_CS", "B_CS", "C_TS"], "Max": [1.0, 0.0, 0.5]}).to_excel(p, index=False)
    elig, excl = load_step_six().eligible_factors(["A_CS", "B_CS", "C_TS", "D_TS"], str(p))
    assert elig == ["A_CS", "C_TS"] and excl == ["B_CS", "D_TS"]   # Max=0 and unlisted are excluded


def test_main_filters_to_eligible():
    s = open(STEP_SIX, encoding="utf-8").read()
    assert "factor_names, excluded = eligible_factors(factor_names)" in s


def test_output_matches_eligible_only_alphas():
    files = ["T60.xlsx", "T2_Top_20_Exposure.csv", "T2_Country_Alphas.xlsx"]
    if not all(os.path.exists(os.path.join(ROOT, f)) for f in files): pytest.skip("data files not present")
    if os.path.getmtime(os.path.join(ROOT, "T2_Country_Alphas.xlsx")) < os.path.getmtime(STEP_SIX):
        pytest.skip("T2_Country_Alphas.xlsx predates the Step Six change; rerun the pipeline")
    t60 = pd.read_excel(os.path.join(ROOT, "T60.xlsx")); t60 = t60.rename(columns={t60.columns[0]: "Date"})
    t60["Date"] = pd.to_datetime(t60["Date"])
    ex = pd.read_csv(os.path.join(ROOT, "T2_Top_20_Exposure.csv")); ex["Date"] = pd.to_datetime(ex["Date"])
    elig, _ = load_step_six().eligible_factors([c for c in t60.columns[1:] if c in ex.columns],
                                              os.path.join(ROOT, "Step Factor Categories.xlsx"))
    d = sorted(set(t60["Date"]) & set(ex["Date"]))[-1]
    a = t60.loc[t60["Date"] == d, elig].iloc[0]
    us = ex[(ex["Date"] == d) & (ex["Country"] == "U.S.")][elig].iloc[0]
    want = float(np.nansum(us.fillna(ex.loc[ex["Date"] == d, elig].mean()) * a))
    out = pd.read_excel(os.path.join(ROOT, "T2_Country_Alphas.xlsx")); out["date"] = pd.to_datetime(out["date"])
    assert abs(out.loc[out["date"] == d, "U.S."].iloc[0] - want) < 1e-9
