"""
tests/test_audit_fixes.py — Guards for the 2026-10-14 fixes after Gemini's Step 0-9 audit (shared by the Momentum
and Value repos). Source-level checks always run; data-level checks are skipped when the data files are absent.
INPUTS (repo root, read-only): Step One Create T2Master.py, Step Two Create Normalized Tidy.py,
  Step Three Top20 Portfolios Fast.py, Step Zero Create P2P Scores.py, T2 Master.xlsx, T2_Country_Final.xlsx
OUTPUTS: none.   USAGE: python -m pytest tests/test_audit_fixes.py -q
"""
import os, re, ast, subprocess, sys, tempfile, shutil
import pandas as pd, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def src(n): return open(os.path.join(ROOT, n), encoding="utf-8").read()
def list_literal(s, name):
    m = re.search(name + r": List\[str\] = (\[[^\]]*\])", s); assert m, name
    return ast.literal_eval(re.sub(r"#.*", "", m.group(1)))

def test_step_one_has_no_two_sided_cleaning():
    s = src("Step One Create T2Master.py")
    assert "APPLY_TWO_SIDED_CLEANING = False" in s
    assert "after_positions" not in s and "med_after" not in s          # no forward-looking regime filter
    assert "if APPLY_TWO_SIDED_CLEANING and sheet_name in process_sheets:" in s

def test_regime_log_labels_use_pass_a_mask():
    assert "'cross-sectional' if bool(xs_mask[col].iloc[i]) else 'time-series'" in src("Step One Create T2Master.py")

def test_currency_quotes_normalized():
    s = src("Step One Create T2Master.py")
    assert 'DIRECT_QUOTED_CCY = {"EUR", "GBP", "AUD", "NZD"}' in s and s.count("to_lcu_per_usd(") >= 3

def test_best_pe_not_double_lagged():
    assert list_literal(src("Step Two Create Normalized Tidy.py"), "lag_one_month") == ["P2P"]

def test_inversion_names_match_sheets():
    inv = list_literal(src("Step Two Create Normalized Tidy.py"), "invert_norm")
    assert len(inv) == len(set(inv)) and "Debt to EV" in inv and "Currency 12" in inv
    p = os.path.join(ROOT, "T2 Master.xlsx")
    if not os.path.exists(p): pytest.skip("T2 Master.xlsx not present")
    sheets = set(pd.ExcelFile(p).sheet_names)
    assert not [n for n in inv if n not in sheets], [n for n in inv if n not in sheets]

@pytest.mark.parametrize("script", ["Step Two Create Normalized Tidy.py", "Step Three Top20 Portfolios Fast.py"])
def test_failures_exit_nonzero(script):
    d = tempfile.mkdtemp()
    try:
        shutil.copy(os.path.join(ROOT, script), d)
        r = subprocess.run([sys.executable, script], cwd=d, capture_output=True, timeout=120)
        assert r.returncode != 0
    finally:
        shutil.rmtree(d)

def test_step_three_skip_list_matches_sheet_names():
    s = src("Step Three Top20 Portfolios Fast.py")
    assert '"Tot Return Index _CS", "Tot Return Index _TS"' in s and '"Tot Return Index_CS"' not in s

def test_p2p_window_is_twelve_observations():
    s = src("Step Zero Create P2P Scores.py")
    assert "iloc[i - 11 : i + 1]" in s and "iloc[-12:]" in s and "iloc[-13:]" not in s

def test_country_final_contributions_sum_to_capped_weight():
    p = os.path.join(ROOT, "T2_Country_Final.xlsx")
    if not os.path.exists(p): pytest.skip("T2_Country_Final.xlsx not present")
    cf = pd.read_excel(p); cf = cf[cf["Country"] != "TOTAL"]; fac = [c for c in cf.columns if c not in ("Country", "Weight")]
    assert (cf["Weight"] - cf[fac].sum(axis=1)).abs().max() < 1e-9
