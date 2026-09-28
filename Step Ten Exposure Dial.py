"""
=============================================================================
SCRIPT NAME: Step Ten Exposure Dial.py
=============================================================================

DESCRIPTION:
    Sets how much of the account the country-ETF book should occupy next month, from
    34-country market breadth, and hands that number to the Schwab trader.

    Rule (research: Experiments Deep Dive/Regime Breadth Overlay/FINDINGS.md, decided by
    Arjun 2026-09-27):
        breadth         = share of the 34 country indices whose total-return level is above
                          its 200-day EMA at the close of the last completed month
        target_exposure = clip(DIAL_LOW + (DIAL_HIGH - DIAL_LOW) * breadth, DIAL_LOW, DIAL_HIGH)
                        = 2 x breadth with the defaults (0% .. 200% of account value)
    The trader scales the country book (the weights in Latest_Country_Alpha_Weights, which
    sum to 1) by target_exposure. Above 1.0 the account borrows on margin; below 1.0 the rest
    sits in cash. Backtest 2003-10..2026-08, gross: Momentum 23.5%/yr vs 16.4% held, Sharpe
    0.88 vs 0.77, max drawdown -49% vs -69%; Value 17.2% vs 12.4%, 0.80 vs 0.67, -41% vs -62%.

    Runs AFTER Step FINALFINAL (it adds a sheet to FINALFINAL's output). Fails loudly (non-zero
    exit, nothing written) if the daily Bloomberg data do not reach the last month-end, if the
    Bloomberg column layout differs from the expected 34 tickers, or if fewer than MIN_VALID
    countries have a 200-day history. There is no fallback data source.

    The breadth definition reproduces the research exactly (weekday rows, local holidays
    forward-filled up to 10 days, EMA span 200 with adjust=False, a country counts once it has
    200 observations).

INPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/Master Database/Country Bloomberg Data Master T Daily.xlsx
        sheet 'Tot Return Index ' (daily total-return levels of the 34 country indices; refreshed
        with the monthly Bloomberg update)
    <repo>/T2_FINAL_T60_VALUE.xlsx  sheet 'Latest_Country_Alpha_Weights' (Step FINALFINAL output)

OUTPUT FILES:
    <repo>/T2_FINAL_T60_VALUE.xlsx  sheets added/replaced:
        'Exposure_Dial'   key/value table read by the trader (target_exposure, asof_date, breadth, ...)
        'Exposure_Detail' per-country index level, EMA200, above/valid flags, base and scaled weight
    <repo>/T2_exposure_dial_log.txt   one appended line per run

    (<repo> = /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value)

VERSION: 1.0
LAST UPDATED: 2026-09-27
AUTHOR: Claude Code (replaces the former 'Step Ten Create Final Report.py')

DEPENDENCIES: pandas, numpy, openpyxl
USAGE:
    cd "/Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value"
    venv/bin/python "Step Ten Exposure Dial.py"                  # as-of = last completed month-end
    venv/bin/python "Step Ten Exposure Dial.py" --asof 2026-08-31
    venv/bin/python "Step Ten Exposure Dial.py" --final-path /tmp/copy.xlsx   # test on a copy
=============================================================================
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---- account-specific configuration (the only lines that differ between the repos) ----
REPO = Path(__file__).resolve().parent
FINAL_PATH = REPO / "T2_FINAL_T60_VALUE.xlsx"
STRATEGY_LABEL = "T2 Factor Timing Fuzzy Value (Value, #167)"

# ---- rule parameters (identical in both repos) ----
DIAL_LOW = 0.0
DIAL_HIGH = 2.0
EMA_SPAN = 200
MIN_VALID = 30
MAX_ASOF_GAP_DAYS = 4        # data must reach within 4 calendar days of the month's last day
DAILY_FILE = Path("/Users/arjundivecha/Dropbox/AAA Backup/Master Database/Country Bloomberg Data Master T Daily.xlsx")
TRI_SHEET = "Tot Return Index "
DIAL_SHEET = "Exposure_Dial"
DETAIL_SHEET = "Exposure_Detail"
LOG_PATH = REPO / "T2_exposure_dial_log.txt"

# Bloomberg ticker (column order of the daily file) -> country name used in the weights
TICKER_COUNTRY = [
    ("MXSG Index", "Singapore"), ("MXAU Index", "Australia"), ("MSDUCA Index", "Canada"),
    ("MSDUGR Index", "Germany"), ("MSDUJN Index", "Japan"), ("MSDUSZ Index", "Switzerland"),
    ("MSDUUK Index", "U.K."), ("CCMP Index", "NASDAQ"), ("SPX Index", "U.S."),
    ("MSDUFR Index", "France"), ("MSDUNE Index", "Netherlands"), ("MSDUSW Index", "Sweden"),
    ("MSDUIT Index", "Italy"), ("MSEUSCF Index", "ChinaA"), ("MXCL Index", "Chile"),
    ("MXID Index", "Indonesia"), ("MXPH Index", "Philippines"), ("MXPL Index", "Poland"),
    ("RTY Index", "US SmallCap"), ("MXMY Index", "Malaysia"), ("MSEUSTW Index", "Taiwan"),
    ("MXMX Index", "Mexico"), ("MXKR Index", "Korea"), ("MXBR Index", "Brazil"),
    ("MXZA Index", "South Africa"), ("MSDUDE Index", "Denmark"), ("MXIN Index", "India"),
    ("MSELTCF Index", "ChinaH"), ("MXHK Index", "Hong Kong"), ("MXTH Index", "Thailand"),
    ("MXTR Index", "Turkey"), ("MSDUSP Index", "Spain"), ("MXVI Index", "Vietnam"),
    ("MXSA Index", "Saudi Arabia"),
]


class DialError(RuntimeError):
    pass


def load_tri():
    if not DAILY_FILE.exists():
        raise DialError(f"daily Bloomberg file not found: {DAILY_FILE}")
    raw = pd.read_excel(DAILY_FILE, TRI_SHEET, header=0)
    tickers = [str(c).strip() for c in raw.columns[1:1 + len(TICKER_COUNTRY)]]
    expected = [t for t, _ in TICKER_COUNTRY]
    if tickers != expected:
        bad = [(i, a, b) for i, (a, b) in enumerate(zip(tickers, expected)) if a != b]
        raise DialError(f"Bloomberg column layout changed; first mismatches (pos, found, expected): {bad[:5]}")
    fields = raw.iloc[0, 1:1 + len(expected)].astype(str).str.strip().unique()
    if list(fields) != ["tot_return_index_gross_dvds"]:
        raise DialError(f"unexpected Bloomberg field row: {fields}")
    body = raw.iloc[1:]
    idx = pd.to_datetime(body.iloc[:, 0], errors="coerce")
    vals = body.iloc[:, 1:1 + len(expected)].apply(pd.to_numeric, errors="coerce")
    vals.columns = [c for _, c in TICKER_COUNTRY]
    vals.index = idx
    vals = vals[vals.index.notna()].sort_index()
    vals = vals[vals.index.dayofweek < 5]
    vals = vals.where(vals > 0)
    return vals


def resolve_asof(tri, asof_arg, today):
    last = tri.dropna(how="all").index.max()
    if asof_arg:
        asof = pd.Timestamp(asof_arg)
    else:
        month_end_of_last = last + pd.offsets.BMonthEnd(0)
        if last == month_end_of_last and last.to_period("M") == today.to_period("M"):
            target_month = today.to_period("M")          # run after the close of the month's last weekday
        else:
            target_month = today.to_period("M") - 1      # normal: previous calendar month
        asof = target_month.to_timestamp(how="end").normalize()
    month_rows = tri.index[(tri.index.to_period("M") == asof.to_period("M")) & (tri.index <= asof)]
    if len(month_rows) == 0:
        raise DialError(f"no daily data in the as-of month {asof.to_period('M')} (latest data {last.date()})")
    asof_row = month_rows.max()
    month_last_day = asof.to_period("M").to_timestamp(how="end").normalize()
    if asof_arg is None and (month_last_day - asof_row).days > MAX_ASOF_GAP_DAYS:
        raise DialError(f"daily Bloomberg data end {asof_row.date()}, more than {MAX_ASOF_GAP_DAYS} days before "
                        f"month-end {month_last_day.date()} — refresh '{DAILY_FILE.name}' and rerun")
    return asof_row, last


def compute_breadth(tri, asof_row):
    P = tri.loc[:asof_row].ffill(limit=10)
    ema = P.ewm(span=EMA_SPAN, adjust=False, ignore_na=True).mean()
    n_obs = P.notna().cumsum()
    row, e, n = P.loc[asof_row], ema.loc[asof_row], n_obs.loc[asof_row]
    valid = row.notna() & (n >= EMA_SPAN)
    above = (row > e) & valid
    if valid.sum() < MIN_VALID:
        raise DialError(f"only {int(valid.sum())} countries have {EMA_SPAN} days of history at {asof_row.date()} "
                        f"(need {MIN_VALID})")
    detail = pd.DataFrame({"Index Level": row, "EMA200": e, "Valid": valid, "Above EMA200": above})
    return float(above.sum() / valid.sum()), int(above.sum()), int(valid.sum()), detail


def main():
    ap = argparse.ArgumentParser(description="Step Ten: breadth exposure dial")
    ap.add_argument("--asof", default=None, help="override the as-of date (YYYY-MM-DD)")
    ap.add_argument("--final-path", default=None, help="write to this copy of the FINAL workbook (testing)")
    ap.add_argument("--today", default=None, help="pretend today is this date (testing the staleness check)")
    args = ap.parse_args()
    global FINAL_PATH
    if args.final_path:
        FINAL_PATH = Path(args.final_path)
    log_path = FINAL_PATH.parent / LOG_PATH.name
    today = pd.Timestamp(args.today) if args.today else pd.Timestamp(datetime.now().date())

    if not FINAL_PATH.exists():
        raise DialError(f"{FINAL_PATH.name} not found — run Step FINALFINAL first")
    weights = pd.read_excel(FINAL_PATH, "Latest_Country_Alpha_Weights")
    if not {"Country", "Country Weight"} <= set(weights.columns):
        raise DialError(f"{FINAL_PATH.name} lacks Country / Country Weight columns")
    unknown = set(weights["Country"]) - {c for _, c in TICKER_COUNTRY}
    if unknown:
        raise DialError(f"countries in {FINAL_PATH.name} not in the breadth universe: {sorted(unknown)}")

    tri = load_tri()
    asof_row, last = resolve_asof(tri, args.asof, today)
    breadth, n_above, n_valid, detail = compute_breadth(tri, asof_row)
    exposure = float(np.clip(DIAL_LOW + (DIAL_HIGH - DIAL_LOW) * breadth, DIAL_LOW, DIAL_HIGH))

    w = weights.set_index("Country")["Country Weight"].astype(float)
    detail["Base Weight"] = w.reindex(detail.index).fillna(0.0)
    detail["Scaled Weight"] = detail["Base Weight"] * exposure
    detail.index.name = "Country"
    computed_at = datetime.now().isoformat(timespec="seconds")
    dial = pd.DataFrame([
        ("target_exposure", exposure),
        ("breadth", breadth),
        ("n_above", n_above),
        ("n_valid", n_valid),
        ("asof_date", asof_row.date().isoformat()),
        ("dial_low", DIAL_LOW),
        ("dial_high", DIAL_HIGH),
        ("rule", f"clip({DIAL_LOW} + ({DIAL_HIGH}-{DIAL_LOW}) x breadth, {DIAL_LOW}, {DIAL_HIGH}); "
                 f"breadth = share of 34 country indices above {EMA_SPAN}-day EMA"),
        ("computed_at", computed_at),
        ("strategy", STRATEGY_LABEL),
        ("source_file", str(DAILY_FILE)),
        ("source_last_date", last.date().isoformat()),
        ("base_weight_sum", float(w.sum())),
    ], columns=["Key", "Value"])

    with pd.ExcelWriter(FINAL_PATH, mode="a", engine="openpyxl", if_sheet_exists="replace") as xw:
        dial.to_excel(xw, sheet_name=DIAL_SHEET, index=False)
        detail.reset_index().to_excel(xw, sheet_name=DETAIL_SHEET, index=False)
    with open(log_path, "a") as f:
        f.write(f"{computed_at}\tasof={asof_row.date()}\tbreadth={breadth:.4f} ({n_above}/{n_valid})\t"
                f"target_exposure={exposure:.4f}\n")

    print(f"Step Ten — {STRATEGY_LABEL}")
    print(f"  as-of {asof_row.date()} (daily data through {last.date()})")
    print(f"  breadth {breadth:.1%} ({n_above} of {n_valid} countries above their {EMA_SPAN}-day EMA)")
    print(f"  TARGET EXPOSURE {exposure:.1%} of account value  (range {DIAL_LOW:.0%}..{DIAL_HIGH:.0%})")
    print(f"  written to {FINAL_PATH} [{DIAL_SHEET}, {DETAIL_SHEET}]")


if __name__ == "__main__":
    try:
        main()
    except DialError as e:
        print(f"STEP TEN FAILED: {e}", file=sys.stderr)
        sys.exit(1)
