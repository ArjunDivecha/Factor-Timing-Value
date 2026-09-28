"""
=============================================================================
SCRIPT NAME: Step Ten Exposure Dial.py
=============================================================================

DESCRIPTION:
    Sets how much of the account the country-ETF book should occupy next month, from
    34-country market breadth, and hands that number to the Schwab trader.

    Rule (research: Experiments Deep Dive/Regime Breadth Overlay/FINDINGS.md, decided by
    Arjun 2026-09-27; data source switched to yfinance 2026-09-28):
        breadth         = share of the 34 country ETFs (the trader's own tickers, AssetList.xlsx
                          sheet 'Yahoo') whose dividend-adjusted close is above its 200-day EMA
                          at the close of the last completed month
        target_exposure = clip(DIAL_LOW + (DIAL_HIGH - DIAL_LOW) * breadth, DIAL_LOW, DIAL_HIGH)
                        = 2 x breadth with the defaults (0% .. 200% of account value)
    The trader scales the country book (the weights in Latest_Country_Alpha_Weights, which sum
    to 1) by target_exposure: above 1.0 the account borrows on margin, below 1.0 the rest sits
    in cash.
    Backtests: 2003-10..2026-08 (index breadth) Momentum 23.5%/yr vs 16.4% held, Value 17.2%
    vs 12.4%; 2012..2026 (ETF breadth, all 34 ETFs live) about equal to 100% long on return
    (T2 14.9% vs 14.7%) with a lower worst drawdown (-29% vs -34%) and a lower Sharpe (0.77 vs
    0.86). ETF breadth tracks the research's index breadth closely (month-end correlation 0.99;
    the dial differs by >10 points in about 1 month in 5).

    Breadth definition: weekday closes, gaps forward-filled up to 10 days, EMA span 200 with
    adjust=False over ~5 years of history, and an ETF counts once it has 200 observations.
    The universe is the same 34 tickers in both repos (SPY/QQQ/IWM for the U.S./NASDAQ/US
    SmallCap slots). Value's trading overrides (VTV/VBR) do not change the signal.

    Runs AFTER Step FINALFINAL (it adds sheets to FINALFINAL's output). It stops loudly
    (non-zero exit, nothing written to the workbook) if the yfinance download fails or is
    incomplete, if the prices do not reach the last month-end, if AssetList.xlsx does not have
    34 tickers, or if fewer than MIN_VALID ETFs have a 200-day history. There is no fallback
    data source.

INPUT FILES:
    <repo>/AssetList.xlsx              sheet 'Yahoo' (34 country ETF tickers, trader order)
    <repo>/T2_FINAL_T60_VALUE.xlsx           sheet 'Latest_Country_Alpha_Weights' (Step FINALFINAL output)
    Yahoo Finance daily prices via yfinance (downloaded at run time, dividend-adjusted)

OUTPUT FILES:
    <repo>/T2_FINAL_T60_VALUE.xlsx  sheets added/replaced:
        'Exposure_Dial'   key/value table read by the trader (target_exposure, asof_date, breadth, ...)
        'Exposure_Detail' per-country ETF, adjusted close, EMA200, above/valid flags, base and scaled weight
    <repo>/outputs/exposure_dial_prices_YYYYMMDD.parquet   the exact prices used (audit trail)
    <repo>/T2_exposure_dial_log.txt   one appended line per run

    (<repo> = /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value)

VERSION: 2.0 (2026-09-28) — yfinance ETF prices replace the daily Bloomberg file. 1.0 (2026-09-27)
    read 'Country Bloomberg Data Master T Daily.xlsx'.
AUTHOR: Claude Code (replaces the former 'Step Ten Create Final Report.py')

DEPENDENCIES: pandas, numpy, openpyxl, pyarrow, yfinance
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
MAX_ASOF_GAP_DAYS = 4        # prices must reach within 4 calendar days of the month's last day
HISTORY_PERIOD = "5y"        # yfinance download window (EMA warm-up plus 200-day validity)
ASSET_LIST_PATH = REPO / "AssetList.xlsx"
DIAL_SHEET = "Exposure_Dial"
DETAIL_SHEET = "Exposure_Detail"
LOG_NAME = "T2_exposure_dial_log.txt"
PRICES_DIR = REPO / "outputs"

# Country order of AssetList.xlsx 'Yahoo' — identical to Step Schwab Trading.load_country_etf_mapping
COUNTRIES = [
    'Singapore', 'Australia', 'Canada', 'Germany', 'Japan',
    'Switzerland', 'U.K.', 'NASDAQ', 'U.S.', 'France', 'Netherlands',
    'Sweden', 'Italy', 'ChinaA', 'Chile', 'Indonesia', 'Philippines',
    'Poland', 'US SmallCap', 'Malaysia', 'Taiwan', 'Mexico', 'Korea',
    'Brazil', 'South Africa', 'Denmark', 'India', 'ChinaH', 'Hong Kong',
    'Thailand', 'Turkey', 'Spain', 'Vietnam', 'Saudi Arabia',
]


class DialError(RuntimeError):
    pass


def load_tickers():
    tickers = pd.read_excel(ASSET_LIST_PATH, sheet_name="Yahoo")["Ticker"].astype(str).str.strip().tolist()
    if len(tickers) != len(COUNTRIES) or len(set(tickers)) != len(tickers):
        raise DialError(f"{ASSET_LIST_PATH.name} 'Yahoo' must list {len(COUNTRIES)} distinct tickers; found {len(tickers)}")
    return dict(zip(COUNTRIES, tickers))


def download_prices(tickers):
    import yfinance as yf
    try:
        data = yf.download(tickers, period=HISTORY_PERIOD, interval="1d", auto_adjust=True,
                           progress=False, threads=True)
    except Exception as exc:  # network / API failure
        raise DialError(f"yfinance download failed: {exc}") from exc
    if data is None or data.empty or "Close" not in data:
        raise DialError("yfinance returned no price data")
    px = data["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    missing = [t for t in tickers if t not in px.columns or px[t].notna().sum() == 0]
    if missing:
        raise DialError(f"yfinance returned no prices for {missing}")
    px = px[tickers]
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px = px[px.index.dayofweek < 5].sort_index()
    return px.where(px > 0)


def resolve_asof(px, asof_arg, today):
    last = px.dropna(how="all").index.max()
    if asof_arg:
        asof = pd.Timestamp(asof_arg)
    else:
        month_end_of_last = last + pd.offsets.BMonthEnd(0)
        if last == month_end_of_last and last.to_period("M") == today.to_period("M") and last < today:
            target_month = today.to_period("M")          # run after the month's last trading day
        else:
            target_month = today.to_period("M") - 1      # normal: previous calendar month
        asof = target_month.to_timestamp(how="end").normalize()
    month_rows = px.index[(px.index.to_period("M") == asof.to_period("M")) & (px.index <= asof)]
    if len(month_rows) == 0:
        raise DialError(f"no prices in the as-of month {asof.to_period('M')} (latest {last.date()})")
    asof_row = month_rows.max()
    month_last_day = asof.to_period("M").to_timestamp(how="end").normalize()
    if asof_arg is None and (month_last_day - asof_row).days > MAX_ASOF_GAP_DAYS:
        raise DialError(f"prices end {asof_row.date()}, more than {MAX_ASOF_GAP_DAYS} days before month-end "
                        f"{month_last_day.date()}")
    return asof_row, last


def compute_breadth(px, asof_row):
    P = px.loc[:asof_row].ffill(limit=10)
    ema = P.ewm(span=EMA_SPAN, adjust=False, ignore_na=True).mean()
    n_obs = P.notna().cumsum()
    row, e, n = P.loc[asof_row], ema.loc[asof_row], n_obs.loc[asof_row]
    valid = row.notna() & (n >= EMA_SPAN)
    above = (row > e) & valid
    if valid.sum() < MIN_VALID:
        raise DialError(f"only {int(valid.sum())} ETFs have {EMA_SPAN} days of history at {asof_row.date()} "
                        f"(need {MIN_VALID})")
    detail = pd.DataFrame({"Adj Close": row, "EMA200": e, "Valid": valid, "Above EMA200": above})
    return float(above.sum() / valid.sum()), int(above.sum()), int(valid.sum()), detail


def main():
    ap = argparse.ArgumentParser(description="Step Ten: breadth exposure dial (yfinance)")
    ap.add_argument("--asof", default=None, help="override the as-of date (YYYY-MM-DD)")
    ap.add_argument("--final-path", default=None, help="write to this copy of the FINAL workbook (testing)")
    ap.add_argument("--today", default=None, help="pretend today is this date (testing the staleness check)")
    args = ap.parse_args()
    final_path = Path(args.final_path) if args.final_path else FINAL_PATH
    testing = args.final_path is not None
    today = pd.Timestamp(args.today) if args.today else pd.Timestamp(datetime.now().date())

    if not final_path.exists():
        raise DialError(f"{final_path.name} not found — run Step FINALFINAL first")
    weights = pd.read_excel(final_path, "Latest_Country_Alpha_Weights")
    if not {"Country", "Country Weight"} <= set(weights.columns):
        raise DialError(f"{final_path.name} lacks Country / Country Weight columns")
    unknown = set(weights["Country"]) - set(COUNTRIES)
    if unknown:
        raise DialError(f"countries in {final_path.name} not in the breadth universe: {sorted(unknown)}")

    c2t = load_tickers()
    tickers = [c2t[c] for c in COUNTRIES]
    px = download_prices(tickers)
    asof_row, last = resolve_asof(px, args.asof, today)
    breadth, n_above, n_valid, detail = compute_breadth(px, asof_row)
    exposure = float(np.clip(DIAL_LOW + (DIAL_HIGH - DIAL_LOW) * breadth, DIAL_LOW, DIAL_HIGH))

    detail.index = pd.Index(COUNTRIES, name="Country")
    detail.insert(0, "ETF", tickers)
    w = weights.set_index("Country")["Country Weight"].astype(float)
    detail["Base Weight"] = w.reindex(detail.index).fillna(0.0)
    detail["Scaled Weight"] = detail["Base Weight"] * exposure
    computed_at = datetime.now().isoformat(timespec="seconds")
    prices_dir = final_path.parent / "outputs" if testing else PRICES_DIR
    prices_dir.mkdir(parents=True, exist_ok=True)
    prices_file = prices_dir / f"exposure_dial_prices_{datetime.now():%Y%m%d}.parquet"
    px.to_parquet(prices_file)
    dial = pd.DataFrame([
        ("target_exposure", exposure),
        ("breadth", breadth),
        ("n_above", n_above),
        ("n_valid", n_valid),
        ("asof_date", asof_row.date().isoformat()),
        ("dial_low", DIAL_LOW),
        ("dial_high", DIAL_HIGH),
        ("rule", f"clip({DIAL_LOW} + ({DIAL_HIGH}-{DIAL_LOW}) x breadth, {DIAL_LOW}, {DIAL_HIGH}); "
                 f"breadth = share of 34 country ETFs above {EMA_SPAN}-day EMA (adjusted closes)"),
        ("computed_at", computed_at),
        ("strategy", STRATEGY_LABEL),
        ("source_file", f"yfinance (auto_adjust, period={HISTORY_PERIOD}); snapshot {prices_file}"),
        ("source_last_date", last.date().isoformat()),
        ("base_weight_sum", float(w.sum())),
    ], columns=["Key", "Value"])

    with pd.ExcelWriter(final_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as xw:
        dial.to_excel(xw, sheet_name=DIAL_SHEET, index=False)
        detail.reset_index().to_excel(xw, sheet_name=DETAIL_SHEET, index=False)
    with open(final_path.parent / LOG_NAME, "a") as f:
        f.write(f"{computed_at}\tasof={asof_row.date()}\tbreadth={breadth:.4f} ({n_above}/{n_valid})\t"
                f"target_exposure={exposure:.4f}\tsource=yfinance\n")

    print(f"Step Ten — {STRATEGY_LABEL}")
    print(f"  as-of {asof_row.date()} (yfinance prices through {last.date()})")
    print(f"  breadth {breadth:.1%} ({n_above} of {n_valid} country ETFs above their {EMA_SPAN}-day EMA)")
    below = detail.loc[detail["Valid"] & ~detail["Above EMA200"], "ETF"].tolist()
    print(f"  below their EMA: {', '.join(below) if below else 'none'}")
    print(f"  TARGET EXPOSURE {exposure:.1%} of account value  (range {DIAL_LOW:.0%}..{DIAL_HIGH:.0%})")
    print(f"  written to {final_path} [{DIAL_SHEET}, {DETAIL_SHEET}]; prices saved to {prices_file}")


if __name__ == "__main__":
    try:
        main()
    except DialError as e:
        print(f"STEP TEN FAILED: {e}", file=sys.stderr)
        sys.exit(1)
