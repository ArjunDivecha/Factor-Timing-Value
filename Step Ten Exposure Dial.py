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
                          at the LATEST completed yfinance close (changed 2026-09-30: no longer
                          the previous month-end; today's bar is ignored until after 4:05pm ET)
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
    adjust=False over ~10 years of history, and an ETF counts once it has 200 observations.
    The universe is the same 34 tickers in both repos (SPY/QQQ/IWM for the U.S./NASDAQ/US
    SmallCap slots). Value's trading overrides (VTV/VBR) do not change the signal.

    Runs AFTER Step FINALFINAL (it adds sheets to FINALFINAL's output). It stops loudly
    (non-zero exit, nothing written to the workbook) if the yfinance download fails or is
    incomplete, if the latest close is more than 4 days old, if AssetList.xlsx does not have
    34 tickers, or if fewer than MIN_VALID ETFs have a 200-day history. There is no fallback
    data source.

CHART HISTORY: for each country the chart uses the ETF's adjusted close from the ETF's first day and the
    Bloomberg total-return index (scaled to join the ETF price on that day) before it. The same rule
    (share above the 200-day EMA, x2) is applied on the US trading-day grid. The last chart value must
    equal the dial. Checked 2026-10-01 against the research's index-breadth dial: month-end correlation
    0.993, 1 month of 308 differing by more than 25 points; and vs the research ETF dial since 2012: 1.000.

INPUT FILES:
    <repo>/AssetList.xlsx              sheet 'Yahoo' (34 country ETF tickers, trader order)
    <repo>/T2_FINAL_T60_VALUE.xlsx           sheet 'Latest_Country_Alpha_Weights' (Step FINALFINAL output)
    Yahoo Finance daily prices via yfinance (downloaded at run time, dividend-adjusted)
    /Users/arjundivecha/Dropbox/AAA Backup/Master Database/Country Bloomberg Data Master T Daily.xlsx
        sheet 'Tot Return Index ' — CHART ONLY (the dial never uses it); if it is missing or does not match
        the ETFs the chart falls back to ETF history only, with a warning

OUTPUT FILES:
    <repo>/T2_FINAL_T60_VALUE.xlsx  sheets added/replaced:
        'Exposure_Dial'   key/value table read by the trader (target_exposure, asof_date, breadth, ...)
        'Exposure_Detail' per-country ETF, adjusted close, EMA200, above/valid flags, base and scaled weight
    <repo>/outputs/exposure_dial_prices_YYYYMMDD.parquet   the exact prices used (audit trail)
    <repo>/T2_exposure_dial_log.txt   one appended line per run
    <repo>/T2_exposure_dial_history.pdf   chart of the dial (leverage) over time, 2000 to today; overwritten
                                          each run. Uses the ETF prices; before an ETF existed it uses the
                                          Bloomberg index (see CHART HISTORY below)

    (<repo> = /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value)

VERSION: 2.2 (2026-10-01) — adds the leverage-over-time chart (T2_exposure_dial_history.pdf), back to 2000;
    price download window 5y -> max. Pre-ETF history comes from the Bloomberg daily total-return index. 2.1 (2026-09-30) — as-of = latest completed yfinance close instead of the previous month-end;
    fails if the latest close is more than 4 days old. 2.0 (2026-09-28) — yfinance ETF prices replace the daily Bloomberg file. 1.0 (2026-09-27)
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
from zoneinfo import ZoneInfo

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
MAX_ASOF_GAP_DAYS = 4        # latest yfinance close must be within 4 calendar days of today
MARKET_CLOSE_ET = (16, 5)    # today's bar counts as complete only after 4:05pm US/Eastern
HISTORY_PERIOD = "max"       # yfinance download window (EMA warm-up, 200-day validity, and the chart history)
CHART_NAME = "T2_exposure_dial_history.pdf"
# Bloomberg daily total-return indices: used ONLY to extend the CHART back before each ETF existed
# (the dial itself uses the ETF prices only). Columns are positional, in COUNTRIES order; row 0
# holds the Bloomberg field names.
BBG_DAILY_FILE = Path("/Users/arjundivecha/Dropbox/AAA Backup/Master Database/"
                      "Country Bloomberg Data Master T Daily.xlsx")
BBG_TRI_SHEET = "Tot Return Index "      # trailing space is part of the sheet name
BBG_MIN_WEEKLY_CORR = 0.5                # sanity check: index vs ETF 5-day returns, last 3 years
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


def drop_partial_today_bar(px, today, use_real_clock):
    """yfinance includes today's still-forming bar while the US market is open. Drop it so the
    signal always uses a completed close. Only applies with the real clock (not --today tests)."""
    if not use_real_clock:
        return px
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if px.dropna(how="all").index.max() == pd.Timestamp(now_et.date()) and \
            (now_et.hour, now_et.minute) < MARKET_CLOSE_ET:
        return px.loc[px.index < pd.Timestamp(now_et.date())]
    return px


def resolve_asof(px, asof_arg, today):
    """As-of row = the LATEST completed yfinance close (not the previous month-end).
    --asof picks the last available close on or before that date instead."""
    last = px.dropna(how="all").index.max()
    if asof_arg:
        rows = px.dropna(how="all").index[px.dropna(how="all").index <= pd.Timestamp(asof_arg)]
        if len(rows) == 0:
            raise DialError(f"no prices on or before {asof_arg} (latest {last.date()})")
        return rows.max(), last
    if (today - last).days > MAX_ASOF_GAP_DAYS:
        raise DialError(f"latest yfinance close is {last.date()}, more than {MAX_ASOF_GAP_DAYS} days "
                        f"before today ({today.date()}) — data is stale")
    return last, last


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


def load_bloomberg_tri():
    """Daily Bloomberg total-return index levels (weekdays), one column per country."""
    raw = pd.read_excel(BBG_DAILY_FILE, BBG_TRI_SHEET, header=0).iloc[1:]
    tri = raw.iloc[:, 1:1 + len(COUNTRIES)].apply(pd.to_numeric, errors="coerce")
    tri.columns = COUNTRIES
    tri.index = pd.to_datetime(raw.iloc[:, 0])
    tri = tri.sort_index()
    return tri[tri.index.dayofweek < 5].where(tri > 0)


def splice_etf_and_bloomberg(px, tri, tickers):
    """Price history for the chart: the ETF's adjusted close wherever the ETF exists, and the
    Bloomberg index (scaled to join the ETF price on its first day) before that.
    Returns (spliced prices, number of countries on ETF prices per date)."""
    # Stay on the ETF (US trading day) grid so the EMA is computed exactly as the dial computes it.
    idx = px.index
    etf = px
    bbg = tri.reindex(tri.index.union(idx)).ffill(limit=10).reindex(idx)
    out, uses_etf = etf.copy(), pd.DataFrame(False, index=idx, columns=etf.columns)
    bad = []
    for c, t in zip(COUNTRIES, tickers):
        e = etf[t]
        first = e.first_valid_index()
        uses_etf[t] = e.index >= first
        # sanity: Bloomberg column order must really be this country (index vs ETF 5-day returns)
        j = pd.concat([e, bbg[c]], axis=1).dropna().iloc[-750:].resample("W").last().pct_change().dropna()
        if len(j) > 50 and j.corr().iloc[0, 1] < BBG_MIN_WEEKLY_CORR:
            bad.append(f"{c}/{t} corr {j.corr().iloc[0, 1]:.2f}")
        pre = e.index < first
        b_first = bbg[c].get(first, np.nan)
        if pre.any() and np.isfinite(b_first):
            out.loc[pre, t] = bbg[c][pre] * (e[first] / b_first)
    if bad:
        raise DialError(f"Bloomberg index columns do not match the ETFs ({'; '.join(bad)})")
    return out, uses_etf.sum(axis=1)


def exposure_history(px, asof_row):
    """Daily target exposure the dial rule would have given on every date up to asof_row
    (same rule as compute_breadth, vectorised). Dates with fewer than MIN_VALID ETFs that have
    200 days of history are left out."""
    P = px.loc[:asof_row].ffill(limit=10)
    ema = P.ewm(span=EMA_SPAN, adjust=False, ignore_na=True).mean()
    valid = P.notna() & (P.notna().cumsum() >= EMA_SPAN)
    above = (P > ema) & valid
    n_valid = valid.sum(axis=1)
    breadth = (above.sum(axis=1) / n_valid.where(n_valid > 0)).where(n_valid >= MIN_VALID).dropna()
    exposure = (DIAL_LOW + (DIAL_HIGH - DIAL_LOW) * breadth).clip(DIAL_LOW, DIAL_HIGH)
    return pd.DataFrame({"breadth": breadth, "exposure": exposure})


def plot_exposure_history(hist, asof_row, exposure, out_pdf, n_etf=None):
    """Light-mode PDF: leverage (target exposure) over time, with the 100% line and today's value.
    If n_etf is given, a lower panel shows how many of the 34 countries are on ETF prices (the rest
    use the Bloomberg index before the ETF existed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if n_etf is None:
        fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
        ax2 = None
    else:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7), facecolor="white", sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
    ax.set_facecolor("white")
    x, y = hist.index, hist["exposure"] * 100
    ax.plot(x, y, color="#1f4e79", lw=0.9, label="Target exposure (2 x breadth)")
    ax.fill_between(x, 100, y, where=y > 100, color="#2e8b57", alpha=0.25, label="Levered (> 100%)")
    ax.fill_between(x, y, 100, where=y < 100, color="#c0392b", alpha=0.20, label="De-risked (< 100%, rest in cash)")
    ax.axhline(100, color="black", lw=0.8, ls="--")
    ax.scatter([x[-1]], [y.iloc[-1]], color="#c0392b", zorder=5)
    ax.annotate(f"{y.iloc[-1]:.0f}%  ({asof_row.date()})", (x[-1], y.iloc[-1]), textcoords="offset points",
                xytext=(-8, 8), ha="right", fontsize=10, fontweight="bold")
    ax.set_ylim(DIAL_LOW * 100 - 5, DIAL_HIGH * 100 + 5)
    ax.set_ylabel("Leverage (% of account value)")
    src = ("ETF prices, with the Bloomberg index used before each ETF existed" if n_etf is not None
           else "ETF prices only")
    ax.set_title(f"{STRATEGY_LABEL}: exposure dial over time\n"
                 f"share of the 34 countries above their {EMA_SPAN}-day EMA x {DIAL_HIGH:.0f}  ({src})", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    if ax2 is not None:
        n = n_etf.reindex(x).ffill()
        ax2.set_facecolor("white")
        ax2.fill_between(x, 0, n, color="#7f8c8d", alpha=0.5, step="post")
        ax2.set_ylim(0, 36)
        ax2.set_ylabel("Countries on\nETF prices")
        ax2.axhline(34, color="black", lw=0.6, ls=":")
        ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


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
    px = drop_partial_today_bar(px, today, use_real_clock=args.today is None)
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

    # Leverage-over-time chart. The dial is already written above, so a chart problem is reported
    # loudly but does not undo or block the dial.
    chart_file = final_path.parent / CHART_NAME
    try:
        try:
            hist_px, n_etf = splice_etf_and_bloomberg(px, load_bloomberg_tri(), tickers)
        except Exception as exc:  # Bloomberg problem: say so, chart ETF-only history instead
            print(f"  WARNING: Bloomberg history not used ({exc}); chart is ETF prices only", file=sys.stderr)
            hist_px, n_etf = px, None
        hist = exposure_history(hist_px, asof_row)
        if hist.empty or abs(float(hist["exposure"].iloc[-1]) - exposure) > 1e-9:
            raise DialError("history's last value does not match the dial just written")
        plot_exposure_history(hist, asof_row, exposure, chart_file, n_etf=n_etf)
        print(f"  leverage history chart ({hist.index[0].date()}..{hist.index[-1].date()}): {chart_file}")
    except Exception as exc:
        print(f"  WARNING: leverage chart NOT written: {exc}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except DialError as e:
        print(f"STEP TEN FAILED: {e}", file=sys.stderr)
        sys.exit(1)
