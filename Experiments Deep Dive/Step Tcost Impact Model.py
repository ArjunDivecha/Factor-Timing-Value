"""
=============================================================================
SCRIPT NAME: Step Tcost Impact Model.py
=============================================================================

WHAT THIS PROGRAM DOES (plain English):
The production turnover study concluded "do not reduce turnover" assuming a
flat ~4 bps one-way cost (quoted half-spread only, from Step Tcost.xlsx). That
number has NO market-impact component. This script estimates the REAL all-in
one-way cost at a given AUM by adding a square-root market-impact model on top
of the quoted half-spread, using each ETF's actual daily dollar volume (ADV)
and daily volatility pulled from Interactive Brokers.

    one_way_cost_c = half_spread_c  +  k * sigma_daily_c * sqrt( participation_c )
    participation_c = (AUM * |dW_c|) / (ADV_c * horizon_days)

It then applies this per-country, per-trade cost to the ACTUAL month-by-month
production country-weight changes (T2_Final_Country_Weights.xlsx) to get:
  - effective blended one-way cost (bps per unit traded)
  - annual net-return drag (%/yr) = mean monthly trading cost * 12
across a sweep of AUM levels and execution horizons. The point is to locate
where the effective cost crosses the ~50-70 bps break-even at which the prior
study said turnover reduction starts ADDING net alpha.

INPUT FILES (full paths):
- ../Step Tcost.xlsx                  (Country, Borrow Cost, Trading Cost[bps half-spread])
- ../AssetList.xlsx  (sheet "Yahoo")  (ETF ticker per country, SAME row order)
- ../T2_Final_Country_Weights.xlsx (All Periods)  (production country weights -> trades)
- Interactive Brokers (TWS/Gateway) for daily ADV + volatility (cached after first pull)

OUTPUT FILES (this directory):
- IBKR_Liquidity.xlsx                 (per-ETF ADV + daily vol cache; reused if IBKR down)
- T2_Impact_Cost_Model.xlsx           (per-country cost at base AUM; AUM sweep; per-trade tail)
- T2_Impact_Cost_Model.pdf            (effective cost vs AUM; annual drag vs AUM; cost tail)

VERSION: 1.0   LAST UPDATED: 2026-06-24   AUTHOR: Devin (impact-cost study)

DEPENDENCIES: ib_insync, pandas, numpy, matplotlib, openpyxl
USAGE: python3 "Step Tcost Impact Model.py"   (IB Gateway/TWS running for the pull)
NOTES:
- IB historical TRADES volume for these US-listed ETFs is in SHARES; $ADV =
  close * volume. (Verified against EWZ ~ $0.5B/day.)
- k (impact coefficient) default 1.0; we report k in {0.5, 1.0, 1.5} sensitivity.
- ADV uses the trailing-63-day MEDIAN (robust to spikes). Vol uses 1y daily std.
- This script changes NOTHING in the production pipeline (read-only analysis).
=============================================================================
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TCOST_PATH = os.path.join(ROOT, "Step Tcost.xlsx")
ASSET_PATH = os.path.join(ROOT, "AssetList.xlsx")
# Auto-detect the country-weights file so this runs in all sister repos
# (classic/value use T2_Final_Country_Weights.xlsx; GDELT uses GDELT_Final_...).
WEIGHTS_PATH = next(
    (os.path.join(ROOT, f) for f in ("T2_Final_Country_Weights.xlsx",
                                     "GDELT_Final_Country_Weights.xlsx")
     if os.path.exists(os.path.join(ROOT, f))),
    os.path.join(ROOT, "T2_Final_Country_Weights.xlsx"))
LIQ_CACHE = os.path.join(HERE, "IBKR_Liquidity.xlsx")
OUT_XLSX = os.path.join(HERE, "T2_Impact_Cost_Model.xlsx")
OUT_PDF = os.path.join(HERE, "T2_Impact_Cost_Model.pdf")

IB_PORTS = (4001, 4002, 7496, 7497)   # gateway live/paper, TWS live/paper
ADV_WINDOW = 63                       # trailing days for median $ADV
K_IMPACT = 1.0                        # central square-root impact coefficient
HORIZON_DAYS = 1                      # trading days to work each rebalance
AUM_BASE = 7_000_000                  # the user's actual portfolio value
AUM_SWEEP = [3e6, 7e6, 15e6, 30e6, 50e6, 100e6, 250e6, 500e6, 1e9]
BREAKEVEN_LOW, BREAKEVEN_HIGH = 50.0, 70.0   # bps; from prior turnover study


# =============================================================================
# LIQUIDITY DATA (IBKR pull -> cache fallback)
# =============================================================================
def pull_ibkr_liquidity(countries, tickers):
    """Pull trailing-1y daily bars per ETF -> median $ADV and daily vol.
    Returns DataFrame[Country, ETF, ADV_USD, Sigma_Daily, Source] or None on failure."""
    try:
        from ib_insync import IB, Stock
    except Exception as e:
        print("ib_insync import failed:", e)
        return None

    ib = IB()
    for port in IB_PORTS:
        try:
            ib.connect("127.0.0.1", port, clientId=92, timeout=8)
            print(f"Connected to IBKR on port {port}: {ib.reqCurrentTime()}")
            break
        except Exception as e:
            print(f"  port {port} failed: {repr(e)[:80]}")
    if not ib.isConnected():
        print("No IBKR connection.")
        return None

    rows = []
    try:
        for country, tk in zip(countries, tickers):
            try:
                c = Stock(tk, "SMART", "USD")
                ib.qualifyContracts(c)
                bars = ib.reqHistoricalData(
                    c, "", "1 Y", "1 day", "TRADES", useRTH=True, formatDate=1)
                if not bars:
                    print(f"  {tk}: no bars"); 
                    rows.append((country, tk, np.nan, np.nan, "MISSING")); continue
                df = pd.DataFrame({"close": [b.close for b in bars],
                                   "vol":   [b.volume for b in bars]})
                df = df[(df["close"] > 0) & (df["vol"] > 0)]
                dollar_vol = df["close"] * df["vol"]
                adv = float(dollar_vol.tail(ADV_WINDOW).median())
                sigma = float(df["close"].pct_change().dropna().tail(252).std())
                rows.append((country, tk, adv, sigma, "IBKR"))
                print(f"  {tk:5s} {country:13s} $ADV={adv:12,.0f}  sigma_d={sigma:6.4f}")
            except Exception as e:
                print(f"  {tk}: error {repr(e)[:70]}")
                rows.append((country, tk, np.nan, np.nan, "ERROR"))
    finally:
        ib.disconnect()

    out = pd.DataFrame(rows, columns=["Country", "ETF", "ADV_USD", "Sigma_Daily", "Source"])
    if out["ADV_USD"].notna().sum() < len(out) * 0.6:
        print("Too many missing ADVs from IBKR.")
        return out if out["ADV_USD"].notna().any() else None
    return out


def get_liquidity(countries, tickers):
    """IBKR first; cache the result. If IBKR unavailable, load cache."""
    liq = pull_ibkr_liquidity(countries, tickers)
    if liq is not None and liq["ADV_USD"].notna().any():
        liq.to_excel(LIQ_CACHE, index=False)
        print(f"Saved liquidity cache -> {LIQ_CACHE}")
    elif os.path.exists(LIQ_CACHE):
        print(f"IBKR unavailable; loading cache {LIQ_CACHE}")
        liq = pd.read_excel(LIQ_CACHE)
    else:
        raise RuntimeError("No IBKR data and no liquidity cache available.")
    # Fill any residual missing ADV/vol with conservative cross-sectional medians
    liq["ADV_USD"] = liq["ADV_USD"].fillna(liq["ADV_USD"].median())
    liq["Sigma_Daily"] = liq["Sigma_Daily"].fillna(liq["Sigma_Daily"].median())
    return liq


# =============================================================================
# IMPACT MODEL
# =============================================================================
def one_way_cost_bps(trade_usd, adv_usd, sigma_daily, half_spread_bps,
                     k=K_IMPACT, horizon=HORIZON_DAYS):
    """Square-root impact + quoted half-spread, in bps (per unit traded)."""
    participation = trade_usd / np.maximum(adv_usd * horizon, 1.0)
    impact_bps = k * (sigma_daily * 1e4) * np.sqrt(participation)
    return half_spread_bps + impact_bps


def monthly_cost_fraction(dW, aum, adv, sigma, hspread, k, horizon):
    """For a month's per-country |dW| vector, return total one-way cost as a
    fraction of AUM = sum_c |dW_c| * one_way_cost_c."""
    trade_usd = aum * dW.values
    cost_bps = one_way_cost_bps(trade_usd, adv.values, sigma.values,
                                hspread.values, k, horizon)
    return float((dW.values * cost_bps / 1e4).sum())


def effective_cost_bps(dW_all, aum, adv, sigma, hspread, k, horizon):
    """Trade-weighted average one-way cost (bps) over all months and the
    annual return drag (%/yr)."""
    monthly = []
    traded = []
    for _, dW in dW_all.iterrows():
        trade_usd = aum * dW.values
        cost_bps = one_way_cost_bps(trade_usd, adv.values, sigma.values,
                                    hspread.values, k, horizon)
        monthly.append((dW.values * cost_bps / 1e4).sum())  # cost as frac of AUM
        traded.append(dW.values.sum())                       # one-way turnover
    monthly = np.array(monthly); traded = np.array(traded)
    eff_bps = (monthly.sum() / traded.sum()) * 1e4 if traded.sum() else np.nan
    annual_drag_pct = monthly.mean() * 12 * 100
    return eff_bps, annual_drag_pct, monthly.mean() * 100


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("Loading inputs...")
    tc = pd.read_excel(TCOST_PATH)
    countries = tc["Country"].tolist()
    hspread = pd.Series(tc["Trading Cost"].values, index=countries)  # bps, one-way half-spread
    tickers = pd.read_excel(ASSET_PATH, sheet_name="Yahoo")["Ticker"].tolist()
    assert len(tc) == len(tickers)

    w = pd.read_excel(WEIGHTS_PATH, sheet_name="All Periods", index_col=0)
    w.index = pd.to_datetime(w.index)
    w = w.reindex(columns=countries)            # align order; missing -> NaN
    w = w.fillna(0.0)
    dW_all = w.diff().abs().iloc[1:]            # per-country one-way trade fraction / month
    print(f"Weights: {w.shape[0]} months {w.index.min().date()}..{w.index.max().date()}; "
          f"mean one-way turnover {0.5*dW_all.sum(axis=1).mean()*100:.1f}%/mo")

    liq = get_liquidity(countries, tickers).set_index("Country").reindex(countries)
    adv = liq["ADV_USD"]; sigma = liq["Sigma_Daily"]
    print(f"\nADV range: ${adv.min():,.0f} (min) .. ${adv.max():,.0f} (max); "
          f"median ${adv.median():,.0f}")
    print(f"Daily vol range: {sigma.min():.4f} .. {sigma.max():.4f}")

    # ---- per-country cost at base AUM (using each country's MEAN monthly trade) ----
    mean_trade = dW_all.mean()                         # avg |dW| per country per month
    typ_trade_usd = AUM_BASE * mean_trade
    pc = pd.DataFrame({
        "ETF": pd.Series(tickers, index=countries),
        "ADV_USD": adv,
        "Sigma_Daily": sigma,
        "HalfSpread_bps": hspread,
        "Avg_Monthly_Trade_%AUM": mean_trade * 100,
        "Avg_Trade_$": typ_trade_usd,
        "Participation_%ADV": typ_trade_usd / adv * 100,
        "Impact_bps": one_way_cost_bps(typ_trade_usd, adv.values, sigma.values, 0.0),
        "AllIn_OneWay_bps": one_way_cost_bps(typ_trade_usd, adv.values, sigma.values,
                                             hspread.values),
    }).sort_values("AllIn_OneWay_bps", ascending=False)
    print(f"\n--- Per-country ALL-IN one-way cost at AUM=${AUM_BASE:,.0f} "
          f"(typical monthly trade) ---")
    print(pc[["ETF", "ADV_USD", "Avg_Trade_$", "Participation_%ADV",
              "HalfSpread_bps", "Impact_bps", "AllIn_OneWay_bps"]].round(2).to_string())

    # ---- AUM sweep ----
    print("\n--- AUM SWEEP (k=%.1f, horizon=%d day) ---" % (K_IMPACT, HORIZON_DAYS))
    sweep_rows = []
    for aum in AUM_SWEEP:
        eff, drag, mcost = effective_cost_bps(dW_all, aum, adv, sigma, hspread,
                                              K_IMPACT, HORIZON_DAYS)
        sweep_rows.append({"AUM_$": aum, "Eff_OneWay_bps": eff,
                           "Monthly_Cost_%": mcost, "Annual_Drag_%/yr": drag})
    sweep = pd.DataFrame(sweep_rows)
    print(sweep.assign(AUM_M=lambda d: d["AUM_$"]/1e6).round(3)
          [["AUM_M", "Eff_OneWay_bps", "Annual_Drag_%/yr"]].to_string(index=False))

    # baseline (flat 4 bps, no impact) for reference
    base_eff, base_drag, _ = effective_cost_bps(
        dW_all, 1.0, adv*0 + 1e18, sigma, hspread, 0.0, 1)  # impact->0
    print(f"\nReference (half-spread only, no impact): eff {base_eff:.2f} bps, "
          f"drag {base_drag:.2f}%/yr")

    # ---- k sensitivity at base AUM ----
    print(f"\n--- k sensitivity at AUM=${AUM_BASE:,.0f} ---")
    ks = []
    for k in (0.5, 1.0, 1.5):
        for hz in (1, 3):
            eff, drag, _ = effective_cost_bps(dW_all, AUM_BASE, adv, sigma, hspread, k, hz)
            ks.append({"k": k, "horizon_days": hz, "Eff_OneWay_bps": eff,
                       "Annual_Drag_%/yr": drag})
    ksens = pd.DataFrame(ks)
    print(ksens.round(2).to_string(index=False))

    eff7, drag7, _ = effective_cost_bps(dW_all, AUM_BASE, adv, sigma, hspread,
                                        K_IMPACT, HORIZON_DAYS)
    print("\n" + "=" * 74)
    print(f"BOTTOM LINE at AUM=${AUM_BASE:,.0f}, k={K_IMPACT}, horizon={HORIZON_DAYS}d:")
    print(f"  Effective one-way cost   : {eff7:.1f} bps   "
          f"(vs {base_eff:.1f} bps spread-only, vs {BREAKEVEN_LOW:.0f}-{BREAKEVEN_HIGH:.0f} bps break-even)")
    print(f"  Annual turnover drag     : {drag7:.2f} %/yr")
    print("=" * 74)

    # ---- save ----
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        pc.round(4).to_excel(xw, sheet_name="PerCountry_at_base_AUM")
        sweep.round(4).to_excel(xw, sheet_name="AUM_Sweep", index=False)
        ksens.round(4).to_excel(xw, sheet_name="k_Sensitivity", index=False)
    print(f"Saved -> {OUT_XLSX}")

    # ---- charts ----
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    aum_m = sweep["AUM_$"] / 1e6

    ax = axes[0]
    ax.plot(aum_m, sweep["Eff_OneWay_bps"], "o-", lw=2, color="tab:blue")
    ax.axhspan(BREAKEVEN_LOW, BREAKEVEN_HIGH, color="tab:red", alpha=0.15,
               label=f"break-even {BREAKEVEN_LOW:.0f}-{BREAKEVEN_HIGH:.0f} bps")
    ax.axvline(AUM_BASE/1e6, color="k", ls="--", alpha=0.6, label=f"${AUM_BASE/1e6:.0f}M (you)")
    ax.set_xscale("log"); ax.set_xlabel("AUM ($M, log)"); ax.set_ylabel("Effective one-way cost (bps)")
    ax.set_title("All-in one-way cost vs AUM"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(aum_m, sweep["Annual_Drag_%/yr"], "o-", lw=2, color="tab:purple")
    ax.axvline(AUM_BASE/1e6, color="k", ls="--", alpha=0.6)
    ax.set_xscale("log"); ax.set_xlabel("AUM ($M, log)"); ax.set_ylabel("Annual turnover drag (%/yr)")
    ax.set_title("Annual net-return drag vs AUM"); ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f%%"))

    ax = axes[2]
    top = pc.head(12).iloc[::-1]
    ax.barh(top["ETF"], top["AllIn_OneWay_bps"], color="tab:orange", alpha=0.8)
    ax.barh(top["ETF"], top["HalfSpread_bps"], color="tab:gray", alpha=0.9, label="half-spread")
    ax.set_xlabel("All-in one-way cost (bps)")
    ax.set_title(f"Most expensive names @ ${AUM_BASE/1e6:.0f}M (typical trade)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")

    fig.suptitle("T2 turnover: real all-in trading cost (square-root impact + spread)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_PDF}")


if __name__ == "__main__":
    main()
