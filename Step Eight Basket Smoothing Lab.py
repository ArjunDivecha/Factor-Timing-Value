"""
=============================================================================
SCRIPT NAME: Step Eight Basket Smoothing Lab.py
=============================================================================

PURPOSE (read-only experiment - changes nothing in the production pipeline):
The turnover decomposition showed ~74% of country turnover (~16.7%/mo, ~200%/yr)
is "Term 2" = basket reshuffling that re-expresses the SAME factor bets. This lab
tests whether we can cut that turnover WITHOUT hurting the factor bet (active
return), by smoothing how each factor's country basket is built.

It rebuilds the final country weights  W_t = B_t @ w_t  under several basket
schemes, holding the Step Five factor bets w_t FIXED:

  baseline : production fuzzy band (15-25% taper)           [must match production]
  ewma3    : rank on a 3-month EWMA of each factor's score   (A1)
  ewma6    : rank on a 6-month EWMA of each factor's score    (A1)
  buffer   : hysteresis band - enter at 25%, hold until 35%   (A2)

For each scheme it reports:
  - mean one-way turnover (%/mo)            <- did turnover drop?
  - gross annual active return vs benchmark <- did the bet survive?
  - net annual active return (after real per-country trading costs)

INPUT FILES:
- T2_rolling_window_weights.xlsx   (factor bets w_t; first sheet, long-only)
- Normalized_T2_MasterCSV.csv      (country x factor scores + 1MRet)
- T2_Final_Country_Weights.xlsx    (production W_t, validation only)
- Portfolio_Data.xlsx              (Returns, Benchmarks)
- Step Tcost.xlsx                  (per-country trading costs)
- step_fuzzy_bands.py, Step Fourteen Target Optimization.py (reused helpers)

OUTPUT FILES:
- T2_Basket_Smoothing_Comparison.xlsx
- T2_Basket_Smoothing_Comparison.pdf

VERSION: 1.0
LAST UPDATED: 2026-06-15
=============================================================================
"""

import importlib.util
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step_fuzzy_bands import (band_matrix, band_weights, fuzzy_taper,
                              _deterministic_ranks, SOFT_BAND_TOP, SOFT_BAND_CUTOFF)

# ---- import Step Fourteen module for shared return/cost helpers ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "step_fourteen", os.path.join(_HERE, "Step Fourteen Target Optimization.py"))
s14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s14)

WEIGHTS_FILE = "T2_rolling_window_weights.xlsx"
FACTOR_FILE = "Normalized_T2_MasterCSV.csv"
PROD_FILE = "T2_Final_Country_Weights.xlsx"

BUFFER_ENTRY = 0.25   # enter the band at rank-pct <= 25%
BUFFER_EXIT = 0.35    # a held name stays until rank-pct > 35%


# =============================================================================
# DATA LOADING
# =============================================================================

def load_weights():
    try:
        w = pd.read_excel(WEIGHTS_FILE, sheet_name="Net_Weights", index_col=0)
    except Exception:
        w = pd.read_excel(WEIGHTS_FILE, index_col=0)
    w.index = pd.to_datetime(w.index)
    return w.sort_index().shift(1).iloc[1:]   # Step Eight vintage shift


def buffered_band_weights(scores, prev_members):
    """One factor, one month: fuzzy weights with sticky (hysteresis) membership.

    A country enters when rank-pct <= BUFFER_ENTRY and is held until rank-pct
    exceeds BUFFER_EXIT. Member weights use a taper widened to BUFFER_EXIT so
    retained names in the 25-35% hold zone keep a (small) nonzero weight.
    Returns (weights Series summing to 1, new member set).
    """
    s = scores.dropna()
    n = len(s)
    if n == 0:
        return pd.Series(dtype=float), set()
    rp = _deterministic_ranks(s, ascending=False) / n
    rp = pd.Series(rp, index=s.index)

    is_member = (rp <= BUFFER_ENTRY) | (rp.index.isin(prev_members) & (rp <= BUFFER_EXIT))
    if not is_member.any():
        return pd.Series(dtype=float), set()

    # widened taper: full < TOP, linear TOP->EXIT, else 0
    raw = np.where(rp < SOFT_BAND_TOP, 1.0,
                   np.clip(1.0 - (rp - SOFT_BAND_TOP) / (BUFFER_EXIT - SOFT_BAND_TOP), 0.0, 1.0))
    raw = pd.Series(raw, index=rp.index).where(is_member, 0.0)
    total = raw.sum()
    if total <= 0:
        return pd.Series(dtype=float), set()
    w = raw / total
    members = set(w.index[w > 1e-12])
    return w, members


# =============================================================================
# BUILD COUNTRY WEIGHTS UNDER A GIVEN SCHEME
# =============================================================================

def build_weights(scheme, w_df, factor_frames, elig_by_date, weight_cols, countries):
    """Reconstruct W_t for every date under one basket scheme.

    scheme: 'baseline' | 'ewma3' | 'ewma6' | 'buffer'
    factor_frames: dict factor -> DataFrame [date x country] of scores
    elig_by_date:  dict date  -> boolean Series (country eligible this month)
    """
    dates = list(w_df.index)
    out = pd.DataFrame(0.0, index=dates, columns=countries)

    # pre-smooth scores for EWMA schemes
    span = {"ewma3": 3, "ewma6": 6}.get(scheme)
    frames = factor_frames
    if span is not None:
        frames = {f: df.ewm(span=span, min_periods=1).mean() for f, df in factor_frames.items()}

    prev_members = {f: set() for f in weight_cols}  # buffer state

    for d in dates:
        w_vec = w_df.loc[d].astype(float)
        w_vec = w_vec[w_vec.abs() > 1e-10]
        if w_vec.empty:
            continue
        elig = elig_by_date.get(d)
        if elig is None:
            continue

        # assemble this month's score pivot (countries x held/available factors)
        present = [f for f in weight_cols if f in frames and d in frames[f].index]
        if not present:
            continue
        V = pd.DataFrame({f: frames[f].loc[d] for f in present})
        V = V.reindex(elig.index)
        V = V.where(elig, other=np.nan)

        if scheme == "buffer":
            B = pd.DataFrame(0.0, index=V.index, columns=present)
            for f in present:
                bw, mem = buffered_band_weights(V[f], prev_members[f])
                prev_members[f] = mem
                if not bw.empty:
                    B.loc[bw.index, f] = bw.values
        else:
            B = band_matrix(V)

        available = B.sum(axis=0) > 1e-12
        held = [f for f in w_vec.index if f in present]
        contrib = {}
        missing = sum(w_vec[f] for f in w_vec.index if f not in present)
        for f in held:
            if available[f]:
                contrib[f] = B[f] * w_vec[f]
            else:
                missing += w_vec[f]
        if abs(missing) > 1e-12:
            avail_cols = [f for f in present if available[f]]
            if avail_cols:
                contrib["MeanFill"] = B[avail_cols].sum(axis=1) * (missing / len(avail_cols))
        if not contrib:
            continue
        Wt = pd.DataFrame(contrib).sum(axis=1)
        out.loc[d, Wt.index] = Wt.reindex(countries).fillna(0.0).values

    return out


# =============================================================================
# METRICS
# =============================================================================

def mean_turnover(weights_df):
    return s14.calculate_turnover_series(weights_df).dropna().mean()


def realized_cost_series(weights_df, returns_df, scaled_costs):
    dates = weights_df.index
    costs = scaled_costs.reindex(weights_df.columns).fillna(scaled_costs.mean())
    vals = []
    for i, d in enumerate(dates):
        if i == 0:
            vals.append(0.0); continue
        pd_ = dates[i - 1]
        rolled = (s14.roll_forward_weights(weights_df.loc[pd_], returns_df.loc[pd_])
                  if pd_ in returns_df.index else weights_df.loc[pd_])
        vals.append(float(((weights_df.loc[d] - rolled).abs() * costs).sum()))
    return pd.Series(vals, index=dates)


def annual_ret(monthly):
    return s14.calculate_performance_stats(monthly.dropna())["Annual Return"] / 100.0


def evaluate(weights_df, returns_df, benchmark, scaled_costs):
    gross = s14.calculate_portfolio_returns(weights_df, returns_df)
    cost = realized_cost_series(weights_df, returns_df, scaled_costs).reindex(gross.index).fillna(0.0)
    net = gross - cost
    common = sorted(set(gross.index).intersection(benchmark.index))
    bench_a = annual_ret(benchmark.loc[common])
    return {
        "Mean Turnover (%/mo)": mean_turnover(weights_df) * 100,
        "Gross Active (%/yr)": (annual_ret(gross.loc[common]) - bench_a) * 100,
        "Net Active (%/yr)": (annual_ret(net.loc[common]) - bench_a) * 100,
        "_net_cum": (1 + net.loc[common]).cumprod(),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Loading data...")
    w_df = load_weights()
    weight_cols = list(w_df.columns)

    factor_df = pd.read_csv(FACTOR_FILE)
    factor_df["date"] = pd.to_datetime(factor_df["date"])
    countries = sorted(factor_df["country"].unique())

    # eligibility: country has a 1MRet this month (live-month fallback = all True)
    print("Building eligibility and per-factor score frames...")
    ret_rows = factor_df[factor_df["variable"] == "1MRet"]
    ret_piv = ret_rows.pivot(index="date", columns="country", values="value")
    elig_by_date = {}
    for d in w_df.index:
        if d in ret_piv.index and ret_piv.loc[d].notna().any():
            elig_by_date[d] = ret_piv.loc[d].reindex(countries).notna()
        else:
            elig_by_date[d] = pd.Series(True, index=countries)

    # per-factor [date x country] score frames (only the 82 weight factors)
    sub = factor_df[factor_df["variable"].isin(weight_cols)]
    factor_frames = {f: g.pivot(index="date", columns="country", values="value")
                     for f, g in sub.groupby("variable")}

    returns_df = pd.read_excel("Portfolio_Data.xlsx", sheet_name="Returns", index_col=0)
    returns_df.index = pd.to_datetime(returns_df.index)
    bench = pd.read_excel("Portfolio_Data.xlsx", sheet_name="Benchmarks", index_col=0)
    bench.index = pd.to_datetime(bench.index)
    benchmark = bench["equal_weight"]

    costs_df = pd.read_excel("Step Tcost.xlsx", sheet_name=0)
    cs = s14.build_country_cost_summary(countries, costs_df)
    scaled_costs = pd.Series(cs["Scaled Trading Cost"].values, index=cs["Country"])

    # ---- baseline + validation ----
    print("Building baseline and validating vs production...")
    W_base = build_weights("baseline", w_df, factor_frames, elig_by_date, weight_cols, countries)
    prod = pd.read_excel(PROD_FILE, sheet_name="All Periods", index_col=0)
    prod.index = pd.to_datetime(prod.index)
    diffs = []
    for d in W_base.index:
        if d in prod.index:
            a = W_base.loc[d].reindex(countries).fillna(0.0)
            b = prod.loc[d].reindex(countries).fillna(0.0)
            diffs.append(float(np.abs(a.values - b.values).sum()))
    print(f"  baseline vs production: mean abs diff/date = {np.mean(diffs):.2e}, max = {np.max(diffs):.2e}")

    schemes = ["baseline", "ewma3", "ewma6", "buffer"]
    results, cums = {}, {}
    for sc in schemes:
        print(f"Building + evaluating scheme: {sc}")
        Wsc = W_base if sc == "baseline" else build_weights(
            sc, w_df, factor_frames, elig_by_date, weight_cols, countries)
        m = evaluate(Wsc, returns_df, benchmark, scaled_costs)
        cums[sc] = m.pop("_net_cum")
        results[sc] = m

    table = pd.DataFrame(results).T
    base_to = table.loc["baseline", "Mean Turnover (%/mo)"]
    table["Turnover Cut (%)"] = (1 - table["Mean Turnover (%/mo)"] / base_to) * 100

    print("\n" + "=" * 78)
    print("BASKET SMOOTHING COMPARISON  (factor bets held fixed)")
    print("=" * 78)
    pd.set_option("display.width", 200)
    print(table.round(2).to_string())
    print("=" * 78)

    with pd.ExcelWriter("T2_Basket_Smoothing_Comparison.xlsx", engine="xlsxwriter") as xw:
        table.round(4).to_excel(xw, sheet_name="Comparison")
    print("Saved T2_Basket_Smoothing_Comparison.xlsx")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for sc in schemes:
        axes[0].plot(cums[sc].index, cums[sc].values, label=sc, lw=1.2)
    axes[0].set_title("Net cumulative return (after real costs)")
    axes[0].set_ylabel("Growth of $1"); axes[0].legend(); axes[0].grid(alpha=0.3)

    x = np.arange(len(schemes))
    axes[1].bar(x - 0.2, table.loc[schemes, "Mean Turnover (%/mo)"], 0.4, label="Turnover %/mo")
    axes[1].bar(x + 0.2, table.loc[schemes, "Net Active (%/yr)"], 0.4, label="Net Active %/yr")
    axes[1].set_xticks(x); axes[1].set_xticklabels(schemes)
    axes[1].set_title("Turnover vs Net Active Return"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("T2_Basket_Smoothing_Comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved T2_Basket_Smoothing_Comparison.pdf")


if __name__ == "__main__":
    main()
