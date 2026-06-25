"""
=============================================================================
SCRIPT NAME: Step Eight Turnover Decomposition.py
=============================================================================

PURPOSE (read-only analysis - changes nothing in the pipeline):
Quantify WHY the final country portfolio turns over ~20%/month even though the
Step Five factor-weight turnover is only ~6%. It does an EXACT month-by-month
path decomposition of the change in country weights:

    W_t = B_t @ w_t          (Step Eight's construction: factor baskets x bets)

    W_t - W_{t-1}
        = (W_t - Wbar)       <- Term 1 "bet change"   (WANTED)
        + (Wbar - W_{t-1})   <- Term 2 "basket refresh" (AVOIDABLE)
    where Wbar = B_t @ w_{t-1}  (THIS month's refreshed baskets, LAST month's bet)

Term 1 + Term 2 sum EXACTLY to the true monthly change, so the split is honest.
One-way turnover = 0.5 * sum(|.|).

It faithfully replicates Step Eight: same shared step_fuzzy_bands band math, same
shift(1) vintage, same mean-fill-of-empty-bands redistribution. The reconstructed
W_t is validated against the production T2_Final_Country_Weights.xlsx before the
decomposition is trusted.

INPUT FILES:
- T2_rolling_window_weights.xlsx      (factor weights w_t; first sheet, long-only)
- Normalized_T2_MasterCSV.csv         (country x factor scores + 1MRet, long format)
- T2_Final_Country_Weights.xlsx       (All Periods - production W_t, for validation)
- step_fuzzy_bands.py                 (shared fuzzy-band utility)

OUTPUT FILES:
- T2_Turnover_Decomposition.xlsx      (Monthly series + Summary sheets)
- T2_Turnover_Decomposition.pdf       (time series of Total / Term1 / Term2 turnover)

VERSION: 1.0
LAST UPDATED: 2026-06-15
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step_fuzzy_bands import band_matrix

WEIGHTS_FILE = "T2_rolling_window_weights.xlsx"
FACTOR_FILE = "Normalized_T2_MasterCSV.csv"
PROD_FILE = "T2_Final_Country_Weights.xlsx"


def load_weights():
    """Factor weights with the SAME vintage shift Step Eight applies."""
    try:
        w = pd.read_excel(WEIGHTS_FILE, sheet_name="Net_Weights", index_col=0)
    except Exception:
        w = pd.read_excel(WEIGHTS_FILE, index_col=0)
    w.index = pd.to_datetime(w.index)
    w = w.sort_index()
    # Step Eight: weights at date d come from d-1 (drop first row)
    w = w.shift(1).iloc[1:]
    return w


def contributions_for(pivot, eligible, w_vec, all_factors):
    """
    Build the country weight vector W = B @ w for ONE month, replicating
    Step Eight's calculate_country_contributions_for_date exactly, but taking
    the factor-weight vector `w_vec` as an argument so we can apply DIFFERENT
    bets to the SAME month's baskets (needed for the counterfactual Wbar).

    pivot       : countries x factors score matrix for this month
    eligible    : boolean Series (country has a 1MRet this month, else live-month all True)
    w_vec       : factor weights to apply (Series indexed by factor)
    all_factors : factor columns present in pivot AND in the weight file
    """
    w = w_vec[w_vec.abs() > 1e-10]
    if w.empty:
        return None

    V = pivot[all_factors].where(eligible, other=np.nan)

    neg_cols = [f for f in all_factors if w.get(f, 0.0) < 0]
    B_own = band_matrix(V, ascending_cols=neg_cols)
    B_top = B_own if not neg_cols else band_matrix(V)
    available = B_own.sum(axis=0) > 1e-12

    held = [f for f in w.index if f in all_factors]
    contributions = {}
    missing_weight = sum(w[f] for f in w.index if f not in all_factors)
    for f in held:
        if available[f]:
            contributions[f] = B_own[f] * w[f]
        else:
            missing_weight += w[f]

    if abs(missing_weight) > 1e-12:
        avail_cols = [f for f in all_factors if available[f]]
        if avail_cols:
            spread = missing_weight / len(avail_cols)
            contributions["MeanFill"] = B_top[avail_cols].sum(axis=1) * spread

    if not contributions:
        return None
    return pd.DataFrame(contributions).sum(axis=1)


def band_membership(pivot, eligible, all_factors, held_factors):
    """Set of (country) names with non-zero weight across the HELD factor bands
    this month -- used to count name entries/exits (membership churn)."""
    V = pivot[all_factors].where(eligible, other=np.nan)
    B = band_matrix(V)
    cols = [f for f in held_factors if f in B.columns]
    if not cols:
        return set()
    member_mass = B[cols].sum(axis=1)
    return set(member_mass.index[member_mass > 1e-12])


def main():
    print("Loading data...")
    w_df = load_weights()
    weight_cols = list(w_df.columns)

    factor_df = pd.read_csv(FACTOR_FILE)
    factor_df["date"] = pd.to_datetime(factor_df["date"])
    by_date = factor_df.groupby("date")

    prod = pd.read_excel(PROD_FILE, sheet_name="All Periods", index_col=0)
    prod.index = pd.to_datetime(prod.index)

    dates = list(w_df.index)
    countries = sorted(factor_df["country"].unique())

    # ---- build per-date pivots / eligibility once ----
    def get_pivot(d):
        try:
            sl = by_date.get_group(d)
        except KeyError:
            return None, None, None
        pivot = sl.pivot(index="country", columns="variable", values="value")
        all_factors = [c for c in weight_cols if c in pivot.columns]
        if not all_factors:
            return None, None, None
        if "1MRet" in pivot.columns and pivot["1MRet"].notna().any():
            eligible = pivot["1MRet"].notna()
        else:
            eligible = pd.Series(True, index=pivot.index)
        return pivot, eligible, all_factors

    # ---- reconstruct W_t for every date (and validate) ----
    print("Reconstructing W_t and validating against production...")
    W = {}
    members = {}
    for d in dates:
        pivot, eligible, all_factors = get_pivot(d)
        if pivot is None:
            continue
        w_vec = w_df.loc[d].astype(float)
        Wt = contributions_for(pivot, eligible, w_vec, all_factors)
        if Wt is None:
            continue
        W[d] = Wt.reindex(countries).fillna(0.0)
        held = [f for f in w_vec.index[w_vec.abs() > 1e-10] if f in all_factors]
        members[d] = band_membership(pivot, eligible, all_factors, held)

    # validation vs production
    val_rows = []
    for d in W:
        if d in prod.index:
            a = W[d].reindex(countries).fillna(0.0)
            b = prod.loc[d].reindex(countries).fillna(0.0)
            val_rows.append(float(np.abs(a.values - b.values).sum()))
    val = np.array(val_rows)
    print(f"  validated {len(val)} dates vs production; "
          f"mean abs weight diff/date = {val.mean():.2e}, max = {val.max():.2e}")

    # ---- exact path decomposition ----
    print("Decomposing turnover (Term 1 bet change vs Term 2 basket refresh)...")
    rows = []
    sorted_dates = sorted(W.keys())
    for i in range(1, len(sorted_dates)):
        d, dprev = sorted_dates[i], sorted_dates[i - 1]
        pivot, eligible, all_factors = get_pivot(d)
        if pivot is None:
            continue

        Wt = W[d]
        Wprev = W[dprev]
        # counterfactual: THIS month's baskets, LAST month's bet
        Wbar = contributions_for(pivot, eligible, w_df.loc[dprev].astype(float), all_factors)
        if Wbar is None:
            continue
        Wbar = Wbar.reindex(countries).fillna(0.0)

        d_total = (Wt - Wprev)
        d_term1 = (Wt - Wbar)      # bet change (wanted)
        d_term2 = (Wbar - Wprev)   # basket refresh (avoidable)

        to_total = 0.5 * d_total.abs().sum()
        to_term1 = 0.5 * d_term1.abs().sum()
        to_term2 = 0.5 * d_term2.abs().sum()

        # membership churn (names rotating across held bands)
        m_now, m_prev = members.get(d, set()), members.get(dprev, set())
        entries = len(m_now - m_prev)
        exits = len(m_prev - m_now)

        rows.append({
            "Date": d,
            "Total_Turnover": to_total,
            "Term1_BetChange": to_term1,
            "Term2_BasketRefresh": to_term2,
            "Term1_plus_Term2": to_term1 + to_term2,
            "Name_Entries": entries,
            "Name_Exits": exits,
        })

    res = pd.DataFrame(rows).set_index("Date")

    # ---- summary ----
    mt = res["Total_Turnover"].mean()
    m1 = res["Term1_BetChange"].mean()
    m2 = res["Term2_BasketRefresh"].mean()
    denom = m1 + m2
    summary = pd.Series({
        "Mean Total one-way turnover (%/mo)": mt * 100,
        "Mean Term1 bet-change (%/mo)": m1 * 100,
        "Mean Term2 basket-refresh (%/mo)": m2 * 100,
        "Term1 share of (T1+T2)": m1 / denom,
        "Term2 share of (T1+T2)": m2 / denom,
        "Mean name entries/mo": res["Name_Entries"].mean(),
        "Mean name exits/mo": res["Name_Exits"].mean(),
        "Annualized Total turnover (%)": mt * 12 * 100,
        "Annualized Term2 (avoidable) (%)": m2 * 12 * 100,
        "Validation mean abs diff/date": val.mean(),
    })

    print("\n" + "=" * 70)
    print("TURNOVER DECOMPOSITION SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:42s}: {v:.4f}")
    print("=" * 70)

    # ---- save ----
    with pd.ExcelWriter("T2_Turnover_Decomposition.xlsx", engine="xlsxwriter") as xw:
        out = res.copy()
        out.index = out.index.strftime("%Y-%m-%d")
        out.to_excel(xw, sheet_name="Monthly")
        summary.to_frame("Value").to_excel(xw, sheet_name="Summary")
    print("Saved T2_Turnover_Decomposition.xlsx")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(res.index, res["Total_Turnover"] * 100, label="Total", color="black", lw=1.3)
    ax.plot(res.index, res["Term1_BetChange"] * 100, label="Term 1: bet change (wanted)",
            color="tab:green", lw=1.0)
    ax.plot(res.index, res["Term2_BasketRefresh"] * 100, label="Term 2: basket refresh (avoidable)",
            color="tab:red", lw=1.0)
    ax.set_ylabel("One-way turnover (% / month)")
    ax.set_title(f"Country turnover decomposition\n"
                 f"Total {mt*100:.1f}%  =  Term1 {m1*100:.1f}% (bet)  +  "
                 f"Term2 {m2*100:.1f}% (basket refresh)  [L1, interaction in text]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("T2_Turnover_Decomposition.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved T2_Turnover_Decomposition.pdf")


if __name__ == "__main__":
    main()
