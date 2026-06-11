#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: Step Four Point Five.py
=============================================================================

INPUT FILES:
- Normalized_T2_MasterCSV.csv:
  Normalized factor data in long format (date, country, variable, value)
- Step Tcost.xlsx (jjunk sheet):
  Trading Cost per country (static, columns: Country, Borrow Cost, Trading Cost)

OUTPUT FILES:
- T2_Tcost.xlsx (Factor_Tcost sheet):
  Weighted-average Trading Cost for each factor portfolio at each point in time.
  Rows = dates, Columns = factors.

DESCRIPTION:
For each factor and each month, replicates the fuzzy-logic portfolio weights
from Step Four (soft 15-25% linear taper) and computes the weighted-average
Trading Cost across the portfolio countries using the static Tcost table.

FUZZY LOGIC (same as Step Four):
- Top 15% of countries by factor score: full weight (1.0)
- 15-25%: linearly decreasing weight
- Below 25%: zero weight
- Weights normalized to sum to 1

VERSION: 1.0
=============================================================================
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SOFT_BAND_TOP    = 0.15   # top 15% -> full weight
SOFT_BAND_CUTOFF = 0.25   # 25%     -> zero weight

DATA_PATH   = "Normalized_T2_MasterCSV.csv"
TCOST_PATH  = "Step Tcost.xlsx"
OUTPUT_PATH = "T2_Tcost.xlsx"


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    # ---- Load data -----------------------------------------------
    print("Loading factor data...")
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"]).dt.to_period("M").dt.to_timestamp()

    print("Loading Tcost data...")
    tcost_df = pd.read_excel(TCOST_PATH, sheet_name="jjunk")
    tcost_map = tcost_df.set_index("Country")["Trading Cost"].to_dict()
    print(f"  {len(tcost_map)} countries with Trading Cost data")

    # ---- Identify factors ----------------------------------------
    features = sorted(set(data["variable"]) - {"1MRet"})
    print(f"  {len(features)} factors to process")

    # ---- Pre-index factor data by (feature, date) ----------------
    print("Pre-indexing factor data...")
    results = {}   # {factor: {date: weighted_avg_tcost}}

    for feature in features:
        feat_data = data[data["variable"] == feature].copy()
        if feat_data.empty:
            continue

        feat_data["value"] = pd.to_numeric(feat_data["value"], errors="coerce")
        feat_data = feat_data.dropna(subset=["value"])
        if feat_data.empty:
            continue

        factor_results = {}

        for date, group in feat_data.groupby("date"):
            # Sort descending by factor score
            group = group.sort_values("value", ascending=False).reset_index(drop=True)
            n = len(group)
            if n == 0:
                continue

            # Fuzzy weights
            rank_pct = (np.arange(n) + 1) / n
            weights = np.zeros(n)
            weights[rank_pct < SOFT_BAND_TOP] = 1.0
            in_band = (rank_pct >= SOFT_BAND_TOP) & (rank_pct <= SOFT_BAND_CUTOFF)
            weights[in_band] = 1.0 - (rank_pct[in_band] - SOFT_BAND_TOP) / (SOFT_BAND_CUTOFF - SOFT_BAND_TOP)

            nonzero = weights > 0
            if not nonzero.any():
                continue

            w = weights[nonzero]
            countries = group["country"].values[nonzero]

            # Map Trading Cost; skip countries not in Tcost table
            tcosts = np.array([tcost_map.get(c, np.nan) for c in countries])
            valid = ~np.isnan(tcosts)
            if not valid.any():
                continue

            w_valid = w[valid]
            t_valid = tcosts[valid]
            w_valid = w_valid / w_valid.sum()   # renormalize after dropping missing

            factor_results[date] = float(np.dot(w_valid, t_valid))

        if factor_results:
            results[feature] = factor_results

    # ---- Assemble DataFrame --------------------------------------
    print(f"\nAssembling results for {len(results)} factors...")
    tcost_df_out = pd.DataFrame(results).sort_index()
    tcost_df_out.index.name = "Date"

    print(f"Output shape: {tcost_df_out.shape}")
    print(f"Date range:   {tcost_df_out.index.min()} to {tcost_df_out.index.max()}")
    print("\nSample (first 5 rows, first 5 factors):")
    print(tcost_df_out.iloc[:5, :5])

    # ---- Save to Excel -------------------------------------------
    print(f"\nSaving to {OUTPUT_PATH}...")
    with pd.ExcelWriter(OUTPUT_PATH, engine="xlsxwriter") as writer:
        tcost_df_out.to_excel(writer, sheet_name="Factor_Tcost", index_label="Date")
        wb  = writer.book
        ws  = writer.sheets["Factor_Tcost"]
        date_fmt = wb.add_format({"num_format": "dd-mmm-yyyy"})
        num_fmt  = wb.add_format({"num_format": "0.0000"})
        ws.set_column(0, 0, 15, date_fmt)
        ws.set_column(1, len(tcost_df_out.columns), 12, num_fmt)

    print("Done!")


if __name__ == "__main__":
    main()
