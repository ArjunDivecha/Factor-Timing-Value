#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
=============================================================================
SCRIPT NAME: Step Six Point Five.py
=============================================================================

PURPOSE:
Creates country alphas using ONLY the factors selected by Step Five's optimizer.
Unlike Step Six (which uses all factors), this script gates each factor by whether
it received a non-zero weight in the Step Five optimization.

COUNTRY ALPHA CALCULATION:
==========================

For each month and each country:

    Country_Alpha = Σ  I(Exposure ≠ 0) × Weight × T60
                     f

Where:
    - f iterates over all factors
    - Weight[f, month]    = optimized factor weight from Step Five
    - T60[f, month]       = trailing 60-month average return for that factor
    - Exposure[country, f, month] = country's exposure to that factor (from Step Three)
    - I(Exposure ≠ 0)     = 1 if the country has non-zero exposure, 0 otherwise

The exposure value itself is NOT used as a multiplier — it only acts as a binary
gate.  If a factor has zero weight in Step Five, it contributes nothing regardless
of exposure.  If a factor has non-zero weight, every country with non-zero
exposure to that factor receives the same alpha contribution (Weight × T60).

EXAMPLE:
    Month M: Step Five assigns   REER_CS = 100% weight, all others = 0%
             T60 for REER_CS   = 0.15
    → Every country with non-zero REER_CS exposure gets alpha = 1.0 × 0.15 = 0.15
    → Countries with zero REER_CS exposure get alpha = 0

INPUT FILES:
1. T2_rolling_window_weights.xlsx
   - Source: Step Five FAST.py
   - Structure: Index = dates (datetime), Columns = factor names, Values = weights

2. T60.xlsx
   - Source: Step Four Create Monthly Top20 Returns FAST.py
   - Structure: Sheet "T60", Index = "Date" (datetime), Columns = factor names,
                Values = trailing 60-month average returns × 100

3. T2_Top_20_Exposure.csv
   - Source: Step Three Top20 Portfolios Fast.py
   - Structure: Columns = Date, Country, <factor1>, <factor2>, ...
                Values = country weights (0 to 1)

4. T2 Master.xlsx (optional)
   - Used only for consistent country ordering in output

OUTPUT FILES:
1. T2_Country_Top_Alphas.xlsx
   - Sheet "Country_Scores": Monthly alpha scores (rows = dates, columns = countries)
   - Sheet "Data_Quality": Completeness metrics by country
   - Sheet "Factor_Contributions": Which factors contributed each month
   - Sheet "Missing_Data_Log": Record of any skipped data points

VERSION: 1.0
LAST UPDATED: 2026-05-26
AUTHOR: Quantitative Research Team

DEPENDENCIES:
- pandas >= 2.0.0
- numpy >= 1.24.0
- openpyxl >= 3.1.0
- tqdm >= 4.65.0

USAGE:
    python "Step Six Point Five.py"
=============================================================================
"""

import pandas as pd
import numpy as np
import time
from tqdm import tqdm

# =============================================================================
# FILE PATHS
# =============================================================================
WEIGHTS_FILE = 'T2_rolling_window_weights.xlsx'
T60_FILE = 'T60.xlsx'
EXPOSURE_FILE = 'T2_Top_20_Exposure.csv'
T2_MASTER_FILE = 'T2 Master.xlsx'
OUTPUT_FILE = 'T2_Country_Top_Alphas.xlsx'


def load_data():
    """
    Load and validate all three input files plus the optional country-order reference.

    Returns:
        tuple: (weights_df, t60_df, exposure_df, country_order)
    """
    print(f"Loading optimized factor weights from {WEIGHTS_FILE} ...")
    weights_df = pd.read_excel(WEIGHTS_FILE, index_col=0)
    weights_df.index = pd.to_datetime(weights_df.index)
    weights_df = weights_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    print(f"  Shape: {weights_df.shape}  |  Date range: "
          f"{weights_df.index.min():%Y-%m-%d} → {weights_df.index.max():%Y-%m-%d}")

    print(f"Loading trailing 60-month averages from {T60_FILE} ...")
    t60_df = pd.read_excel(T60_FILE, sheet_name=0, index_col=0)
    t60_df.index = pd.to_datetime(t60_df.index)
    t60_df.index.name = 'Date'
    print(f"  Shape: {t60_df.shape}  |  Date range: "
          f"{t60_df.index.min():%Y-%m-%d} → {t60_df.index.max():%Y-%m-%d}")

    print(f"Loading country factor exposures from {EXPOSURE_FILE} ...")
    exposure_df = pd.read_csv(EXPOSURE_FILE)
    exposure_df['Date'] = pd.to_datetime(exposure_df['Date'])
    print(f"  Shape: {exposure_df.shape}  |  Countries: {exposure_df['Country'].nunique()}  |  "
          f"Date range: {exposure_df['Date'].min():%Y-%m-%d} → {exposure_df['Date'].max():%Y-%m-%d}")

    print(f"Loading country order from {T2_MASTER_FILE} ...")
    try:
        master_df = pd.read_excel(T2_MASTER_FILE, sheet_name='1MRet')
        country_order = master_df.columns[1:].tolist()
        print(f"  Found {len(country_order)} countries in reference file")
    except Exception as e:
        print(f"  Warning: could not load country order ({e}); using alphabetical")
        country_order = None

    return weights_df, t60_df, exposure_df, country_order


def compute_country_alphas(weights_df, t60_df, exposure_df):
    """
    For each common date, compute country alphas as:

        Country_Alpha[c, t] = Σ_f  I(Exposure[c,f,t] ≠ 0) × Weight[f,t] × T60[f,t]

    Returns:
        pivot_df:    DataFrame (index=date, columns=country, values=alpha score)
        factor_info: DataFrame tracking which factors contributed each month
        skip_log:    list of dicts recording skipped data points
    """
    # Determine the set of factors present in all three files
    weight_factors = set(weights_df.columns)
    t60_factors = set(t60_df.columns)
    exposure_factors = set(c for c in exposure_df.columns if c not in ('Date', 'Country'))
    common_factors = sorted(weight_factors & t60_factors & exposure_factors)
    print(f"\nFactors present in all three files: {len(common_factors)}")

    # Determine common dates across all three files
    weight_dates = set(weights_df.index)
    t60_dates = set(t60_df.index)
    exposure_dates = set(exposure_df['Date'])
    common_dates = sorted(weight_dates & t60_dates & exposure_dates)
    print(f"Common dates across all three files: {len(common_dates)}")
    if not common_dates:
        raise ValueError("No common dates found across the three input files.")
    print(f"Date range: {common_dates[0]:%Y-%m-%d} → {common_dates[-1]:%Y-%m-%d}")

    results = []
    factor_info_rows = []
    skip_log = []

    for dt in tqdm(common_dates, desc="Processing months"):
        # 1. Get the weight vector and T60 vector for this month
        w = weights_df.loc[dt, common_factors].values.astype(float)
        t60 = t60_df.loc[dt, common_factors].values.astype(float)

        # Replace any NaN in T60 with 0 (factor has no history yet)
        t60_nan_mask = np.isnan(t60)
        t60 = np.where(t60_nan_mask, 0.0, t60)

        # Weighted factor alpha = weight × T60
        weighted_alpha = w * t60

        # Track which factors are active this month
        active_mask = w != 0
        active_factors = [f for f, a in zip(common_factors, active_mask) if a]
        factor_info_rows.append({
            'date': dt,
            'num_active_factors': int(active_mask.sum()),
            'active_factors': ', '.join(active_factors),
            'total_weighted_alpha': float(weighted_alpha.sum()),
        })

        # 2. Get all countries for this date
        month_exp = exposure_df[exposure_df['Date'] == dt]
        if month_exp.empty:
            skip_log.append({'date': dt, 'reason': 'no exposure data for this date'})
            continue

        for _, row in month_exp.iterrows():
            country = row['Country']
            exposures = row[common_factors].values.astype(float)

            # Binary gate: 1 where country has non-zero exposure, 0 otherwise
            has_exposure = (exposures != 0) & (~np.isnan(exposures))

            # Country alpha = sum of (weighted_alpha where country has exposure)
            country_alpha = float(np.sum(weighted_alpha * has_exposure))

            results.append({
                'date': dt,
                'country': country,
                'alpha_score': country_alpha,
                'factors_contributing': int((has_exposure & active_mask).sum()),
            })

    # Build output DataFrames
    results_df = pd.DataFrame(results)
    pivot_df = results_df.pivot_table(index='date', columns='country', values='alpha_score')
    factor_info_df = pd.DataFrame(factor_info_rows)

    return pivot_df, factor_info_df, results_df, skip_log


def main():
    start_time = time.time()

    # 1. Load data
    weights_df, t60_df, exposure_df, country_order = load_data()

    # 2. Compute country alphas
    pivot_df, factor_info_df, results_df, skip_log = compute_country_alphas(
        weights_df, t60_df, exposure_df
    )

    # 3. Reorder columns to match T2 Master country order
    if country_order:
        ordered = [c for c in country_order if c in pivot_df.columns]
        extra = [c for c in pivot_df.columns if c not in ordered]
        pivot_df = pivot_df[ordered + extra]
        print(f"Columns reordered to match {T2_MASTER_FILE} ({len(ordered)} matched, "
              f"{len(extra)} extra)")

    # 4. Data quality summary
    missing_months = pivot_df.isna().sum()
    quality_df = pd.DataFrame({
        'country': missing_months.index,
        'missing_months': missing_months.values,
        'completeness_pct': 100 * (1 - missing_months.values / len(pivot_df)),
    }).sort_values('completeness_pct')

    # 5. Factor-count pivot (how many factors contributed per country-month)
    factor_count_pivot = results_df.pivot_table(
        index='date', columns='country', values='factors_contributing'
    )

    # 6. Write to Excel
    print(f"\nWriting results to {OUTPUT_FILE} ...")
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
        pivot_df.to_excel(writer, sheet_name='Country_Scores')
        quality_df.to_excel(writer, sheet_name='Data_Quality', index=False)
        factor_info_df.to_excel(writer, sheet_name='Factor_Contributions', index=False)
        factor_count_pivot.to_excel(writer, sheet_name='Factor_Counts')

        if skip_log:
            pd.DataFrame(skip_log).to_excel(writer, sheet_name='Skip_Log', index=False)

        workbook = writer.book
        date_fmt = workbook.add_format({'num_format': 'yyyy-mm-dd'})

        for sheet_name in ['Country_Scores', 'Factor_Counts']:
            ws = writer.sheets[sheet_name]
            ws.set_column(0, 0, 14, date_fmt)

        if 'Factor_Contributions' in writer.sheets:
            ws = writer.sheets['Factor_Contributions']
            ws.set_column(0, 0, 14, date_fmt)
            ws.set_column(2, 2, 80)

    elapsed = time.time() - start_time

    # 7. Summary
    print("\n" + "=" * 70)
    print("STEP SIX POINT FIVE — SUMMARY")
    print("=" * 70)
    print(f"  Date range       : {pivot_df.index.min():%Y-%m-%d} → {pivot_df.index.max():%Y-%m-%d}")
    print(f"  Months processed : {len(pivot_df)}")
    print(f"  Countries        : {len(pivot_df.columns)}")
    print(f"  Output file      : {OUTPUT_FILE}")
    print(f"  Execution time   : {elapsed:.1f}s")

    avg_active = factor_info_df['num_active_factors'].mean()
    print(f"  Avg active factors/month: {avg_active:.1f}")

    print("\nPreview (first 5 months, first 5 countries):")
    preview_cols = pivot_df.columns[:5].tolist()
    print(pivot_df[preview_cols].head())

    print("\nActive-factor summary (last 5 months):")
    print(factor_info_df[['date', 'num_active_factors', 'active_factors']].tail())

    print("\nData quality (countries with most missing data):")
    print(quality_df.head())


if __name__ == "__main__":
    main()
