"""
=============================================================================
SCRIPT NAME: test_wider_band_smooth.py  (EXPERIMENT ONLY)
=============================================================================

WHAT THIS PROGRAM DOES:
Tests wider fuzzy band + EMA smoothing on Step Eight's country weight
calculation to reduce turnover. Replicates Step Eight's exact logic but with:
  - SOFT_BAND_TOP = 0.20 (was 0.15)
  - SOFT_BAND_CUTOFF = 0.35 (was 0.25)
  - EMA smoothing on final country weights (alpha = 0.5)

Overwrites T2_Final_Country_Weights.xlsx so Step Nine works normally.

INPUT FILES:
- T2_rolling_window_weights.xlsx : Factor weights
- Normalized_T2_MasterCSV.csv : Factor data per country
- T2 Master.xlsx : Country sort order

OUTPUT FILES:
- T2_Final_Country_Weights.xlsx : Country weights with wider band + smoothing

VERSION: 1.0   LAST UPDATED: 2026-06-10
=============================================================================
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

HERE = Path(__file__).parent

SOFT_BAND_TOP = 0.20
SOFT_BAND_CUTOFF = 0.35
SMOOTH_ALPHA = 0.5

weights_file = HERE / "T2_rolling_window_weights.xlsx"
factor_file = HERE / "Normalized_T2_MasterCSV.csv"
master_file = HERE / "T2 Master.xlsx"
output_file = HERE / "T2_Final_Country_Weights.xlsx"
country_final_file = HERE / "T2_Country_Final.xlsx"

print("=" * 70)
print("WIDER FUZZY BAND + EMA SMOOTHING TEST")
print(f"  SOFT_BAND_TOP={SOFT_BAND_TOP}, SOFT_BAND_CUTOFF={SOFT_BAND_CUTOFF}")
print(f"  EMA alpha={SMOOTH_ALPHA}")
print("=" * 70)

print("\nLoading factor weights...")
try:
    feature_weights_df = pd.read_excel(weights_file, sheet_name='Net_Weights', index_col=0)
except Exception:
    feature_weights_df = pd.read_excel(weights_file, index_col=0)

feature_weights_df = feature_weights_df.sort_index()
feature_weights_df = feature_weights_df.shift(1).iloc[1:]
print(f"  {feature_weights_df.index[0].strftime('%Y-%m')} to {feature_weights_df.index[-1].strftime('%Y-%m')}")

print("Loading factor data...")
factor_df = pd.read_csv(factor_file)
factor_df['date'] = pd.to_datetime(factor_df['date'])
all_countries = factor_df['country'].unique()
all_dates = list(feature_weights_df.index)
by_date = factor_df.groupby('date')

print("\nComputing country weights (wider band)...")
all_weights = pd.DataFrame(0.0, index=all_dates, columns=all_countries)
all_weights = all_weights.astype(float)

for date in tqdm(all_dates):
    date_dt = pd.to_datetime(date)
    w = feature_weights_df.loc[date_dt].astype(float)
    w = w[w.abs() > 1e-10]
    if w.empty:
        continue
    try:
        slice_df = by_date.get_group(date_dt)
    except KeyError:
        continue
    pivot = slice_df.pivot(index='country', columns='variable', values='value')
    common_factors = pivot.columns.intersection(w.index)
    if len(common_factors) == 0:
        continue
    V = pivot[common_factors]
    w_vec = w.loc[common_factors]
    rank_desc = V.rank(axis=0, method='first', ascending=False)
    rank_asc = V.rank(axis=0, method='first', ascending=True)
    counts = V.notna().sum(axis=0).replace(0, np.nan)
    counts_mat = np.tile(counts.values, (len(V.index), 1))
    rank_desc_pct = rank_desc.values / counts_mat
    rank_asc_pct = rank_asc.values / counts_mat
    pos_mask = (w_vec.values > 0).astype(float)
    pos_mask_mat = np.tile(pos_mask, (len(V.index), 1))
    rank_pct = np.where(pos_mask_mat > 0, rank_desc_pct, rank_asc_pct)

    full_mask = (rank_pct < SOFT_BAND_TOP).astype(float)
    in_band = (rank_pct >= SOFT_BAND_TOP) & (rank_pct <= SOFT_BAND_CUTOFF)
    taper = 1.0 - (rank_pct - SOFT_BAND_TOP) / (SOFT_BAND_CUTOFF - SOFT_BAND_TOP)
    taper = np.where(in_band, taper, 0.0)
    fuzzy = full_mask + taper

    col_sums = fuzzy.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    fuzzy_norm = fuzzy / col_sums

    w_mat = np.tile(w_vec.values, (len(V.index), 1))
    contrib = fuzzy_norm * w_mat
    country_weights = contrib.sum(axis=1)
    if hasattr(country_weights, 'index'):
        all_weights.loc[date_dt, country_weights.index] = country_weights.values
    else:
        all_weights.loc[date_dt, V.index] = country_weights

print("\nApplying EMA smoothing on country weights...")
smoothed = pd.DataFrame(0.0, index=all_weights.index, columns=all_weights.columns)
smoothed = smoothed.astype(float)
for c in all_weights.columns:
    raw = all_weights[c].values
    s = raw.copy()
    for i in range(1, len(raw)):
        s[i] = SMOOTH_ALPHA * raw[i] + (1 - SMOOTH_ALPHA) * s[i - 1]
    smoothed[c] = s

row_sums = smoothed.sum(axis=1)
row_sums[row_sums == 0] = 1.0
smoothed = smoothed.div(row_sums, axis=0)
print(f"  Weight sum range: {smoothed.sum(axis=1).min():.4f} to {smoothed.sum(axis=1).max():.4f}")

print(f"\nSaving to {output_file}...")
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    smoothed.to_excel(writer, sheet_name='All Periods')
    summary_stats = pd.DataFrame({
        'Mean Weight': smoothed.mean(),
        'Std Dev': smoothed.std(),
        'Min Weight': smoothed.min(),
        'Max Weight': smoothed.max(),
        'Days with Weight': (smoothed.abs() > 0).sum()
    }).sort_values('Mean Weight', ascending=False)
    summary_stats.to_excel(writer, sheet_name='Summary Statistics')
    latest_date = smoothed.dropna(how='all').index[-1]
    latest = smoothed.loc[latest_date].sort_values(ascending=False)
    latest_df = pd.DataFrame({
        'Country': latest.index,
        'Weight': latest.values,
        'vs_avg': latest.values - smoothed.mean().loc[latest.index].values
    })
    latest_df.to_excel(writer, sheet_name='Latest Weights', index=False)

print(f"Writing {country_final_file}...")
latest_date = smoothed.dropna(how='all').index[-1]
latest_weights = smoothed.loc[latest_date]
master_df = pd.read_excel(master_file, index_col=0)
country_cols = [c for c in master_df.columns if c in latest_weights.index]
final = pd.DataFrame({
    'Country': country_cols,
    'Weight': [latest_weights.get(c, 0) for c in country_cols]
})
with pd.ExcelWriter(country_final_file, engine='xlsxwriter') as writer:
    final.to_excel(writer, sheet_name='Final Weights', index=False)

print(f"\nTop 10 countries by avg weight:")
avg = smoothed.mean().sort_values(ascending=False)
for c in avg.head(10).index:
    print(f"  {c:20s} {avg[c]:8.4f}")

active_countries = (smoothed > 0.001).sum(axis=1)
print(f"\nAvg active countries per month: {active_countries.mean():.1f}")
print(f"Min active: {active_countries.min()}, Max active: {active_countries.max()}")

print("\nDone. Ready to run Step Nine.")
