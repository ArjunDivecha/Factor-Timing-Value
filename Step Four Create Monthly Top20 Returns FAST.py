#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: Step Four Create Monthly Top20 Returns FAST.py (Fuzzy Logic)
=============================================================================

INPUT FILES:
- Normalized_T2_MasterCSV.csv: 
  Normalized factor data in long format (date, country, variable, value)
  Contains all country factor scores and 1-month returns for portfolio construction
- Portfolio_Data.xlsx (Benchmarks sheet):
  Equal weight benchmark returns for calculating excess returns

OUTPUT FILES:
- T2_Optimizer.xlsx (Monthly_Net_Returns sheet):
  Monthly excess returns for each factor portfolio (portfolio return - benchmark return)
  Used as input for portfolio optimization in later steps
- T60.xlsx (T60 sheet):
  60-month trailing averages of excess returns for each factor
  Provides smoothed performance trends for analysis

VERSION: 4.2 - FAST Optimized Fuzzy Logic Implementation
LAST UPDATED: 2025-12-03
AUTHOR: Claude Code (optimized for speed)

OPTIMIZATIONS:
- Pre-merged factor+returns data for each date to avoid repeated merges
- Pre-indexed data lookups using dictionaries
- Vectorized numpy operations for weight calculations

DESCRIPTION:
This script creates factor-based investment portfolios and calculates their excess returns.
It's the data preparation step for portfolio optimization. Here's what it does:

1. PORTFOLIO CREATION: For each factor (like GDP growth, inflation, etc.):
   - Each month, ranks all countries by their factor score
   - Uses fuzzy logic with soft 15-25% linear taper (not hard 20% cutoff)
   - Top 15% countries get full weight, 15-25% get linearly decreasing weight
   - Calculates the weighted return of selected countries

2. EXCESS RETURN CALCULATION: 
   - Subtracts the benchmark return from each portfolio's return
   - This shows how much better (or worse) the factor strategy performed
   - Positive excess returns mean the factor strategy beat the benchmark

3. DATA SMOOTHING:
   - Creates 60-month rolling averages to reduce noise
   - Helps identify long-term factor performance trends
   - Fills missing data using cross-sectional averages

FUZZY LOGIC FEATURES:
- Soft 15-25% linear taper instead of hard 20% cutoff
- Top 15% countries receive full weight (1.0)
- Countries ranked 15-25% receive linearly decreasing weights
- Countries below 25% receive zero weight
- Weights normalized to sum to 1 for equal capital allocation
- Proper handling of missing data and edge cases

DEPENDENCIES:
- pandas >= 2.0.0
- numpy >= 1.24.0
- xlsxwriter (for Excel formatting)

USAGE:
python "Step Four Create Monthly Top20 Returns FAST.py"

NOTES:
- Excludes multi-month return variables and technical indicators from analysis
- Missing factor values are handled by skipping those country-date combinations
- Cross-sectional mean filling ensures no missing data in final output
- Results are scaled to percentage points (multiplied by 100)
=============================================================================
"""

import os
import warnings
from typing import Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Fuzzy Logic Configuration (Same as Step Three)
# ------------------------------------------------------------------
SOFT_BAND_TOP = 0.15     # 15% ⇒ full weight
SOFT_BAND_CUTOFF = 0.25  # 25% ⇒ zero weight


# ------------------------------------------------------------------
# Fuzzy Logic Portfolio Analysis - OPTIMIZED
# ------------------------------------------------------------------
def analyze_portfolios(
    data: pd.DataFrame,
    features: list,
    benchmark_returns: pd.Series,
) -> Dict[str, pd.Series]:
    """
    Build factor portfolios with fuzzy logic (soft 15–25% linear taper) and
    return monthly **net** returns (portfolio – benchmark).
    
    OPTIMIZED VERSION - produces identical output to original.

    Parameters
    ----------
    data : DataFrame
        Long-format table with columns {date,country,variable,value}.
    features : list[str]
        Factor names to evaluate (ex-'1MRet').
    benchmark_returns : Series
        Monthly equal-weight benchmark (index: datetime).

    Returns
    -------
    monthly_net_returns : dict[str, Series]
        Net return series per factor.
    """
    monthly_net_returns: Dict[str, pd.Series] = {}
    
    print("Pre-indexing data...")
    
    # Get return data once and clean it
    returns_data = data[data['variable'] == '1MRet'].copy()
    returns_data['value'] = pd.to_numeric(returns_data['value'], errors='coerce')
    returns_data = returns_data.rename(columns={'value': 'return_value'})
    
    # Create date-indexed dictionary for returns: {date: DataFrame with country, return_value}
    returns_by_date = {
        date: group[['country', 'return_value']].copy()
        for date, group in returns_data.groupby('date')
    }
    
    # Pre-merge returns with each feature for each date
    # This is the key optimization - do all merges upfront
    print("Pre-merging factor and returns data...")
    feature_merged_cache = {}
    
    for feature in features:
        feature_data = data[data['variable'] == feature].copy()
        if feature_data.empty:
            continue
            
        feature_data['value'] = pd.to_numeric(feature_data['value'], errors='coerce')
        feature_data = feature_data.dropna(subset=['value'])
        
        if feature_data.empty:
            continue
        
        feature_data = feature_data.rename(columns={'value': 'factor_value'})
        
        # Pre-merge and pre-sort for each date
        feature_merged_cache[feature] = {}
        for date, group in feature_data.groupby('date'):
            if date not in returns_by_date:
                continue
            
            # Merge with returns
            merged = pd.merge(
                group[['country', 'factor_value']],
                returns_by_date[date],
                on='country'
            )
            
            # Drop NaN and sort by factor descending
            merged = merged.dropna(subset=['factor_value', 'return_value'])
            if merged.empty:
                continue
            
            merged = merged.sort_values('factor_value', ascending=False).reset_index(drop=True)
            
            # Store as numpy arrays for fast computation
            feature_merged_cache[feature][date] = (
                merged['factor_value'].values,
                merged['return_value'].values
            )
    
    print(f"Processing {len(features)} features...")
    processed = 0
    
    for feature in features:
        # Skip if no cached data for this feature
        if feature not in feature_merged_cache:
            continue
        
        feat_by_date = feature_merged_cache[feature]
        feature_dates = sorted(feat_by_date.keys())
        
        if not feature_dates:
            continue
        
        # Initialize results
        portfolio_returns = pd.Series(index=feature_dates, dtype=float)
        
        # Process each date with valid data
        for date in feature_dates:
            factor_values, return_values = feat_by_date[date]
            n = len(factor_values)
            
            if n == 0:
                continue
            
            # Compute rank percentile (already sorted descending)
            rank_pct = (np.arange(n) + 1) / n
            
            # Compute weights using vectorized numpy operations
            weights = np.zeros(n)
            
            # Full weight for top band (< 15%)
            top_mask = rank_pct < SOFT_BAND_TOP
            weights[top_mask] = 1.0
            
            # Linearly decreasing weight inside the grey band (15% - 25%)
            in_band = (rank_pct >= SOFT_BAND_TOP) & (rank_pct <= SOFT_BAND_CUTOFF)
            weights[in_band] = 1.0 - (rank_pct[in_band] - SOFT_BAND_TOP) / (SOFT_BAND_CUTOFF - SOFT_BAND_TOP)
            
            # Filter to non-zero weights
            nonzero_mask = weights > 0
            if not nonzero_mask.any():
                continue
            
            weights_filtered = weights[nonzero_mask]
            returns_filtered = return_values[nonzero_mask]
            
            # Normalize weights to sum to 1
            weights_filtered = weights_filtered / weights_filtered.sum()
            
            # Portfolio return = weighted sum
            portfolio_return = np.dot(weights_filtered, returns_filtered)
            portfolio_returns[date] = portfolio_return
        
        # Drop any NaN values
        portfolio_returns = portfolio_returns.dropna()
        
        # Skip if no valid returns
        if portfolio_returns.empty:
            continue
        
        # Calculate net returns (portfolio - benchmark)
        aligned_benchmark = benchmark_returns.reindex(portfolio_returns.index)
        valid_idx = aligned_benchmark.notna()
        if any(valid_idx):
            net_returns = portfolio_returns[valid_idx] - aligned_benchmark[valid_idx]
            monthly_net_returns[feature] = net_returns
        
        processed += 1
        if processed % 20 == 0:
            print(f"  Processed {processed}/{len(features)} features...")

    return monthly_net_returns


# ------------------------------------------------------------------
# Excel Output Helpers (unchanged, except debug prints preserved)
# ------------------------------------------------------------------
def save_net_returns_to_excel(net_returns: Dict[str, pd.Series], output_path: str):
    """Save monthly net returns and trailing-60M averages to Excel."""
    print("\nDebug information:")
    print(f"Number of factors: {len(net_returns)}")
    print(f"Sample factors: {list(net_returns)[:5]}")

    # Optional deep-dive checks
    target_factors = [
        "LT_Growth_TS",
        "10Yr Bond 12_CS",
        "10Yr Bond 12_TS",
        "10Yr Bond_CS",
        "10Yr Bond_TS",
    ]
    for fac in target_factors:
        if fac in net_returns:
            print(f"\nFactor {fac} exists:")
            print(net_returns[fac].head())
        else:
            print(f"\nFactor {fac} does NOT exist")

    # Dict[Series] → DataFrame
    net_df = pd.DataFrame(net_returns)

    # Exclude multi-month returns and other unwanted columns
    cols_excl = [
        "3MRet",
        "6MRet",
        "9MRet",
        "12MRet",
        "120MA_CS",
        "120MA_TS",
        "12MTR_CS",
        "12MTR_TS",
        "Agriculture_CS",
        "Agriculture 12_CS",
        "Copper_CS",
        "Copper 12_CS",
        "Gold_CS",
        "Gold 12_CS",
        "Oil_CS",
        "Oil 12_CS",
        "BEST EPS_CS",
        "Currency_CS",
        "MCAP_CS",
        "MCAP_TS",
        "MCAP Adj_CS",
        "MCAP Adj_TS",
        "PX_LAST_CS",
        "PX_LAST_TS",
        "Tot Return Index _CS",
        "Tot Return Index _TS",
        "Trailing EPS_CS",
        "Trailing EPS_TS",
    ]
    print("\nExcluding the following factors:")
    for col in cols_excl:
        print(f"- {col}" + ("" if col in net_df else " (not found)"))

    net_df = net_df[[c for c in net_df.columns if c not in cols_excl]]

    print(f"\nDataFrame shape: {net_df.shape}")
    print("Preview:")
    print(net_df.iloc[:5, :5])

    # ------------------------------------------------------------------
    # Trailing 60-month averages (T60.xlsx)
    # ------------------------------------------------------------------
    print("\nWriting trailing 60-month averages to T60.xlsx …")
    filled = net_df.apply(lambda row: row.fillna(row.mean()), axis=1)

    # Add an extra future month before rolling
    next_month = filled.index[-1] + pd.DateOffset(months=1)
    filled.loc[next_month] = np.nan

    t60 = filled.shift(1).rolling(60, min_periods=1).mean() * 100

    with pd.ExcelWriter("T60.xlsx", engine="xlsxwriter") as writer:
        t60.to_excel(writer, sheet_name="T60", index_label="Date")
        wb, ws = writer.book, writer.sheets["T60"]
        ws.set_column(0, 0, 15, wb.add_format({"num_format": "dd-mmm-yyyy"}))
        ws.set_column(1, len(t60.columns), 12, wb.add_format({"num_format": "0.0000"}))
    print("T60.xlsx saved.")

    # ------------------------------------------------------------------
    # Main net-return sheet (T2_Optimizer.xlsx)
    # ------------------------------------------------------------------
    net_df = net_df.apply(lambda row: row.fillna(row.mean()), axis=1) * 100
    net_df.sort_index(inplace=True)
    net_df.to_excel(
        output_path, sheet_name="Monthly_Net_Returns", index_label="Date"
    )
    print(f"T2_Optimizer.xlsx saved to {output_path}")


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------
def run_portfolio_analysis(data_path: str, benchmark_path: str, output_path: str):
    """Load data, run analysis, save results."""
    print("Loading data …")
    data = pd.read_csv(data_path)
    data["date"] = (
        pd.to_datetime(data["date"]).dt.to_period("M").dt.to_timestamp()
    )

    bench = pd.read_excel(benchmark_path, sheet_name="Benchmarks", index_col=0)
    bench.index = pd.to_datetime(bench.index).to_period("M").to_timestamp()
    benchmark_returns = bench["equal_weight"]

    # All variables except the return series
    features = sorted(set(data["variable"]) - {"1MRet"})

    print("Analyzing portfolios …")
    net_returns = analyze_portfolios(data, features, benchmark_returns)

    print("Saving results …")
    save_net_returns_to_excel(net_returns, output_path)
    print("Done!")


if __name__ == "__main__":
    DATA_PATH = "Normalized_T2_MasterCSV.csv"
    BENCHMARK_PATH = "Portfolio_Data.xlsx"
    OUTPUT_PATH = "T2_Optimizer.xlsx"

    run_portfolio_analysis(DATA_PATH, BENCHMARK_PATH, OUTPUT_PATH)
