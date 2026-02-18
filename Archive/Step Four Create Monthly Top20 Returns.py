#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: Step Four Create Monthly Top20 Returns.py (Fuzzy Logic)
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

VERSION: 4.1 - Fuzzy Logic Implementation
LAST UPDATED: 2025-07-18
AUTHOR: Claude Code

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
python "Step Four Create Monthly Top20 Returns.py"

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
SOFT_BAND_TOP   = 0.15   # 15% ⇒ full weight
SOFT_BAND_CUTOFF = 0.25  # 25% ⇒ zero weight

# ------------------------------------------------------------------
# Fuzzy Logic Portfolio Analysis (Modified to use Step Three's approach)
# ------------------------------------------------------------------
def analyze_portfolios(
    data: pd.DataFrame,
    features: list,
    benchmark_returns: pd.Series,
) -> Dict[str, pd.Series]:
    """
    Build factor portfolios with fuzzy logic (soft 15–25% linear taper) and
    return monthly **net** returns (portfolio – benchmark).
    
    Modified to use Step Three's fuzzy logic approach:
    - Soft 15-25% linear taper (not hard 20% cutoff)
    - Linear weight calculation between bands
    - Weight normalization to sum to 1
    - Proper NaN handling
    - Date-by-date processing for accuracy

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
    
    # Get return data once and clean it
    returns_data = data[data['variable'] == '1MRet'].copy()
    returns_data['value'] = pd.to_numeric(returns_data['value'], errors='coerce')

    for feature in features:
        # Get data for this feature
        feature_data = data[data['variable'] == feature].copy()
        
        # Skip if no data for this feature
        if feature_data.empty:
            continue
            
        # Convert empty strings to NaN and drop rows with NaN values
        feature_data['value'] = pd.to_numeric(feature_data['value'], errors='coerce')
        feature_data = feature_data.dropna(subset=['value'])  # type: ignore
        
        # Skip if all values are NaN
        if feature_data.empty:
            continue
            
        # Get dates that have actual data (non-NaN values)
        feature_dates = sorted(feature_data['date'].unique())
        
        # Skip if no valid dates
        if not feature_dates:
            continue
        
        # Initialize results
        portfolio_returns = pd.Series(index=feature_dates, dtype=float)
        
        # Process each date with valid data (using fuzzy logic from Step Three)
        for date in feature_dates:
            try:
                # Get data for this date
                curr_feature_data = feature_data[feature_data['date'] == date]
                curr_returns_data = returns_data[returns_data['date'] == date]
                
                # Skip if either is empty
                if len(curr_feature_data) == 0 or len(curr_returns_data) == 0:
                    continue
                
                # Merge feature and returns data (same as Step Three)
                merged = pd.merge(
                    curr_feature_data[['country', 'value']],  # type: ignore
                    curr_returns_data[['country', 'value']],  # type: ignore
                    on='country',
                    suffixes=('_factor', '_return')
                )
                
                # Skip if no matches or if all values are NaN
                if merged.empty:
                    continue
                
                # Drop any remaining NaN values
                merged = merged.dropna(subset=['value_factor', 'value_return'])
                if merged.empty:
                    continue
                
                # FUZZY LOGIC IMPLEMENTATION (Same as Step Three)
                # Rank by factor (high-to-low)
                merged = merged.sort_values('value_factor', ascending=False).reset_index(drop=True)
                merged['rank_pct'] = (merged.index + 1) / len(merged)  # 0-1 percentile rank
                
                # Linear weights using fuzzy logic
                in_band = (merged['rank_pct'] >= SOFT_BAND_TOP) & (merged['rank_pct'] <= SOFT_BAND_CUTOFF)
                merged['weight'] = 0.0
                
                # Full weight for top band (< 15%)
                merged.loc[merged['rank_pct'] < SOFT_BAND_TOP, 'weight'] = 1.0
                
                # Linearly decreasing weight inside the grey band (15% - 25%)
                merged.loc[in_band, 'weight'] = (
                    1.0 - (merged.loc[in_band, 'rank_pct'] - SOFT_BAND_TOP)
                          / (SOFT_BAND_CUTOFF - SOFT_BAND_TOP)
                )
                
                # Remove zero-weight rows for efficiency
                merged = merged[merged['weight'] > 0]
                
                if merged.empty:
                    continue
                
                # Normalize weights to sum to 1 (equal total capital each month)
                merged['weight'] /= merged['weight'].sum()
                
                # Portfolio return = weighted mean (same as Step Three)
                portfolio_return = np.dot(merged['weight'], merged['value_return'])
                portfolio_returns[date] = portfolio_return
                    
            except Exception as e:
                continue
        
        # Drop any NaN values
        portfolio_returns = portfolio_returns.dropna()
        
        # Skip if no valid returns
        if portfolio_returns.empty:
            continue
            
        # Calculate net returns (same as original Step Four)
        aligned_benchmark = benchmark_returns.reindex(portfolio_returns.index)
        valid_idx = aligned_benchmark.notna()
        if any(valid_idx):
            net_returns = portfolio_returns[valid_idx] - aligned_benchmark[valid_idx]
            monthly_net_returns[feature] = net_returns

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