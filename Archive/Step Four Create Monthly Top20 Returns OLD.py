#!/usr/bin/env python3
"""
Portfolio Factor Analysis Module - Optimizer Version

INPUT FILES:
1. Normalized_T2_MasterCSV.csv
   - Normalized factor and return data from Step Two
   - Format: CSV with columns (date, country, variable, value)
   - Required variables:
     * 1MRet: Monthly returns for performance calculation
     * Factor variables: Used for country sorting and portfolio formation
   - All data must be properly normalized and cleaned

2. Portfolio_Data.xlsx
   - Sheet: 'Benchmarks'
   - Equal-weight benchmark returns for performance comparison
   - Format: Excel with date index and return column
   - Monthly frequency aligned with factor data

OUTPUT FILES:
1. T2_Optimizer.xlsx
   - Monthly net returns for factor portfolios
   - Format: Excel workbook with single sheet
   - Sheet: 'Monthly_Net_Returns'
     * Rows: Dates (monthly)
     * Columns: Factor names
     * Values: Net returns relative to benchmark (in %)
   - Excludes multi-month return factors (3MRet, 6MRet, 9MRet, 12MRet)

2. T60.xlsx
   - 60-month trailing average (shifted by one month) of the net returns
   - Format: Excel workbook with single sheet 'T60'
   - Index: Dates (monthly)
   - Columns: Factor names
   - Values: Average monthly returns (decimal form) before percentage conversion

This module analyzes the performance of factor-based country portfolios and saves
monthly net returns (relative to benchmark) to an Excel file. It:
1. Sorts countries by each factor
2. Forms equal-weighted portfolios of top 20% countries
3. Calculates net returns relative to equal-weight benchmark
4. Saves monthly net returns to Excel with features as columns

Author: Codeium AI
Version: 1.2
Last Updated: 2025-06-13
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import warnings
import os

# Configure warnings
warnings.filterwarnings('ignore')

def analyze_portfolios(data: pd.DataFrame, features: list, benchmark_returns: pd.Series) -> Dict[str, pd.Series]:
    """
    Analyze portfolios for all features and calculate net returns.
    Properly handles empty values in the input data.

    Args:
        data: DataFrame with the complete dataset
        features: List of factor features
        benchmark_returns: Benchmark returns series
    
    Returns:
        Dictionary containing monthly net returns for each feature
    """
    monthly_net_returns = {}
    
    for feature in features:
        # Get data for this feature
        feature_data = data[data['variable'] == feature].copy()
        
        # Skip if no data for this feature
        if feature_data.empty:
            continue
            
        # Convert empty strings to NaN and drop rows with NaN values
        feature_data['value'] = pd.to_numeric(feature_data['value'], errors='coerce')
        feature_data = feature_data.dropna(subset=['value'])
        
        # Skip if all values are NaN
        if feature_data.empty:
            continue
            
        # Get dates that have actual data (non-NaN values)
        feature_dates = sorted(feature_data['date'].unique())
        
        # Skip if no valid dates
        if not feature_dates:
            continue
            
        # Get return data
        returns_data = data[data['variable'] == '1MRet'].copy()
        returns_data['value'] = pd.to_numeric(returns_data['value'], errors='coerce')
        
        # Initialize results
        portfolio_returns = pd.Series(index=feature_dates)
        
        # Process each date with valid data
        for date in feature_dates:
            try:
                # Get data for this date
                curr_feature_data = feature_data[feature_data['date'] == date]
                curr_returns_data = returns_data[returns_data['date'] == date]
                
                # Skip if either is empty
                if curr_feature_data.empty or curr_returns_data.empty:
                    continue
                
                # Merge feature and returns data
                portfolio_data = pd.merge(
                    curr_feature_data[['country', 'value']],
                    curr_returns_data[['country', 'value']],
                    on='country',
                    suffixes=('_factor', '_return')
                )
                
                # Skip if no matches or if all values are NaN
                if portfolio_data.empty:
                    continue
                
                # Calculate portfolio return
                n_select = max(1, int(len(portfolio_data) * 0.2))
                selected = portfolio_data.nlargest(n_select, 'value_factor')
                portfolio_return = selected['value_return'].mean()
                portfolio_returns[date] = portfolio_return
                    
            except Exception as e:
                continue
        
        # Drop any NaN values
        portfolio_returns = portfolio_returns.dropna()
        
        # Skip if no valid returns
        if portfolio_returns.empty:
            continue
            
        # Calculate net returns
        aligned_benchmark = benchmark_returns.reindex(portfolio_returns.index)
        valid_idx = aligned_benchmark.notna()
        if any(valid_idx):
            net_returns = portfolio_returns[valid_idx] - aligned_benchmark[valid_idx]
            monthly_net_returns[feature] = net_returns
    
    return monthly_net_returns

def save_net_returns_to_excel(net_returns: Dict[str, pd.Series], output_path: str):
    """
    Save monthly net returns to Excel file, excluding specified factors.

    Args:
        net_returns: Dictionary of net returns series for each feature
        output_path: Path to save the Excel file
    """
    # Print some debug info
    print("\nDebug information:")
    print(f"Number of factors: {len(net_returns)}")
    sample_factors = list(net_returns.keys())[:5]  # First 5 factors
    print(f"Sample factors: {sample_factors}")
    
    # Check for specific factors
    target_factors = ["LT_Growth_TS", "10Yr Bond 12_CS", "10Yr Bond 12_TS", "10Yr Bond_CS", "10Yr Bond_TS"]
    for factor in target_factors:
        if factor in net_returns:
            print(f"\nFactor {factor} exists in results")
            print(f"First few values:")
            print(net_returns[factor].head())
        else:
            print(f"\nFactor {factor} does NOT exist in results")
    
    # Convert dictionary of series to DataFrame
    net_returns_df = pd.DataFrame(net_returns)
    
    # Filter out unwanted return columns and additional factors to exclude
    columns_to_exclude = [
        # Original exclusions
        '3MRet', '6MRet', '9MRet', '12MRet',
        # Additional factors to exclude
        '120MA_CS', '120MA_TS', '12MTR_CS', '12MTR_TS',
        'Agriculture_CS', 'Agriculture 12_CS', 
        'Copper_CS', 'Copper 12_CS',
        'Gold_CS', 'Gold 12_CS',
        'Oil_CS', 'Oil 12_CS',
        'BEST EPS_CS', 'Currency_CS',
        'MCAP_CS', 'MCAP_TS',
        'MCAP Adj_CS', 'MCAP Adj_TS',
        'PX_LAST_CS', 'PX_LAST_TS',
        'Tot Return Index _CS', 'Tot Return Index _TS',
        'Trailing EPS_CS', 'Trailing EPS_TS'
    ]
    
    # Print factors being excluded
    print("\nExcluding the following factors:")
    for col in columns_to_exclude:
        if col in net_returns_df.columns:
            print(f"- {col}")
        else:
            print(f"- {col} (not found in data)")
    
    net_returns_df = net_returns_df[[col for col in net_returns_df.columns if col not in columns_to_exclude]]
    
    # Print info about the DataFrame
    print(f"\nDataFrame shape: {net_returns_df.shape}")
    print("First few rows and columns:")
    print(net_returns_df.iloc[:5, :5])  # First 5 rows and 5 columns
    
    # Before converting to percentages, compute the trailing 60-month average for T60.xlsx
    print("\nWriting trailing 60-month averages to T60.xlsx ...")
    
    # Fill any missing values with the average of available factor values for each month
    # This replaces NaNs in a row (date) with the mean of that row, preserving the monthly context
    filled_returns = net_returns_df.apply(lambda row: row.fillna(row.mean()), axis=1)
    
    # EXTEND the dataframe by one month FIRST before calculating T60
    last_date = filled_returns.index[-1]
    next_month = last_date + pd.DateOffset(months=1)
    
    # Add the next month as a row with NaN values (we'll fill it in T60 calculation)
    filled_returns.loc[next_month] = np.nan
    
    # Now calculate trailing 60-month averages of the PREVIOUS 60 months (not including current month)
    # This means we shift the data first, then calculate the rolling mean
    trailing60 = filled_returns.shift(1).rolling(window=60, min_periods=1).mean()
    
    # Multiply values by 100 before writing to T60.xlsx as requested
    trailing60 = trailing60 * 100
    
    # Save trailing60 to Excel with basic formatting
    with pd.ExcelWriter('T60.xlsx', engine='xlsxwriter') as writer:
        trailing60.to_excel(writer, sheet_name='T60', index_label="Date")
        workbook = writer.book
        worksheet = writer.sheets['T60']
        date_fmt = workbook.add_format({'num_format': 'dd-mmm-yyyy'})
        worksheet.set_column(0, 0, 15, date_fmt)
        num_fmt = workbook.add_format({'num_format': '0.0000'})
        worksheet.set_column(1, len(trailing60.columns), 12, num_fmt)
    print("T60.xlsx saved with values multiplied by 100 and 'Date' in cell A1.")

    # Fill any remaining missing values in the main DataFrame using the row (monthly) mean
    net_returns_df = net_returns_df.apply(lambda row: row.fillna(row.mean()), axis=1)
    
    # Convert returns to percentages for better readability
    net_returns_df = net_returns_df * 100
    
    # Sort index (dates) in ascending order
    net_returns_df.sort_index(inplace=True)
    
    # Save to Excel with 'Date' in cell A1
    net_returns_df.to_excel(output_path, sheet_name='Monthly_Net_Returns', index_label="Date")

def run_portfolio_analysis(data_path: str, benchmark_path: str, output_path: str):
    """
    Run the portfolio analysis and save results.

    Args:
        data_path: Path to the input CSV data file
        benchmark_path: Path to the benchmark Excel file
        output_path: Path to save the output Excel file
    """
    # Load and prepare data
    print("Loading data...")
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data['date'] = data['date'].dt.to_period('M').dt.to_timestamp()  # Convert to first of month
    
    # Load benchmark returns
    benchmark_data = pd.read_excel(benchmark_path, sheet_name='Benchmarks', index_col=0)
    benchmark_data.index = pd.to_datetime(benchmark_data.index)
    benchmark_data.index = benchmark_data.index.to_period('M').to_timestamp()  # Convert to first of month
    benchmark_returns = benchmark_data['equal_weight']
    
    # Get features (all variables except '1MRet')
    features = sorted(list(set(data['variable'].unique()) - {'1MRet'}))
    
    # Run analysis
    print("Analyzing portfolios...")
    net_returns = analyze_portfolios(data, features, benchmark_returns)
    
    # Save results
    print("Saving results...")
    save_net_returns_to_excel(net_returns, output_path)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # File paths - using relative paths in current directory
    DATA_PATH = "Normalized_T2_MasterCSV.csv"
    BENCHMARK_PATH = "Portfolio_Data.xlsx"
    OUTPUT_PATH = "T2_Optimizer.xlsx"
    
    # Run analysis
    run_portfolio_analysis(DATA_PATH, BENCHMARK_PATH, OUTPUT_PATH)
