"""
# ===============================================================================
# STEP TWO POINT FIVE: CREATE BENCHMARK RETURNS
# ===============================================================================
# Version: 1.0
# Last Updated: 2024-07-01
# 
# DESCRIPTION:
# This script calculates and saves benchmark returns for stock market analysis.
# It processes monthly returns data for multiple countries and creates three benchmark portfolios:
# 1. Equal-weighted portfolio (average of all country returns)
# 2. Market-cap weighted portfolio (weighted by market capitalization)
# 3. US market returns (as a reference benchmark)
#
# INPUT FILES:
# - T2 Master.xlsx
#   - Sheet '1MRet': Monthly returns for each country/market
#   - Sheet 'Mcap Weights': Market capitalization weights for each country/market
#
# OUTPUT FILES:
# - Portfolio_Data.xlsx
#   - Sheet 'Returns': Processed returns data
#   - Sheet 'Weights': Processed market cap weights
#   - Sheet 'Benchmarks': Calculated benchmark returns (equal-weight, market-cap weight, US market)
#
# MISSING DATA HANDLING:
# - If the last row contains NaN values (most recent date with incomplete data), 
#   it's handled separately and NaN values are preserved in the output
# - If weights data has fewer rows than returns data, zeros are added for missing rows
# - If weights data has more rows than returns data, extra rows are truncated
# - Only common assets between returns and weights are used in calculations
# - For any missing country data, the program will use available data only
#
# USAGE:
# Run this script directly with Python:
#   python "Step Two Point Five Create Benchmark Rets.py"
# ===============================================================================
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from pathlib import Path
import sys

def read_data(file_path: str = 'T2 Master.xlsx', return_sheet: str = '1MRet') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read monthly returns and market cap weights from Excel file.
    
    Args:
        file_path: Path to the Excel file
        return_sheet: Name of the sheet containing returns (e.g. '1MRet', '12MRet')
        
    Returns:
        Tuple of (returns DataFrame, market cap weights DataFrame)
    """
    print(f"Reading data from {file_path}...")
    
    try:
        # Read returns from specified sheet
        returns = pd.read_excel(file_path, sheet_name=return_sheet, index_col=0)
        
        # Check if the last row has NaN values (date with no returns)
        has_empty_last_row = returns.iloc[-1].isna().all()
        print(f"Last row has no data: {has_empty_last_row}")
        
        # Convert index to datetime
        returns.index = pd.to_datetime(returns.index)
        
        print(f"\n{return_sheet} data shape:", returns.shape)
        print("Returns date range:", returns.index[0], "to", returns.index[-1])
        print("\nReturn columns:", returns.columns.tolist()[:5], "... (showing first 5)")
        
        # Read market cap weights from 'Mcap Weights' sheet
        # But ignore its date index and use the returns index instead
        weights_raw = pd.read_excel(file_path, sheet_name='Mcap Weights', index_col=0)
        print("\nWeights data shape (raw):", weights_raw.shape)
        
        # Create a new weights dataframe with the same index as returns
        # This assumes that the rows in both sheets are in the same order
        if len(weights_raw) != len(returns):
            print(f"WARNING: Number of rows in weights ({len(weights_raw)}) doesn't match returns ({len(returns)})")
            # If weights has fewer rows, we'll pad it with zeros for the missing rows
            if len(weights_raw) < len(returns):
                missing_rows = len(returns) - len(weights_raw)
                padding = pd.DataFrame(0, index=returns.index[-missing_rows:], columns=weights_raw.columns)
                weights_data = weights_raw.values
                weights = pd.DataFrame(weights_data, index=returns.index[:-missing_rows], columns=weights_raw.columns)
                weights = pd.concat([weights, padding])
                print(f"Added {missing_rows} rows of zeros to weights")
            else:
                # If weights has more rows, we'll use only the first rows that match returns
                weights_data = weights_raw.iloc[:len(returns)].values
                weights = pd.DataFrame(weights_data, index=returns.index, columns=weights_raw.columns)
                print(f"Truncated weights to match returns length")
        else:
            weights_data = weights_raw.values
            weights = pd.DataFrame(weights_data, index=returns.index, columns=weights_raw.columns)
        
        print("\nWeights data shape (aligned):", weights.shape)
        print("Weights date range:", weights.index[0], "to", weights.index[-1])
        print("\nWeight columns:", weights.columns.tolist()[:5], "... (showing first 5)")
        
        # Ensure both dataframes have the same columns
        common_assets = returns.columns.intersection(weights.columns)
        
        if len(common_assets) == 0:
            print("ERROR: No common assets between returns and weights!")
            sys.exit(1)
        
        returns = returns.loc[:, common_assets]
        weights = weights.loc[:, common_assets]
        
        print("\nAfter alignment:")
        print("Date range:", returns.index[0], "to", returns.index[-1])
        print("Common assets:", len(common_assets))
        print("Common columns:", common_assets.tolist()[:5], "... (showing first 5)")
        print("Total rows in returns:", len(returns))
        print("Total rows in weights:", len(weights))
        
        return returns, weights
    except Exception as e:
        print(f"ERROR in read_data: {str(e)}")
        raise

def prepare_benchmark_data(returns: pd.DataFrame, weights: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate benchmark returns including equal-weight, market-cap weight, and US market.
    
    Args:
        returns: DataFrame of asset returns
        weights: DataFrame of market cap weights
        
    Returns:
        Dict containing benchmark return series:
        - equal_weight: Equal-weighted portfolio returns
        - mcap_weight: Market-cap weighted portfolio returns
        - us_market: US market (SPX) returns
    """
    print("\nCalculating benchmark returns...")
    
    try:
        # Handle NaN values in the last row
        has_nan_last_row = returns.iloc[-1].isna().all()
        
        # Calculate on all data except possibly the last row with NaNs
        calc_returns = returns[:-1] if has_nan_last_row else returns
        calc_weights = weights[:-1] if has_nan_last_row else weights
        
        # Calculate equal-weight returns
        equal_weight = calc_returns.mean(axis=1)
        print("Equal-weight portfolio calculated")
        
        # Calculate market-cap weighted returns
        mcap_weight = (calc_returns * calc_weights).sum(axis=1)
        print("Market-cap weighted portfolio calculated")
        
        # Check if 'U.S.' is in the columns
        if 'U.S.' not in calc_returns.columns:
            print(f"WARNING: 'U.S.' not found in returns columns. Available columns: {calc_returns.columns.tolist()[:5]}...")
            # Use S&P 500 or another US market proxy if available
            if 'SPX' in calc_returns.columns:
                us_market = calc_returns['SPX']
                print("Using 'SPX' as US market proxy")
            elif 'S&P 500' in calc_returns.columns:
                us_market = calc_returns['S&P 500']
                print("Using 'S&P 500' as US market proxy")
            else:
                # Use the first column as a fallback
                us_market = calc_returns.iloc[:, 0]
                print(f"Using '{calc_returns.columns[0]}' as US market proxy (fallback)")
        else:
            us_market = calc_returns['U.S.']
            print("US market returns extracted (U.S.)")
        
        # Create the benchmarks dictionary
        benchmarks = {
            'equal_weight': equal_weight,
            'mcap_weight': mcap_weight,
            'us_market': us_market
        }
        
        # If there was a NaN row at the end, add NaN values for that date to all benchmarks
        if has_nan_last_row:
            last_date = returns.index[-1]
            for key in benchmarks:
                benchmarks[key] = pd.concat([benchmarks[key], pd.Series([np.nan], index=[last_date])])
        
        # Print summary statistics (excluding the NaN row)
        print("\nBenchmark Statistics:")
        for name, series in benchmarks.items():
            # Calculate stats excluding NaN values
            clean_series = series.dropna()
            print(f"\n{name}:")
            print(f"Mean monthly return: {clean_series.mean():.4%}")
            print(f"Monthly volatility: {clean_series.std():.4%}")
            print(f"Annualized return: {(1 + clean_series.mean())**12 - 1:.4%}")
            print(f"Annualized volatility: {clean_series.std() * np.sqrt(12):.4%}")
        
        return benchmarks
    except Exception as e:
        print(f"ERROR in prepare_benchmark_data: {str(e)}")
        raise

def save_data(returns: pd.DataFrame, 
             weights: pd.DataFrame, 
             benchmarks: Dict[str, pd.Series],
             output_file: str = 'Portfolio_Data.xlsx') -> None:
    """
    Save the processed data to a new Excel file.
    
    Args:
        returns: DataFrame of asset returns
        weights: DataFrame of market cap weights
        benchmarks: Dict of benchmark returns
        output_file: Name of the output Excel file
    """
    print(f"\nSaving data to {output_file}...")
    
    try:
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            # Save all returns
            returns.to_excel(writer, sheet_name='Returns')
            
            # Save all weights
            weights.to_excel(writer, sheet_name='Weights')
            
            # Save benchmarks
            benchmark_df = pd.DataFrame(benchmarks)
            benchmark_df.to_excel(writer, sheet_name='Benchmarks')
            
            # Apply proper date formatting to sheets with date indices
            workbook = writer.book
            date_format = workbook.add_format({'num_format': 'dd-mmm-yyyy'})
            
            # Format date columns in all sheets (they all have date indices)
            for sheet_name in ['Returns', 'Weights', 'Benchmarks']:
                if sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    worksheet.set_column(0, 0, 12, date_format)  # Format index column (dates)
        
        print("Data saved successfully!")
        print(f"Date range in output file: {returns.index[0]} to {returns.index[-1]}")
        print(f"Total rows in output file: {len(returns)}")
    except Exception as e:
        print(f"ERROR in save_data: {str(e)}")
        raise

def main():
    """
    Main function that orchestrates the entire process:
    1. Read data from input Excel file
    2. Calculate benchmark returns
    3. Save processed data to output Excel file
    """
    try:
        print("Starting benchmark returns calculation...")
        
        # Define absolute paths
        input_file = 'T2 Master.xlsx'
        output_file = 'Portfolio_Data.xlsx'
         
        # Create output directory if it doesn't exist
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Process data
        returns, weights = read_data(input_file)
        benchmarks = prepare_benchmark_data(returns, weights)
        save_data(returns, weights, benchmarks, output_file)
        
        print("Script completed successfully!")
    except Exception as e:
        print(f"ERROR in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 