#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step Thirteen Create Country alphas from Factor Alphas.py

This script calculates country alpha scores by combining factor scores with country exposures to each factor.
For each country and month, it computes a total score based on its exposure to each factor multiplied by the factor's alpha.

INPUT FILES:
- T60.xlsx
  Contains factor alphas/scores for each month in the first sheet
  The first column contains dates without an explicit header
  
- Normalized_T2_MasterCSV.csv
  Contains country exposures to factors (how much each country is exposed to each factor)
  
- T2 Master.xlsx
  Reference file for country order in output

OUTPUT FILES:
- T2_Country_Alphas.xlsx
  Excel file with:
  - Country_Scores sheet: Total country scores for each month (dates as rows, countries as columns)
  - Data_Quality sheet: Report on missing data by country
  - Factor_Counts sheet: Number of factors used for each country/month

Processing Logic:
1. For each month, get factor scores from T60.xlsx
2. For each month, get country exposures from Normalized_T2_MasterCSV.csv
3. For each factor, multiply factor score by country exposure
4. Sum up scores across all factors for each country
5. Output results with countries in the same order as T2 Master.xlsx

Version: 1.0 (2025-06-10)
"""

import pandas as pd
import numpy as np
import time
import datetime
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# File paths
T60_FILE = 'T60.xlsx'
MASTER_CSV_FILE = 'Normalized_T2_MasterCSV.csv'
T2_MASTER_FILE = 'T2 Master.xlsx'  # Reference file for country order
OUTPUT_FILE = 'T2_Country_Alphas.xlsx'

def load_data():
    """
    Load the factor scores and country exposure data.
    
    Returns:
        tuple: (factor_df, country_df, country_order)
    """
    print(f"Loading factor data from {T60_FILE}...")
    # Read T60.xlsx without assuming 'Date' is in cell A1
    factor_df = pd.read_excel(T60_FILE, sheet_name=0)
    # Rename the first column to 'Date' for consistency
    factor_df = factor_df.rename(columns={factor_df.columns[0]: 'Date'})
    
    print(f"Loading country exposure data from {MASTER_CSV_FILE}...")
    country_df = pd.read_csv(MASTER_CSV_FILE)
    
    # Convert date columns to datetime
    factor_df['Date'] = pd.to_datetime(factor_df['Date'])
    country_df['date'] = pd.to_datetime(country_df['date'])
    
    # Get country order from T2 Master.xlsx
    print(f"Loading country order reference from {T2_MASTER_FILE}...")
    try:
        # Extract country order from the first sheet (1MRet)
        country_order_df = pd.read_excel(T2_MASTER_FILE, sheet_name='1MRet')
        # Get countries from columns (skip the first 'Country' column)
        country_order = country_order_df.columns[1:].tolist()
        print(f"Found {len(country_order)} countries in reference file")
    except Exception as e:
        print(f"Warning: Could not load country order from {T2_MASTER_FILE}: {e}")
        print("Using countries as they appear in the data instead")
        country_order = None
    
    return factor_df, country_df, country_order

def get_available_dates(factor_df, country_df):
    """
    Get dates that are available in both datasets.
    
    Args:
        factor_df (DataFrame): Factor scores dataframe
        country_df (DataFrame): Country exposure dataframe
        
    Returns:
        list: List of common dates as datetime objects
    """
    factor_dates = set(factor_df['Date'])
    country_dates = set(country_df['date'])
    
    common_dates = list(factor_dates.intersection(country_dates))
    common_dates.sort()
    
    return common_dates

def process_month(date_obj, factor_df, country_df):
    """
    Process a single month to calculate the sum of factor scores for each country.
    
    Args:
        date_obj (datetime): Date to process
        factor_df (DataFrame): Factor scores dataframe
        country_df (DataFrame): Country exposure dataframe
        
    Returns:
        DataFrame: DataFrame with countries as index and total score
    """
    # Filter factor scores for the current date
    factor_row = factor_df[factor_df['Date'] == date_obj].iloc[0].drop('Date')
    
    # Filter country exposures for the current date
    country_data = country_df[country_df['date'] == date_obj]
    
    # Create a pivot table of country exposures
    country_pivot = country_data.pivot_table(
        index='country', 
        columns='variable', 
        values='value'
    )
    
    # Get the list of countries
    countries = country_pivot.index.tolist()
    
    # Create a DataFrame to store the total country scores
    total_scores = pd.Series(0.0, index=countries)
    valid_factor_count = pd.Series(0, index=countries)
    
    # Process each factor
    for factor_name in factor_row.index:
        if factor_name in country_pivot.columns:
            # Get factor score
            factor_score = factor_row[factor_name]
            
            # Get country exposures to this factor
            exposures = country_pivot[factor_name].copy()
            
            # Skip factors with all NaN values
            if exposures.isna().all():
                continue
            
            # Handle missing exposure values by filling with mean
            missing_mask = exposures.isna()
            if missing_mask.any():
                mean_exposure = exposures.mean(skipna=True)
                exposures.fillna(mean_exposure, inplace=True)
            
            # Calculate country scores for this factor
            country_scores = exposures * factor_score
            
            # Add to the total scores
            total_scores += country_scores
            
            # Count valid factors for each country
            valid_factor_count += ~missing_mask
    
    # Create a DataFrame with the date and total scores for each country
    result = pd.DataFrame({'date': date_obj, 'total_score': total_scores})
    result['valid_factors'] = valid_factor_count
    
    # Include the date and reset the index to make country a column
    result = result.reset_index()
    # Rename the 'index' column to 'country' to ensure consistent column naming
    result = result.rename(columns={'index': 'country'})
    
    return result

def main():
    """
    Main function to execute the workflow.
    """
    start_time = time.time()
    
    # Load data
    factor_df, country_df, country_order = load_data()
    
    # Get common dates
    common_dates = get_available_dates(factor_df, country_df)
    print(f"Found {len(common_dates)} common dates between datasets")
    print(f"Date range: {common_dates[0].strftime('%Y-%m-%d')} to {common_dates[-1].strftime('%Y-%m-%d')}")
    
    # Determine the number of processes to use
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    # Process all months with progress bar
    print("Processing all months...")
    all_results = []
    
    # Create a progress bar
    with tqdm(total=len(common_dates)) as pbar:
        # Process each date
        for date_obj in common_dates:
            result = process_month(date_obj, factor_df, country_df)
            all_results.append(result)
            pbar.update(1)
    
    # Combine results from all months
    print("Combining results...")
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Debug column names
    print("\nVerifying column names:")
    print(combined_results.columns.tolist())
    
    # Create a pivot table with dates as rows and countries as columns
    print("\nCreating pivot table...")
    country_col = 'country'
    if 'country' not in combined_results.columns:
        # Check if there's a similar column
        for col in combined_results.columns:
            if 'country' in col.lower():
                country_col = col
                print(f"Using '{country_col}' instead of 'country'")
                break
    
    # Create the pivot table
    pivot_table = combined_results.pivot_table(
        index='date', 
        columns=country_col, 
        values='total_score'
    )
    
    # Reorder countries according to T2 Master.xlsx if country_order is available
    if country_order:
        print(f"Reordering columns to match reference country order")
        # Get intersection of available countries in pivot table and reference order
        available_countries = [c for c in country_order if c in pivot_table.columns]
        # Add any missing countries that might be in pivot_table but not in country_order
        for country in pivot_table.columns:
            if country not in available_countries:
                available_countries.append(country)
        # Reorder the pivot table columns
        pivot_table = pivot_table[available_countries]
    
    # Create a quality report for the pivot table
    missing_data = pivot_table.isna().sum()
    country_quality = pd.DataFrame({
        'country': missing_data.index,
        'missing_months': missing_data.values,
        'completeness_pct': 100 * (1 - missing_data.values / len(common_dates))
    }).sort_values('completeness_pct')
    
    # Create a factor count table using the same country column
    factor_count_pivot = combined_results.pivot_table(
        index='date', 
        columns=country_col, 
        values='valid_factors'
    )
    
    # Write results to Excel with proper date formatting
    print(f"Writing results to {OUTPUT_FILE}...")
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
        # Write main pivot table with country scores
        pivot_table.to_excel(writer, sheet_name='Country_Scores')
        
        # Write quality report
        country_quality.to_excel(writer, sheet_name='Data_Quality', index=False)
        
        # Write factor count pivot table
        factor_count_pivot.to_excel(writer, sheet_name='Factor_Counts')
        
        # Apply date formatting to all sheets
        workbook = writer.book
        date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
        
        # Format dates in Country_Scores sheet
        worksheet = writer.sheets['Country_Scores']
        worksheet.set_column(0, 0, 12, date_format)  # Format date column
        
        # Format dates in Factor_Counts sheet
        worksheet = writer.sheets['Factor_Counts']
        worksheet.set_column(0, 0, 12, date_format)  # Format date column
    
    # Calculate summary statistics
    execution_time = time.time() - start_time
    
    print("\nSummary:")
    print(f"- Date range: {common_dates[0].strftime('%Y-%m-%d')} to {common_dates[-1].strftime('%Y-%m-%d')}")
    print(f"- Number of months processed: {len(common_dates)}")
    print(f"- Number of countries: {len(pivot_table.columns)}")
    print(f"- Output file: {OUTPUT_FILE}")
    print(f"- Execution time: {execution_time:.2f} seconds")
    
    # Preview results
    print("\nPreview of country scores (first 5 months, first 5 countries):")
    preview_countries = pivot_table.columns[:5].tolist()
    print(pivot_table[preview_countries].head())
    
    print("\nData quality issues (countries with most missing data):")
    print(country_quality.head())

if __name__ == "__main__":
    main()
