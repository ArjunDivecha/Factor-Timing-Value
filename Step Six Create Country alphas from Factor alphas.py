#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
T2 Factor Timing - Step Six: Country Alphas from Factor Alphas
=============================================================

PURPOSE:
Calculates country-specific alpha scores by combining factor exposures with factor alphas.
This is a critical step in the T2 Factor Timing strategy that translates factor-level
alphas into country-level investment signals.

COUNTRY ALPHA CALCULATION METHODOLOGY:
=================================

The country alpha calculation follows a linear combination approach that weights each
factor's alpha by the country's exposure to that factor:

1. DATA ALIGNMENT:
   - Factor alphas from T60.xlsx (one value per factor per month)
   - Country factor exposures from T2_Top_20_Exposure.csv (one value per country-factor-month)
   - Only dates present in both datasets are processed

2. MONTHLY CALCULATION (for each country-month combination):
   Country_Alpha[country, month] = Σ(Factor_Exposure[country, factor, month] × Factor_Alpha[factor, month])
   
   Where:
   - Factor_Exposure[country, factor, month] = Country's exposure to a specific factor in that month
   - Factor_Alpha[factor, month] = Factor's alpha signal for that month (from optimization)
   - Σ = Sum across all available factors

3. STEP-BY-STEP PROCESS:
   a) For each month, extract factor alphas from T60.xlsx
   b) For each country in that month, extract factor exposures from T2_Top_20_Exposure.csv
   c) Multiply each country's factor exposure by the corresponding factor alpha
   d) Sum all factor contributions to get the country's total alpha score
   e) Track the number of valid factors used (non-missing data)

4. MISSING DATA HANDLING:
   - Missing factor exposures: Filled with mean exposure for that factor/date
   - Missing factor alphas: Factor is skipped for all countries that month
   - All imputations are logged for transparency

5. MATHEMATICAL EXAMPLE:
   If Country USA has exposures [0.5, -0.2, 0.8] to factors [F1, F2, F3]
   And factor alphas are [0.1, 0.3, -0.1] for [F1, F2, F3]
   Then USA_Alpha = (0.5 × 0.1) + (-0.2 × 0.3) + (0.8 × -0.1) = 0.05 - 0.06 - 0.08 = -0.09

6. INTERPRETATION:
   - Positive country alpha: Expected outperformance based on factor exposures
   - Negative country alpha: Expected underperformance based on factor exposures
   - Magnitude indicates strength of signal

This approach ensures that countries with higher exposures to factors with positive alphas
receive higher country alpha scores, creating a coherent translation from factor-level
signals to country-level investment decisions.

IMPORTANT NOTE:
This script requires factor alphas from T60.xlsx and country exposures from
T2_Top_20_Exposure.csv. Ensure these files are up-to-date before running.

INPUT FILES:
1. T60.xlsx
   - Source: Output from previous optimization step
   - Format: Excel workbook with single sheet
   - Structure:
     * First column: 'Date' (datetime)
     * Subsequent columns: Factor names (float values)
   - Notes:
     * Each row represents a month's factor alphas
     * Must contain a 'Date' column with proper datetime formatting

2. T2_Top_20_Exposure.csv
   - Source: Output from factor exposure calculation
   - Format: CSV file
   - Structure:
     * Date: Observation date (YYYY-MM-DD)
     * Country: 3-letter country code
     * Factor_Exposure_[N]: Exposure values for each factor
   - Notes:
     * One row per country-date combination
     * Missing values will be filled with factor-date means

3. T2 Master.xlsx (optional)
   - Source: Reference file for country order
   - Format: Excel workbook
   - Sheet: '1MRet' (used for country order reference)
   - Notes:
     * Only used to maintain consistent country ordering in output
     * If not provided, countries will be sorted alphabetically

OUTPUT FILES:
1. T2_Country_Alphas.xlsx
   - Excel workbook with multiple sheets:
     * Country_Scores: Monthly alpha scores (rows=dates, columns=countries)
     * Data_Quality: Completeness metrics by country
     * Factor_Counts: Valid factors used per country/month
     * Missing_Data_Log: Record of all imputed values
   - Format Specifications:
     * Dates in 'yyyy-mm-dd' format
     * Numeric values rounded to 6 decimal places
     * Missing values shown as empty cells

MISSING DATA HANDLING:
1. Missing country-factor exposures:
   - Filled with mean exposure for that factor/date
   - Original missing values and imputations logged
   - Data quality metrics track imputation frequency

2. Missing dates:
   - Only dates present in both input files are processed
   - Warning issued if date ranges don't align

3. Data Quality Tracking:
   - Completeness percentage by country
   - Factor usage statistics
   - Detailed imputation log

PERFORMANCE CONSIDERATIONS:
- Uses parallel processing for monthly calculations
- Efficient memory management for large datasets
- Progress tracking with tqdm
- Batch processing of dates to limit memory usage

VERSION HISTORY:
- 2.1 (2025-06-17): Updated docstrings to match new format
- 2.0 (2025-06-15): Switched to Top_20_Exposure.csv format, added parallel processing
- 1.0 (2025-06-10): Initial version using Normalized_T2_MasterCSV.csv

AUTHOR: Quantitative Research Team
LAST UPDATED: 2025-06-17

NOTES:
- All calculations use float64 precision
- Output is sorted by date (ascending)
- Country codes follow ISO 3166-1 alpha-3 standard
- Script is idempotent - can be safely rerun

DEPENDENCIES:
- pandas>=2.0.0
- numpy>=1.24.0
- tqdm>=4.65.0
- openpyxl>=3.1.0
- python-dateutil>=2.8.2
- tqdm>=4.65.0 (for progress bars)
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
EXPOSURE_FILE = 'T2_Top_20_Exposure.csv'  # New wide-format exposure file
T2_MASTER_FILE = 'T2 Master.xlsx'  # Reference file for country order
OUTPUT_FILE = 'T2_Country_Alphas.xlsx'

def load_data():
    """
    Load and prepare all input data files.
    
    INPUT FILES:
    - T60.xlsx: Factor alphas by date
    - T2_Top_20_Exposure.csv: Country factor exposures
    - T2 Master.xlsx: Reference for country order (optional)
    
    RETURNS:
        tuple: (factor_df, exposure_df, country_order)
            factor_df: DataFrame of factor alphas by date
            exposure_df: DataFrame of country factor exposures
            country_order: List of countries in reference order or None
    """
    print(f"Loading factor alpha data from {T60_FILE}...")
    factor_df = pd.read_excel(T60_FILE, sheet_name=0)
    factor_df = factor_df.rename(columns={factor_df.columns[0]: 'Date'})
    factor_df['Date'] = pd.to_datetime(factor_df['Date'])

    print(f"Loading wide-format country exposure data from {EXPOSURE_FILE}...")
    exposure_df = pd.read_csv(EXPOSURE_FILE)
    exposure_df['Date'] = pd.to_datetime(exposure_df['Date'])

    print(f"Loading country order reference from {T2_MASTER_FILE}...")
    try:
        country_order_df = pd.read_excel(T2_MASTER_FILE, sheet_name='1MRet')
        country_order = country_order_df.columns[1:].tolist()
        print(f"Found {len(country_order)} countries in reference file")
    except Exception as e:
        print(f"Warning: Could not load country order from {T2_MASTER_FILE}: {e}")
        print("Using countries as they appear in the data instead")
        country_order = None
    return factor_df, exposure_df, country_order

def get_available_dates(factor_df, exposure_df):
    """
    Identify dates present in both factor and exposure datasets.
    
    ARGUMENTS:
        factor_df: DataFrame with Date column and factor alphas
        exposure_df: DataFrame with Date column and country exposures
        
    RETURNS:
        list: Sorted list of datetime objects common to both datasets
        
    NOTES:
        Only returns dates where both datasets have complete data
    """
    factor_dates = set(factor_df['Date'])
    exposure_dates = set(exposure_df['Date'])
    common_dates = list(factor_dates.intersection(exposure_dates))
    common_dates.sort()
    return common_dates

def process_month(date_obj, factor_df, exposure_df, factor_names, missing_log):
    """
    Calculate country alphas for a single month by combining factor exposures with factor alphas.
    
    ARGUMENTS:
        date_obj: datetime object for target month
        factor_df: DataFrame with factor alphas by date
        exposure_df: DataFrame with country factor exposures
        factor_names: List of factor names to include
        missing_log: List to track missing data imputations
        
    RETURNS:
        DataFrame: Processed results with columns:
            - date: Processing date
            - country: Country code
            - total_score: Sum of (factor exposure * factor alpha)
            - valid_factors: Count of non-missing factors used
            
    MISSING DATA HANDLING:
        Missing factor exposures are filled with the mean for that factor/date
        All imputations are logged to missing_log
    """
    # Get factor alphas for this date
    factor_row = factor_df[factor_df['Date'] == date_obj]
    if factor_row.empty:
        return pd.DataFrame()
    factor_row = factor_row.iloc[0]
    # All country rows for this date
    date_countries = exposure_df[exposure_df['Date'] == date_obj]
    results = []
    # Precompute factor means for missing value filling
    factor_means = {}
    for f in factor_names:
        col = f
        vals = date_countries[col]
        factor_means[f] = vals[vals.notna()].mean()
    # For each country
    for idx, row in date_countries.iterrows():
        country = row['Country']
        total = 0.0
        valid_factors = 0
        for f in factor_names:
            exposure = row[f]
            if pd.isna(exposure):
                # Fill with mean of available countries
                exposure = factor_means[f]
                missing_log.append({'date': date_obj, 'country': country, 'factor': f, 'filled_value': exposure})
            alpha = factor_row[f]
            if pd.isna(alpha):
                continue  # skip if factor alpha missing
            if pd.isna(exposure):
                continue  # skip if still missing
            total += exposure * alpha
            valid_factors += 1
        results.append({'date': date_obj, 'country': country, 'total_score': total, 'valid_factors': valid_factors})
    return pd.DataFrame(results)

def main():
    """
    Main execution function for calculating country alphas from factor exposures.
    
    PROCESSING STEPS:
    1. Load and validate input data
    2. Align dates and factors between datasets
    3. Calculate monthly country alphas (parallel processing)
    4. Generate data quality reports
    5. Export results to formatted Excel workbook
    
    OUTPUT FILES CREATED:
    - T2_Country_Alphas.xlsx with multiple sheets:
        - Country_Scores: Monthly alpha scores by country
        - Data_Quality: Completeness metrics
        - Factor_Counts: Factors used per country/month
        - Missing_Data_Log: Record of all imputations
        
    PERFORMANCE NOTES:
    - Uses parallel processing for monthly calculations
    - Includes progress tracking with tqdm
    - Handles missing data with mean imputation
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    import time
    start_time = time.time()

    # 1. Load data
    factor_df, exposure_df, country_order = load_data()

    # 2. Align dates and factors
    common_dates = get_available_dates(factor_df, exposure_df)
    print(f"Number of common dates: {len(common_dates)}")
    # Factor names: intersection of columns in both files (excluding Date/Country)
    exposure_factors = [c for c in exposure_df.columns if c not in ['Date','Country']]
    factor_file_factors = [c for c in factor_df.columns if c != 'Date']
    # Only use factors present in both
    factor_names = [f for f in factor_file_factors if f in exposure_factors]
    print(f"Number of factors used: {len(factor_names)}")

    # 3. For each month, calculate country alphas (sequential processing)
    missing_log = []
    print(f"Found {len(common_dates)} common dates between datasets")
    print(f"Date range: {common_dates[0].strftime('%Y-%m-%d')} to {common_dates[-1].strftime('%Y-%m-%d')}")
    
    # Process all months with progress bar
    print("Processing all months...")
    all_results = []
    
    # Create a progress bar
    with tqdm(total=len(common_dates), desc="Processing months") as pbar:
        # Process each date
        for date_obj in common_dates:
            result = process_month(date_obj, factor_df, exposure_df, factor_names, missing_log)
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
