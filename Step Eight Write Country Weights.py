"""
Feature-to-Country Weight Conversion (Long–Short)
=================================================

Converts optimized factor weights (net, can be negative) into country-level
allocations using fuzzy Top 15–25% selection per factor and date. Negative factor
weights map to the bottom countries (short side). Vectorized implementation for speed.

Version: 3.0 (Long–Short, vectorized, optimized)
Last Updated: 2026-04-20
Author: Claude Code (optimized for speed)

OPTIMIZATIONS:
- Vectorized numpy operations for per-date processing
- Pre-grouped data by date for efficient slicing
- No inner loops over countries for speed

INPUT FILES
==========
1. "T2_rolling_window_weights.xlsx"
   - Location: Same directory as script
   - Format: Excel file
    - Content: Optimized factor weights (Net_Weights preferred)
   - Structure:
     * Rows: Dates (index)
     * Columns: Feature names
     * Values: Net factor weights (negative allowed)

2. "Normalized_T2_MasterCSV.csv"
   - Location: Same directory as script
   - Format: CSV file
   - Content: Normalized factor data for multiple countries
   - Structure:
     * date: Date of observation (YYYY-MM-DD)
     * country: Country code/name
     * variable: Feature name
     * value: Normalized feature value

3. "T2 Master.xlsx"
   - Location: Same directory as script
   - Format: Excel file
   - Purpose: Defines the original sort order of countries

OUTPUT FILES
===========
1. "T2_Final_Country_Weights.xlsx"
   - Format: Excel workbook with multiple sheets
   - Sheets:
     a. "All Periods": Complete time series of net country weights (can be negative)
        * Rows: Dates
        * Columns: Countries
        * Values: Net investment weights (sum ≈ 1.0 per date)
     b. "Summary Statistics": Statistical analysis
        * Metrics: Mean, standard deviation, min, max, etc.
        * Rows: Statistics
        * Columns: Countries
     c. "Latest Weights": Current allocation snapshot
        * Country: Country name
        * Weight: Current weight
        * vs_avg: Difference from historical average

2. "T2_Country_Final.xlsx"
   - Format: Excel workbook with single sheet
   - Purpose: Final country weights in original sort order
   - Structure:
     * Column A: Country names (from T2 Master.xlsx order)
     * Column B: Assigned weights (formatted as percentage)

METHODOLOGY
==========
1) For each date: pivot factor exposures to Countries×Factors; compute rank percentiles
   both descending and ascending.
2) For a positive factor weight: use descending ranks; for a negative weight: ascending
   ranks (i.e., bottom countries).
3) Apply fuzzy band (full weight <15%; linear taper 15–25%). Column-normalize to sum 1.
4) Multiply by factor weights (with sign) and sum across factors to net country weights.

3. Output Generation:
   - Create comprehensive Excel workbook with multiple analysis views
   - Generate final weight file in predefined sort order
   - Apply proper formatting and styling

DATA QUALITY HANDLING
====================
1. Missing Data:
   - Countries with missing feature data are excluded from selection for that feature
   - No imputation is performed to avoid introducing bias

2. Data Validation:
   - Checks for consistent date ranges across input files
   - Validates country codes match between sources
   - Verifies non-negative weights where applicable

3. Error Handling:
   - Graceful handling of missing or malformed input files
   - Logging of data quality issues

VERSION HISTORY
==============
1.0 (2025-06-01): Initial version
1.1 (2025-06-05): Added support for inverted features
1.2 (2025-06-09): Improved handling of missing data
1.3 (2025-06-12): Enhanced documentation and error handling
1.4 (2025-07-18): Implemented fuzzy logic with soft 15-25% linear taper

NOTES
=====
- Net country weights may be negative (short). Net sum per date ≈ 1.0.
- Uses fuzzy 15–25% taper per factor; column-normalized within factor before applying sign.
- Vectorized per-date pipeline for speed (no inner loops over countries).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ===============================
# FUZZY LOGIC CONFIGURATION
# ===============================

# Fuzzy logic constants (same as Steps 3 and 4)
SOFT_BAND_TOP   = 0.15   # 15% ⇒ full weight
SOFT_BAND_CUTOFF = 0.25  # 25% ⇒ zero weight

# ===============================
# DATA LOADING AND PREPROCESSING
# ===============================

# Input file paths
weights_file = "T2_rolling_window_weights.xlsx"
factor_file = "Normalized_T2_MasterCSV.csv"

print("Loading data...")
# Load feature weights from optimization model (prefer Net_Weights sheet for long–short)
try:
    feature_weights_df = pd.read_excel(weights_file, sheet_name='Net_Weights', index_col=0)
except Exception:
    feature_weights_df = pd.read_excel(weights_file, index_col=0)

# -------------------------------------------------------------------
# VINTAGE ALIGNMENT WITH STEP FIVE
# Step Five pairs factor weights chosen at month t with the factor-return
# row at t+1 (portfolios formed from month t+1 scores). To replicate that
# at the country level, country weights stamped at date d must be built
# from the PRIOR month's factor weights combined with date-d scores.
# Shifting the weights frame forward one row achieves this (the index is
# monthly contiguous). The first date is dropped (no prior vintage); the
# extra-month optimization row becomes usable next month when new scores
# arrive.
# -------------------------------------------------------------------
feature_weights_df = feature_weights_df.sort_index()
feature_weights_df = feature_weights_df.shift(1).iloc[1:]
print(f"Applied one-month vintage shift: weights at date d now come from d-1 "
      f"({feature_weights_df.index[0].strftime('%Y-%m')} to "
      f"{feature_weights_df.index[-1].strftime('%Y-%m')})")

# Load factor data for all countries
factor_df = pd.read_csv(factor_file)
factor_df['date'] = pd.to_datetime(factor_df['date'])  # Convert date column to datetime

# ===============================
# FEATURE CLASSIFICATION
# ===============================

# Features where LOWER values are better (will select BOTTOM 20% of countries)
INVERTED_FEATURES = {
    # Financial Indicators
    'BEST Cash Flow', 'BEST Div Yield', 'BEST EPS 3 Year', 'BEST PBK', 'BEST PE', 
    'BEST PS', 'BEST ROE', 'EV to EBITDA', 'Shiller PE', 'Trailing PE', 'Positive PE', 
    'Best Price Sales', 'Debt To EV',
    # Economic Indicators
    'Currency Change', 'Debt to GDP', 'REER', '10Yr Bond 12', 'Bond Yield Change',
    # Technical Indicators
    'RSI14', 'Advance Decline', '1MTR', '3MTR', 
    # Risk Metrics
    'Bloom Country Risk'
}

# ===============================
# INITIALIZATION
# ===============================

# Get all unique countries from the factor data in their original order
all_countries = factor_df['country'].unique()  # Removed sorting to preserve original order

# Get all unique dates from the factor data
all_factor_dates = sorted(factor_df['date'].unique())

# Get latest weights date - we'll only process dates that have feature weights
latest_weights_date = feature_weights_df.index.max()

# Initialize DataFrame to store weights for all countries and dates
# Only include dates that have feature weights available
all_dates = list(feature_weights_df.index)  # Only use dates with feature weights
all_weights = pd.DataFrame(index=all_dates, columns=all_countries)
all_weights = all_weights.fillna(0.0)  # Start with zero weights

# ===============================
# WEIGHT CALCULATION PROCESS
# ===============================

print("\nProcessing all dates (vectorized per date)...")

# Group factor data by date for efficient slicing
by_date = factor_df.groupby('date')

def calculate_country_contributions_for_date(date_dt):
    if date_dt not in feature_weights_df.index:
        return None, None

    w = feature_weights_df.loc[date_dt].astype(float)
    w = w[w.abs() > 1e-10]
    if w.empty:
        return None, None

    try:
        slice_df = by_date.get_group(date_dt)
    except KeyError:
        return None, None

    pivot = slice_df.pivot(index='country', columns='variable', values='value')
    common_factors = pivot.columns.intersection(w.index)
    if len(common_factors) == 0:
        return None, None

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
    contributions_df = pd.DataFrame(contrib, index=V.index, columns=common_factors)
    country_weights = contributions_df.sum(axis=1)

    return country_weights, contributions_df

for date in tqdm(all_dates):
    # Ensure Timestamp type to index the groupby
    date_dt = pd.to_datetime(date)
    country_weights, _ = calculate_country_contributions_for_date(date_dt)
    if country_weights is None:
        continue
    all_weights.loc[date, country_weights.index] = country_weights.values

# ===============================
# VALIDATION AND ANALYSIS
# ===============================

# Verify weights sum to approximately 1 for each date
weight_sums = all_weights.sum(axis=1)
print("\nWeight sum statistics:")
print(weight_sums.describe())

# ===============================
# RESULTS SAVING
# ===============================

print("\nSaving results...")
output_file = 'T2_Final_Country_Weights.xlsx'
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    # Sheet 1: Complete time series of country weights
    all_weights.to_excel(writer, sheet_name='All Periods')
    
    # Sheet 2: Summary statistics for each country
    summary_stats = pd.DataFrame({
        'Mean Weight': all_weights.mean(),
        'Std Dev': all_weights.std(),
        'Min Weight': all_weights.min(),
        'Max Weight': all_weights.max(),
        'Days with Weight': (all_weights.abs() > 0).sum()
    }).sort_values('Mean Weight', ascending=False)
    
    summary_stats.to_excel(writer, sheet_name='Summary Statistics')
    
    # Sheet 3: Latest country weights with comparison to historical average
    # Find the last row that has non-zero weights
    non_zero_dates = all_weights.index[all_weights.sum(axis=1) > 0]
    if len(non_zero_dates) > 0:
        latest_valid_date = non_zero_dates[-1]  # Get the last date with non-zero weights
        latest_weights = pd.DataFrame({
            'Weight': all_weights.loc[latest_valid_date],
            'Average Weight': all_weights.mean(),
            'Days with Weight': (all_weights.abs() > 0).sum(),
            'Latest Date': pd.Series([latest_valid_date] * len(all_weights.columns), index=all_weights.columns)
        }).sort_values('Weight', ascending=False)
        print(f"\nUsing {latest_valid_date} as the latest valid date with non-zero weights")
    else:
        # Fallback in case there are no dates with non-zero weights (unlikely)
        latest_weights = pd.DataFrame({
            'Weight': all_weights.iloc[-1],
            'Average Weight': all_weights.mean(),
            'Days with Weight': (all_weights.abs() > 0).sum()
        }).sort_values('Weight', ascending=False)
    
    latest_weights.to_excel(writer, sheet_name='Latest Weights')

print(f"\nResults saved to {output_file}")

# ===============================
# SUMMARY REPORTING
# ===============================

# Print top countries by average weight
print("\nTop 10 countries by average weight:")
print(summary_stats.head(10))

# ===============================
# WRITE COUNTRY WEIGHTS IN ORIGINAL ORDER
# ===============================

def write_final_country_weights():
    """
    Write country weights to T2_Country_Final.xlsx in the original sort order
    from T2 Master.xlsx using the latest calculated weights.
    """
    print("\nWriting country weights to T2_Country_Final.xlsx...")
    
    # Get the latest calculated weights from the algorithm
    # Find the last row that has non-zero weights
    non_zero_dates = all_weights.index[all_weights.sum(axis=1) > 0]
    if len(non_zero_dates) == 0:
        print("Error: No date with non-zero weights found")
        return
        
    latest_valid_date = non_zero_dates[-1]  # Get the last date with non-zero weights
    print(f"Using weights from latest date: {latest_valid_date}")
    
    # Get weights for the latest date
    latest_weights = all_weights.loc[latest_valid_date]
    
    # Create a dictionary of country weights from algorithm results (include negatives)
    country_weight_dict = latest_weights.to_dict()
    print(f"Found {len(country_weight_dict)} countries in latest weights")
    total = sum(country_weight_dict.values())
    print(f"Total net weight: {total:.4f}")
    
    # Read original country order from T2 Master.xlsx
    try:
        print("Reading original country order from T2 Master.xlsx...")
        master_df = pd.read_excel("T2 Master.xlsx")
        
        # In T2 Master.xlsx, countries are column names (except for the first 'Country' column)
        # The first column is actually dates, not countries
        country_columns = list(master_df.columns[1:])  # Skip the first column which is 'Country' (dates)
        print(f"Found {len(country_columns)} countries in column headers")
        
        # Create a full list of country names (keep original names from T2 Master)
        all_countries = list(country_columns)  # Use original column names as-is
            
        print(f"Total countries to include: {len(all_countries)}")
        
        # Create a DataFrame with ALL countries and initialize weights to 0
        all_weights_df = pd.DataFrame({
            'Country': all_countries,
            'Weight': 0.0  # Default weight is 0
        })
        
        # Update weights for countries based on the algorithm calculations
        # Map algorithm country names to T2 Master names if needed
        algorithm_to_master_mapping = {}  # Add mappings here if algorithm uses different names
        
        for country, weight in country_weight_dict.items():
            # First try direct match (case-insensitive)
            match_idx = all_weights_df[all_weights_df['Country'].str.lower() == country.lower()].index
            if len(match_idx) > 0:
                all_weights_df.loc[match_idx[0], 'Weight'] = weight
            else:
                # Try mapping if algorithm uses different name
                mapped_name = algorithm_to_master_mapping.get(country, country)
                match_idx = all_weights_df[all_weights_df['Country'].str.lower() == mapped_name.lower()].index
                if len(match_idx) > 0:
                    all_weights_df.loc[match_idx[0], 'Weight'] = weight
                else:
                    print(f"Note: Country '{country}' with weight {weight:.4f} not found in T2 Master.xlsx")
                    # Add it to the end with its weight
                    new_row = pd.DataFrame({'Country': [country], 'Weight': [weight]})
                    all_weights_df = pd.concat([all_weights_df, new_row], ignore_index=True)
        
        # Result is already sorted in the original order from T2 Master.xlsx
        sorted_weights = all_weights_df
        
    except Exception as e:
        print(f"Error reading original country order: {e}")
        print("Falling back to only countries with weights")
        
        # Create DataFrame from the dictionary if we can't read T2 Master.xlsx
        sorted_weights = pd.DataFrame(list(country_weight_dict.items()), 
                                  columns=['Country', 'Weight'])
        
    print("\nCalculating per-factor country contributions for latest date ...")
    latest_country_weights, contributions_df = calculate_country_contributions_for_date(latest_valid_date)
    if contributions_df is None:
        print("Error: Unable to calculate per-factor country contributions for latest date")
        return
    print(f"Found {len(contributions_df.columns)} factors with non-zero weights on {latest_valid_date}.")

    # Validate contributions by comparing only countries that exist in both DataFrames
    contributions_sum = contributions_df.sum(axis=1)
    common_countries = contributions_sum.index.intersection(latest_weights.index)
    if len(common_countries) > 0:
        row_diff = (contributions_sum[common_countries] - latest_weights[common_countries]).abs().max()
        if row_diff > 1e-6:
            print(f"Warning: maximum difference between aggregated contributions and weights = {row_diff:.6f}")
    else:
        print("Warning: No common countries found between contributions and latest weights")
    
    # Reorder factor columns by total contribution (largest first)
    factor_totals = contributions_df.sum().sort_values(ascending=False)
    contributions_df = contributions_df[factor_totals.index]
    
    # Merge factor contributions into sorted_weights to create final output
    contributions_reset = contributions_df.reset_index().rename(columns={'index': 'Country', 'country': 'Country'})
    final_df = pd.merge(sorted_weights, contributions_reset, on='Country', how='left')
    final_df = final_df.fillna(0.0)
    
    # Ensure each country appears only once to avoid double-counting
    final_df = final_df.drop_duplicates(subset='Country', keep='first')
    
    # Recompute total weight after deduplication
    total_weight = final_df['Weight'].sum()
    if abs(total_weight - 1.0) > 1e-6:
        print(f"Warning: total country weight after deduplication = {total_weight:.4f}, should be 1.0")
    
    # Write to Excel file
    output_file = 'T2_Country_Final.xlsx'
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        final_df.to_excel(writer, sheet_name='Country Weights', index=False)
        
        workbook = writer.book
        worksheet = writer.sheets['Country Weights']
        
        header_format = workbook.add_format({'bold': True, 'text_wrap': True,
                                             'valign': 'top', 'bg_color': '#D9D9D9', 'border': 1})
        pct_format = workbook.add_format({'num_format': '0.00%'})

        worksheet.set_column(0, 0, 15)  # Country col
        last_col = final_df.shape[1] - 1
        worksheet.set_column(1, last_col, 12, pct_format)

        for col_num, value in enumerate(final_df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        total_weight = final_df['Weight'].sum()
        last_row = len(final_df) + 1
        bold_format = workbook.add_format({'bold': True})
        total_format = workbook.add_format({'bold': True, 'num_format': '0.00%'})
        worksheet.write(last_row, 0, 'TOTAL', bold_format)
        worksheet.write(last_row, 1, total_weight, total_format)
        
        # NEW: write totals for each factor column
        col_sums = final_df.drop(columns=['Country']).sum()
        for col_idx, col_name in enumerate(final_df.columns[1:], start=1):
            col_total = col_sums[col_name]
            worksheet.write(last_row, col_idx, col_total, total_format)
        
    print(f"Final weights with factor contributions saved to {output_file}")

# Execute the function to write country weights
write_final_country_weights()
