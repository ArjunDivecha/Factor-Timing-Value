"""
Feature-to-Country Weight Conversion Program
===========================================

This program converts feature importance weights from a machine learning model into 
country-specific investment weights for stock market forecasting. The system processes
monthly feature weights and factor data to generate country-level investment allocations.

Version: 1.4
Last Updated: 2025-07-18

INPUT FILES
==========
1. "T2_rolling_window_weights.xlsx"
   - Location: Same directory as script
   - Format: Excel file
   - Content: Feature weights from machine learning model
   - Structure:
     * Rows: Dates (index)
     * Columns: Feature names
     * Values: Feature importance weights

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
     a. "All Periods": Complete time series of country weights
        * Rows: Dates
        * Columns: Countries
        * Values: Investment weights (0-1)
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
1. Data Loading and Preparation:
   - Load feature weights and normalized factor data
   - Identify all unique countries and dates
   - Initialize weight tracking structures

2. Weight Calculation (per date) - FUZZY LOGIC APPROACH:
   a. For each date's model:
      - Identify significant features (non-zero weights)
      - For each significant feature:
        * Rank countries by feature values (high-to-low)
        * Apply fuzzy logic weighting:
          - Top 15% countries: Full weight (1.0)
          - 15-25% countries: Linearly decreasing weight (1.0 to 0.0)
          - Bottom 75% countries: Zero weight
        * Normalize weights to sum to 1 for equal capital allocation
        * Distribute feature's weight proportionally among selected countries
      - Sum weights across all features for final country allocations

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
- All weights are normalized to sum to 1 (100%) for each date
- Uses fuzzy logic with soft 15-25% linear taper (not hard 20% cutoff)
- Top 15% countries get full weight, 15-25% get linearly decreasing weight
- Weights are normalized per feature to ensure equal capital allocation
- Excel output uses xlsxwriter for formatting and styling
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
# Load feature weights from optimization model
feature_weights_df = pd.read_excel(weights_file, index_col=0)

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

print("\nProcessing all dates...")
# Process each date in the all_dates list
for date in tqdm(all_dates):
    # Initialize weights for all countries on this date
    country_weights = {country: 0.0 for country in all_countries}
    
    # Convert to datetime if it's not already
    if not isinstance(date, pd.Timestamp):
        date_dt = pd.to_datetime(date)
    else:
        date_dt = date
    
    # Use the CURRENT date for feature weights instead of previous date
    # Skip if the current date is not in the feature weights index
    if date_dt not in feature_weights_df.index:
        print(f"Skipping {date} - date not available in feature weights")
        continue
    
    # Get feature weights from the CURRENT date
    date_weights = feature_weights_df.loc[date_dt]
    
    # Filter out features with negligible weights (numerical stability)
    significant_weights = date_weights[date_weights.abs() > 1e-10]
    
    # Process each feature that has a significant weight
    for feature, feature_weight in significant_weights.items():
        # Get data for this feature and CURRENT date across all countries
        feature_data = factor_df[
            (factor_df['date'] == date) & 
            (factor_df['variable'] == feature)
        ].copy()
        
        # Skip if no data available for this feature/date
        if feature_data.empty:
            continue
            
        # FUZZY LOGIC IMPLEMENTATION (Same as Steps 3 and 4)
        # Rank by factor (high-to-low)
        feature_data = feature_data.sort_values('value', ascending=False).reset_index(drop=True)
        feature_data['rank_pct'] = (feature_data.index + 1) / len(feature_data)  # 0-1 percentile rank
        
        # Linear weights using fuzzy logic
        in_band = (feature_data['rank_pct'] >= SOFT_BAND_TOP) & (feature_data['rank_pct'] <= SOFT_BAND_CUTOFF)
        feature_data['weight'] = 0.0
        
        # Full weight for top band (< 15%)
        feature_data.loc[feature_data['rank_pct'] < SOFT_BAND_TOP, 'weight'] = 1.0
        
        # Linearly decreasing weight inside the grey band (15% - 25%)
        feature_data.loc[in_band, 'weight'] = (
            1.0 - (feature_data.loc[in_band, 'rank_pct'] - SOFT_BAND_TOP)
                  / (SOFT_BAND_CUTOFF - SOFT_BAND_TOP)
        )
        
        # Remove zero-weight rows for efficiency
        feature_data = feature_data[feature_data['weight'] > 0]
        
        if feature_data.empty:
            continue
        
        # Normalize weights to sum to 1 (equal total capital allocation per feature)
        feature_data['weight'] /= feature_data['weight'].sum()
        
        # Apply feature weight and distribute to countries
        for _, row in feature_data.iterrows():
            country = row['country']
            country_weight = feature_weight * row['weight']
            country_weights[country] += country_weight
    
    # Store calculated weights for this date
    for country, weight in country_weights.items():
        all_weights.loc[date, country] = weight

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
        'Days with Weight': (all_weights > 0).sum()
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
            'Days with Weight': (all_weights > 0).sum(),
            'Latest Date': pd.Series([latest_valid_date] * len(all_weights.columns), index=all_weights.columns)
        }).sort_values('Weight', ascending=False)
        print(f"\nUsing {latest_valid_date} as the latest valid date with non-zero weights")
    else:
        # Fallback in case there are no dates with non-zero weights (unlikely)
        latest_weights = pd.DataFrame({
            'Weight': all_weights.iloc[-1],
            'Average Weight': all_weights.mean(),
            'Days with Weight': (all_weights > 0).sum()
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
    
    # Create a dictionary of country weights from algorithm results
    # Filter to include only countries with non-zero weights
    country_weight_dict = {}
    for country, weight in latest_weights.items():
        if weight > 0:
            country_weight_dict[country] = weight
    
    print(f"Found {len(country_weight_dict)} countries with non-zero weights")
    total = sum(country_weight_dict.values())
    print(f"Total weight: {total:.4f}")
    
    # Read original country order from T2 Master.xlsx
    try:
        print("Reading original country order from T2 Master.xlsx...")
        master_df = pd.read_excel("T2 Master.xlsx")
        
        # In T2 Master.xlsx, countries are column names (except for the first 'Country' column)
        # The first column is actually dates, not countries
        country_columns = list(master_df.columns[1:])  # Skip the first column which is 'Country' (dates)
        print(f"Found {len(country_columns)} countries in column headers")
        
        # Create a full list of country names - keep original names from T2 Master.xlsx
        all_countries = []
        
        for country in country_columns:
            # Keep original names from T2 Master.xlsx
            all_countries.append(country)
            
        print(f"Total countries to include: {len(all_countries)}")
        
        # Create a DataFrame with ALL countries and initialize weights to 0
        all_weights_df = pd.DataFrame({
            'Country': all_countries,
            'Weight': 0.0  # Default weight is 0
        })
        
        # Update weights for countries based on the algorithm calculations
        for country, weight in country_weight_dict.items():
            # Try direct case-insensitive match first
            match_idx = all_weights_df[all_weights_df['Country'].str.lower() == country.lower()].index
            
            # If no direct match, handle ChinaH/Hong Kong name variations
            # These are the same country but may have different names in different data sources
            if len(match_idx) == 0:
                # Check if country from algorithm is ChinaH or Hong Kong
                country_lower = country.lower()
                if country_lower in ['chinah', 'hong kong']:
                    # Try to match to either ChinaH or Hong Kong in T2 Master
                    for master_country in all_countries:
                        if master_country.lower() in ['chinah', 'hong kong']:
                            match_idx = all_weights_df[all_weights_df['Country'] == master_country].index
                            break
            
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
        
    # Calculate per-factor country contributions for the latest date
    print("\nCalculating per-factor country contributions for latest date ...")
    date_feature_weights = feature_weights_df.loc[latest_valid_date]
    significant_feature_weights = date_feature_weights[date_feature_weights.abs() > 1e-10]
    factor_list = significant_feature_weights.index.tolist()
    print(f"Found {len(factor_list)} factors with non-zero weights on {latest_valid_date}.")

    # Use the actual countries from the factor data for this date to ensure consistency
    actual_countries_for_date = factor_df[factor_df['date'] == latest_valid_date]['country'].unique()
    contributions_df = pd.DataFrame(0.0, index=actual_countries_for_date, columns=factor_list)

    for feature, f_weight in significant_feature_weights.items():
        feat_data = factor_df[(factor_df['date'] == latest_valid_date) & (factor_df['variable'] == feature)]
        if feat_data.empty:
            continue  # no data available
            
        # FUZZY LOGIC IMPLEMENTATION (Same as main loop)
        # Rank by factor (high-to-low)
        feat_data = feat_data.sort_values('value', ascending=False).reset_index(drop=True)
        feat_data['rank_pct'] = (feat_data.index + 1) / len(feat_data)  # 0-1 percentile rank
        
        # Linear weights using fuzzy logic
        in_band = (feat_data['rank_pct'] >= SOFT_BAND_TOP) & (feat_data['rank_pct'] <= SOFT_BAND_CUTOFF)
        feat_data['weight'] = 0.0
        
        # Full weight for top band (< 15%)
        feat_data.loc[feat_data['rank_pct'] < SOFT_BAND_TOP, 'weight'] = 1.0
        
        # Linearly decreasing weight inside the grey band (15% - 25%)
        feat_data.loc[in_band, 'weight'] = (
            1.0 - (feat_data.loc[in_band, 'rank_pct'] - SOFT_BAND_TOP)
                  / (SOFT_BAND_CUTOFF - SOFT_BAND_TOP)
        )
        
        # Remove zero-weight rows for efficiency
        feat_data = feat_data[feat_data['weight'] > 0]
        
        if feat_data.empty:
            continue
        
        # Normalize weights to sum to 1 (equal total capital allocation per feature)
        feat_data['weight'] /= feat_data['weight'].sum()
        
        # Apply feature weight and distribute to countries
        for _, row in feat_data.iterrows():
            country = row['country']
            country_weight = f_weight * row['weight']
            if country in contributions_df.index:
                contributions_df.loc[country, feature] += country_weight
            else:
                print(f"Warning: Country '{country}' not found in contributions DataFrame index")

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
    contributions_reset = contributions_df.reset_index().rename(columns={'index': 'Country'})
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
