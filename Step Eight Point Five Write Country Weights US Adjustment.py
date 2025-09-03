"""
US Weight Adjustment for Country Portfolio
========================================

This program adjusts the US weight in the country portfolio according to the formula:
New US Weight = 25% + (50% * US Alpha)

It then rescales all other countries proportionally to maintain a total weight of 100%.

INPUT FILES
==========
1. "T2_Final_Country_Weights.xlsx" - Original country weights
2. "T2_Country_Alphas.xlsx" - Country alphas for US weight adjustment

OUTPUT FILES
===========
Overwrites with adjusted weights1. "T2_Final_Country_Weights.xlsx" - 
2. "T2_Country_Final.xlsx" - Overwrites with adjusted final weights

Version: 1.0
Last Updated: 2025-07-29
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

# ===============================
# INPUT FILE DEFINITIONS
# ===============================

original_weights_file = "T2_Final_Country_Weights.xlsx"
alphas_file = "T2_Country_Alphas.xlsx"

print("Loading data...")
# Load original country weights
original_weights_df = pd.read_excel(original_weights_file, sheet_name='All Periods')
original_weights_df = original_weights_df.set_index('Unnamed: 0')

# Load alpha data for US weight adjustment
alphas_df = pd.read_excel(alphas_file)
alphas_df['date'] = pd.to_datetime(alphas_df['date'])

# ===============================
# US WEIGHT ADJUSTMENT PROCESS
# ===============================

print("\nAdjusting US weights...")
# Create a copy for adjusted weights
adjusted_weights = original_weights_df.copy()

# Get all dates
all_dates = adjusted_weights.index.tolist()

# Process each date to adjust US weight
for date in tqdm(all_dates):
    # Convert to datetime if it's not already
    if not isinstance(date, pd.Timestamp):
        date = pd.to_datetime(date)
    
    # Get the alpha value for US for this date
    alpha_row = alphas_df[alphas_df['date'] == date]
    
    if not alpha_row.empty:
        us_alpha = alpha_row['U.S.'].iloc[0]
        
        # Calculate new US weight: 25% + (50% * US Alpha)
        new_us_weight = 0.25 + (0.5 * us_alpha)
        
        # Ensure new US weight is within reasonable bounds [0, 1]
        new_us_weight = max(0.0, min(1.0, new_us_weight))
        
        # Get current US weight
        current_us_weight = adjusted_weights.loc[date, 'U.S.']
        
        # Print debugging information
        print(f"Date: {date.strftime('%Y-%m-%d')}, US Alpha: {us_alpha:.4f}, Original US Weight: {current_us_weight:.6f}, New US Weight: {new_us_weight:.6f}")
        
        # Get sum of all other countries' weights
        other_countries_sum = adjusted_weights.loc[date].sum() - current_us_weight
        
        # If other countries have positive weights, rescale them
        if other_countries_sum > 0:
            # Calculate scaling factor for other countries
            scaling_factor = (1 - new_us_weight) / other_countries_sum
            
            # Apply new US weight
            adjusted_weights.loc[date, 'U.S.'] = new_us_weight
            
            # Rescale all other countries
            for country in adjusted_weights.columns:
                if country != 'U.S.':
                    adjusted_weights.loc[date, country] *= scaling_factor
        else:
            # If other countries have no weight, assign all to US
            adjusted_weights.loc[date, 'U.S.'] = 1.0
            for country in adjusted_weights.columns:
                if country != 'U.S.':
                    adjusted_weights.loc[date, country] = 0.0
    else:
        print(f"Warning: No alpha data found for date {date}")

# ===============================
# VALIDATION AND ANALYSIS
# ===============================

# Verify weights sum to approximately 1 for each date
weight_sums = adjusted_weights.sum(axis=1)
print("\nAdjusted weight sum statistics:")
print(weight_sums.describe())

# ===============================
# RESULTS SAVING
# ===============================

print("\nSaving adjusted results...")
output_file = 'T2_Final_Country_Weights.xlsx'

# Reset index to match original format
adjusted_weights_reset = adjusted_weights.reset_index()

with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    # Sheet 1: Complete time series of country weights
    adjusted_weights_reset.to_excel(writer, sheet_name='All Periods', index=False)
    
    # Copy other sheets from original file
    try:
        original_xl = pd.ExcelFile(original_weights_file, engine='openpyxl')
        for sheet in original_xl.sheet_names:
            if sheet != 'All Periods':
                sheet_df = pd.read_excel(original_weights_file, sheet_name=sheet, engine='openpyxl')
                sheet_df.to_excel(writer, sheet_name=sheet, index=False)
    except Exception as e:
        print(f"Warning: Could not copy other sheets from original file: {e}")
        # Create minimal sheets if we can't copy them
        summary_stats = pd.DataFrame({
            'Mean Weight': adjusted_weights.mean(),
            'Std Dev': adjusted_weights.std(),
            'Min Weight': adjusted_weights.min(),
            'Max Weight': adjusted_weights.max(),
            'Latest Weight': adjusted_weights.iloc[-1]
        }).sort_values('Mean Weight', ascending=False)
        summary_stats.to_excel(writer, sheet_name='Summary Statistics')
        
        # Sheet 3: Latest weights with comparison to average
        latest_valid_date = adjusted_weights.dropna().index.max() if not adjusted_weights.dropna().empty else adjusted_weights.index[-1]
        latest_weights = adjusted_weights.loc[latest_valid_date]
        
        if not latest_weights.empty:
            latest_weights_df = pd.DataFrame({
                'Weight': latest_weights,
                'Average Weight': adjusted_weights.mean(),
                'vs_avg': latest_weights - adjusted_weights.mean(),
                'Days with Weight': (adjusted_weights > 0).sum()
            }).sort_values('Weight', ascending=False)
        else:
            # Fallback in case there are no dates with non-zero weights (unlikely)
            latest_weights_df = pd.DataFrame({
                'Weight': adjusted_weights.iloc[-1],
                'Average Weight': adjusted_weights.mean(),
                'vs_avg': adjusted_weights.iloc[-1] - adjusted_weights.mean(),
                'Days with Weight': (adjusted_weights > 0).sum()
            }).sort_values('Weight', ascending=False)
        
        latest_weights_df.to_excel(writer, sheet_name='Latest Weights')

print(f"\nAdjusted weights saved to {output_file}")

# ===============================
# UPDATE FINAL WEIGHTS FILE
# ===============================

print("\nUpdating final country weights file...")
# Load the existing final weights file
final_weights_df = pd.read_excel('T2_Country_Final.xlsx')

# Get the latest date with valid weights
latest_date = adjusted_weights.dropna().index.max() if not adjusted_weights.dropna().empty else adjusted_weights.index[-1]
latest_weights = adjusted_weights.loc[latest_date]

# Update weights in the final file
for idx, row in final_weights_df.iterrows():
    country = row['Country']
    if country == 'TOTAL':
        final_weights_df.loc[idx, 'Weight'] = 1.0
    elif country in latest_weights.index:
        final_weights_df.loc[idx, 'Weight'] = latest_weights[country]

# Update the TOTAL row
final_weights_df.loc[final_weights_df['Country'] == 'TOTAL', 'Weight'] = final_weights_df['Weight'].sum() - 1.0  # Subtract the TOTAL row itself

# Save the updated final weights file
final_output_file = 'T2_Country_Final.xlsx'
with pd.ExcelWriter(final_output_file, engine='xlsxwriter') as writer:
    final_weights_df.to_excel(writer, sheet_name='Country Weights', index=False)
    
    workbook = writer.book
    worksheet = writer.sheets['Country Weights']
    
    header_format = workbook.add_format({'bold': True, 'text_wrap': True,
                                         'valign': 'top', 'bg_color': '#D9D9D9', 'border': 1})
    pct_format = workbook.add_format({'num_format': '0.00%'})

    worksheet.set_column(0, 0, 15)  # Country col
    last_col = final_weights_df.shape[1] - 1
    worksheet.set_column(1, last_col, 12, pct_format)

    for col_num, value in enumerate(final_weights_df.columns.values):
        worksheet.write(0, col_num, value, header_format)

    # Format the TOTAL row
    total_row_index = len(final_weights_df)  # 1-based indexing for Excel
    bold_format = workbook.add_format({'bold': True})
    total_format = workbook.add_format({'bold': True, 'num_format': '0.00%'})
    worksheet.write(total_row_index, 0, 'TOTAL', bold_format)
    for col_idx in range(1, len(final_weights_df.columns)):
        worksheet.write(total_row_index, col_idx, final_weights_df.iloc[-1, col_idx], total_format)

print(f"Final adjusted weights saved to {final_output_file}")

print("\nProcess completed successfully!")
