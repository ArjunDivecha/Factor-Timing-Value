"""
=============================================================================
SCRIPT NAME: T2_FINALFINAL.py
=============================================================================

INPUT FILES:
- T2_Country_Alphas.xlsx
    - Sheet: 'Country_Scores'
    - Contains country alpha scores across dates. Each row is a date; columns are countries.
- T2_Optimized_Country_Weights.xlsx
    - Sheet: 'Latest_Weights'
    - Contains latest optimized country weights. Columns: 'Country', 'Weight' (or similar).

OUTPUT FILES:
- T2_FINAL_T60_VALUE.xlsx
    - Sheet: 'Latest_Country_Alpha_Weights' (columns: Country, Country Alpha, Country Weight)

VERSION: 1.1
LAST UPDATED: 2025-07-17
AUTHOR: Cascade

DESCRIPTION:
This script extracts the most recent country alpha scores from the last row of the 'Country_Scores' sheet in T2_Country_Alphas.xlsx and the latest weights from the 'Latest_Weights' sheet in T2_Optimized_Country_Weights.xlsx, merges them by country, and writes a single Excel sheet with columns: Country, Country Alpha, Country Weight.

=============================================================================
"""

import pandas as pd
import os

# --- File names (local only) ---
ALPHAS_FILE = 'T2_Country_Alphas.xlsx'
WEIGHTS_FILE = 'T2_Optimized_Country_Weights.xlsx'
OUTPUT_FILE = 'T2_FINAL_T60_VALUE.xlsx'

def main():
    # Check input files exist
    for f in [ALPHAS_FILE, WEIGHTS_FILE]:
        if not os.path.exists(f):
            print(f"Error: Required file '{f}' not found in current directory.")
            return

    # Read the last row of country alphas (latest scores)
    alphas_df = pd.read_excel(ALPHAS_FILE, sheet_name='Country_Scores', index_col=0)
    latest_scores = alphas_df.iloc[-1]
    latest_scores = latest_scores.reset_index()
    latest_scores.columns = ['Country', 'Country Alpha']

    # Read the latest weights
    weights_df = pd.read_excel(WEIGHTS_FILE, sheet_name='Latest_Weights')
    # Try to find the right column for weights
    weight_col = 'Weight' if 'Weight' in weights_df.columns else weights_df.columns[-1]
    weights_df = weights_df[['Country', weight_col]].copy()
    weights_df.columns = ['Country', 'Country Weight']

    # Merge on Country
    merged = pd.merge(latest_scores, weights_df, on='Country', how='inner')

    # Save to Excel (single sheet)
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
        merged.to_excel(writer, sheet_name='Latest_Country_Alpha_Weights', index=False)
        worksheet = writer.sheets['Latest_Country_Alpha_Weights']
        for idx, col in enumerate(merged.columns):
            max_len = max(merged[col].astype(str).map(len).max(), len(str(col)))
            worksheet.set_column(idx, idx, max_len + 2)
    print(f"Successfully created '{OUTPUT_FILE}' with columns: Country, Country Alpha, Country Weight.")


if __name__ == "__main__":
    main()
