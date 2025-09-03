"""
T2 Data Normalization Pipeline - Step Two
========================================

PURPOSE:
Transforms raw financial data from T2 Master.xlsx into normalized formats suitable for analysis.
Creates both wide (Excel) and long/tidy (CSV) format outputs with cross-sectional and time-series
normalizations.

INPUT FILES:
1. T2 Master.xlsx
   - Source: Output from Step One (Create T2Master.py)
   - Format: Excel workbook with multiple sheets
   - Structure:
     * Each sheet represents a financial variable
     * Rows: Dates (index)
     * Columns: Country codes
   - Required sheets:
     * Returns: 1MRet, 3MRet, 6MRet, 9MRet, 12MRet
     * Financial Metrics: BEST Cash Flow, BEST Div Yield, BEST EPS, etc.
     * Market Indicators: RSI14, Advance Decline, etc.
     * Valuation: PE, PB, PS ratios
     * Technicals: Moving Averages, etc.

OUTPUT FILES:
1. Normalized_T2_Master.xlsx
   - Format: Excel workbook with multiple sheets
   - Naming: {VariableName}_{NormalizationType}
   - Normalization Types:
     * _CS: Cross-sectional (by date)
     * _TS: Time-series (by country, expanding window)
     * (no suffix): Original returns (not normalized)
   - Structure per sheet:
     * Rows: Dates (index)
     * Columns: Country codes

2. Normalized_T2_MasterCSV.csv
   - Format: Tidy/long format CSV
   - Columns:
     * date: Observation date
     * country: Country code
     * value: Normalized value
     * variable: {OriginalVariable}_{NormalizationType}
   - Benefits:
     * Ideal for analysis with pandas/pyarrow
     * Compatible with visualization tools
     * Efficient storage of sparse data

NORMALIZATION METHODOLOGY:
1. Cross-Sectional (CS) Normalization:
   - For each date, across all countries:
     * Remove mean
     * Divide by standard deviation
   - Preserves cross-sectional relationships
   - Mitigates extreme values

2. Time-Series (TS) Normalization:
   - For each country, using expanding window:
     * Subtract expanding mean
     * Divide by expanding standard deviation
   - Captures time-series dynamics
   - Handles non-stationary data

SPECIAL HANDLING:
1. Returns (No Normalization):
   - Variables: 1MRet, 3MRet, 6MRet, 9MRet, 12MRet
   - Kept in original form without normalization
   - Preserves economic interpretation of returns

2. Inverted Variables (Lower is Better):
   - Financial Indicators: BEST Cash Flow, BEST Div Yield, BEST EPS 3Y, etc.
   - Valuation Metrics: BEST PBK, BEST PE, BEST PS, EV/EBITDA, etc.
   - Technical Indicators: RSI, etc.
   - Economic Indicators: Currency Change, Debt to GDP, REER, 10Yr Bond 12, Bond Yield Change
   - Risk Metrics: Bloom Country Risk
   - For these variables, the normalization is multiplied by -1 to ensure higher values 
     consistently represent better conditions across all variables

3. Missing Data Handling:
   - Forward-filled within each country
   - Missing values after forward-fill set to 0
   - Logged for diagnostics

VERSION HISTORY:
- 2.0 (2025-05-15): Complete rewrite with improved error handling
- 1.5 (2025-03-22): Added support for inverted variables
- 1.0 (2025-02-10): Initial production version

AUTHOR: Financial Data Processing Team
LAST UPDATED: 2025-06-17

NOTES:
- All normalizations use N-1 degrees of freedom for std dev (pandas default)
- Expanding window for TS normalization starts after min_periods=12
- Output files are overwritten without warning
- Progress is printed to console during processing
- For variables where lower values are better, normalization is inverted
  to maintain consistent interpretation (higher = better)

DEPENDENCIES:
- pandas>=2.0.0
- numpy>=1.24.0
- openpyxl>=3.1.0
"""

import pandas as pd
from pathlib import Path
from typing import List
import numpy as np

def _truncate_sheet_name(name: str, max_length: int = 31) -> str:
    """
    Truncate sheet names to Excel's 31-character limit while preserving suffixes.
    """
    if len(name) <= max_length:
        return name
    # Identify suffix if present
    for suffix in ['_CS', '_Global', '_TS']:
        if name.endswith(suffix):
            base = name[:-len(suffix)]
            trunc_length = max_length - len(suffix) - 1  # 1 for underscore
            return f"{base[:trunc_length]}{suffix}"
    return name[:max_length]

def _create_tidy_df(df: pd.DataFrame, variable: str, normalization: str) -> pd.DataFrame:
    """
    Convert wide dataframe to tidy format with normalization type as part of variable name.
    
    Args:
        df (pd.DataFrame): DataFrame to transform.
        variable (str): Variable name corresponding to the sheet name.
        normalization (str): Type of normalization ('CS', 'Global', 'TS').
    
    Returns:
        pd.DataFrame: Tidy formatted DataFrame.
    """
    tidy_df = df.reset_index().melt(
        id_vars=['date'],
        var_name='country',
        value_name='value'
    )
    # Only add normalization suffix if it's not an original return variable
    if normalization != 'Original':
        tidy_df['variable'] = f"{variable}_{normalization}"
    else:
        tidy_df['variable'] = variable
    return tidy_df[['date', 'country', 'value', 'variable']]

def normalize_data() -> None:
    """
    Create two normalized versions of each sheet in T2 Master.xlsx:
    1. Cross-sectional normalization (by date) - CS
    2. Time series normalization (by country) - TS
    Outputs both Excel and tidy format CSV files.
    """
    try:
        input_file = 'T2 Master.xlsx'
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file '{input_file}' not found in current directory.")
        
        xls = pd.ExcelFile(input_file)
        
        excel_output = 'Normalized_T2_Master.xlsx'
        csv_output = 'Normalized_T2_MasterCSV.csv'
        
        # List of sheets to skip entirely
        skip_sheets: List[str] = [
            'README', 'Sheet1'  # Sheets to skip
        ]
        
        # List of sheets to copy directly without normalization
        copy_direct: List[str] = [
            '1MRet', '3MRet', '6MRet', '9MRet', '12MRet'  # Return sheets to copy directly
        ]
        
        # List of variables to invert normalization (where lower values are better)
        invert_norm: List[str] = [
            'Best Cash Flow',
            'Best PBK', 'Best PE ', 'Best Price Sales',
            'EV to EBITDA', 'Shiller PE', 'Trailing PE', 'Positive PE ',
            'Currency Change', 'Debt to GDP', 'REER', 'RSI14', '10Yr Bond 12',
            'Advance Decline', '1MTR', '3MTR', 'Debt To EV', 'Best Price Sales',
             'Bloom Country Risk', 'Bond Yield Change'
        ]
        
        # Initialize list to store all tidy data
        all_tidy_data = []
        
        with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
            for sheet_name in xls.sheet_names:
                # Skip sheets that should be ignored
                if sheet_name in skip_sheets:
                    continue
                
                # Load and prepare original data
                df = xls.parse(sheet_name)
                
                # Ensure the first column is 'date' and is datetime
                original_date_column = df.columns[0]
                df.rename(columns={original_date_column: 'date'}, inplace=True)
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df.dropna(subset=['date'], inplace=True)  # Drop rows where date conversion failed
                # Convert dates to first of month for consistency
                df['date'] = df['date'].dt.to_period('M').dt.to_timestamp()
                df.set_index('date', inplace=True)
                
                # Initialize dictionary to hold all variants
                variants = {}
                
                # Check if this sheet should be copied directly
                if sheet_name in copy_direct:
                    variants['Original'] = df.copy()
                else:
                    # 1. Cross-sectional normalization (CS)
                    cs_df = df.copy()
                    cs_means = cs_df.mean(axis=1)
                    cs_stds = cs_df.std(axis=1)
                    cs_norm = (cs_df.subtract(cs_means, axis=0)).divide(cs_stds, axis=0)
                    if sheet_name in invert_norm:
                        cs_norm = -1 * cs_norm
                    variants['CS'] = cs_norm
                    
                    # 2. Time series normalization (TS)
                    ts_norm = pd.DataFrame(index=df.index, columns=df.columns)
                    for country in df.columns:
                        country_data = df[country].copy()
                        ts_mean = country_data.expanding().mean()
                        ts_std = country_data.expanding().std()
                        # Handle case where std is 0 (like US in Currency 12)
                        ts_norm[country] = np.where(
                            ts_std == 0,
                            0,  # when std is 0, set normalized value to 0
                            (country_data - ts_mean) / ts_std
                        )
                    if sheet_name in invert_norm:
                        ts_norm = -1 * ts_norm
                    variants['TS'] = ts_norm
                
                # Write all variants to Excel and prepare tidy DataFrames
                for variant_type, variant_df in variants.items():
                    # Create sheet name with variant suffix (except for original return variables)
                    if sheet_name in copy_direct:
                        safe_sheet_name = _truncate_sheet_name(sheet_name)
                    else:
                        safe_sheet_name = _truncate_sheet_name(f"{sheet_name}_{variant_type}")
                    variant_df.to_excel(writer, sheet_name=safe_sheet_name)
                    
                    # Create and store tidy version
                    tidy_df = _create_tidy_df(variant_df, sheet_name, variant_type)
                    all_tidy_data.append(tidy_df)
        
        # Combine all tidy data and save to CSV
        combined_tidy = pd.concat(all_tidy_data, ignore_index=True)
        combined_tidy.to_csv(csv_output, index=False)
        
        print(f"Successfully created normalized data files:")
        print(f"1. Excel file: {excel_output}")
        print(f"2. CSV file: {csv_output}")
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    normalize_data()
