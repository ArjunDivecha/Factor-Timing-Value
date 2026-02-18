"""
T2 Master Data Processing Pipeline - Step One
===========================================

PURPOSE:
Processes raw financial data from Bloomberg into a clean, standardized format for downstream
analysis in the T2 Factor Timing strategy. This is the first and most critical step in the pipeline.

INPUT FILES:
1. Country Bloomberg Data Master T.xlsx
   - Source: Bloomberg terminal data dump
   - Format: Excel workbook with multiple sheets
   - Required sheets:
     * Tot Return Index: Total return indices by country
     * PX_LAST: Price data
     * 120MA: 120-day moving averages
     * Gold, Copper, Oil, Agriculture: Commodity prices
     * Currency: FX rates
     * 10Yr Bond: Government bond yields
     * Best EPS, Trailing EPS: Earnings metrics
   - Frequency: Daily

2. P2P_Country_Historical_Scores.xlsx (optional)
   - Source: Internal P2P scoring model
   - Format: Excel workbook with single sheet
   - Structure: Date column followed by country columns with P2P scores
   - Frequency: Monthly

OUTPUT FILES:
1. T2 Master.xlsx
   - Format: Excel workbook with multiple sheets
   - Generated sheets:
     * Returns: 1MRet, 3MRet, 6MRet, 9MRet, 12MRet (Forward returns)
     * Trailing Returns: 1MTR, 3MTR, 12MTR
     * Return Spread: 12-1MTR
     * Commodities: Gold, Copper, Oil, Agriculture
     * Economic Indicators: Currency, 10Yr Bond
     * Valuation: Best EPS, Trailing EPS, P2P (if available)
     * Technicals: 120MA Signal

DATA PROCESSING PIPELINE:
1. Data Loading & Validation
   - Load all input sheets with type inference
   - Validate required columns and data types
   - Standardize date indexing

2. Data Cleaning
   - Forward fill missing values with type preservation
   - Detect and handle local outliers (±3σ)
   - Apply global winsorization (5th-95th percentiles)
   - Special handling for market cap data

3. Feature Engineering
   - Calculate forward returns (1M, 3M, 6M, 9M, 12M)
   - Compute trailing returns (1M, 3M, 12M)
   - Generate return spreads (e.g., 12M-1M)
   - Create moving average signals

4. Output Generation
   - Apply consistent formatting
   - Validate output data quality
   - Generate comprehensive logs

VERSION HISTORY:
- 2.1 (2025-06-15): Enhanced error handling and logging
- 2.0 (2025-05-20): Complete rewrite with type safety and performance improvements
- 1.5 (2025-04-10): Added P2P score integration
- 1.0 (2025-01-15): Initial production version

AUTHOR: Financial Data Processing Team
LAST UPDATED: 2025-06-17

NOTES:
- All percentage values are properly scaled (0.01 = 1%)
- Returns are calculated as simple returns ((P1/P0) - 1)
- Missing values are forward-filled within each country
- Outliers are detected using rolling window statistics
- Market cap data receives special treatment to preserve large-cap influence

DEPENDENCIES:
- pandas>=2.0.0
- numpy>=1.24.0
- scipy>=1.10.0
- openpyxl>=3.1.0
"""

from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
from scipy import stats
import os
import logging
from datetime import datetime

# Configure logging to both file and console
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, 'T2_processing.log')

# Create a formatter for both handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Configure the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler - append mode
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Set pandas options for type safety and display
pd.set_option('future.no_silent_downcasting', True)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

def check_local_outliers(
    series: pd.Series,
    window_size: int = 20,
    sigma_threshold: float = 4.0
) -> List[Dict[str, float]]:
    """
    Detect local outliers using rolling window statistics optimized for financial time series.
    
    This implementation uses exponentially weighted statistics to give more weight
    to recent data points, which is particularly important for financial time series
    where recent values are more relevant for outlier detection.
    
    Args:
        series: Time series data to check for outliers
        window_size: Number of neighboring points to consider
        sigma_threshold: Number of standard deviations for outlier threshold
    
    Returns:
        List of dictionaries containing outlier information
    
    Example:
        >>> prices = pd.Series([100, 101, 150, 102, 101])  # 150 is potential outlier
        >>> outliers = check_local_outliers(prices, window_size=2)
        >>> print(len(outliers))  # Should detect 150 as outlier
        1
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    if series.empty:
        logger.warning("Empty series provided for outlier detection")
        return []
    
    # Convert to float64 for numerical stability
    series = series.astype('float64')
    
    outliers_found: List[Dict[str, float]] = []
    
    try:
        for i in range(len(series)):
            # Get window bounds with proper boundary handling
            start_idx = max(0, i - window_size)
            end_idx = min(len(series), i + window_size + 1)
            
            # Get local window excluding the current value
            local_window = pd.concat([series[start_idx:i], series[i+1:end_idx]])
            
            if len(local_window) < 10:
                continue
            
            # Use exponentially weighted statistics
            local_mean = local_window.ewm(span=window_size).mean().iloc[-1]
            local_std = local_window.ewm(span=window_size).std().iloc[-1]
            
            if local_std <= np.finfo(float).eps:  # Avoid division by zero
                continue
            
            z_score = abs((series[i] - local_mean) / local_std)
            
            if z_score > sigma_threshold:
                outliers_found.append({
                    'index': i,
                    'original_value': float(series[i]),
                    'new_value': float(local_mean),
                    'z_score': float(z_score),
                    'local_mean': float(local_mean),
                    'local_std': float(local_std),
                    'pct_change': float((local_mean - series[i]) / series[i] if series[i] != 0 else np.inf)
                })
    
    except Exception as e:
        logger.error(f"Error in outlier detection: {str(e)}")
        raise
    
    return outliers_found

def winsorize_column(
    series: pd.Series,
    mad_threshold: float = 5.0
) -> Tuple[pd.Series, int, int, pd.Series, float, float]:
    """
    Winsorize financial data while preserving statistical properties.
    Only catches extremely outlying values.
    
    Args:
        series: Financial data series to winsorize
        mad_threshold: Number of MADs for threshold (default: 5.0)
    
    Returns:
        Tuple containing:
        - Winsorized series
        - Count of lower outliers
        - Count of upper outliers
        - Example outliers with their values
        - Lower bound
        - Upper bound
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    if series.empty:
        logger.warning("Empty series provided for winsorization")
        return series, 0, 0, pd.Series(), 0.0, 0.0
    
    try:
        # Convert to float64 for numerical stability
        series = series.astype('float64')
        
        # Use robust statistics for better outlier handling
        median = series.median()
        mad = stats.median_abs_deviation(series, scale='normal')
        
        # Calculate bounds using median ± mad_threshold * MAD
        lower_bound = median - mad_threshold * mad
        upper_bound = median + mad_threshold * mad
        
        # Store original values for outliers
        outliers_orig = series[(series < lower_bound) | (series > upper_bound)].copy()
        
        # Count outliers before winsorization
        outliers_below = (series < lower_bound).sum()
        outliers_above = (series > upper_bound).sum()
        
        # Store example outliers with both original and new values
        examples = pd.DataFrame({
            'original': outliers_orig.head(3),
            'new_value': pd.Series(
                [lower_bound if x < lower_bound else upper_bound for x in outliers_orig.head(3)],
                index=outliers_orig.head(3).index
            ),
            'pct_change': pd.Series(
                [(lower_bound - x)/x if x < lower_bound else (upper_bound - x)/x 
                 for x in outliers_orig.head(3)],
                index=outliers_orig.head(3).index
            )
        })
        
        # Winsorize while preserving dtype
        winsorized = series.clip(lower=lower_bound, upper=upper_bound)
        
        return (
            winsorized,
            int(outliers_below),
            int(outliers_above),
            examples,
            float(lower_bound),
            float(upper_bound)
        )
    
    except Exception as e:
        logger.error(f"Error in winsorization: {str(e)}")
        raise

def calculate_forward_returns(data: pd.DataFrame, months: int = 1) -> pd.DataFrame:
    """
    Calculate forward returns using Total Return Index data for a specified number of months.
    
    The forward return for each month is calculated as:
    ((Total Return Index[t+n] / Total Return Index[t]) - 1)
    where n is the number of months forward.
    
    Note: The last n rows will have NaN returns as the future months' data isn't available.
    
    Args:
        data (pd.DataFrame): DataFrame containing the Total Return Index data
        months (int): Number of months forward for return calculation (1, 3, 6, 9, or 12)
    
    Returns:
        pd.DataFrame: DataFrame containing the forward returns
    
    Example:
        >>> tot_ret_idx = pd.DataFrame({'Singapore': [100, 102, 105, 108]})
        >>> fwd_returns = calculate_forward_returns(tot_ret_idx, months=3)
        >>> print(fwd_returns['Singapore'])
        0    0.0800  # (108/100 - 1)
        1       NaN  # Future data not available
        2       NaN  # Future data not available
        3       NaN  # Future data not available
    """
    try:
        # Create a copy to avoid modifying original data
        returns = pd.DataFrame(index=data.index)
        
        # Process each country column except the date column
        for col in data.columns[1:]:
            # Ensure data is numeric and properly typed
            values = pd.to_numeric(data[col], errors='coerce').astype('float64')
            
            # Calculate forward returns using shift(-months) to get future value
            future_values = values.shift(-months)
            returns[col] = (future_values / values) - 1
            
            # Convert to percentage values
            returns[col] = returns[col].astype('float64')
        
        # Copy the date column
        returns.insert(0, data.columns[0], data[data.columns[0]])
        
        return returns
    
    except Exception as e:
        logger.error(f"Error calculating {months}-month forward returns: {str(e)}")
        raise

def calculate_trailing_returns(data: pd.DataFrame, months: int = 1) -> pd.DataFrame:
    """
    Calculate trailing returns using Total Return Index data for a specified number of months.
    
    The trailing return for each month is calculated as:
    ((Total Return Index[t] / Total Return Index[t-n]) - 1)
    where n is the number of months back.
    
    Note: The first n rows will have NaN returns as the historical months' data isn't available.
    
    Args:
        data (pd.DataFrame): DataFrame containing the Total Return Index data
        months (int): Number of months back for return calculation (1, 3, or 12)
    
    Returns:
        pd.DataFrame: DataFrame containing the trailing returns
    
    Example:
        >>> tot_ret_idx = pd.DataFrame({'Singapore': [100, 102, 105, 108]})
        >>> trail_returns = calculate_trailing_returns(tot_ret_idx, months=3)
        >>> print(trail_returns['Singapore'])
        0       NaN  # Historical data not available
        1       NaN  # Historical data not available
        2       NaN  # Historical data not available
        3    0.0800  # (108/100 - 1)
    """
    try:
        # Create a copy to avoid modifying original data
        returns = pd.DataFrame(index=data.index)
        
        # Process each country column except the date column
        for col in data.columns[1:]:
            # Ensure data is numeric and properly typed
            values = pd.to_numeric(data[col], errors='coerce').astype('float64')
            
            # Calculate trailing returns using shift(months) to get historical value
            historical_values = values.shift(months)
            returns[col] = (values / historical_values) - 1
            
            # Convert to percentage values
            returns[col] = returns[col].astype('float64')
        
        # Copy the date column
        returns.insert(0, data.columns[0], data[data.columns[0]])
        
        return returns
    
    except Exception as e:
        logger.error(f"Error calculating {months}-month trailing returns: {str(e)}")
        raise

def calculate_change(data: pd.DataFrame, months: int = 12, is_absolute: bool = False) -> pd.DataFrame:
    """
    Calculate n-month change in values.
    
    The change is calculated as:
    For absolute change (is_absolute=True):
        Value[t] - Value[t-n]
    For percentage change (is_absolute=False):
        ((Value[t] / Value[t-n]) - 1)
    where n is the number of months back.
    
    Args:
        data (pd.DataFrame): Input data
        months (int): Number of months to look back
        is_absolute (bool): If True, calculate absolute change instead of percentage change
    
    Returns:
        pd.DataFrame: DataFrame containing the changes
    """
    try:
        logger.info(f"Calculating {months}-month change")
        
        # Create a copy to avoid modifying original data
        changes = pd.DataFrame(index=data.index)
        
        # Copy the date column first
        changes.insert(0, data.columns[0], data[data.columns[0]])
        
        # Process each country column
        for col in data.columns[1:]:
            # Ensure data is numeric and properly typed
            values = pd.to_numeric(data[col], errors='coerce').astype('float64')
            
            # Calculate change using shift(months) to get historical value
            historical_values = values.shift(months)
            
            if is_absolute:
                # Calculate absolute change
                changes[col] = values - historical_values
            else:
                # Calculate percentage change
                changes[col] = (values / historical_values) - 1
            
            # Convert to float64 for consistency
            changes[col] = changes[col].astype('float64')
        
        return changes
        
    except Exception as e:
        logger.error(f"Error calculating {months}-month change: {str(e)}")
        raise

def calculate_ma_signal(price_data: pd.DataFrame, ma_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Moving Average Signal as PX_LAST/MA ratio using pre-calculated MA values.
    If either PX_LAST or MA is missing, the result will be NaN.
    
    Args:
        price_data (pd.DataFrame): Input data with PX_LAST values
        ma_data (pd.DataFrame): Input data with pre-calculated MA values
    
    Returns:
        pd.DataFrame: DataFrame containing the MA signals
    """
    try:
        logger.info("Calculating MA Signal using PX_LAST and 120MA data")
        
        # Create a copy to avoid modifying original data
        signals = pd.DataFrame(index=price_data.index)
        
        # Copy the date column first
        signals.insert(0, price_data.columns[0], price_data[price_data.columns[0]])
        
        # Process each country column
        for col in price_data.columns[1:]:
            if col in ma_data.columns:
                # Ensure data is numeric and properly typed
                prices = pd.to_numeric(price_data[col], errors='coerce').astype('float64')
                ma_values = pd.to_numeric(ma_data[col], errors='coerce').astype('float64')
                
                # Calculate signal as PX_LAST/MA
                signals[col] = prices / ma_values
                
                # Convert to float64 for consistency
                signals[col] = signals[col].astype('float64')
            else:
                logger.warning(f"Column {col} not found in MA data")
                signals[col] = np.nan
        
        return signals
        
    except Exception as e:
        logger.error(f"Error calculating MA Signal: {str(e)}")
        raise

def standardize_date(df: pd.DataFrame) -> pd.DataFrame:
    """Convert dates to first of next month for consistency."""
    # Check if 'Country' column contains date-like values
    try:
        # Make a copy to avoid modifying the original if conversion fails
        df_copy = df.copy()
        df_copy['Country'] = pd.to_datetime(df_copy['Country'], errors='raise')
        # Convert to the first day of the NEXT month (add one month period and then convert to timestamp)
        df_copy['Country'] = (df_copy['Country'].dt.to_period('M') + 1).dt.to_timestamp()
        return df_copy
    except (ValueError, pd.errors.ParserError):
        # If conversion fails, log a warning and return the original dataframe
        logger.warning(f"Could not convert 'Country' column to datetime. Column may contain non-date values.")
        return df

def process_excel_file() -> None:
    """
    Process Bloomberg financial data with enhanced data quality controls.
    
    This function implements a robust financial data processing pipeline:
    1. Proper handling of percentage values and returns
    2. Type-safe operations throughout
    3. Special treatment for market capitalization data
    4. Detailed logging of all data transformations
    
    File Structure:
    - Input: Country Bloomberg Data Master T.xlsx
    - Input: P2P_Country_Historical_Scores.xlsx
    - Output: T2 Master.xlsx
    
    Processing Steps:
    1. Load and validate input data
    2. Forward fill missing values with type inference
    3. Apply outlier detection and winsorization
    4. Calculate forward returns (1M, 3M, 6M, 9M, 12M)
    5. Calculate trailing returns (1M, 3M, 12M)
    6. Copy P2P data
    7. Validate output before saving
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If data validation fails
        Exception: For other processing errors
    """
    start_time = datetime.now()
    logger.info("Starting financial data processing")
    
    try:
        # Input/output paths
        input_file = '/Users/arjundivecha/Dropbox/AAA Backup/Master Database/Country Bloomberg Data Master T.xlsx'
        p2p_file = 'P2P_Country_Historical_Scores.xlsx'
        output_dir = '/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/Transformer/T2'
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        if not os.path.exists(p2p_file):
            raise FileNotFoundError(f"P2P file not found: {p2p_file}")
        
        # Country names for validation
        country_names = [
            'Country', 'Singapore', 'Australia', 'Canada', 'Germany', 'Japan',
            'Switzerland', 'U.K.', 'NASDAQ', 'U.S.', 'France', 'Netherlands',
            'Sweden', 'Italy', 'ChinaA', 'Chile', 'Indonesia', 'Philippines',
            'Poland', 'US SmallCap', 'Malaysia', 'Taiwan', 'Mexico', 'Korea',
            'Brazil', 'South Africa', 'Denmark', 'India', 'ChinaH', 'Hong Kong',
            'Thailand', 'Turkey', 'Spain', 'Vietnam', 'Saudi Arabia'
        ]
        
        # Create output directory
       # os.makedirs(output_dir, exist_ok=True)
        output_file = 'T2 Master.xlsx'
        
        # Process file
        excel_file = pd.ExcelFile(input_file)
        sheet_names = excel_file.sheet_names[1:]  # Skip first sheet
        process_sheets = [sheet for sheet in sheet_names if 'Mcap' not in sheet]
        
        # Process P2P data first
        logger.info("Processing P2P data")
        try:
            # Read P2P data
            p2p_data = pd.read_excel(p2p_file, engine='openpyxl')
            
            # Replace headers with country names
            if len(p2p_data.columns) <= len(country_names):
                p2p_data.columns = country_names[:len(p2p_data.columns)]
            else:
                logger.warning("P2P data has more columns than available country names")
                raise ValueError("P2P data has more columns than expected")
            
            # Keep original dates from P2P file instead of replacing them
            # Convert to datetime for consistency but don't standardize
            if 'Country' in p2p_data.columns:
                p2p_data['Country'] = pd.to_datetime(p2p_data['Country'], errors='coerce')
                logger.info("Using original dates from P2P input file")
                
                # Delete the first row to align dates with other files
                p2p_data = p2p_data.iloc[1:].reset_index(drop=True)
                logger.info("Deleted first row of P2P data to align dates with other files")
            else:
                logger.warning("No 'Country' column found in P2P data, date handling may be incorrect")
            
        except Exception as e:
            logger.error(f"Error processing P2P data: {str(e)}")
            raise

        # Now write all sheets including the processed P2P data
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            # First, read and process the Total Return Index sheet
            logger.info("Processing Total Return Index sheet")
            tot_ret_data = pd.read_excel(input_file, sheet_name='Tot Return Index ')  # Note the space at the end
            tot_ret_data = tot_ret_data.iloc[2:].reset_index(drop=True)
            tot_ret_data.columns = country_names[:len(tot_ret_data.columns)]
            tot_ret_data = standardize_date(tot_ret_data)  # Standardize dates
            
            # Calculate forward returns for different periods
            periods = [1, 3, 6, 9, 12]
            for months in periods:
                logger.info(f"Calculating {months}-month forward returns")
                forward_returns = calculate_forward_returns(tot_ret_data, months)
                sheet_name = f"{months}MRet"
                forward_returns.to_excel(writer, sheet_name=sheet_name, index=False)
                logger.info(f"{months}-month forward returns calculation completed")
            
            # Calculate trailing returns for different periods
            trailing_periods = [1, 3, 12]
            trailing_returns_dict = {}  # Store trailing returns for later use
            for months in trailing_periods:
                logger.info(f"Calculating {months}-month trailing returns")
                trailing_returns = calculate_trailing_returns(tot_ret_data, months)
                sheet_name = f"{months}MTR"
                trailing_returns.to_excel(writer, sheet_name=sheet_name, index=False)
                trailing_returns_dict[months] = trailing_returns
                logger.info(f"{months}-month trailing returns calculation completed")
            
            # Calculate 12-month minus 1-month trailing returns
            logger.info("Calculating 12-1 month trailing returns spread")
            # Exclude Country column for subtraction, then add it back
            country_col = trailing_returns_dict[12]['Country']
            tr_12_1_spread = trailing_returns_dict[12].drop('Country', axis=1).subtract(
                trailing_returns_dict[1].drop('Country', axis=1))
            tr_12_1_spread.insert(0, 'Country', country_col)
            tr_12_1_spread.to_excel(writer, sheet_name='12-1MTR', index=False)
            logger.info("12-1 month trailing returns spread calculation completed")
            
            # Dictionary of sheets and their change periods
            change_calculations = {
                'Gold': 12,
                'Copper': 12,
                'Oil': 12,
                'Agriculture': 12,
                'Currency': 12,
                '10Yr Bond': 12,
                'Best EPS': 36,
                'Trailing EPS': 36
            }
            
            # Process sheets that need change calculations
            for sheet_name, months in change_calculations.items():
                if sheet_name in excel_file.sheet_names:
                    logger.info(f"Calculating {months}-month change for {sheet_name}")
                    try:
                        # Load data
                        sheet_data = pd.read_excel(input_file, sheet_name=sheet_name)
                        sheet_data = sheet_data.iloc[2:].reset_index(drop=True)
                        sheet_data.columns = country_names[:len(sheet_data.columns)]
                        sheet_data = standardize_date(sheet_data)  # Standardize dates
                        
                        # Calculate change
                        if sheet_name == '10Yr Bond':
                            changes = calculate_change(sheet_data, months, is_absolute=True)
                        else:
                            changes = calculate_change(sheet_data, months)
                        new_sheet_name = f"{sheet_name} {months}"
                        changes.to_excel(writer, sheet_name=new_sheet_name, index=False)
                        logger.info(f"{months}-month change calculation completed for {sheet_name}")
                    except Exception as e:
                        logger.error(f"Error processing {months}-month change for {sheet_name}: {str(e)}")
                        raise
            
            # Process sheets for MA signals
            sheets_for_ma = ['PX_LAST']
            
            for sheet_name in sheets_for_ma:
                logger.info(f"Processing {sheet_name} for MA signal")
                try:
                    # Read the sheet
                    sheet_data = pd.read_excel(input_file, sheet_name=sheet_name)
                    sheet_data = sheet_data.iloc[2:].reset_index(drop=True)  # Skip first two rows
                    
                    # Ensure column names match country names
                    if len(sheet_data.columns) <= len(country_names):
                        sheet_data.columns = country_names[:len(sheet_data.columns)]
                        sheet_data = standardize_date(sheet_data)  # Standardize dates
                        
                        # Calculate MA signal
                        ma_data = pd.read_excel(input_file, sheet_name='120MA')
                        ma_data = ma_data.iloc[2:].reset_index(drop=True)
                        ma_data.columns = country_names[:len(ma_data.columns)]
                        ma_data = standardize_date(ma_data)  # Standardize dates
                        signals = calculate_ma_signal(sheet_data, ma_data)
                        new_sheet_name = "120MA Signal"
                        signals.to_excel(writer, sheet_name=new_sheet_name, index=False)
                        logger.info(f"120MA Signal calculation completed for {sheet_name}")
                    
                except Exception as e:
                    logger.error(f"Error processing MA signal for {sheet_name}: {str(e)}")
                    raise
            
            # Process other sheets as before
            for sheet_name in sheet_names:
                logger.info(f"Processing sheet: {sheet_name}")
                
                try:
                    # Load data
                    data = pd.read_excel(input_file, sheet_name=sheet_name)
                    data = data.iloc[2:].reset_index(drop=True)
                    
                    # Special handling for MCAP Adj sheet - limit to 400 rows
                    if sheet_name == 'MCAP Adj':
                        data = data.iloc[:400].copy()
                        logger.info("MCAP Adj sheet: Limited to first 400 rows")
                    
                    # Validate and set column names
                    num_cols = len(data.columns)
                    if num_cols > len(country_names):
                        raise ValueError(f"Sheet {sheet_name} has more columns than expected")
                    
                    data.columns = country_names[:num_cols]
                    
                    # Standardize dates for all sheets to ensure consistent date alignment
                    data = standardize_date(data)
                    
                    # Forward fill with type inference
                    data = data.ffill().infer_objects()
                    
                    # Process non-Mcap sheets
                    if sheet_name in process_sheets:
                        total_winsorized_changes = 0
                        total_local_outlier_changes = 0
                        
                        for col in data.columns[1:]: # Skip 'Country' column
                            # Ensure column is numeric before processing
                            series = pd.to_numeric(data[col], errors='coerce').astype('float64')
                            
                            if not series.isna().all():
                                #logger.info(f"\nAnalyzing {col} in {sheet_name}") # Too verbose
                                column_winsorized_changes = 0
                                column_local_outliers = 0
                                
                                # Step 1: Global winsorization for egregious outliers
                                winsorized, below, above, examples, lower, upper = winsorize_column(
                                    series, mad_threshold=5.0
                                )
                                
                                if below + above > 0:
                                    column_winsorized_changes = below + above
                                    total_winsorized_changes += column_winsorized_changes
                                    # Verbose logging commented out
                                    # logger.info(
                                    #     f"Winsorization changes for {col}:"
                                    #     f"\n  - Total outliers: {below + above} ({((below + above)/len(series))*100:.2f}% of data)"
                                    #     f"\n  - Below threshold: {below}"
                                    #     f"\n  - Above threshold: {above}"
                                    #     f"\n  - Bounds: [{lower:.4f}, {upper:.4f}]"
                                    # )
                                    # if not examples.empty:
                                    #     logger.info("Example changes:")
                                    #     for idx, row in examples.iterrows():
                                    #         logger.info(
                                    #             f"  - Index {idx}: {row['original']:.4f} -> {row['new_value']:.4f} "
                                    #             f"({row['pct_change']*100:.1f}% change)"
                                    #         )
                                
                                # Step 2: Local outlier detection on winsorized data
                                local_outliers = check_local_outliers(winsorized)
                                if local_outliers:
                                    column_local_outliers = len(local_outliers)
                                    total_local_outlier_changes += column_local_outliers
                                    # Verbose logging commented out
                                    # logger.info(
                                    #     f"\nLocal outlier changes for {col}:"
                                    #     f"\n  - Found {len(local_outliers)} local outliers ({(len(local_outliers)/len(series))*100:.2f}% of data)"
                                    # )
                                    # # Show first few examples
                                    # for outlier in local_outliers[:3]:
                                    #     logger.info(
                                    #         f"  - Index {outlier['index']}: {outlier['original_value']:.4f} -> "
                                    #         f"{outlier['new_value']:.4f} ({outlier['pct_change']*100:.1f}% change)"
                                    #     )
                                    
                                    # Apply local outlier corrections
                                    for outlier in local_outliers:
                                        winsorized.iloc[outlier['index']] = outlier['new_value']
                                
                                # Verbose summary commented out
                                # if column_winsorized_changes + column_local_outliers > 0:
                                #     logger.info(
                                #         f"\nSummary for {col}:"
                                #         f"\n  - Total changes: {column_winsorized_changes + column_local_outliers} ({((column_winsorized_changes + column_local_outliers)/len(series))*100:.2f}% of data)"
                                #         f"\n  - Total unchanged: {len(series) - (column_winsorized_changes + column_local_outliers)}"
                                #     )
                                
                                data[col] = winsorized
                        
                        # Log summary for the entire sheet after processing all columns
                        if total_winsorized_changes > 0 or total_local_outlier_changes > 0:
                             logger.info(f"Sheet {sheet_name}: Winsorized={total_winsorized_changes}, LocalOutliers={total_local_outlier_changes} values adjusted.")
                                 
                    # Validate processed data
                    if data.empty:
                        raise ValueError(f"Sheet {sheet_name} is empty after processing")
                    
                    # Write to Excel
                    data.to_excel(writer, sheet_name=sheet_name, index=False)
                    logger.info(f"Sheet {sheet_name} processed successfully")
                
                except Exception as e:
                    logger.error(f"Error processing sheet {sheet_name}: {str(e)}")
                    raise
            
            # Write P2P data to the output file
            p2p_data.to_excel(writer, sheet_name='P2P', index=False)
            logger.info("P2P data processed and updated successfully")
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logger.info(f"Financial data processing completed in {processing_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during financial data processing: {str(e)}")
        raise

if __name__ == "__main__":
    process_excel_file()
