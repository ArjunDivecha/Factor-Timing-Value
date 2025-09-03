#!/usr/bin/env python3
"""
T2 Factor Portfolio Analysis - Step Three
========================================

PURPOSE:
Analyzes factor performance by constructing and evaluating portfolios of top 20% countries
ranked by various factors. Generates comprehensive performance metrics, visualizations,
and exposure matrices for downstream analysis.

INPUT FILES:
1. Normalized_T2_MasterCSV.csv
   - Source: Output from Step Two (Create Normalized Tidy.py)
   - Format: Long-format CSV with columns:
     * date: Observation date (YYYY-MM-DD)
     * country: Country code
     * variable: Factor or return identifier
     * value: Normalized value or return
   - Required variables:
     * 1MRet: Monthly returns (for performance calculation)
     * Various factor scores (for portfolio formation)
   - Data should be properly normalized (CS/TS) from previous step

2. Portfolio_Data.xlsx
   - Sheet: 'Benchmarks'
   - Contains benchmark returns for comparison
   - Format:
     * Index: DatetimeIndex aligned with portfolio data
     * Column: 'Equal Weight' - Equal-weighted benchmark returns
   - Frequency: Monthly, aligned with factor data

OUTPUT FILES:
1. T2 Top20.xlsx
   - Comprehensive portfolio analysis workbook with sheets:
     * Performance: Key statistics for each factor portfolio
       - Annualized return, volatility, Sharpe ratio
       - Max drawdown, Calmar ratio, Sortino ratio
       - Win rate, profit factor, turnover
     * Monthly_Returns: Time series of portfolio returns
       - Rows: Dates (monthly)
       - Columns: Factor names
     * Holdings: Monthly portfolio compositions
       - Rows: Dates (monthly)
       - Columns: Countries
       - Values: 1 if held, 0 otherwise
     * Risk_Metrics: Detailed risk analytics
       - Rolling volatility (12M, 24M)
       - Drawdown analysis
       - Correlation matrix

2. T2 Top20.pdf
   - Visual performance report with:
     * Page 1: Cumulative returns (log scale)
       - Factor portfolios vs benchmark
       - Rolling 12M Sharpe ratio
     * Page 2: Risk analysis
       - Rolling 12M volatility
       - Drawdown curve
       - Underwater plot
     * Page 3: Portfolio characteristics
       - Turnover (1M, 3M, 12M)
       - Holdings concentration
       - Factor exposure over time

3. Top_20_Exposure.csv
   - Binary exposure matrix for all factors
   - Format:
     * Rows: Date-Country pairs
     * Columns: Factor names
     * Values: 1 if country in top 20% for factor, else 0
   - Example:
     Date,Country,Value,Momentum,Quality
     2024-05-31,USA,1,0,1
     2024-05-31,JPN,0,1,0
     2024-05-31,IND,1,1,0

METHODOLOGY:
1. Portfolio Formation:
   - For each factor and month:
     * Rank countries by factor score
     * Select top 20% countries
     * Equal-weight selected countries
     * Hold for one month

2. Performance Calculation:
   - Returns: Monthly rebalanced, gross of costs
   - Risk-free rate: Assumed 0 for Sharpe ratio
   - Turnover: Average monthly churn

VERSION HISTORY:
- 2.1 (2025-06-10): Added exposure matrix output
- 2.0 (2025-05-20): Complete rewrite with new metrics
- 1.5 (2025-04-15): Enhanced visualization
- 1.0 (2025-03-01): Initial production version

AUTHOR: Quantitative Research Team
LAST UPDATED: 2025-06-17

NOTES:
- All performance metrics are annualized where applicable
- Assumes monthly rebalancing at month-end
- Turnover calculation accounts for both additions and deletions
- Missing factor values are filled with 0 (neutral exposure)

DEPENDENCIES:
- pandas>=2.0.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- openpyxl>=3.1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import warnings
import os
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

# Configure warnings and plotting
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')  # Use correct seaborn style name
sns.set_theme()  # Set seaborn defaults

def analyze_portfolios(data: pd.DataFrame, features: list, benchmark_returns: pd.Series) -> Tuple[Dict[str, pd.Series], Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Analyze portfolios for all features.

    Args:
        data: DataFrame with the complete dataset
        features: List of factor features
        benchmark_returns: Benchmark returns series
    
    Returns:
        Tuple containing monthly returns dict, holdings dict, and results DataFrame
    """
    results = []
    monthly_returns = {}
    monthly_holdings = {}
    
    # Get unique dates and countries
    dates = sorted(data['date'].unique())
    countries = sorted(data['country'].unique())
    
    # Initialize placeholders for holdings DataFrame
    holdings_template = pd.DataFrame(0, index=dates, columns=countries)
    
    # Get all return data once and ensure proper date format
    returns_data = data[data['variable'] == '1MRet'].copy()
    returns_data['date'] = pd.to_datetime(returns_data['date'])
    returns_data = returns_data.sort_values('date')
    
    for feature in features:
        feature_data = data[data['variable'] == feature].copy()
        feature_data['date'] = pd.to_datetime(feature_data['date'])
        
        # Only keep dates where we have at least one valid value
        valid_dates = feature_data.groupby('date')['value'].apply(lambda x: x.notna().any())
        valid_dates = valid_dates[valid_dates].index
        feature_data = feature_data[feature_data['date'].isin(valid_dates)]
        
        feature_data = feature_data.sort_values('date')
        
        # Initialize storage for results
        portfolio_returns = pd.Series(0.0, index=dates)
        holdings = holdings_template.copy()
        
        # Process each date
        for date in dates:
            curr_feature_data = feature_data[feature_data['date'] == date]
            curr_returns_data = returns_data[returns_data['date'] == date]
            
            # If no feature data is available or all values are NaN, set return to NaN
            if len(curr_feature_data) == 0 or curr_feature_data['value'].isna().all():
                portfolio_returns[date] = np.nan
                continue
                
            if len(curr_returns_data) > 0:
                # Merge feature and returns data
                portfolio_data = pd.merge(
                    curr_feature_data[['country', 'value']],
                    curr_returns_data[['country', 'value']],
                    on='country',
                    suffixes=('_factor', '_return')
                )
                
                if len(portfolio_data) > 0:
                    # Select top 20% of countries from available data
                    n_select = max(1, int(len(portfolio_data) * 0.2))
                    selected = portfolio_data.nlargest(n_select, 'value_factor')
                    
                    # Calculate equal-weighted return
                    portfolio_return = selected['value_return'].mean()
                    selected_countries = selected['country'].tolist()
                else:
                    portfolio_return = np.nan
                    selected_countries = []
                    
                # Store results
                portfolio_returns[date] = portfolio_return
                if selected_countries:  # Only update holdings if we selected specific countries
                    holdings.loc[date, selected_countries] = 1

        # Ensure returns are properly aligned with benchmark dates
        portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
        portfolio_returns = portfolio_returns.reindex(benchmark_returns.index)
        
        monthly_returns[feature] = portfolio_returns
        monthly_holdings[feature] = holdings
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(portfolio_returns, benchmark_returns)
        
        # Calculate turnover
        turnover = calculate_turnover(holdings)
        
        # Combine all metrics
        result = {'Feature': feature, 'Average Turnover (%)': turnover}
        result.update(metrics)
        results.append(result)
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df[[
            'Feature',
            'Avg Excess Return (%)',
            'Volatility (%)',
            'Information Ratio',
            'Maximum Drawdown (%)',
            'Hit Ratio (%)',
            'Skewness',
            'Kurtosis',
            'Beta',
            'Tracking Error (%)',
            'Calmar Ratio',
            'Average Turnover (%)'
        ]]
    
    return monthly_returns, monthly_holdings, results_df

def calculate_performance_metrics(returns, benchmark_returns):
    """Calculate comprehensive performance metrics."""
    # Align series and convert to float
    returns = pd.to_numeric(returns, errors='coerce')
    benchmark_returns = pd.to_numeric(benchmark_returns, errors='coerce')
    
    # Ensure proper date alignment
    returns.index = pd.to_datetime(returns.index)
    benchmark_returns.index = pd.to_datetime(benchmark_returns.index)
    
    # Drop any NaN values and align the series
    valid_idx = returns.notna() & benchmark_returns.notna()
    returns = returns[valid_idx]
    benchmark_returns = benchmark_returns[valid_idx]
    
    if len(returns) == 0:
        return {
            'Avg Excess Return (%)': 0,
            'Volatility (%)': 0,
            'Information Ratio': 0,
            'Maximum Drawdown (%)': 0,
            'Hit Ratio (%)': 0,
            'Skewness': 0,
            'Kurtosis': 0,
            'Beta': 0,
            'Tracking Error (%)': 0,
            'Calmar Ratio': 0
        }
    
    # Calculate excess returns
    excess_returns = returns - benchmark_returns
    
    # Basic statistics with proper annualization
    avg_monthly_excess = excess_returns.mean()
    avg_excess_return = avg_monthly_excess * 12 * 100  # Simple annualization for excess returns
    
    volatility = returns.std() * np.sqrt(12) * 100  # Annualized volatility
    tracking_error = excess_returns.std() * np.sqrt(12) * 100  # Annualized tracking error
    information_ratio = avg_monthly_excess * np.sqrt(12) / (excess_returns.std()) if excess_returns.std() != 0 else 0
    
    # Drawdown analysis using cumulative excess returns
    cum_returns = (1 + excess_returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max * 100
    max_drawdown = drawdowns.min()
    
    # Hit ratio calculation
    hit_ratio = (excess_returns > 0).mean() * 100
    
    # Higher moments
    skewness = excess_returns.skew()
    kurtosis = excess_returns.kurtosis()
    
    # Beta calculation using aligned returns
    covariance = returns.cov(benchmark_returns)
    benchmark_variance = benchmark_returns.var()
    beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
    
    # Calmar ratio (annualized excess return / max drawdown)
    calmar_ratio = -avg_excess_return / max_drawdown if max_drawdown != 0 else 0
    
    return {
        'Avg Excess Return (%)': round(avg_excess_return, 2),
        'Volatility (%)': round(volatility, 2),
        'Information Ratio': round(information_ratio, 2),
        'Maximum Drawdown (%)': round(max_drawdown, 2),
        'Hit Ratio (%)': round(hit_ratio, 2),
        'Skewness': round(skewness, 2),
        'Kurtosis': round(kurtosis, 2),
        'Beta': round(beta, 2),
        'Tracking Error (%)': round(tracking_error, 2),
        'Calmar Ratio': round(calmar_ratio, 2)
    }

def calculate_turnover(holdings_df):
    """Calculate average monthly turnover."""
    if len(holdings_df) <= 1:
        return 0
        
    turnovers = []
    prev_holdings = None
    
    for date in holdings_df.index:
        curr_holdings = set(holdings_df.loc[date].dropna().index)
        if prev_holdings is not None and len(curr_holdings) > 0:
            # Calculate one-way turnover
            turnover = len(curr_holdings.symmetric_difference(prev_holdings)) / (2 * len(curr_holdings)) * 100
            turnovers.append(turnover)
        prev_holdings = curr_holdings
    
    return round(np.mean(turnovers), 2) if turnovers else 0

def create_performance_charts(returns_dict, benchmark_returns, output_path):
    """Create performance visualization charts."""
    n_features = len(returns_dict)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(15, n_rows * 4))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    
    # Use Seaborn for better aesthetics
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    
    for i, (feature, returns) in enumerate(returns_dict.items()):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Ensure dates are properly formatted as datetime
        returns.index = pd.to_datetime(returns.index)
        benchmark_returns.index = pd.to_datetime(benchmark_returns.index)
        
        # Calculate excess returns
        excess_returns = returns - benchmark_returns
        
        # Find first valid data point
        first_valid_idx = excess_returns.first_valid_index()
        if first_valid_idx:
            # Only keep data from first valid point onwards
            excess_returns = excess_returns[first_valid_idx:]
            
            # Initialize cumulative return series starting at 0
            cum_returns = pd.Series(0.0, index=excess_returns.index)
            
            # Calculate cumulative returns properly
            running_sum = 0
            for date in excess_returns.index:
                if pd.notna(excess_returns[date]):
                    running_sum += excess_returns[date]
                cum_returns[date] = running_sum
            
            # Plot
            ax.plot(cum_returns.index, cum_returns * 100, label='Excess Return', linewidth=1.5)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax.set_title(feature, fontsize=12, weight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        
        # Format x-axis with years only
        ax.xaxis.set_major_locator(mdates.YearLocator(2))  # Show every 2 years
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_portfolio_analysis(
    data_path: str,
    benchmark_path: str,
    output_dir: str
) -> None:
    """
    Executes a comprehensive portfolio analysis by processing factor and return data,
    forming portfolios of top 20% countries, and generating performance metrics.

    Parameters:
    -----------
    data_path : str
        Path to the CSV file containing factor and return data.
        Required format: columns for date, country, variable, value
    benchmark_path : str
        Path to the Excel file containing benchmark data with equal-weight returns
    output_dir : str
        Directory where output files (Excel and PDF) will be saved

    The function:
    1. Loads and processes factor and return data
    2. Forms portfolios based on country rankings
    3. Calculates performance metrics vs benchmark
    4. Generates Excel reports and visualization charts
    """
    print("\nStarting portfolio analysis...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Variables to skip in analysis
    skip_variables = [
        # Return variables
        '1MRet', '3MRet', '6MRet', '9MRet', '12MRet',
        # Moving averages
        '120MA_CS', '129MA_TS',
        # Commodity-related
        'Agriculture_TS', 'Agriculture_CS',
        'Copper_TS', 'Copper_CS',
        'Gold_CS', 'Gold_TS',
        'Oil_CS', 'Oil_TS',
        # Market and price related
        'MCAP Adj_CS', 'MCAP Adj_TS',
        'MCAP_CS', 'MCAP_TS',
        'PX_LAST_CS', 'PX_LAST_TS',
        'Tot Return Index_CS', 'Tot Return Index_TS',
        # Currency
        'Currency_CS', 'Currency_TS',
        # Earnings
        'BEST EPS_CS', 'BEST EPS_TS',
        'Trailing EPS_CS', 'Trailing EPS_TS'
    ]
    
    try:
        # Load and prepare data
        data = pd.read_csv(data_path)
        data['date'] = pd.to_datetime(data['date'])
        # Convert all dates to first of month
        data['date'] = data['date'].dt.to_period('M').dt.to_timestamp()
        
        # Load benchmark data
        benchmark_data = pd.read_excel(benchmark_path, sheet_name='Benchmarks', index_col=0)
        benchmark_data.index = pd.to_datetime(benchmark_data.index)
        # Convert benchmark dates to first of month
        benchmark_data.index = benchmark_data.index.to_period('M').to_timestamp()
        benchmark_returns = benchmark_data['equal_weight']
        
        # Get unique features excluding return variables
        features = sorted([f for f in data['variable'].unique() if f not in skip_variables])
        
        print(f"Analyzing {len(features)} features...")
        monthly_returns, monthly_holdings, results = analyze_portfolios(data, features, benchmark_returns)
        
        # Save results
        output_excel = 'T2 Top20.xlsx'
        output_pdf = 'T2 Top20.pdf'
        
        print("\nSaving results...")
        # Save results to Excel - single sheet
        results = results.sort_values('Information Ratio', ascending=False)
        results.to_excel(
            os.path.join(output_dir, output_excel),
            index=False,
            float_format='%.2f'
        )
        
        print("\nCreating performance charts...")
        # Create charts
        create_performance_charts(
            monthly_returns,
            benchmark_returns,
            os.path.join(output_dir, output_pdf)
        )
        
        print(f"\nAnalysis complete!")
        print(f"Results saved to: {os.path.join(output_dir, output_excel)}")
        print(f"Charts saved to: {os.path.join(output_dir, output_pdf)}")

        # --- BEGIN: Write Top_20_Exposure.csv ---
        # This section generates a CSV showing, for each date and country, which factors the country is in the top 20 for.
        # Columns: Date, Country, [Factor1], [Factor2], ... (factors in input order)
        # Each row: 1 if country is in top 20 for that factor on that date, else 0
        # Example row: 2024-05-31,USA,1,0,1
        try:
            exposure_rows = []
            # For each factor, use the holdings (from monthly_holdings) to determine top 20 countries per date
            # monthly_holdings: dict of {factor: DataFrame [date x country], 1 if in top 20, else 0}
            # We'll iterate over all dates and countries, and for each, collect 1/0 for each factor
            all_dates = set()
            all_countries = set()
            for df in monthly_holdings.values():
                all_dates.update(df.index)
                all_countries.update(df.columns)
            all_dates = sorted(pd.to_datetime(list(all_dates)))
            all_countries = sorted(all_countries)
            # For each date and country, build a row
            for date in all_dates:
                for country in all_countries:
                    row = [date.strftime('%Y-%m-%d'), country]
                    for factor in features:
                        # 1 if country is in top 20 for this factor at this date, else 0
                        holding = monthly_holdings[factor]
                        val = holding.loc[date, country] if (date in holding.index and country in holding.columns) else 0
                        row.append(int(val))
                    exposure_rows.append(row)
            # Build DataFrame
            exposure_cols = ['Date', 'Country'] + features
            exposure_df = pd.DataFrame(exposure_rows, columns=exposure_cols)
            # Write to CSV
            exposure_path = os.path.join(output_dir, 'T2_Top_20_Exposure.csv')
            exposure_df.to_csv(exposure_path, index=False)
            print(f"Top 20 exposure matrix saved to: {exposure_path}")
        except Exception as ex:
            print(f"[T2_Top_20_Exposure.csv] Error: {ex}")
        # --- END: Write Top_20_Exposure.csv ---
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # File paths
    DATA_PATH = "Normalized_T2_MasterCSV.csv"
    BENCHMARK_PATH = "Portfolio_Data.xlsx"
    OUTPUT_DIR = "."  # Changed output directory to current directory
    
    # Run analysis
    run_portfolio_analysis(DATA_PATH, BENCHMARK_PATH, OUTPUT_DIR)
