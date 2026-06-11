"""
T2 Factor Timing - Step Nine: Calculate Portfolio Returns (Long–Short)
=====================================================================

PURPOSE:
Calculates and analyzes the performance of a country-weighted investment portfolio by applying
country weights to historical returns data. This is a critical performance measurement step
that validates the T2 Factor Timing strategy against an equal-weight benchmark.

IMPORTANT NOTES:
- Returns in Portfolio_Data.xlsx use the FORWARD convention from Step One:
  row t of the Returns sheet already holds the return earned from month t to t+1
  (1MRet is built with shift(-1)). Therefore month-t weights are applied to
  row t directly — adding another one-month shift here would double-lag the
  signal. The equal-weight benchmark row t uses the same forward convention,
  so portfolio and benchmark at row t cover the same return month.
- Handles missing data and edge cases in turnover calculations
- Generates both detailed Excel reports and publication-quality PDF visualizations
- All calculations are in local currency (no FX adjustments)
- Supports long–short: country weights may be negative and are NOT renormalized by |weights|

INPUT FILES:
1. T2_Final_Country_Weights.xlsx
   - Source: Output from Step Eight (Write Country Weights)
   - Format: Excel workbook with sheet "All Periods"
   - Structure:
     * Index: Dates (datetime64[ns])
     * Columns: 3-letter country codes (str)
     * Values: Net portfolio weights (float, can be negative)
   - Notes:
     * Net weights typically sum ≈ 1.0 per date
     * Must include all countries in Portfolio_Data.xlsx

2. Portfolio_Data.xlsx
   - Source: Historical returns database
   - Format: Excel workbook with two sheets:
     a. "Returns":
        * Index: Dates (datetime64[ns])
        * Columns: 3-letter country codes (str)
        * Values: Monthly returns (float, decimal)
     b. "Benchmarks":
        * Index: Dates (datetime64[ns])
        * Columns: Benchmark names (str)
        * Values: Monthly returns (float, decimal)
   - Notes:
     * Must include at least 'Equal Weight' benchmark
     * Dates should align with weights file

OUTPUT FILES:
1. T2_Final_Portfolio_Returns.xlsx
   - Excel workbook with multiple sheets:
     a. "Monthly Returns":
        * Date-indexed time series of portfolio, benchmark, and net returns
        * All values in decimal format (0.01 = 1%)
     b. "Cumulative Returns":
        * Cumulative performance of portfolio vs. benchmark
        * Includes drawdown calculations
     c. "Statistics":
        * Comprehensive performance metrics (annualized returns, volatility, Sharpe ratio)
        * Win rates, maximum drawdown, and turnover statistics
     d. "Net Returns":
        * Active returns (portfolio minus benchmark)
        * Cumulative active return
     e. "Turnover Analysis":
        * Monthly and annualized turnover metrics
        * Cumulative turnover and trading costs

2. T2_Final_Portfolio_Returns.pdf
   - Professional-quality visualizations:
     a. Cumulative Returns:
        * Portfolio vs. equal-weight benchmark
        * Log-scale y-axis for long-term comparison
        * Drawdown periods highlighted
     b. Net Returns:
        * Rolling 12-month active returns
        * Histogram of monthly active returns
     c. Turnover Analysis:
        * 12-month moving average turnover
        * Cumulative trading costs (assuming 10bps per trade)

METHODOLOGY:
1. Portfolio Construction:
   - Weights from month t are applied to returns row t, which (forward
     convention) is the return earned from month t to t+1 — no look-ahead.
   - Net returns computed as Σ w_net × r (no renormalization by |weights|)
   - Missing returns are treated as 0 (no impact)

2. Performance Metrics:
   - Returns: Annualized arithmetic and geometric means
   - Risk: Annualized standard deviation of returns
   - Risk-Adjusted: Sharpe ratio (assuming 0% risk-free rate)
   - Drawdown: Maximum peak-to-trough decline
   - Turnover: One-way portfolio turnover (sum of absolute weight changes)

3. Statistical Significance:
   - T-statistics for mean returns
   - Skewness and kurtosis of return distributions
   - Hit rate (percentage of positive months)

VERSION HISTORY:
- 1.2 (2025-04-23): Added turnover analysis and improved visualizations
- 1.1 (2025-03-15): Enhanced performance metrics and reporting
- 1.0 (2025-02-01): Initial version

AUTHOR: Quantitative Research Team
LAST UPDATED: 2025-06-17

NOTES:
- All returns are in local currency (unhedged)
- Assumes monthly rebalancing at month-end
- Transaction costs are not included in returns but are estimated separately
- Benchmark is always equal-weighted portfolio of all countries

DEPENDENCIES:
- pandas>=2.0.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- openpyxl>=3.1.0
- scipy>=1.10.0 (for statistical calculations)
- fpdf>=1.7.2 (for PDF report generation)

MISSING DATA HANDLING:
- Analysis is restricted to dates common to both weights and returns datasets
- Rows whose forward returns are all-NaN (the most recent stub month) drop out
  naturally because the return row is empty
- No explicit imputation is performed in this script

VERSION HISTORY (continued):
- 1.3 (2026-06-09): Fixed double-lag bug — weights at month t are now applied
  to returns row t (already forward t→t+1 per Step One convention) instead of
  row t+1, and the equal-weight benchmark is taken from the same row so
  portfolio and benchmark cover the same return month.
"""

# Missing data handling implementation
# Analysis is restricted to dates common to both weights and returns datasets

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# VISUALIZATION SETUP
# ===============================

# Set plot style for consistent, professional visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]

# ===============================
# DATA LOADING
# ===============================

print("Loading data...")
# Input file paths
weights_file = 'T2_Final_Country_Weights.xlsx'
portfolio_data_file = 'Portfolio_Data.xlsx'  # Updated path to current directory

# Load country weights from the T2_Final_Country_Weights.xlsx file
weights_df = pd.read_excel(weights_file, sheet_name='All Periods', index_col=0)

# Load historical returns data and benchmark data
returns_df = pd.read_excel(portfolio_data_file, sheet_name='Returns', index_col=0)
benchmark_df = pd.read_excel(portfolio_data_file, sheet_name='Benchmarks', index_col=0)

# ===============================
# DATA PREPROCESSING
# ===============================

# Convert all index dates to datetime and standardize to month-end timestamps
weights_df.index = pd.to_datetime(weights_df.index)
weights_df.index = weights_df.index.to_period('M').to_timestamp()  # Standardize to month-end

returns_df.index = pd.to_datetime(returns_df.index)
returns_df.index = returns_df.index.to_period('M').to_timestamp()  # Standardize to month-end

benchmark_df.index = pd.to_datetime(benchmark_df.index)
benchmark_df.index = benchmark_df.index.to_period('M').to_timestamp()  # Standardize to month-end

# NOTE on timing: Portfolio_Data.xlsx returns use the FORWARD convention
# (Step One builds 1MRet with shift(-1)), so row t already holds the return
# earned from month t to t+1. Month-t weights are therefore applied to row t
# directly. Shifting to row t+1 here would double-lag the signal.

# Find common dates between weights and returns datasets
# We need dates that exist in both weights and returns datasets
weights_dates = set(weights_df.index)
returns_dates = set(returns_df.index)
common_dates = sorted(list(weights_dates.intersection(returns_dates)))

# Drop trailing dates whose forward returns are entirely missing
# (the most recent month is a stub with no realized t -> t+1 return yet)
while common_dates and returns_df.loc[common_dates[-1]].isna().all():
    common_dates = common_dates[:-1]

# Use all available dates with weights and returns - don't filter based on a fixed date
first_available_date = common_dates[0] if common_dates else None

print(f"\nAnalysis period: {common_dates[0]} to {common_dates[-1]}")
print(f"Number of months: {len(common_dates)}")

# ===============================
# PORTFOLIO RETURN CALCULATION
# ===============================

print("\nCalculating portfolio returns...")
# Initialize array to store portfolio returns
portfolio_returns = np.zeros(len(common_dates))

# Calculate portfolio returns for each date in the common dates
for i, date in enumerate(common_dates):
    # Get weights for this date
    weights = weights_df.loc[date]
    
    # Returns row t already holds the forward return t -> t+1 (Step One
    # convention), so apply month-t weights to row t. No extra shift.
    forward_returns = returns_df.loc[date]
    
    # Calculate weighted return for this date
    # Only use countries that have both weights and returns data
    common_countries = set(weights.index).intersection(forward_returns.index)
    
    # Skip if no common countries
    if len(common_countries) == 0:
        portfolio_returns[i] = np.nan
        continue
    
    # Calculate weighted return (supports long–short). Do NOT renormalize by total weight.
    weighted_return = 0.0
    for country in common_countries:
        w = weights[country]
        r = forward_returns[country]
        if not np.isnan(w) and not np.isnan(r):
            weighted_return += w * r
    portfolio_returns[i] = weighted_return

# ===============================
# TURNOVER CALCULATION
# ===============================

print("Calculating portfolio turnover...")

def calculate_turnover(weights_df, common_dates):
    """
    Calculate portfolio turnover for each period.
    
    Turnover = Sum of |Weight(t) - Weight(t-1)| for all countries
    This measures the one-way turnover (percentage of portfolio that changes each period)
    
    Parameters:
    weights_df: DataFrame with weights for each country and date
    common_dates: List of dates to analyze
    
    Returns:
    turnover_series: Series with turnover for each date
    """
    turnover_data = []
    
    for i, date in enumerate(common_dates):
        if i == 0:
            # First period has no previous weights, so turnover is NaN
            turnover_data.append(np.nan)
            continue
            
        # Get current and previous weights
        current_weights = weights_df.loc[date]
        previous_weights = weights_df.loc[common_dates[i-1]]
        
        # Get all countries that appear in either period
        all_countries = set(current_weights.index).union(set(previous_weights.index))
        
        # Calculate turnover as sum of absolute weight changes
        turnover = 0
        for country in all_countries:
            current_weight = current_weights.get(country, 0)
            previous_weight = previous_weights.get(country, 0)
            
            # Handle NaN values by treating them as 0
            if pd.isna(current_weight):
                current_weight = 0
            if pd.isna(previous_weight):
                previous_weight = 0
                
            turnover += abs(current_weight - previous_weight)
        
        # One-way turnover is half of the sum of absolute changes
        # (since buying one asset and selling another creates 2x turnover in the sum)
        turnover_data.append(turnover / 2)
    
    return pd.Series(turnover_data, index=common_dates)

# Calculate turnover series
turnover_series = calculate_turnover(weights_df, common_dates)

# Create results DataFrame with portfolio and benchmark returns
results = pd.DataFrame({
    'Portfolio': portfolio_returns,
    'Equal Weight': benchmark_df.loc[common_dates, 'equal_weight'],
    'Turnover': turnover_series
})

# Calculate net returns (active returns = portfolio minus benchmark)
results['Net Return'] = results['Portfolio'] - results['Equal Weight']

# Verify that net return calculation is correct
portfolio_mean = results['Portfolio'].mean() * 12 * 100
benchmark_mean = results['Equal Weight'].mean() * 12 * 100
expected_net_mean = portfolio_mean - benchmark_mean

# Calculate cumulative returns (growth of $1 invested)
cumulative_returns = (1 + results[['Portfolio', 'Equal Weight']]).cumprod()
cumulative_net = (1 + results['Net Return']).cumprod()

# Calculate cumulative turnover
cumulative_turnover = results['Turnover'].cumsum()

# ===============================
# PERFORMANCE STATISTICS
# ===============================

# Function to calculate key performance metrics
def calculate_stats(returns, turnover=None):
    stats = {}
    stats['Annual Return'] = returns.mean() * 12 * 100  # Annualized return in %
    stats['Annual Vol'] = returns.std() * np.sqrt(12) * 100  # Annualized volatility in %
    stats['Sharpe Ratio'] = (returns.mean() * 12) / (returns.std() * np.sqrt(12))  # Annualized Sharpe
    stats['Max Drawdown'] = ((1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1).min() * 100  # Max drawdown in %
    stats['Hit Rate'] = (returns > 0).mean() * 100  # Percentage of positive months
    stats['Skewness'] = returns.skew()  # Distribution skewness (asymmetry)
    stats['Kurtosis'] = returns.kurtosis()  # Distribution kurtosis (tail thickness)
    
    # Add turnover statistics if provided
    if turnover is not None:
        stats['Avg Monthly Turnover'] = turnover.mean() * 100  # Average monthly turnover in %
        stats['Annual Turnover'] = turnover.mean() * 12 * 100  # Annualized turnover in %
        stats['Max Monthly Turnover'] = turnover.max() * 100  # Maximum monthly turnover in %
        stats['Min Monthly Turnover'] = turnover.min() * 100  # Minimum monthly turnover in %
        stats['Turnover Volatility'] = turnover.std() * 100  # Volatility of turnover in %
    
    return pd.Series(stats)

# Calculate statistics for portfolio, benchmark, and net returns
portfolio_stats = calculate_stats(results['Portfolio'], results['Turnover'])
equal_weight_stats = calculate_stats(results['Equal Weight'])
net_return_stats = calculate_stats(results['Net Return'])

# For Equal Weight and Net Return, add turnover fields with 0 values instead of NaN
# This provides more meaningful display while maintaining data integrity
turnover_fields = ['Avg Monthly Turnover', 'Annual Turnover', 'Max Monthly Turnover', 
                  'Min Monthly Turnover', 'Turnover Volatility']
for field in turnover_fields:
    if field in portfolio_stats:
        equal_weight_stats[field] = 0.0  # Equal Weight has fixed allocation
        net_return_stats[field] = 0.0    # Net Return isn't a portfolio with turnover

stats = pd.DataFrame({
    'Portfolio': portfolio_stats,
    'Equal Weight': equal_weight_stats,
    'Net Return': net_return_stats
})

# Verify that the net return annual return matches the difference between portfolio and benchmark
print(f"\nVerification of Net Return calculation:")
print(f"Portfolio Annual Return: {portfolio_mean:.6f}%")
print(f"Equal Weight Annual Return: {benchmark_mean:.6f}%")
print(f"Expected Net Annual Return: {expected_net_mean:.6f}%")
print(f"Calculated Net Annual Return: {stats.loc['Annual Return', 'Net Return']:.6f}%")
print(f"Difference: {abs(expected_net_mean - stats.loc['Annual Return', 'Net Return']):.6f}%")

# ===============================
# VISUALIZATION
# ===============================

# Create multi-panel figure for performance visualization
plt.figure(figsize=(15, 12))  # Increased height for 3 plots

# Plot 1: Cumulative Total Returns - Compare portfolio to benchmark
plt.subplot(3, 1, 1)
cumulative_returns['Portfolio'].plot(label='Portfolio', color='blue')
cumulative_returns['Equal Weight'].plot(label='Equal Weight', color='red', alpha=0.7)
plt.title('Cumulative Total Returns')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.legend()

# Plot 2: Cumulative Net Returns - Show excess return over benchmark
plt.subplot(3, 1, 2)
cumulative_net.plot(label='Cumulative Net Return', color='green')
plt.axhline(y=1, color='r', linestyle='--', alpha=0.3)  # Reference line at 1.0
plt.title('Cumulative Net Returns (Portfolio - Equal Weight)')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.legend()

# Plot 3: Monthly Turnover - Show portfolio turnover over time
plt.subplot(3, 1, 3)
(results['Turnover'] * 100).plot(label='Monthly Turnover', color='orange', alpha=0.7)
plt.title('Monthly Portfolio Turnover')
plt.ylabel('Turnover (%)')
plt.xlabel('Date')
plt.grid(True)
plt.legend()

# Save the figure with high resolution
plt.tight_layout()
plt.savefig('T2_Final_Portfolio_Returns.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

# ===============================
# RESULTS SAVING
# ===============================

print("\nSaving results...")
output_file = 'T2_Final_Portfolio_Returns.xlsx'

# Create turnover analysis DataFrame
turnover_analysis = pd.DataFrame({
    'Monthly Turnover': results['Turnover'],
    'Cumulative Turnover': cumulative_turnover,
    'Annualized Turnover': results['Turnover'] * 12
}, index=results.index)

with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    # Sheet 1: Monthly returns for all strategies
    results[['Portfolio', 'Equal Weight', 'Net Return']].to_excel(writer, sheet_name='Monthly Returns')
    
    # Sheet 2: Cumulative returns over time
    cumulative_returns.to_excel(writer, sheet_name='Cumulative Returns')
    
    # Sheet 3: Performance statistics
    stats.to_excel(writer, sheet_name='Statistics')
    
    # Sheet 4: Net returns analysis
    pd.DataFrame({
        'Net Return': results['Net Return'],
        'Cumulative Net': cumulative_net
    }).to_excel(writer, sheet_name='Net Returns')
    
    # Sheet 5: Turnover analysis
    turnover_analysis.to_excel(writer, sheet_name='Turnover Analysis')
    
    # Access workbook and worksheet objects for formatting
    workbook = writer.book
    
    # Format date columns in all sheets
    date_format = workbook.add_format({'num_format': 'dd-mmm-yyyy'})
    
    for sheet_name in ['Monthly Returns', 'Cumulative Returns', 'Net Returns', 'Turnover Analysis']:
        worksheet = writer.sheets[sheet_name]
        # Set date column format (column A, index 0)
        worksheet.set_column(0, 0, 15, date_format)
        # Set other columns width for better visibility
        worksheet.set_column(1, 10, 12)

print(f"\nResults saved to {output_file}")

# ===============================
# DETAILED PERFORMANCE REPORTING
# ===============================

# Print performance statistics table
print("\nPortfolio Statistics:")
print("--------------------")
print(stats)

# Calculate additional net return statistics
net_returns = results['Net Return']
positive_months = (net_returns > 0).sum()
negative_months = (net_returns < 0).sum()
avg_positive = net_returns[net_returns > 0].mean() * 100  # Convert to percentage
avg_negative = net_returns[net_returns < 0].mean() * 100  # Convert to percentage

# Print detailed net return analysis
print("\nNet Return Analysis:")
print("-------------------")
print(f"Positive Months: {positive_months} ({positive_months/len(net_returns)*100:.1f}%)")
print(f"Negative Months: {negative_months} ({negative_months/len(net_returns)*100:.1f}%)")
print(f"Average Positive Return: {avg_positive:.2f}%")
print(f"Average Negative Return: {avg_negative:.2f}%")
print(f"Win/Loss Ratio: {abs(avg_positive/avg_negative):.2f}")  # Ratio of avg win to avg loss

# Print turnover analysis
turnover_stats = results['Turnover'].dropna()
print("\nTurnover Analysis:")
print("-----------------")
print(f"Average Monthly Turnover: {turnover_stats.mean()*100:.2f}%")
print(f"Annualized Turnover: {turnover_stats.mean()*12*100:.2f}%")
print(f"Maximum Monthly Turnover: {turnover_stats.max()*100:.2f}%")
print(f"Minimum Monthly Turnover: {turnover_stats.min()*100:.2f}%")
print(f"Turnover Volatility: {turnover_stats.std()*100:.2f}%")
print(f"Total Cumulative Turnover: {cumulative_turnover.iloc[-1]*100:.2f}%")

# Print most recent net returns
print("\nMost Recent Net Returns:")
print(results['Net Return'].tail())

# Print most recent turnover
print("\nMost Recent Turnover:")
print((results['Turnover'] * 100).tail())

# Calculate and print correlation with benchmark
correlation = results[['Portfolio', 'Equal Weight']].corr().iloc[0,1]
print(f"\nCorrelation with Equal Weight: {correlation:.2f}")

# Calculate tracking error (annualized standard deviation of active returns)
tracking_error = (results['Portfolio'] - results['Equal Weight']).std() * np.sqrt(12) * 100
print(f"Tracking Error: {tracking_error:.2f}%")

# Calculate information ratio (active return / tracking error)
active_return = results['Portfolio'].mean() - results['Equal Weight'].mean()
info_ratio = (active_return * 12) / (tracking_error / 100)
print(f"Information Ratio: {info_ratio:.2f}")

# Print most recent performance
print("\nMost Recent Returns:")
print(results[['Portfolio', 'Equal Weight', 'Net Return', 'Turnover']].tail())
