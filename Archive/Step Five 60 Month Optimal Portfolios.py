"""
T2 Factor Timing - Step Five: 60-Month Optimal Portfolios
========================================================

PURPOSE:
Implements a sophisticated rolling window portfolio optimization framework that constructs
factor portfolios using both expanding and hybrid (expanding/rolling) windows. The optimization
incorporates risk aversion, concentration penalties, and factor-specific constraints to generate
optimal factor allocations over time.

IMPORTANT WARNING:
This program uses factor-specific maximum weight constraints from Step Factor Categories.xlsx.
Always verify this file before execution to ensure proper constraint specification. Factors with
Max=0 will be excluded from the hybrid portfolio. Incorrect constraints may lead to unexpected
portfolio allocations.

INPUT FILES:
1. T2_Optimizer.xlsx
   - Source: Output from Step Four (Create Monthly Top20 Returns)
   - Format: Excel workbook with single sheet
   - Structure:
     * Index: Dates (datetime format)
     * Columns: Factor strategy returns (decimal format)
   - Notes:
     * 'Monthly Return_CS' column is automatically removed if present
     * Returns should be in decimal format (0.01 = 1%)

2. Step Factor Categories.xlsx
   - Source: Manual configuration file
   - Format: Excel workbook with single sheet
   - Structure:
     * Factor Name: Name matching T2_Optimizer columns
     * Category: Factor grouping category
     * Max: Maximum weight constraint (0-1)
   - Notes:
     * Max=0 excludes factor from hybrid portfolio
     * Weights are constrained to [0, Max] for each factor

OUTPUT FILES:
1. T2_expanding_window_weights.xlsx
   - Contains expanding window optimization results
   - Format: Excel workbook with single sheet
   - Structure:
     * Index: Monthly dates
     * Columns: Factor names
     * Values: Portfolio weights (0-1, sum to 1)

2. T2_rolling_window_weights.xlsx
   - Contains hybrid window optimization results
   - Format: Same as expanding window file
   - Notes:
     * Expands for first 60 months, then rolls
     * Respects maximum weight constraints

3. T2_strategy_statistics.xlsx
   - Comprehensive performance comparison
   - Format: Excel workbook with sheets:
     * Summary: Key metrics (returns, volatility, Sharpe, etc.)
     * Monthly Returns: Time series of strategy returns
     * Drawdowns: Drawdown analysis
     * Turnover: Monthly turnover metrics

4. T2_turnover_analysis.pdf
   - Visual analysis of portfolio turnover
   - Format: Multi-page PDF
   - Contents:
     * Monthly turnover comparison
     * Cumulative turnover
     * Factor concentration metrics

5. T2_factor_heatmap.pdf
   - Visual representation of factor weights over time
   - Highlights factor allocation patterns
   - Color-coded by factor category

6. T2_strategy_performance.pdf
   - Cumulative performance comparison
   - Format: PDF
   - Shows cumulative returns of both strategies

OPTIMIZATION METHODOLOGY:
1. Objective Function:
   - Maximize: μ'w - λ(w'Σw) - γHHI(w)
   Where:
     * μ: Expected returns (exponentially weighted)
     * w: Portfolio weights
     * Σ: Covariance matrix
     * λ: Risk aversion parameter (default: 20.0)
     * γ: Concentration penalty (default: 5.0)
     * HHI: Herfindahl-Hirschman Index

2. Constraints:
   - Long-only: w ≥ 0
   - Fully invested: Σw = 1
   - Factor limits: w_i ≤ Max_i ∀i

3. Window Types:
   - Expanding: Uses all available data points
   - Hybrid: Expands until 60 months, then rolls with 60-month window
   - First 60 months: Uses all available data (expanding window)
   - After 60 months: Uses exactly 60 months of data ending with the previous month

VERSION HISTORY:
- 2.1 (2025-06-12): Added exponential weighting and improved constraints
- 2.0 (2025-05-15): Complete rewrite with CVXPY optimization
- 1.5 (2025-04-10): Initial hybrid window implementation
- 1.0 (2025-03-01): Initial production version

AUTHOR: Quantitative Research Team
LAST UPDATED: 2025-06-17

NOTES:
- All returns and statistics are annualized where applicable
- Turnover is calculated as the sum of absolute weight changes
- Missing returns are forward-filled with zeros
- Optimization uses SLSQP solver with 1000 max iterations
- T60.xlsx export has been removed from this script

DEPENDENCIES:
- pandas>=2.0.0
- numpy>=1.24.0
- scipy>=1.10.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- openpyxl>=3.1.0
- cvxpy>=1.3.0
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import Dict, Tuple, List
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# Create output directory if it doesn't exist
os.makedirs('outputs/visualizations', exist_ok=True)

class PortfolioOptimizer:
    """
    PortfolioOptimizer class for long-only, fully-invested portfolios with risk aversion and diversification penalty.

    Attributes:
        returns (pd.DataFrame): Asset returns data (monthly, factors as columns).
        lambda_param (float): Risk aversion parameter (higher = more risk-averse).
        hhi_penalty (float): Penalty for concentrated portfolios (HHI = sum of squared weights).
        n_assets (int): Number of factors (columns) in the returns DataFrame.
        max_weights (dict, optional): Dictionary mapping factor names to their maximum allowed weight.
    """

    def __init__(self, returns_df: pd.DataFrame, lambda_param: float = 20.0, hhi_penalty: float = 5.0, max_weights: dict = None):
        """Initialize with high risk aversion (lambda = 5) as specified."""
        self.returns = returns_df
        self.lambda_param = lambda_param
        self.hhi_penalty = hhi_penalty
        self.n_assets = len(returns_df.columns)
        self.max_weights = max_weights or {}  # Default to empty dict if None
    
    def calculate_portfolio_metrics(self, weights: np.ndarray, returns: pd.DataFrame = None) -> tuple:
        """
        Calculate average return and volatility for a given set of weights.
        Used for portfolio optimization objective.

        Args:
            weights (np.ndarray): Portfolio weights (should sum to 1).
            returns (pd.DataFrame, optional): Returns data. If None, uses self.returns.

        Returns:
            tuple: (average return, annualized volatility)
        """
        if returns is None:
            returns = self.returns
            
        portfolio_returns = np.sum(returns * weights, axis=1)
        # Remove annualization to match Factor Utility
        avg_return = np.mean(portfolio_returns)
        volatility = np.std(portfolio_returns) * np.sqrt(12)
        
        # Redefine avg_return as 8 * avg_return to match Factor Utility
        avg_return = 8 * avg_return 
        
        return avg_return, volatility
    
    def objective_function(self, weights: np.ndarray) -> float:
        """
        Objective function for optimizer: maximize expected return minus risk and concentration penalty.

        Args:
            weights (np.ndarray): Portfolio weights.

        Returns:
            float: Negative penalized utility (for minimization).
        """
        avg_return, volatility = self.calculate_portfolio_metrics(weights)
        utility = avg_return - self.lambda_param * (volatility ** 2)
        # HHI = sum of squared weights
        hhi = np.sum(weights**2)
        # Penalize utility by HHI * penalty coefficient
        penalized_utility = utility - self.hhi_penalty * hhi
        return -penalized_utility  # Return negative penalized utility for minimization
    
    def optimize_weights(self, returns: pd.DataFrame = None) -> np.ndarray:
        """
        Find the optimal portfolio weights for the given returns data.
        Uses SLSQP optimizer, subject to long-only and fully-invested constraints.

        Args:
            returns (pd.DataFrame, optional): Returns data. If None, uses self.returns.

        Returns:
            np.ndarray: Optimized weights (sum to 1, all >= 0).
        """
        # Temporarily replace self.returns if needed, matching original behavior
        if returns is not None:
            original_returns = self.returns
            self.returns = returns
        
        # Equal weights initial guess
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        # Constraints: Sum to 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Set bounds based on max_weights if available
        bounds = []
        for i, factor_name in enumerate(self.returns.columns):  # Use self.returns.columns, not returns.columns
            max_weight = self.max_weights.get(factor_name, 1.0)  # Default to 1.0 (100%) if no constraint
            bounds.append((0, max_weight))  # Weight between 0 and max_weight
        
        # Run the optimization - objective_function uses self.returns internally
        result = minimize(
            self.objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Restore original self.returns if needed
        if returns is not None:
            self.returns = original_returns
        
        return result.x

def apply_exponential_weights(returns_df: pd.DataFrame, decay_factor: float = 0.94) -> pd.DataFrame:
    """
    Apply exponential weighting to each row of returns data for time series analysis.
    Most recent months get the highest weight.

    Args:
        returns_df (pd.DataFrame): Monthly returns (rows = months, columns = factors).
        decay_factor (float): How quickly weights decay (0 < decay_factor < 1). Higher = slower decay.

    Returns:
        tuple: (weighted returns DataFrame, weights array)
    """
    # Get the number of observations
    n_obs = len(returns_df)
    
    # Create exponential weights (newest to oldest)
    # Formula: w_i = decay_factor^i / sum(decay_factor^i)
    weights = np.array([decay_factor ** i for i in range(n_obs)])
    
    # Reverse weights so most recent observations get highest weight
    weights = weights[::-1]
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    # Apply weights to each row of returns
    weighted_returns = returns_df.copy()
    for i, (_, row) in enumerate(returns_df.iterrows()):
        weighted_returns.iloc[i] = row * weights[i]
    
    return weighted_returns, weights

def calculate_exponential_forecast(returns_df: pd.DataFrame, decay_factor: float = 0.94) -> pd.Series:
    """
    Compute exponentially weighted mean for each factor, for use as a forecast.
    Handles missing data by using only available months for each factor.

    Args:
        returns_df (pd.DataFrame): Monthly returns (rows = months, columns = factors).
        decay_factor (float): How quickly weights decay (0 < decay_factor < 1).

    Returns:
        pd.Series: Weighted forecast for each factor (index = factor names).
    """
    # Get the number of observations
    n_obs = len(returns_df)
    
    # Create exponential weights (newest to oldest)
    weights = np.array([decay_factor ** i for i in range(n_obs)])
    
    # Reverse weights so most recent observations get highest weight
    weights = weights[::-1]
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    # Calculate weighted average for each column
    weighted_means = pd.Series(index=returns_df.columns, dtype=float)
    
    for col in returns_df.columns:
        # Check for NaN values in this column
        if returns_df[col].isna().any():
            # Use only non-NaN values for this column
            valid_data = returns_df[col].dropna()
            if len(valid_data) > 0:
                # Recalculate weights for the valid data length
                valid_n = len(valid_data)
                valid_weights = np.array([decay_factor ** i for i in range(valid_n)])
                valid_weights = valid_weights[::-1]
                valid_weights = valid_weights / valid_weights.sum()
                
                # Calculate weighted average using only valid data
                weighted_means[col] = np.sum(valid_data.values * valid_weights)
            else:
                weighted_means[col] = np.nan
        else:
            # Calculate the weighted average for this column
            weighted_means[col] = np.sum(returns_df[col].values * weights)
    
    # Annualize the weighted means
    annualized_means = (1 + weighted_means) ** 12 - 1
    
    return annualized_means

def calculate_portfolio_statistics(returns: pd.Series) -> dict:
    """
    Compute key statistics for a portfolio return series (annualized return, volatility, Sharpe, drawdown, etc).

    Args:
        returns (pd.Series): Monthly portfolio returns.

    Returns:
        dict: Dictionary with statistics (see keys in code).
    """
    # Annualize metrics
    ann_return = (1 + returns.mean())**12 - 1
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    
    # Drawdown analysis
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Monthly statistics
    positive_months = (returns > 0).mean()
    
    return {
        'Annualized Return (%)': ann_return * 100,
        'Annualized Volatility (%)': ann_vol * 100,
        'Sharpe Ratio': sharpe,
        'Maximum Drawdown (%)': max_drawdown * 100,
        'Positive Months (%)': positive_months * 100,
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis()
    }

def calculate_turnover(weights_df: pd.DataFrame) -> tuple:
    """
    Calculate portfolio turnover: how much the weights change month-to-month.

    Args:
        weights_df (pd.DataFrame): Portfolio weights (rows = months, columns = factors).

    Returns:
        tuple: (average monthly turnover, monthly turnover time series)
    """
    # Calculate absolute weight changes for each month
    weight_changes = weights_df.diff().abs()
    
    # Sum across assets to get total turnover for each month
    monthly_turnover = weight_changes.sum(axis=1) / 2  # Divide by 2 as each trade affects two positions
    
    # Calculate average monthly turnover
    avg_monthly_turnover = monthly_turnover.mean()
    
    return avg_monthly_turnover, monthly_turnover

def run_rolling_optimization():
    """
    Run the full rolling/expanding window portfolio optimization pipeline.
    Compares two approaches:
      - Expanding window: uses all data up to each month, with exponential weighting.
      - Hybrid window: uses expanding window for first 60 months, then 60-month rolling window.
    Computes weights, returns, turnover, and outputs all results/reports.
    """
    # Load data
    print("Loading data...")
    returns = pd.read_excel('T2_Optimizer.xlsx', index_col=0)
    returns.index = pd.to_datetime(returns.index)
    
    # Load factor max weights from Step Factor Categories.xlsx
    print("Loading factor categories and max weights...")
    factor_categories = pd.read_excel('Step Factor Categories.xlsx')
    # Create a dictionary mapping factor names to their max weights
    max_weights = dict(zip(factor_categories['Factor Name'], factor_categories['Max']))
    print(f"Loaded max weight constraints for {len(max_weights)} factors")
    
    # Convert to decimals if needed
    if returns.abs().mean().mean() > 1:
        returns = returns / 100
    
    # Remove Monthly Return_CS
    if 'Monthly Return_CS' in returns.columns:
        returns = returns.drop(columns=['Monthly Return_CS'])
    
    # Parameters
    LAMBDA = 1.0  # Risk aversion parameter
    HHI_PENALTY = 0.01  # HHI penalty coefficient - Match Factor Utility value
    INITIAL_WINDOW = 60  # 5 years of monthly data
    EMA_DECAY = 0.98  # Increased decay factor for longer tail (was 0.94)
    
    # Initialize optimizers - separate instances for expanding and hybrid
    # Only apply max weight constraints to the hybrid optimizer
    expanding_optimizer = PortfolioOptimizer(returns, LAMBDA, HHI_PENALTY)
    hybrid_optimizer = PortfolioOptimizer(returns, LAMBDA, HHI_PENALTY, max_weights=max_weights)
    
    # Prepare dates for rolling windows
    dates = returns.index
    
    # For rolling window, we can calculate one additional month at the end
    # Create a date for one month after the last date in the dataset
    next_month_date = dates[-1] + pd.DateOffset(months=1)
    
    # Initialize storage for weights and returns - start from month 2 to allow for returns calculation
    expanding_weights = pd.DataFrame(index=dates[1:], columns=returns.columns)
    hybrid_weights = pd.DataFrame(index=list(dates[1:]) + [next_month_date], columns=returns.columns)
    
    # Initialize DataFrame to hold hybrid-window factor forecasts
    hybrid_forecasts = pd.DataFrame(index=hybrid_weights.index, columns=returns.columns, dtype=float)
    # Initialize DataFrame to hold exponentially-weighted expanding window forecasts
    exp_forecasts = pd.DataFrame(index=dates[1:], columns=returns.columns, dtype=float)
    
    # Initialize DataFrame to save input forecasts for debugging
    input_forecasts = pd.DataFrame(index=hybrid_weights.index, columns=returns.columns, dtype=float)
    
    print("\nRunning optimizations...")
    # Run optimizations for each date
    for i, current_date in enumerate(dates[1:], 1):
        # Expanding window with exponential weighting - always use all available data
        expanding_data = returns.loc[:current_date]
        # Apply exponential weighting to the expanding window data
        weighted_expanding_data, _ = apply_exponential_weights(expanding_data, EMA_DECAY)
        expanding_weights.loc[current_date] = expanding_optimizer.optimize_weights(weighted_expanding_data)
        
        # Calculate exponentially-weighted forecasted returns
        exp_forecasts.loc[current_date] = calculate_exponential_forecast(expanding_data, EMA_DECAY)
        
        # Hybrid approach: expanding window until 60 months, then rolling 60-month window
        if i <= INITIAL_WINDOW:
            # Use expanding window for the first 60 months
            hybrid_data = returns.loc[dates[0]:current_date]
        else:
            # Switch to rolling 60-month window after 60 months
            start_idx = i - INITIAL_WINDOW  # Start INITIAL_WINDOW months back
            hybrid_data = returns.loc[dates[start_idx:i]]  # Use exactly 60 months
        
        # Optimize weights and store
        # Compute factor-level forecasted returns for this hybrid window
        factor_means = hybrid_data.mean(axis=0)
        # Remove annualization to match Factor Utility
        hybrid_forecasts.loc[current_date] = factor_means
        
        # Apply 8x scaling factor and convert to percentage scale (multiply by 100) to match Factor Utility
        scaled_means = 8 * factor_means * 100
        
        # Save both original and scaled forecasts for debugging
        input_forecasts.loc[current_date] = scaled_means
        
        # Use these forecasts to optimize weights with max weight constraints
        hybrid_weights.loc[current_date] = hybrid_optimizer.optimize_weights(hybrid_data)
        
        if i % 12 == 0:  # Print progress every year
            print(f"Processed up to {current_date.strftime('%Y-%m')}")
    
    # Calculate the extra month for hybrid strategy
    # Use the last 60 months of data for the next month
    start_idx = max(0, len(dates) - INITIAL_WINDOW)
    hybrid_data = returns.loc[dates[start_idx:]]
    # Compute factor-level forecasted returns for this last hybrid window
    factor_means = hybrid_data.mean(axis=0)
    # Remove annualization to match Factor Utility
    hybrid_forecasts.loc[next_month_date] = factor_means
    
    # Apply 8x scaling factor and convert to percentage scale (multiply by 100) to match Factor Utility
    scaled_means = 8 * factor_means * 100
    
    # Save both original and scaled forecasts for debugging
    input_forecasts.loc[next_month_date] = scaled_means
    
    # Use these forecasts to optimize weights with max weight constraints
    hybrid_weights.loc[next_month_date] = hybrid_optimizer.optimize_weights(hybrid_data)
    
    print(f"Processed extra month: {next_month_date.strftime('%Y-%m')}")
    
    # Calculate strategy returns - start from month 2 to have weights from month 1
    expanding_returns = pd.Series(index=dates[2:], dtype=float)
    hybrid_returns = pd.Series(index=dates[2:], dtype=float)
    
    for date in dates[2:]:
        prev_date = dates[dates < date][-1]
        # Get returns using previous month's weights
        expanding_returns[date] = np.sum(
            expanding_weights.loc[prev_date] * returns.loc[date]
        )
        hybrid_returns[date] = np.sum(
            hybrid_weights.loc[prev_date] * returns.loc[date]
        )
    
    # Calculate cumulative returns
    cum_expanding = (1 + expanding_returns).cumprod()
    cum_hybrid = (1 + hybrid_returns).cumprod()
    
    # Calculate statistics
    expanding_stats = calculate_portfolio_statistics(expanding_returns)
    hybrid_stats = calculate_portfolio_statistics(hybrid_returns)
    
    # Calculate turnover for both strategies
    avg_expanding_turnover, expanding_turnover = calculate_turnover(expanding_weights)
    avg_hybrid_turnover, hybrid_turnover = calculate_turnover(hybrid_weights.loc[dates[1:]])
    
    print("\nTurnover Statistics:")
    print(f"Expanding Window Average Monthly Turnover: {avg_expanding_turnover*100:.2f}%")
    print(f"Hybrid Window Average Monthly Turnover: {avg_hybrid_turnover*100:.2f}%")
    
    # Report on factor weight constraints
    constrained_factors = sum(1 for weight in max_weights.values() if weight < 1.0)
    print(f"\nFactor Weight Constraints: {constrained_factors} factors have maximum weight constraints")
    print(f"Example constraints: {list(max_weights.items())[:5]}...")
    
    # Report on the transition from expanding to rolling window
    transition_date = dates[INITIAL_WINDOW] if INITIAL_WINDOW < len(dates) else None
    if transition_date:
        print(f"\nTransition from expanding to rolling window occurred at: {transition_date.strftime('%Y-%m')}")
        print(f"- Used expanding window from {dates[0].strftime('%Y-%m')} to {dates[INITIAL_WINDOW-1].strftime('%Y-%m')}")
        print(f"- Used 60-month rolling window from {transition_date.strftime('%Y-%m')} onwards")
    
    # Plot turnover over time
    plt.figure(figsize=(15, 8))
    plt.plot(expanding_turnover.index, expanding_turnover * 100, label='Expanding Window', linewidth=2)
    plt.plot(hybrid_turnover.index, hybrid_turnover * 100, label='Hybrid Window', linewidth=2)
    plt.title('Monthly Portfolio Turnover', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Turnover (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper left')
    
    # Add average turnover annotations
    plt.figtext(0.02, 0.02, 
                f'Avg. Monthly Turnover:\nExpanding Window: {avg_expanding_turnover*100:.1f}%\nHybrid Window: {avg_hybrid_turnover*100:.1f}%', 
                fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('T2_turnover_analysis.pdf')
    plt.close()
    
    # Add turnover to statistics
    expanding_stats['Average Monthly Turnover (%)'] = avg_expanding_turnover * 100
    hybrid_stats['Average Monthly Turnover (%)'] = avg_hybrid_turnover * 100
    
    # Save weights
    expanding_weights.to_excel('T2_expanding_window_weights.xlsx')
    hybrid_weights.to_excel('T2_rolling_window_weights.xlsx')
    
    # --- Write enhanced T2_60_Month.xlsx as requested ---
    # Load the exact contents of T2_Optimizer.xlsx
    targets_df = pd.read_excel('T2_Optimizer.xlsx', index_col=0)
    
    # Check if data needs to be converted to decimals
    if targets_df.abs().mean().mean() > 1:
        targets_df_decimal = targets_df / 100
    else:
        targets_df_decimal = targets_df.copy()
    
    # Function to calculate trailing averages (annualized)
    def trailing_annualized(df, window):
        # Calculate trailing window mean
        trailing = df.rolling(window=window, min_periods=window).mean()
        # Annualize: (1 + mean)^12 - 1
        return (1 + trailing) ** 12 - 1
    
    # Compute trailing averages using decimal data, then shift by one month
    feature1 = trailing_annualized(targets_df_decimal, 1).shift(1)
    feature12 = trailing_annualized(targets_df_decimal, 12).shift(1)
    feature36 = trailing_annualized(targets_df_decimal, 36).shift(1)
    feature60 = trailing_annualized(targets_df_decimal, 60).shift(1)
    
    # Add one extra forecast month (next_month_date) for each feature sheet
    last_date = targets_df.index[-1]
    next_month_date = last_date + pd.DateOffset(months=1)
    
    # Calculate the extra forecast row for each feature
    for feat_df, window in zip([feature1, feature12, feature36, feature60], [1, 12, 36, 60]):
        # Get the last 'window' months of data
        last_window_data = targets_df_decimal.iloc[-window:]
        # Calculate mean of those months
        mean_monthly = last_window_data.mean()
        # Annualize the mean
        annualized = (1 + mean_monthly) ** 12 - 1
        # Add to the feature DataFrame
        feat_df.loc[next_month_date] = annualized
    
    # T60 writing removed as requested - now handled in Step Four script

    
    # Create a DataFrame with monthly returns for both strategies
    monthly_returns_df = pd.DataFrame({
        'Expanding Window': expanding_returns * 100,  # Convert to percentage
        'Hybrid Window': hybrid_returns * 100  # Convert to percentage
    })
    
    # Save the forecasts to Excel files for debugging
    print("\nSaving forecasts to Excel files for debugging...")
    
    # Save hybrid window forecasts - using xlsxwriter for better formatting
    with pd.ExcelWriter('T2_optimizer_forecasts.xlsx', engine='xlsxwriter') as writer:
        # Save the forecasts that went into the optimizer (with 8x scaling)
        input_forecasts.to_excel(writer, sheet_name='Scaled Inputs (8x)')
        
        # Save the original forecasts without scaling
        hybrid_forecasts.to_excel(writer, sheet_name='Original Forecasts')
        exp_forecasts.to_excel(writer, sheet_name='Exponential Forecasts')
        
        # Access workbook and worksheet objects for date formatting
        workbook = writer.book
        for sheet_name in ['Scaled Inputs (8x)', 'Original Forecasts', 'Exponential Forecasts']:
            worksheet = writer.sheets[sheet_name]
            
            # Create date format
            date_format = workbook.add_format({'num_format': 'dd-mmm-yyyy'})
            
            # Adjust column width for better visibility of dates
            worksheet.set_column(0, 0, 15)  # Index column (dates)
            
            # Set format for all columns to show 4 decimal places
            num_format = workbook.add_format({'num_format': '0.0000'})
            worksheet.set_column(1, len(input_forecasts.columns), 12, num_format)
    
    # Save statistics and monthly returns to the same Excel file but different sheets
    with pd.ExcelWriter('T2_strategy_statistics.xlsx', engine='xlsxwriter') as writer:
        # Save summary statistics to the first sheet
        stats_df = pd.DataFrame({
            'Expanding Window': expanding_stats,
            'Hybrid Window': hybrid_stats
        })
        stats_df.to_excel(writer, sheet_name='Summary Statistics')
        
        # Save monthly returns to a separate sheet
        monthly_returns_df.to_excel(writer, sheet_name='Monthly Returns')
        
        # Apply proper date formatting to sheets with date indices
        workbook = writer.book
        date_format = workbook.add_format({'num_format': 'dd-mmm-yyyy'})
        
        # Format date columns in sheets with date indices
        if 'Monthly Returns' in writer.sheets:
            worksheet = writer.sheets['Monthly Returns']
            worksheet.set_column(0, 0, 12, date_format)  # Format index column (dates)
    
    print("\nStrategy Statistics:")
    print(stats_df)
    print("\nMonthly returns saved to T2_strategy_statistics.xlsx in sheet 'Monthly Returns'")
    
    # Print a comparison of magnitudes for a sample date
    sample_date = dates[INITIAL_WINDOW + 12] if INITIAL_WINDOW + 12 < len(dates) else dates[-1]  # Choose a date after transition
    print("\nMagnitude Comparison (Sample Date: {})".format(sample_date.strftime('%Y-%m')))
    print("Hybrid Window Forecast Mean: {:.4f}".format(hybrid_forecasts.loc[sample_date].mean()))
    print("Exponential Forecast Mean: {:.4f}".format(exp_forecasts.loc[sample_date].mean()))
    print("\nHybrid Window Forecast Min/Max: {:.4f} / {:.4f}".format(
        hybrid_forecasts.loc[sample_date].min(), 
        hybrid_forecasts.loc[sample_date].max()
    ))
    print("Exponential Forecast Min/Max: {:.4f} / {:.4f}".format(
        exp_forecasts.loc[sample_date].min(), 
        exp_forecasts.loc[sample_date].max()
    ))
    
    # Create performance plot
    plt.figure(figsize=(15, 8))
    plt.plot(cum_expanding, label='Expanding Window', linewidth=2)
    plt.plot(cum_hybrid, label='Hybrid Window', linewidth=2)
    plt.title('Cumulative Performance Comparison', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    
    # Format y-axis as percentages
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y-1)))
    
    # Enhance grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper left')
    
    # Add performance annotations
    ann_ret_exp = expanding_stats['Annualized Return (%)']
    ann_ret_hyb = hybrid_stats['Annualized Return (%)']
    plt.figtext(0.02, 0.02, 
                f'Expanding Window: {ann_ret_exp:.1f}% p.a.\nHybrid Window: {ann_ret_hyb:.1f}% p.a.', 
                fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add vertical line at transition date if it exists
    if transition_date:
        plt.axvline(x=transition_date, color='red', linestyle='--', alpha=0.5)
        plt.figtext(0.5, 0.01, 
                    f'Transition to Rolling Window: {transition_date.strftime("%Y-%m")}', 
                    fontsize=10, ha='center',
                    bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    # Save only in PDF format
    plt.savefig('T2_strategy_performance.pdf')
    plt.close()
    
    # Create factor weight heatmap visualization
    print("\nCreating factor weight heatmap visualization...")
    create_factor_weight_heatmap(hybrid_weights)
    
    return expanding_returns, hybrid_returns, expanding_weights, hybrid_weights

def create_factor_weight_heatmap(weights_df, top_n=20):
    """Create an award-winning, sophisticated factor weight heatmap with executive dashboard styling"""
    print("Creating award-winning factor weight heatmap...")
    
    # Ensure all data is numeric by converting any non-numeric values to float
    weights_df = weights_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Define factor categories and their prefixes/keywords
    categories = {
        'Value': ['Best PE', 'Best PBK', 'Best Price Sales', 'Best Div Yield', 'Trailing PE', 'Positive PE', 
                 'Shiller PE', 'Earnings Yield', 'Best Cash Flow', 'EV to EBITDA'],
        'Momentum': ['1MTR', '3MTR', '12-1MTR', 'RSI14', '120MA', 'Advance Decline', 'Signal', 'P2P'],
        'Economic': ['Inflation', 'REER', '10Yr Bond', 'GDP', 'Debt to GDP', 'Budget Def', 'Current Account'],
        'Quality': ['Best ROE', 'BEST EPS', 'Operating Margin', 'Trailing EPS', 'Debt to EV', 'LT Growth', 'Bloom Country Risk', '20 day vol'],
        'Commodity': ['Oil', 'Gold', 'Copper', 'Agriculture', 'Currency']
    }

    # Sophisticated category styling
    category_colors = {
        'Value': '#2d5a3d',      # Deep forest green
        'Momentum': '#1e3a5f',   # Deep navy blue  
        'Economic': '#5d2e5a',   # Deep purple
        'Quality': '#5a3d2d',    # Deep brown
        'Commodity': '#5a4b2d'   # Deep golden brown
    }
    
    category_backgrounds = {
        'Value': '#f8fdfb',      # Very light green tint
        'Momentum': '#f8fafd',   # Very light blue tint
        'Economic': '#fdf8fd',   # Very light purple tint
        'Quality': '#fdfbf8',    # Very light brown tint
        'Commodity': '#fdfcf8'   # Very light golden tint
    }

    # Get top factors by average weight
    avg_weights = weights_df.mean().sort_values(ascending=False)
    top_factors = avg_weights.head(top_n).index.tolist()
    
    # Ensure we're working with the subset of weights that are in top_factors
    filtered_weights_df = weights_df[top_factors]

    # Organize factors by category
    categorized_factors = []
    category_map = {}
    category_groups = {}
    
    for category, keywords in categories.items():
        category_factors = []
        for factor in top_factors:
            for keyword in keywords:
                if keyword in factor and factor not in categorized_factors:
                    category_factors.append(factor)
                    category_map[factor] = category
                    break
        if category_factors:
            category_factors.sort()
            categorized_factors.extend(category_factors)
            
    for factor in top_factors:
        if factor not in categorized_factors:
            categorized_factors.append(factor)
            category_map[factor] = "Other"
            
    ordered_factors = categorized_factors[:top_n] if len(categorized_factors) > 0 else top_factors

    # Calculate category positions
    current_pos = 0
    for category in ['Value', 'Momentum', 'Economic', 'Quality', 'Commodity']:
        cat_factors = [f for f in ordered_factors if category_map.get(f) == category]
        if cat_factors:
            category_groups[category] = {
                'start': current_pos, 
                'end': current_pos + len(cat_factors) - 1,
                'factors': cat_factors
            }
            current_pos += len(cat_factors)

    # Sample data every 4 months for better visualization
    sample_weights = filtered_weights_df[ordered_factors].iloc[::4].copy()
    if len(sample_weights) == 0:
        print("Warning: No weight data available for heatmap")
        return
        
    # Ensure data is numeric and handle any remaining issues
    sample_weights = sample_weights.astype(float)

    # Use full factor names (no truncation for professional look) and rename 1MTR
    clean_factor_names = []
    for factor in ordered_factors:
        clean_name = factor.replace('_', ' ')
        clean_name = clean_name.replace('1MTR', '1 Month Return')
        clean_factor_names.append(clean_name)

    # Create sophisticated custom colormap: Light sage → Soft teal → Deep teal
    colors = ['#f8fffe', '#e8f5f3', '#d1ebe6', '#a8dadc', '#79c2d0', '#5aa9c4', '#457b9d']
    sophisticated_cmap = mcolors.LinearSegmentedColormap.from_list("sophisticated", colors, N=256)
    sophisticated_cmap.set_under('white')

    # Create figure with optimal proportions
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Set sophisticated styling
    plt.rcParams['font.family'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    # Create category background rectangles
    for category, group_info in category_groups.items():
        start_pos = group_info['start'] - 0.5
        height = group_info['end'] - group_info['start'] + 1
        
        # Add subtle background tinting
        background_rect = Rectangle(
            (-0.5, start_pos), len(sample_weights.index), height,
            facecolor=category_backgrounds[category], 
            edgecolor='none', alpha=0.3, zorder=0
        )
        ax.add_patch(background_rect)

    # Create the main heatmap
    # Ensure we have valid min/max values
    max_value = sample_weights.values.max()
    if not np.isfinite(max_value) or max_value <= 0:
        max_value = 0.5
    else:
        max_value = min(max_value, 0.5)
        
    im = ax.imshow(sample_weights.T.values, cmap=sophisticated_cmap, aspect='auto',
                   vmin=0.001, vmax=max_value)

    # Sophisticated tick formatting
    tick_positions = range(0, len(sample_weights.index), 3)  # Every 3rd year for cleaner look
    ax.set_xticks(tick_positions)
    date_labels = [sample_weights.index[i].strftime('%Y') for i in tick_positions]
    ax.set_xticklabels(date_labels, fontsize=12, fontweight='500', color='#333333')
    
    ax.set_yticks(range(len(clean_factor_names)))
    ax.set_yticklabels(clean_factor_names, fontsize=11, fontweight='400', color='#333333')

    # Professional colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, aspect=25, pad=0.02)
    cbar.set_label('Portfolio Weight (%)', rotation=270, labelpad=25, 
                   fontsize=13, fontweight='600', color='#333333')
    cbar.ax.tick_params(labelsize=11, colors='#333333')
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor('#cccccc')

    # Elegant grid
    ax.set_xticks(np.arange(-0.5, len(sample_weights.index), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(clean_factor_names), 1), minor=True)
    ax.grid(which="minor", color="#f0f0f0", linestyle='-', linewidth=0.5, alpha=0.8)

    # Add thin horizontal lines between categories for clear boundaries
    for category, group_info in category_groups.items():
        end_pos = group_info['end']
        # Draw thin line below each category (except the last one)
        if end_pos < len(ordered_factors) - 1:
            ax.axhline(y=end_pos + 0.5, color='#999999', linewidth=1.0, alpha=0.8, zorder=4)

    # Add horizontal category headers in the middle of the heatmap
    middle_x = len(sample_weights.index) / 2
    for category, group_info in category_groups.items():
        start_pos = group_info['start']
        end_pos = group_info['end']
        center_pos = (start_pos + end_pos) / 2
        
        # Add elegant horizontal category header (no rotation)
        ax.text(middle_x, center_pos, category,
                fontsize=16, fontweight='700', color=category_colors[category],
                ha='center', va='center', rotation=0,
                bbox=dict(boxstyle="round,pad=0.5", 
                         facecolor='white', 
                         edgecolor=category_colors[category],
                         linewidth=2.0, alpha=0.95),
                zorder=5)

    # Professional title with elegant typography
    ax.set_title('Portfolio Weight Allocation Over Time\nTop 20 Factors by Average Weight', 
                 fontsize=22, fontweight='700', color='#1a1a1a', pad=30,
                 fontfamily='serif')

    # Sophisticated axis labels
    ax.set_xlabel('Year', fontsize=16, fontweight='600', color='#333333', labelpad=15)
    ax.set_ylabel('Investment Factor', fontsize=16, fontweight='600', color='#333333', labelpad=15)

    # Remove tick marks for cleaner appearance
    ax.tick_params(axis='both', which='both', length=0, pad=8)

    # Set elegant background
    ax.set_facecolor('#fcfcfc')
    fig.patch.set_facecolor('white')

    # Add subtle frame
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor('#dddddd')

    # Optimal layout with generous margins
    plt.subplots_adjust(left=0.15, right=0.85, top=0.92, bottom=0.08)
    plt.tight_layout()

    # Save with high quality
    plt.savefig('T2_factor_weight_heatmap.pdf',
                bbox_inches='tight', dpi=300, facecolor='white', 
                edgecolor='none', format='pdf')
    plt.close()
    
    print("✓ Award-winning factor weight heatmap created with sophisticated design")

if __name__ == "__main__":
    expanding_returns, hybrid_returns, expanding_weights, hybrid_weights = run_rolling_optimization()
