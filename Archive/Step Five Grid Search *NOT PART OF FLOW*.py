"""
Rolling Window Portfolio Optimization Analysis with HHI Penalty Variation

INPUT FILES:
1. T2_Optimizer.xlsx
   - Monthly returns data from Step Four
   - Format: Excel workbook with single sheet
   - Index: Dates in datetime format
   - Columns: Factor strategy returns
   - Values: Returns in decimal or percentage format
   - Note: 'Monthly Return_CS' column will be removed if present
2. T60.xlsx
   - Factor alpha forecasts data
   - Format: Excel workbook with single sheet
   - Index: Dates in datetime format
   - Columns: Factor alpha forecasts

OUTPUT FILES:
None - All file output operations have been removed. Analysis results are displayed in console only.

This module implements a rolling window portfolio optimization strategy with a loop over different
HHI penalty values:

1. Lambda (risk aversion): Fixed at 1.0
2. HHI penalty: Variable (0, 0.1, 0.2, 0.3)
3. Lookback period: Fixed at 60 months

The analysis outputs key metrics for each HHI penalty value:
- Annualized Returns
- Sharpe Ratios
- Portfolio Turnover
- Portfolio HHI (concentration)

Rolling Window Strategy:
- Uses a 60-month window of historical data 
- IMPORTANT: Uses an expanding window approach until reaching the specified lookback period
- Once enough data is available, uses exactly 60 months of data ending with the previous month
- No maximum limit on asset weights (can concentrate up to 100% in a single asset)

Dependencies:
- numpy
- pandas
- scipy.optimize
- matplotlib

Version: 1.5
Last Updated: 2024
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

class PortfolioOptimizer:
    """
    Portfolio optimization class.

    Attributes:
        returns (pd.DataFrame): Asset returns data.
        lambda_param (float): Risk aversion parameter.
        hhi_penalty (float): Penalty coefficient for HHI concentration.
        n_assets (int): Number of assets in the portfolio.
        alphas_df (pd.DataFrame): Factor alpha forecasts data.
    """

    def __init__(self, returns_df: pd.DataFrame, lambda_param: float = 1.0, hhi_penalty: float = 0.1, alphas_df: pd.DataFrame = None):
        """Initialize with specified lambda and HHI penalty values."""
        self.returns = returns_df
        self.lambda_param = lambda_param
        self.hhi_penalty = hhi_penalty
        self.n_assets = len(returns_df.columns)
        self.alphas_df = alphas_df
    
    def calculate_portfolio_metrics(self, weights: np.ndarray, returns: pd.DataFrame = None, current_date: pd.Timestamp = None) -> tuple:
        """
        Calculate portfolio metrics for given weights and optional returns data.

        Args:
            weights (np.ndarray): Portfolio weights.
            returns (pd.DataFrame, optional): Returns data. Defaults to None.
            current_date (pd.Timestamp, optional): Current date for alpha lookup. Defaults to None.

        Returns:
            tuple: Average return and volatility.
        """
        if returns is None:
            returns = self.returns
            
        portfolio_returns = np.sum(returns * weights, axis=1)
        volatility = np.std(portfolio_returns) * np.sqrt(12)
        
        # Use forecast returns from alphas if available and current_date is provided
        if self.alphas_df is not None and current_date is not None:
            # Get the alpha forecasts for the current date
            alpha_row = self.alphas_df[self.alphas_df['Date'] == current_date]
            if not alpha_row.empty:
                # Map factor names from returns columns to alpha column names
                factor_alphas = []
                for factor in returns.columns:
                    alpha_col = f"{factor}_PRED"
                    if alpha_col in alpha_row.columns:
                        factor_alphas.append(alpha_row[alpha_col].iloc[0])
                    else:
                        # Fallback to historical average if alpha not found
                        historical_avg = (1 + np.mean(returns[factor]))**12 - 1
                        factor_alphas.append(historical_avg)
                
                # Calculate portfolio forecast return using alphas
                factor_alphas = np.array(factor_alphas)
                avg_return = np.sum(weights * factor_alphas)
            else:
                # Fallback to historical average if date not found
                avg_return = (1 + np.mean(portfolio_returns))**12 - 1
        else:
            # Original calculation using historical average
            avg_return = (1 + np.mean(portfolio_returns))**12 - 1
        
        # Apply the 8x multiplier as in original code
        avg_return = 8 * avg_return 
        
        return avg_return, volatility
    
    def objective_function(self, weights: np.ndarray, current_date: pd.Timestamp = None) -> float:
        """
        Calculate negative utility (including HHI penalty).

        Args:
            weights (np.ndarray): Portfolio weights.
            current_date (pd.Timestamp, optional): Current date for alpha lookup. Defaults to None.

        Returns:
            float: Negative utility (including HHI penalty).
        """
        avg_return, volatility = self.calculate_portfolio_metrics(weights, current_date=current_date)
        utility = avg_return - self.lambda_param * (volatility ** 2)
        # HHI = sum of squared weights
        hhi = np.sum(weights**2)
        # Penalize utility by HHI * penalty coefficient
        penalized_utility = utility - self.hhi_penalty * hhi
        return -penalized_utility  # Return negative penalized utility for minimization
    
    def optimize_weights(self, returns: pd.DataFrame = None, current_date: pd.Timestamp = None) -> np.ndarray:
        """
        Optimize weights for given returns data.

        Args:
            returns (pd.DataFrame, optional): Returns data. Defaults to None.
            current_date (pd.Timestamp, optional): Current date for alpha lookup. Defaults to None.

        Returns:
            np.ndarray: Optimized weights.
        """
        if returns is not None:
            original_returns = self.returns
            self.returns = returns
            
        initial_weights = np.ones(self.n_assets) / self.n_assets
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        # Allow weights up to 100% in a single asset
        bounds = tuple((0, 1.0) for _ in range(self.n_assets))
        
        result = minimize(
            lambda x: self.objective_function(x, current_date),
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if returns is not None:
            self.returns = original_returns
            
        return result.x

def calculate_portfolio_statistics(returns: pd.Series) -> dict:
    """
    Calculate comprehensive portfolio statistics.

    Args:
        returns (pd.Series): Portfolio returns.

    Returns:
        dict: Portfolio statistics.
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
    Calculate average monthly turnover and turnover series.

    Args:
        weights_df (pd.DataFrame): Portfolio weights over time.

    Returns:
        tuple: Average monthly turnover and monthly turnover series.
    """
    # Calculate absolute weight changes for each month
    weight_changes = weights_df.diff().abs()
    
    # Sum across assets to get total turnover for each month
    monthly_turnover = weight_changes.sum(axis=1) / 2  # Divide by 2 as each trade affects two positions
    
    # Calculate average monthly turnover
    avg_monthly_turnover = monthly_turnover.mean()
    
    return avg_monthly_turnover, monthly_turnover

def calculate_portfolio_hhi(weights_df: pd.DataFrame) -> tuple:
    """
    Calculate average HHI (Herfindahl-Hirschman Index) concentration for the portfolio.
    
    Args:
        weights_df (pd.DataFrame): Portfolio weights over time.
        
    Returns:
        tuple: Average HHI and monthly HHI series.
    """
    # Calculate HHI for each month (sum of squared weights)
    monthly_hhi = (weights_df ** 2).sum(axis=1)
    
    # Calculate average HHI
    avg_hhi = monthly_hhi.mean()
    
    # Perfect diversification (equal weighting) would have HHI = 1/n
    n_assets = weights_df.shape[1]
    equal_weight_hhi = 1 / n_assets
    
    # Normalized HHI (scaled to 0-1, where 0 is perfect diversification)
    # normalized_hhi = (avg_hhi - equal_weight_hhi) / (1 - equal_weight_hhi)
    
    return avg_hhi, monthly_hhi

def run_rolling_optimization(returns: pd.DataFrame, lambda_param: float = 1.0, 
                             hhi_penalty: float = 0.1, lookback_period: int = 60, alphas_df: pd.DataFrame = None):
    """
    Run rolling window portfolio optimization analysis for specific parameter values.
    Uses an expanding window until enough data is available for the full lookback period.

    Args:
        returns (pd.DataFrame): Asset returns data
        lambda_param (float): Risk aversion parameter
        hhi_penalty (float): HHI concentration penalty
        lookback_period (int): Number of months to look back for the rolling window
        alphas_df (pd.DataFrame): Factor alpha forecasts data

    Returns:
        tuple: Strategy returns, portfolio statistics, turnover, and average HHI
    """
    # Initialize optimizer with specified parameters
    optimizer = PortfolioOptimizer(returns, lambda_param, hhi_penalty, alphas_df)
    
    # Prepare dates for rolling windows
    dates = returns.index
    
    # For rolling window, we can calculate one additional month at the end
    # Create a date for one month after the last date in the dataset
    next_month_date = dates[-1] + pd.DateOffset(months=1)
    
    # Initialize storage for weights and returns - now starting from the 2nd month 
    # so we can calculate returns using at least 1 month of data
    rolling_weights = pd.DataFrame(index=list(dates[1:]) + [next_month_date], columns=returns.columns)
    
    # Run optimizations for each date
    for i, current_date in enumerate(dates[1:], 1):
        # Use expanding window until we reach the lookback period, then switch to rolling window
        if i <= lookback_period:
            # Expanding window: use all available data up to the previous month
            start_idx = 0
            end_idx = max(0, i - 1)  # Use data up to the previous month
        else:
            # Rolling window: use exactly lookback_period months ending at previous month
            start_idx = max(0, i - lookback_period)
            end_idx = max(0, i - 1)  # Use data up to the previous month
            
        if end_idx >= start_idx:  # Ensure we have data to work with
            rolling_data = returns.loc[dates[start_idx : end_idx + 1]]
            rolling_weights.loc[current_date] = optimizer.optimize_weights(rolling_data, current_date)
    
    # Calculate the extra month for rolling strategy
    # Use the last lookback_period months of data for the next month
    start_idx = max(0, len(dates) - lookback_period)
    rolling_data = returns.loc[dates[start_idx:]]
    rolling_weights.loc[next_month_date] = optimizer.optimize_weights(rolling_data, next_month_date)
    
    # Calculate strategy returns starting from the 2nd period to allow for expanding window
    rolling_returns = pd.Series(index=dates[2:], dtype=float)
    
    for date in dates[2:]:
        prev_date = dates[dates < date][-1]
        # Get returns using previous month's weights
        rolling_returns[date] = np.sum(
            rolling_weights.loc[prev_date] * returns.loc[date]
        )
    
    # Calculate statistics
    stats = calculate_portfolio_statistics(rolling_returns)
    
    # Calculate turnover (excluding the extra month for turnover calculation)
    avg_turnover, _ = calculate_turnover(rolling_weights.loc[dates[1:]])
    
    # Calculate HHI (portfolio concentration)
    avg_hhi, _ = calculate_portfolio_hhi(rolling_weights.loc[dates[1:]])
    
    # Calculate and print the maximum weight for any asset at any time
    max_weight = rolling_weights.max().max() * 100
    max_weight_asset = rolling_weights.max().idxmax()
    max_weight_date = rolling_weights.idxmax()[max_weight_asset]
    
    print(f"Max allocation: {max_weight:.2f}% in {max_weight_asset} on {max_weight_date.strftime('%Y-%m-%d')}")
    
    # Print details about the expanding window period
    expanding_end = min(lookback_period, len(dates)-1)
    print(f"Used expanding window from {dates[0].strftime('%Y-%m-%d')} to {dates[expanding_end].strftime('%Y-%m-%d')}")
    if expanding_end < len(dates)-1:
        print(f"Used {lookback_period}-month rolling window thereafter")
    
    return rolling_returns, rolling_weights.loc[dates[1:]]  # Return returns and weights (excluding extra month)

def run_hhi_analysis():
    """
    Run analysis across different HHI penalty values while keeping Lambda fixed at 1.0
    and lookback period fixed at 60 months
    """
    # Load data
    print("Loading data...")
    returns = pd.read_excel('T2_Optimizer.xlsx', index_col=0)
    returns.index = pd.to_datetime(returns.index)
    
    # Convert to decimals if needed
    if returns.abs().mean().mean() > 1:
        returns = returns / 100
    
    # Remove Monthly Return_CS
    if 'Monthly Return_CS' in returns.columns:
        returns = returns.drop(columns=['Monthly Return_CS'])
    
    # Load alpha forecasts
    alphas_df = pd.read_excel('T60.xlsx')
    alphas_df['Date'] = pd.to_datetime(alphas_df['Date'])
    
    # Define parameter grids for 2D search
    lambda_params = [5.0, 50.0, 500.0]
    hhi_penalties = [0.0, 0.5, 1.0]
    
    # Fixed parameters
    lookback_period = 60
    
    # Initialize results storage as 3D structure (lambda x HHI x metrics)
    metrics = ['Annualized Return (%)', 'Sharpe Ratio', 'Turnover (%)', 
               'Volatility (%)', 'Max Drawdown (%)', 'Positive Months (%)', 
               'Average HHI', 'Normalized HHI']
    
    # Create result grids for each metric
    result_grids = {}
    for metric in metrics:
        result_grids[metric] = pd.DataFrame(index=lambda_params, columns=hhi_penalties)
    
    # Total combinations to process
    total_combinations = len(lambda_params) * len(hhi_penalties)
    current_combination = 0
    
    print(f"Starting 2D Grid Search: {len(lambda_params)} lambda values × {len(hhi_penalties)} HHI penalties = {total_combinations} combinations")
    print("=" * 80)
    
    # Grid search loop
    for lambda_param in lambda_params:
        for hhi_penalty in hhi_penalties:
            current_combination += 1
            print(f"\nProcessing combination {current_combination}/{total_combinations}")
            print(f"Lambda: {lambda_param}, HHI Penalty: {hhi_penalty}")
            print("-" * 40)
            
            # Run optimization with current parameters
            portfolio_returns, weights_df = run_rolling_optimization(
                returns, lambda_param, hhi_penalty, lookback_period, alphas_df
            )
            
            # Calculate performance metrics
            annualized_return = (1 + portfolio_returns.mean())**12 - 1
            sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(12)
            volatility = portfolio_returns.std() * np.sqrt(12)
            
            # Calculate drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Calculate turnover
            weight_changes = weights_df.diff().abs().sum(axis=1)
            avg_turnover = weight_changes.mean()
            
            # Calculate positive months percentage
            positive_months = (portfolio_returns > 0).mean()
            
            # Calculate HHI metrics
            hhi_values = (weights_df**2).sum(axis=1)
            avg_hhi = hhi_values.mean()
            normalized_hhi = (avg_hhi - 1/len(weights_df.columns)) / (1 - 1/len(weights_df.columns))
            
            # Store results in grids
            result_grids['Annualized Return (%)'][hhi_penalty][lambda_param] = annualized_return * 100
            result_grids['Sharpe Ratio'][hhi_penalty][lambda_param] = sharpe_ratio
            result_grids['Volatility (%)'][hhi_penalty][lambda_param] = volatility * 100
            result_grids['Max Drawdown (%)'][hhi_penalty][lambda_param] = max_drawdown * 100
            result_grids['Turnover (%)'][hhi_penalty][lambda_param] = avg_turnover * 100
            result_grids['Positive Months (%)'][hhi_penalty][lambda_param] = positive_months * 100
            result_grids['Average HHI'][hhi_penalty][lambda_param] = avg_hhi
            result_grids['Normalized HHI'][hhi_penalty][lambda_param] = normalized_hhi
            
            # Print current results
            print(f"Annualized Return: {annualized_return*100:.2f}%")
            print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
            print(f"Volatility: {volatility*100:.2f}%")
            print(f"Max Drawdown: {max_drawdown*100:.2f}%")
            print(f"Turnover: {avg_turnover*100:.2f}%")
            print(f"Positive Months: {positive_months*100:.1f}%")
            print(f"Average HHI: {avg_hhi:.3f}")
            print(f"Normalized HHI: {normalized_hhi:.3f}")
    
    print("\n" + "=" * 80)
    print("2D GRID SEARCH RESULTS")
    print("=" * 80)
    
    # Display result grids
    for metric in metrics:
        print(f"\n{metric} Grid:")
        print("Lambda \\ HHI Penalty", end="")
        for hhi in hhi_penalties:
            print(f"{hhi:>12.1f}", end="")
        print()
        print("-" * (20 + 12 * len(hhi_penalties)))
        
        for lam in lambda_params:
            print(f"{lam:>18.1f}", end="")
            for hhi in hhi_penalties:
                value = result_grids[metric][hhi][lam]
                if 'HHI' in metric and 'Normalized' not in metric:
                    print(f"{value:>12.3f}", end="")
                elif metric == 'Sharpe Ratio':
                    print(f"{value:>12.3f}", end="")
                else:
                    print(f"{value:>12.2f}", end="")
            print()
    
    # Find optimal combinations for key metrics
    print("\n" + "=" * 80)
    print("OPTIMAL PARAMETER COMBINATIONS")
    print("=" * 80)
    
    # Best Sharpe Ratio
    best_sharpe_idx = result_grids['Sharpe Ratio'].stack().idxmax()
    best_sharpe_hhi, best_sharpe_lambda = best_sharpe_idx
    print(f"Best Sharpe Ratio: {result_grids['Sharpe Ratio'][best_sharpe_hhi][best_sharpe_lambda]:.3f}")
    print(f"  Parameters: Lambda={best_sharpe_lambda}, HHI Penalty={best_sharpe_hhi}")
    
    # Best Return
    best_return_idx = result_grids['Annualized Return (%)'].stack().idxmax()
    best_return_hhi, best_return_lambda = best_return_idx
    print(f"Best Annualized Return: {result_grids['Annualized Return (%)'][best_return_hhi][best_return_lambda]:.2f}%")
    print(f"  Parameters: Lambda={best_return_lambda}, HHI Penalty={best_return_hhi}")
    
    # Lowest Drawdown (closest to 0)
    best_dd_idx = result_grids['Max Drawdown (%)'].stack().idxmax()  # Max because drawdowns are negative
    best_dd_hhi, best_dd_lambda = best_dd_idx
    print(f"Lowest Max Drawdown: {result_grids['Max Drawdown (%)'][best_dd_hhi][best_dd_lambda]:.2f}%")
    print(f"  Parameters: Lambda={best_dd_lambda}, HHI Penalty={best_dd_hhi}")
    
    return result_grids

if __name__ == "__main__":
    results = run_hhi_analysis()
