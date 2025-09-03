"""
Factor Timing Strategy Grid Search - Step Six (Archive - Not Part of Main Flow)

=============================================================================
INPUT FILES:
- T2_Optimizer.xlsx: Excel file containing factor returns data with dates in first column and factor returns as percentages in subsequent columns
- Step Factor Categories.xlsx: Excel file with factor filtering criteria containing columns "Factor Name" and "Max" (where Max=1 indicates factors to include)

OUTPUT FILES:
- Terminal display only: Three grids showing annualized returns, Sharpe ratios, and monthly turnover for different factor/lookback combinations
- No files are written to disk by this program

=============================================================================

WHAT THIS PROGRAM DOES:
This program helps find the best combination of two important settings for our factor timing strategy:
1. How many top factors to use (1, 3, 5, 7, 10, or 15 factors)
2. How far back to look when evaluating factors (24, 36, 60, 72, or 90 months)

The program tests all 30 possible combinations (6 factor counts × 5 lookback periods) and shows you:
- Which combination gives the highest returns
- Which combination has the best risk-adjusted returns (Sharpe ratio)
- How much trading each combination requires (turnover)

IMPORTANT: Only factors marked with Max=1 in the Factor Categories file are used.
Factors marked with Max=0 are completely ignored.

The program displays three tables in the terminal showing performance for each combination,
making it easy to identify the optimal settings for the factor timing strategy.

Version: 1.0
Last Updated: December 2024
Author: T2 Factor Timing System
=============================================================================
"""

import pandas as pd
import numpy as np
import time
from tabulate import tabulate

def factor_timing_strategy(excel_path: str, n_top_factors: int = 3, lookback: int = 36, allowed_factors: list = None) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Run the factor timing strategy with specific settings to see how it performs.
    
    WHAT THIS FUNCTION DOES:
    This function implements our factor timing strategy. Each month, it:
    1. Looks back at the specified number of months of factor performance
    2. Calculates a score for each factor based on momentum, hit rate, and Sharpe ratio
    3. Selects the top N factors with positive scores
    4. Weights them based on their 12-month trailing returns (higher returns get more weight)
    5. Returns the strategy performance and position history
    
    INPUTS:
    - excel_path: Full path to the Excel file with factor returns data
    - n_top_factors: How many top factors to select each month (default: 3)
    - lookback: How many months to look back when scoring factors (default: 36)
    - allowed_factors: List of factor names to consider (if None, uses all factors)
    
    OUTPUTS:
    - strategy_returns: Monthly returns of the strategy as a pandas Series
    - positions: DataFrame showing which factors were selected and their weights each month
    - factor_scores: DataFrame showing the calculated scores for all factors over time
    
    The strategy only invests in factors with positive scores and weights them by their
    12-month trailing returns rather than equal weighting.
    """
    
    # Read data
    df = pd.read_excel(excel_path)
    df.index = pd.to_datetime(df.iloc[:,0])
    returns_df = df.iloc[:,1:].astype(float) / 100
    
    # Filter for allowed factors if specified
    if allowed_factors is not None:
        # Keep only factors in the allowed_factors list
        available_factors = set(returns_df.columns)
        filtered_factors = [f for f in allowed_factors if f in available_factors]
        
        if len(filtered_factors) == 0:
            raise ValueError(f"None of the allowed factors found in {excel_path}. Check factor names match exactly.")
            
        returns_df = returns_df[filtered_factors]
        print(f"Filtered returns data to {len(filtered_factors)} allowed factors.")
    
    # Initialize factor scores dataframe
    factor_scores = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)
    
    # Calculate scores for each date
    for date in returns_df.index[lookback:]:
        hist_data = returns_df.loc[:date]
        
        for factor in returns_df.columns:
            # Calculate components of factor score
            momentum = hist_data[factor].tail(lookback).mean()
            hit_rate = (hist_data[factor].tail(lookback) > 0).mean()
            vol = hist_data[factor].tail(lookback).std()
            sharpe = momentum / vol if vol != 0 else 0
            
            # Combined score weights momentum, hit rate and risk-adjusted return
            factor_scores.loc[date, factor] = momentum * hit_rate * (1 + sharpe)
    
    # Generate positions
    positions = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)
    
    for date in factor_scores.index[lookback:]:
        # Select factors with positive scores
        positive_factors = factor_scores.loc[date][factor_scores.loc[date] > 0]
        
        if len(positive_factors) > 0:
            # Take top N factors
            top_factors = positive_factors.nlargest(min(n_top_factors, len(positive_factors)))
            
            # Calculate 12-month trailing returns for top factors
            trailing_returns = {}
            for factor in top_factors.index:
                # Get the last 12 months of returns for this factor
                hist_data = returns_df[factor].loc[:date].tail(12)
                # Calculate cumulative return over 12 months
                twelve_month_return = (1 + hist_data).prod() - 1
                trailing_returns[factor] = max(twelve_month_return, 0)  # Only use positive trailing returns
            
            # Calculate weights proportional to 12-month trailing returns
            total_trailing = sum(trailing_returns.values())
            if total_trailing > 0:
                for factor, trailing_ret in trailing_returns.items():
                    positions.loc[date, factor] = trailing_ret / total_trailing
    
    # Calculate strategy returns using previous month's weights
    strategy_returns = pd.Series(index=returns_df.index[lookback+1:], dtype=float)
    dates = returns_df.index
    
    for i, date in enumerate(dates[lookback+1:], lookback+1):
        prev_date = dates[dates < date][-1]  # Get the previous month's date
        # Calculate returns using previous month's weights
        strategy_returns[date] = np.sum(
            positions.loc[prev_date] * returns_df.loc[date]
        )
    
    return strategy_returns, positions, factor_scores

def calculate_performance_metrics(returns: pd.Series, positions: pd.DataFrame) -> tuple[float, float, float]:
    """
    Calculate key performance statistics for a strategy.
    
    WHAT THIS FUNCTION DOES:
    Takes the monthly returns from a strategy and calculates three important metrics:
    1. Annualized return: How much the strategy makes per year on average
    2. Sharpe ratio: Risk-adjusted return (higher is better, above 1.0 is good)
    3. Monthly turnover: How much trading the strategy does each month (lower is usually better)
    
    INPUTS:
    - returns: pandas Series of monthly strategy returns (as decimals, not percentages)
    - positions: DataFrame showing factor weights over time
    
    OUTPUTS:
    - ann_return: Annualized return as a decimal (0.10 = 10% per year)
    - sharpe: Sharpe ratio (risk-adjusted return measure)
    - turnover: Average monthly turnover as a decimal (0.20 = 20% turnover per month)
    
    These metrics help compare different strategy configurations to find the best one.
    """
    monthly_mean = returns.mean()
    monthly_vol = returns.std()
    
    ann_return = (1 + monthly_mean)**12 - 1
    ann_vol = monthly_vol * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    
    # Calculate turnover (average monthly absolute change in positions)
    turnover = positions.diff().abs().sum(axis=1).mean()
    
    return ann_return, sharpe, turnover

def run_grid_search(excel_path: str, n_factors_list: list[int], lookbacks_list: list[int], allowed_factors: list = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Test all combinations of factor counts and lookback periods to find the best settings.
    
    WHAT THIS FUNCTION DOES:
    This is the main function that runs our comprehensive grid search. It:
    1. Tests every combination of factor count (1,3,5,7,10,15) and lookback period (24,36,60,72,90 months)
    2. For each combination, runs the complete factor timing strategy
    3. Calculates performance metrics (returns, Sharpe ratio, turnover) for each combination
    4. Returns three grids showing all results for easy comparison
    
    INPUTS:
    - excel_path: Full path to the Excel file containing factor returns data
    - n_factors_list: List of different factor counts to test (e.g., [1,3,5,7,10,15])
    - lookbacks_list: List of different lookback periods to test (e.g., [24,36,60,72,90])
    - allowed_factors: List of factor names to consider (if None, uses all factors)
    
    OUTPUTS:
    - returns_grid: DataFrame with annualized returns for each combination
    - sharpe_grid: DataFrame with Sharpe ratios for each combination  
    - turnover_grid: DataFrame with monthly turnover for each combination
    
    Each grid has lookback periods as rows and factor counts as columns, making it easy
    to see which combination performs best on each metric.
    """
    # Initialize result grids
    returns_grid = pd.DataFrame(index=[f"{lb}m" for lb in lookbacks_list], 
                               columns=[f"{n}f" for n in n_factors_list])
    sharpe_grid = pd.DataFrame(index=[f"{lb}m" for lb in lookbacks_list], 
                              columns=[f"{n}f" for n in n_factors_list])
    turnover_grid = pd.DataFrame(index=[f"{lb}m" for lb in lookbacks_list], 
                                columns=[f"{n}f" for n in n_factors_list])
    
    total_combinations = len(n_factors_list) * len(lookbacks_list)
    completed = 0
    
    print(f"Running grid search across {total_combinations} combinations...")
    start_time = time.time()
    
    for lookback in lookbacks_list:
        for n_factors in n_factors_list:
            # Run strategy with current parameters
            returns, positions, _ = factor_timing_strategy(excel_path, n_top_factors=n_factors, 
                                                       lookback=lookback, allowed_factors=allowed_factors)
            
            # Calculate performance metrics
            ann_return, sharpe, turnover = calculate_performance_metrics(returns, positions)
            
            # Store results
            returns_grid.loc[f"{lookback}m", f"{n_factors}f"] = ann_return
            sharpe_grid.loc[f"{lookback}m", f"{n_factors}f"] = sharpe
            turnover_grid.loc[f"{lookback}m", f"{n_factors}f"] = turnover
            
            # Update progress
            completed += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / completed
            remaining = avg_time * (total_combinations - completed)
            
            print(f"Completed {completed}/{total_combinations} combinations. " +
                  f"Estimated time remaining: {remaining:.1f}s", end="\r")
    
    print(f"\nGrid search completed in {time.time() - start_time:.1f} seconds.")
    
    return returns_grid, sharpe_grid, turnover_grid

def format_grid(grid: pd.DataFrame, is_return: bool = True) -> pd.DataFrame:
    """
    Format the results grids for nice display in the terminal.
    
    WHAT THIS FUNCTION DOES:
    Takes the raw numerical results and formats them for easy reading:
    - Returns and turnover are shown as percentages (10.25%)
    - Sharpe ratios are shown as decimals with 2 decimal places (1.25)
    - Missing values are shown as "N/A"
    
    INPUTS:
    - grid: DataFrame containing the raw numerical results
    - is_return: True for returns/turnover (show as %), False for Sharpe ratios
    
    OUTPUTS:
    - Formatted DataFrame with nice string formatting for terminal display
    """
    if is_return:
        return grid.applymap(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
    else:
        return grid.applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")

if __name__ == "__main__":
    # Define grid search parameters
    n_factors_list = [1, 3, 5, 7, 10, 15]
    lookbacks_list = [24, 36, 60, 72, 90]
    
    # Load factor categories to filter only those with Max=1
    print("Loading factor categories from Step Factor Categories.xlsx...")
    factor_categories = pd.read_excel("Step Factor Categories.xlsx")
    allowed_factors = factor_categories[factor_categories["Max"] == 1]["Factor Name"].tolist()
    
    print(f"Found {len(allowed_factors)} factors with Max=1 that will be included in the grid search:")
    print(", ".join(allowed_factors))
    print()
    
    # Run grid search with filtered factors
    returns_grid, sharpe_grid, turnover_grid = run_grid_search("T2_Optimizer.xlsx", 
                                                          n_factors_list, 
                                                          lookbacks_list, 
                                                          allowed_factors=allowed_factors)
    
    # Format grids for display
    returns_formatted = format_grid(returns_grid, is_return=True)
    sharpe_formatted = format_grid(sharpe_grid, is_return=False)
    turnover_formatted = format_grid(turnover_grid, is_return=True)
    
    # Display results
    print("\n" + "="*80)
    print("ANNUALIZED RETURNS GRID")
    print("="*80)
    print("Rows: Lookback Periods, Columns: Number of Factors")
    print(tabulate(returns_formatted, headers='keys', tablefmt='grid'))
    
    print("\n" + "="*80)
    print("SHARPE RATIOS GRID")
    print("="*80)
    print("Rows: Lookback Periods, Columns: Number of Factors")
    print(tabulate(sharpe_formatted, headers='keys', tablefmt='grid'))
    
    print("\n" + "="*80)
    print("MONTHLY TURNOVER GRID")
    print("="*80)
    print("Rows: Lookback Periods, Columns: Number of Factors")
    print(tabulate(turnover_formatted, headers='keys', tablefmt='grid'))
    
    # Find and display best combinations
    max_return_idx = returns_grid.stack().idxmax()
    max_sharpe_idx = sharpe_grid.stack().idxmax()
    min_turnover_idx = turnover_grid.stack().idxmin()
    
    print("\n" + "="*80)
    print("BEST COMBINATIONS")
    print("="*80)
    print(f"Best Return: {returns_grid.stack().max()*100:.2f}% with {max_return_idx[1]} factors and {max_return_idx[0]} lookback")
    print(f"Best Sharpe: {sharpe_grid.stack().max():.2f} with {max_sharpe_idx[1]} factors and {max_sharpe_idx[0]} lookback")
    print(f"Lowest Turnover: {turnover_grid.stack().min()*100:.2f}% with {min_turnover_idx[1]} factors and {min_turnover_idx[0]} lookback")
