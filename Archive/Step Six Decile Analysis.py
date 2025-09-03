"""
Factor Timing Strategy Decile Analysis - Step Six (New Decile Version)

=============================================================================
INPUT FILES:
- T2_Optimizer.xlsx: Excel file containing factor returns data with dates in first column and factor returns as percentages in subsequent columns
- Step Factor Categories.xlsx: Excel file with factor filtering criteria containing columns "Factor Name" and "Max" (where Max=1 indicates factors to include)

OUTPUT FILES:
- Terminal display only: Three grids showing annualized returns, Sharpe ratios, and monthly turnover for different decile/lookback combinations
- outputs/visualizations/decile_cumulative_returns_60m.pdf: Cumulative return chart for all 10 deciles using 60-month lookback

=============================================================================

WHAT THIS PROGRAM DOES:
This program analyzes factor timing strategy performance across all 10 deciles instead of just top N factors.
It helps find the best combination of two important settings:
1. Which decile of factors to use (1st decile = top 10%, 2nd decile = next 10%, etc.)
2. How far back to look when evaluating factors (24, 36, 60, 72, or 90 months)

The program tests all 50 possible combinations (10 deciles × 5 lookback periods) and shows you:
- Which decile gives the highest returns for each lookback period
- Which decile has the best risk-adjusted returns (Sharpe ratio)
- How much trading each decile requires (turnover)

IMPORTANT: Only factors marked with Max=1 in the Factor Categories file are used.
Factors marked with Max=0 are completely ignored.

The program displays three tables in the terminal showing performance for each combination,
making it easy to identify the optimal decile and lookback settings.

Version: 1.0
Last Updated: January 2025
Author: T2 Factor Timing System
=============================================================================
"""

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from tabulate import tabulate

def factor_timing_strategy_decile(excel_path: str, decile: int = 1, lookback: int = 36, allowed_factors: list = None) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Run the factor timing strategy using a specific decile of factors.
    
    WHAT THIS FUNCTION DOES:
    This function implements our factor timing strategy using deciles. Each month, it:
    1. Looks back at the specified number of months of factor performance
    2. Calculates a score for each factor based on momentum, hit rate, and Sharpe ratio
    3. Ranks all factors and selects those in the specified decile (1=top 10%, 2=next 10%, etc.)
    4. Weights selected factors based on their 12-month trailing returns
    5. Returns the strategy performance and position history
    
    INPUTS:
    - excel_path: Full path to the Excel file with factor returns data
    - decile: Which decile to select (1=top 10%, 2=next 10%, ..., 10=bottom 10%)
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
    
    # Generate positions using decile approach
    positions = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)
    
    for date in factor_scores.index[lookback:]:
        # Get all factor scores for this date
        scores = factor_scores.loc[date]
        
        # Only consider factors with positive scores
        positive_scores = scores[scores > 0]
        
        if len(positive_scores) >= 10:  # Need at least 10 factors for meaningful deciles
            # Sort factors by score (highest to lowest)
            sorted_factors = positive_scores.sort_values(ascending=False)
            
            # Calculate decile boundaries
            n_factors = len(sorted_factors)
            decile_size = n_factors // 10
            
            # Select factors in the specified decile
            if decile <= 10:
                start_idx = (decile - 1) * decile_size
                if decile == 10:  # Last decile gets remaining factors
                    end_idx = n_factors
                else:
                    end_idx = decile * decile_size
                
                decile_factors = sorted_factors.iloc[start_idx:end_idx]
                
                if len(decile_factors) > 0:
                    # Calculate 12-month trailing returns for decile factors
                    trailing_returns = {}
                    for factor in decile_factors.index:
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

def run_decile_grid_search(excel_path: str, deciles_list: list[int], lookbacks_list: list[int], allowed_factors: list = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Test all combinations of deciles and lookback periods to find the best settings.
    
    WHAT THIS FUNCTION DOES:
    This is the main function that runs our comprehensive decile grid search. It:
    1. Tests every combination of decile (1-10) and lookback period (24,36,60,72,90 months)
    2. For each combination, runs the complete factor timing strategy
    3. Calculates performance metrics (returns, Sharpe ratio, turnover) for each combination
    4. Returns three grids showing all results for easy comparison
    
    INPUTS:
    - excel_path: Full path to the Excel file containing factor returns data
    - deciles_list: List of deciles to test (e.g., [1,2,3,4,5,6,7,8,9,10])
    - lookbacks_list: List of different lookback periods to test (e.g., [24,36,60,72,90])
    - allowed_factors: List of factor names to consider (if None, uses all factors)
    
    OUTPUTS:
    - returns_grid: DataFrame with annualized returns for each combination
    - sharpe_grid: DataFrame with Sharpe ratios for each combination  
    - turnover_grid: DataFrame with monthly turnover for each combination
    
    Each grid has lookback periods as rows and deciles as columns, making it easy
    to see which combination performs best on each metric.
    """
    # Initialize result grids
    returns_grid = pd.DataFrame(index=[f"{lb}m" for lb in lookbacks_list], 
                               columns=[f"D{d}" for d in deciles_list])
    sharpe_grid = pd.DataFrame(index=[f"{lb}m" for lb in lookbacks_list], 
                              columns=[f"D{d}" for d in deciles_list])
    turnover_grid = pd.DataFrame(index=[f"{lb}m" for lb in lookbacks_list], 
                                columns=[f"D{d}" for d in deciles_list])
    
    total_combinations = len(deciles_list) * len(lookbacks_list)
    completed = 0
    
    print(f"Running decile grid search across {total_combinations} combinations...")
    start_time = time.time()
    
    for lookback in lookbacks_list:
        for decile in deciles_list:
            try:
                # Run strategy with current parameters
                returns, positions, _ = factor_timing_strategy_decile(excel_path, decile=decile, 
                                                                   lookback=lookback, allowed_factors=allowed_factors)
                
                # Calculate performance metrics
                ann_return, sharpe, turnover = calculate_performance_metrics(returns, positions)
                
                # Store results
                returns_grid.loc[f"{lookback}m", f"D{decile}"] = ann_return
                sharpe_grid.loc[f"{lookback}m", f"D{decile}"] = sharpe
                turnover_grid.loc[f"{lookback}m", f"D{decile}"] = turnover
                
            except Exception as e:
                print(f"\nError with decile {decile}, lookback {lookback}: {e}")
                # Store NaN for failed combinations
                returns_grid.loc[f"{lookback}m", f"D{decile}"] = np.nan
                sharpe_grid.loc[f"{lookback}m", f"D{decile}"] = np.nan
                turnover_grid.loc[f"{lookback}m", f"D{decile}"] = np.nan
            
            # Update progress
            completed += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / completed
            remaining = avg_time * (total_combinations - completed)
            
            print(f"Completed {completed}/{total_combinations} combinations. " +
                  f"Estimated time remaining: {remaining:.1f}s", end="\r")
    
    print(f"\nDecile grid search completed in {time.time() - start_time:.1f} seconds.")
    
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

def create_cumulative_returns_plot(excel_path: str, lookback: int, allowed_factors: list, output_path: str = "outputs/visualizations/decile_cumulative_returns_60m.pdf"):
    """
    Create a cumulative returns plot for all 10 deciles using the specified lookback period.
    
    WHAT THIS FUNCTION DOES:
    This function generates a comprehensive chart showing how $1 invested in each decile
    would have grown over time. It:
    1. Runs the factor timing strategy for all 10 deciles using the specified lookback
    2. Calculates cumulative returns for each decile
    3. Creates a line plot showing all 10 deciles on the same chart
    4. Saves the plot as a PDF file
    
    INPUTS:
    - excel_path: Full path to the Excel file with factor returns data
    - lookback: Lookback period to use for all deciles
    - allowed_factors: List of factor names to consider
    - output_path: Path where to save the PDF chart
    
    OUTPUTS:
    - Saves a PDF chart showing cumulative returns for all deciles
    - Returns a dictionary of cumulative return series for each decile
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Store cumulative returns for each decile
    cumulative_returns = {}
    
    print(f"\nGenerating cumulative returns data for {lookback}-month lookback...")
    
    # Calculate returns for each decile
    for decile in range(1, 11):
        try:
            print(f"Processing decile {decile}...", end=" ")
            returns, positions, _ = factor_timing_strategy_decile(excel_path, decile=decile, 
                                                               lookback=lookback, allowed_factors=allowed_factors)
            
            # Calculate cumulative returns (starting from $1)
            cumulative_returns[f"Decile {decile}"] = (1 + returns).cumprod()
            print("✓")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            # Create empty series for failed deciles
            cumulative_returns[f"Decile {decile}"] = pd.Series(dtype=float)
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Define colors for each decile (gradient from best to worst)
    colors = plt.cm.RdYlGn(np.linspace(0.9, 0.1, 10))  # Green to red gradient
    
    # Plot each decile
    for i, (decile_name, cum_returns) in enumerate(cumulative_returns.items()):
        if len(cum_returns) > 0:
            plt.plot(cum_returns.index, cum_returns.values, 
                    label=decile_name, color=colors[i], linewidth=2, alpha=0.8)
    
    # Formatting
    plt.title(f'Cumulative Returns by Factor Decile ({lookback}-Month Lookback)\n' +
              f'Growth of $1 Investment', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Value ($)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Format y-axis as currency
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Tight layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nCumulative returns chart saved to: {output_path}")
    
    # Print summary statistics
    print(f"\nCumulative Returns Summary ({lookback}-month lookback):")
    print("-" * 70)
    
    for decile_name, cum_returns in cumulative_returns.items():
        if len(cum_returns) > 0:
            final_value = cum_returns.iloc[-1]
            total_return = (final_value - 1) * 100
            print(f"{decile_name}: ${final_value:.2f} (Total Return: {total_return:+6.1f}%)")
        else:
            print(f"{decile_name}: No data available")
    
    return cumulative_returns

if __name__ == "__main__":
    # Define grid search parameters - all 10 deciles
    deciles_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    lookbacks_list = [24, 36, 60, 72, 90]
    
    # Load factor categories to filter only those with Max=1
    print("Loading factor categories from Step Factor Categories.xlsx...")
    factor_categories = pd.read_excel("Step Factor Categories.xlsx")
    allowed_factors = factor_categories[factor_categories["Max"] == 1]["Factor Name"].tolist()
    
    print(f"Found {len(allowed_factors)} factors with Max=1 that will be included in the decile analysis:")
    print(", ".join(allowed_factors))
    print()
    
    # Run decile grid search with filtered factors
    returns_grid, sharpe_grid, turnover_grid = run_decile_grid_search("T2_Optimizer.xlsx", 
                                                                    deciles_list, 
                                                                    lookbacks_list, 
                                                                    allowed_factors=allowed_factors)
    
    # Format grids for display
    returns_formatted = format_grid(returns_grid, is_return=True)
    sharpe_formatted = format_grid(sharpe_grid, is_return=False)
    turnover_formatted = format_grid(turnover_grid, is_return=True)
    
    # Display results
    print("\n" + "="*100)
    print("ANNUALIZED RETURNS GRID BY DECILE")
    print("="*100)
    print("Rows: Lookback Periods, Columns: Deciles (D1=Top 10%, D2=Next 10%, ..., D10=Bottom 10%)")
    print(tabulate(returns_formatted, headers='keys', tablefmt='grid'))
    
    print("\n" + "="*100)
    print("SHARPE RATIOS GRID BY DECILE")
    print("="*100)
    print("Rows: Lookback Periods, Columns: Deciles (D1=Top 10%, D2=Next 10%, ..., D10=Bottom 10%)")
    print(tabulate(sharpe_formatted, headers='keys', tablefmt='grid'))
    
    print("\n" + "="*100)
    print("MONTHLY TURNOVER GRID BY DECILE")
    print("="*100)
    print("Rows: Lookback Periods, Columns: Deciles (D1=Top 10%, D2=Next 10%, ..., D10=Bottom 10%)")
    print(tabulate(turnover_formatted, headers='keys', tablefmt='grid'))
    
    # Find and display best combinations
    max_return_idx = returns_grid.stack().idxmax()
    max_sharpe_idx = sharpe_grid.stack().idxmax()
    min_turnover_idx = turnover_grid.stack().idxmin()
    
    print("\n" + "="*100)
    print("BEST COMBINATIONS BY DECILE")
    print("="*100)
    print(f"Best Return: {returns_grid.stack().max()*100:.2f}% with {max_return_idx[1]} and {max_return_idx[0]} lookback")
    print(f"Best Sharpe: {sharpe_grid.stack().max():.2f} with {max_sharpe_idx[1]} and {max_sharpe_idx[0]} lookback")
    print(f"Lowest Turnover: {turnover_grid.stack().min()*100:.2f}% with {min_turnover_idx[1]} and {min_turnover_idx[0]} lookback")
    
    # Additional analysis: Show performance trend across deciles for best lookback
    print("\n" + "="*100)
    print("DECILE PERFORMANCE ANALYSIS")
    print("="*100)
    
    # Find best performing lookback period based on average Sharpe across all deciles
    avg_sharpe_by_lookback = sharpe_grid.mean(axis=1)
    best_lookback = avg_sharpe_by_lookback.idxmax()
    
    print(f"Best average performing lookback period: {best_lookback}")
    print(f"Performance by decile for {best_lookback} lookback:")
    print("-" * 60)
    
    for decile in deciles_list:
        col_name = f"D{decile}"
        ret = returns_grid.loc[best_lookback, col_name]
        sharpe = sharpe_grid.loc[best_lookback, col_name]
        turnover = turnover_grid.loc[best_lookback, col_name]
        
        if pd.notnull(ret):
            print(f"Decile {decile:2d} (Top {decile*10-9:2d}-{decile*10:2d}%): " +
                  f"Return={ret*100:6.2f}%, Sharpe={sharpe:5.2f}, Turnover={turnover*100:5.2f}%")
        else:
            print(f"Decile {decile:2d} (Top {decile*10-9:2d}-{decile*10:2d}%): No data")
    
    # Generate cumulative returns chart for 60-month lookback
    print("\n" + "="*100)
    print("CUMULATIVE RETURNS VISUALIZATION")
    print("="*100)
    
    cumulative_returns_60m = create_cumulative_returns_plot("T2_Optimizer.xlsx", 
                                                           lookback=60, 
                                                           allowed_factors=allowed_factors)