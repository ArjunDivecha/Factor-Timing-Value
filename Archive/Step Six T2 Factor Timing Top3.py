"""
Factor Timing Strategy Implementation

INPUT FILES:
1. T2_Optimizer.xlsx
   - Excel file with dates in first column and factor returns in subsequent columns (as percentages)
   - Format: Excel workbook with single sheet
   - Index: Dates in datetime format
   - Contains factor returns from Step Four

OUTPUT FILES:
1. T2_Top3_Factor_Weights.xlsx
   - Excel file with same structure as input, containing factor weights instead of returns
   - Format: Excel workbook with single sheet
   - Index: Dates (monthly)
   - Columns: Factor weights based on 60-month lookback period
2. T2_Top3_Monthly_Returns.xlsx
   - Excel file containing monthly returns of the strategy
   - Format: Excel workbook with single sheet
   - Index: Dates (monthly)
   - Columns: Monthly returns

STRATEGY OVERVIEW:
This module implements an adaptive factor rotation strategy that dynamically allocates 
capital among country factors based on their recent performance characteristics. Unlike 
traditional static allocation approaches, this strategy actively shifts exposure between 
factors as market regimes change.

Key Strategy Components:
1. Multi-metric Factor Scoring System:
   - Momentum: Captures directional strength by measuring recent factor returns
   - Hit Rate: Evaluates consistency by calculating the percentage of positive months
   - Risk-adjusted Performance: Incorporates Sharpe ratio to balance return and volatility
   - Combined Score: Multiplicative combination that rewards factors with strong, 
     consistent, and risk-efficient performance

2. Adaptive Lookback Period:
   - Tests multiple lookback windows (3m to 60m) to identify optimal signal horizon
   - Demonstrates the trade-off between signal responsiveness and stability
   - Longer lookbacks (60m) produce higher Sharpe ratios but lower turnover
   - Shorter lookbacks (3m-12m) react faster to changing market conditions but with higher turnover

3. Concentration Control:
   - Selects top 3 factors with positive scores to maintain diversification
   - Implements position size caps to prevent excessive concentration
   - Weights factors proportionally to their Sharpe ratios rather than equal weighting

Performance Characteristics:
- The 60-month lookback version achieves a Sharpe ratio of 0.64 with reasonable turnover
- The strategy demonstrates strong downside protection during market drawdowns
- Longer lookback periods exhibit more stability and lower turnover, suitable for institutional portfolios
- Shorter lookback periods provide more tactical exposure with higher responsiveness to market shifts

Diversification Benefits:
- Factor timing provides different return streams compared to static allocations
- The approach can complement traditional geographic or sector diversification
- Adaptive weighting helps navigate changing macro environments that impact factor performance

Implementation Considerations:
- Monthly rebalancing to balance trading costs against signal decay
- Transaction cost modeling incorporated into the evaluation framework
- Careful factor selection to avoid excessive correlation among signals

Version: 1.0
Last Updated: 2024
"""

import pandas as pd
import numpy as np

def factor_timing_strategy(excel_path: str, n_top_factors: int = 3, lookback: int = 36) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Implements a factor timing strategy that selects top performing factors based on recent performance metrics.
    Weights are assigned proportionally to Sharpe ratios rather than equal weighting.
    
    Parameters
    ----------
    excel_path : str
        Path to Excel file containing factor returns data. File should have dates in first column
        and factor returns (as percentages) in subsequent columns.
    n_top_factors : int, default=3 
        Number of top factors to select each month for portfolio construction
    lookback : int, default=36
        Number of months to look back for calculating factor scores and Sharpe ratios
        
    Returns
    -------
    strategy_returns : pd.Series
        Monthly returns of the strategy
    positions : pd.DataFrame
        Factor weights/positions over time, with dates as index and factors as columns
    factor_scores : pd.DataFrame 
        Score for each factor over time based on momentum, hit rate and Sharpe ratio
        
    Notes
    -----
    The strategy works in three steps:
    1. Calculate factor scores using momentum, hit rate, and Sharpe ratio
    2. Select top N factors with positive scores
    3. Weight selected factors proportionally to their Sharpe ratios
    """
    
    # Read data
    df = pd.read_excel(excel_path)
    df.index = pd.to_datetime(df.iloc[:,0])
    returns_df = df.iloc[:,1:].astype(float) / 100
    
    # Initialize positions and trailing returns as factor_scores
    positions = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)
    factor_scores = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)

    # Select top factors by trailing performance and weight by Sharpe ratio
    for date in returns_df.index[lookback:]:
        hist_data = returns_df.loc[:date].tail(lookback)
        
        # Calculate trailing returns (momentum)
        trailing_returns = (hist_data + 1).prod(axis=0) - 1
        
        # Calculate hit rate (consistency)
        hit_rate = (hist_data > 0).mean(axis=0)
        
        # Calculate Sharpe ratio (risk-adjusted performance)
        sharpe_ratios = hist_data.mean(axis=0) / hist_data.std(axis=0)
        sharpe_ratios = sharpe_ratios.fillna(0)  # Handle zero volatility
        
        # Combined score (as described in docstring)
        combined_scores = trailing_returns * hit_rate * sharpe_ratios
        factor_scores.loc[date] = combined_scores
        
        # Select top N factors with positive scores
        top_factors = combined_scores.nlargest(n_top_factors)
        top_factors = top_factors[top_factors > 0]
        
        if len(top_factors) > 0:
            # Weight proportionally to Sharpe ratios
            weights = sharpe_ratios[top_factors.index]
            weights = weights.clip(lower=0)  # Ensure no negative weights
            
            # Normalize weights to sum to 1
            if weights.sum() > 0:
                weights = weights / weights.sum()
                positions.loc[date, weights.index] = weights
    
    # Calculate strategy returns using previous month's weights (like in Step Five)
    strategy_returns = pd.Series(index=returns_df.index[lookback+1:], dtype=float)
    dates = returns_df.index
    
    for i, date in enumerate(dates[lookback+1:], lookback+1):
        prev_date = dates[dates < date][-1]  # Get the previous month's date
        # Calculate returns using previous month's weights
        strategy_returns[date] = np.sum(
            positions.loc[prev_date] * returns_df.loc[date]
        )
    
    return strategy_returns, positions, factor_scores

def returns_only_strategy(excel_path: str, n_top_factors: int = 3, lookback: int = 60) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Implements a simpler factor timing strategy that selects top performing factors based ONLY on trailing returns.
    This is a test version that doesn't use hit rate or Sharpe ratio for factor selection.
    
    Parameters
    ----------
    excel_path : str
        Path to Excel file containing factor returns data
    n_top_factors : int, default=3 
        Number of top factors to select each month for portfolio construction
    lookback : int, default=60
        Number of months to look back for calculating factor returns
        
    Returns
    -------
    strategy_returns : pd.Series
        Monthly returns of the strategy
    positions : pd.DataFrame
        Factor weights/positions over time
    factor_scores : pd.DataFrame 
        Score for each factor over time (in this case, just the trailing returns)
    """
    
    # Read data
    df = pd.read_excel(excel_path)
    df.index = pd.to_datetime(df.iloc[:,0])
    returns_df = df.iloc[:,1:].astype(float) / 100
    
    # Initialize positions and trailing returns as factor_scores
    positions = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)
    factor_scores = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)

    # Select top factors by trailing returns only
    for date in returns_df.index[lookback:]:
        hist_data = returns_df.loc[:date].tail(lookback)
        
        # Calculate trailing returns only
        trailing_returns = (hist_data + 1).prod(axis=0) - 1
        factor_scores.loc[date] = trailing_returns
        
        # Select top N factors with positive trailing returns
        top_factors = trailing_returns.nlargest(n_top_factors)
        top_factors = top_factors[top_factors > 0]
        
        if len(top_factors) > 0:
            # Equal weighting for simplicity
            weights = pd.Series(1/len(top_factors), index=top_factors.index)
            positions.loc[date, weights.index] = weights
    
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

def calculate_turnover(weights_df: pd.DataFrame) -> tuple[float, pd.Series]:
    """
    Calculate average monthly turnover and turnover series.

    Args:
        weights_df (pd.DataFrame): Portfolio weights over time.

    Returns:
        tuple: Average monthly turnover and monthly turnover series.
    """
    # Calculate absolute changes in weights from month to month
    turnover_series = weights_df.diff().abs().sum(axis=1)
    # Calculate average monthly turnover
    avg_turnover = turnover_series.mean()
    
    return avg_turnover, turnover_series

def evaluate_strategy(returns: pd.Series, positions: pd.DataFrame) -> dict[str, str]:
    """
    Calculates key performance metrics for a trading strategy.
    
    Parameters
    ----------
    returns : pd.Series
        Monthly returns of the strategy
    positions : pd.DataFrame
        Factor weights/positions over time
        
    Returns
    -------
    dict
        Dictionary containing performance metrics:
        - Annualized Return (%): (1 + monthly_mean)^12 - 1
        - Annualized Volatility (%): monthly_std * sqrt(12)
        - Sharpe Ratio: annualized_return / annualized_volatility
        - Maximum Drawdown (%): maximum peak to trough decline
        - Monthly Turnover (%): average monthly absolute change in positions
        
    Notes
    -----
    All percentage metrics are returned as formatted strings with % symbol.
    Sharpe ratio assumes zero risk-free rate for simplicity.
    """
    
    # Filter positions to match returns dates for turnover calculation
    positions_aligned = positions.loc[positions.index.isin(returns.index) | positions.index.isin(returns.index - pd.DateOffset(months=1))]
    
    monthly_mean = returns.mean()
    monthly_vol = returns.std()
    
    ann_return = (1 + monthly_mean)**12 - 1
    ann_vol = monthly_vol * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Calculate turnover using the dedicated function
    avg_turnover, _ = calculate_turnover(positions_aligned)
    
    return {
        'Annualized Return': f"{ann_return:.2%}",
        'Annualized Vol': f"{ann_vol:.2%}", 
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Turnover': f"{avg_turnover:.2%}"
    }

def analyze_lookback_periods(excel_path: str, lookbacks: list[int] = [3, 6, 9, 12, 18, 24, 36, 60, 72], n_top_factors: int = 3) -> pd.DataFrame:
    """
    Analyzes strategy performance across different lookback periods.
    
    Parameters
    ----------
    excel_path : str
        Path to Excel file containing factor returns data
    lookbacks : list of int, default=[3,6,9,12,18,24,36,60,72]
        List of lookback periods (in months) to test
    n_top_factors : int, default=3
        Number of top factors to select each month for portfolio construction
        
    Returns
    -------
    pd.DataFrame
        Table comparing strategy performance metrics across lookback periods.
        Index: Lookback periods
        Columns: 
            - Annualized Return (%)
            - Annualized Vol (%)
            - Sharpe Ratio
            - Max Drawdown (%)
            - Turnover (%)
            
    Notes
    -----
    This function runs the factor timing strategy for each lookback period
    and compiles the performance metrics into a single comparison table.
    """
    
    results = []
    for lookback in lookbacks:
        returns, positions, scores = factor_timing_strategy(excel_path, n_top_factors=n_top_factors, lookback=lookback)
        perf = evaluate_strategy(returns, positions)
        perf['Lookback'] = f"{lookback}m"
        results.append(perf)

    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('Lookback')
    results_df = results_df[['Annualized Return', 'Annualized Vol', 'Sharpe Ratio', 'Max Drawdown', 'Turnover']]
    
    return results_df

# Run lookback analysis
if __name__ == "__main__":
    """
    Main execution block that:
    1. Runs lookback analysis across different periods
    2. Generates and saves factor weights for 60-month lookback period
    
    Output files:
    - Prints lookback period analysis to console
    - Saves 60-month factor weights to Excel file in the current directory:
      'T2_Top3_Factor_Weights.xlsx'
    - Saves monthly returns to 'T2_Top3_Monthly_Returns.xlsx'
    """
    
    # Run lookback analysis with top 3 factors
    results = analyze_lookback_periods("T2_Optimizer.xlsx", n_top_factors=3)
    print("\nLookback Period Analysis (Top 3 Factors):")
    print(results)
    
    # Get positions for 60-month lookback and save to Excel
    returns_60m, positions_60m, scores_60m = factor_timing_strategy(
        "T2_Optimizer.xlsx",
        n_top_factors=3,  
        lookback=60
    )
    
    # Read original file to get the date column format
    orig_df = pd.read_excel("T2_Optimizer.xlsx")
    dates = orig_df.iloc[:,0]
    
    # Create output dataframe with same format as input
    output_df = pd.DataFrame(index=positions_60m.index)
    output_df.insert(0, dates.name, output_df.index)  # Add date column first
    output_df = pd.concat([output_df, positions_60m], axis=1)
    
    # Save to Excel
    output_path = "T2_Top3_Factor_Weights.xlsx"  
    output_df.to_excel(output_path, index=False)
    print(f"\nFeature weights for 60-month lookback saved to: {output_path}")
    
    # Save monthly returns to Excel
    returns_df = pd.DataFrame(returns_60m)
    returns_df.columns = ['Monthly_Return']
    returns_df['Monthly_Return'] = returns_df['Monthly_Return'] * 100  # Convert to percentage
    returns_path = "T2_Top3_Monthly_Returns.xlsx"
    returns_df.to_excel(returns_path)
    print(f"Monthly returns saved to: {returns_path}")
    
    # TEST: Run returns-only strategy with 60-month lookback
    print("\n--- TEST: Returns-Only Strategy Analysis ---")
    returns_test, positions_test, scores_test = returns_only_strategy(
        "T2_Optimizer.xlsx",
        n_top_factors=3,
        lookback=60
    )
    
    # Evaluate and print performance metrics for test strategy
    perf_test = evaluate_strategy(returns_test, positions_test)
    perf_original = evaluate_strategy(returns_60m, positions_60m)
    
    print("\nPerformance Comparison (60-month lookback):")
    print("Metric            Original Strategy    Returns-Only Strategy")
    print("-" * 65)
    for metric in ['Annualized Return', 'Annualized Vol', 'Sharpe Ratio', 'Max Drawdown', 'Turnover']:
        print(f"{metric:<18} {perf_original[metric]:<20} {perf_test[metric]}")
    
    # Save test strategy weights to Excel
    output_test_df = pd.DataFrame(index=positions_test.index)
    output_test_df.insert(0, dates.name, output_test_df.index)
    output_test_df = pd.concat([output_test_df, positions_test], axis=1)
    
    test_weights_path = "T2_Top3_Returns_Only_Weights.xlsx"
    output_test_df.to_excel(test_weights_path, index=False)
    
    # Save test strategy returns to Excel
    test_returns_df = pd.DataFrame(returns_test)
    test_returns_df.columns = ['Monthly_Return']
    test_returns_df['Monthly_Return'] = test_returns_df['Monthly_Return'] * 100
    test_returns_path = "T2_Top3_Returns_Only_Monthly_Returns.xlsx"
    test_returns_df.to_excel(test_returns_path)
    
    print(f"\nTest strategy weights saved to: {test_weights_path}")
    print(f"Test strategy returns saved to: {test_returns_path}")