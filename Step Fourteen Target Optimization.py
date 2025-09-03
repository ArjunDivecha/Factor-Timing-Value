"""
=============================================================================
SCRIPT NAME: Step Fourteen Target Optimization.py
=============================================================================

INPUT FILES:
- T2_Final_Country_Weights.xlsx: 
  Target country weights from factor timing strategy (All Periods sheet)
- Portfolio_Data.xlsx: 
  Monthly country returns for portfolio roll-forward (Returns sheet)
- T2_Country_Alphas.xlsx: 
  Country alpha estimates for optimization objective
- T2_Top_20_Exposure.csv:
  Country factor exposures for factor analysis

OUTPUT FILES:
- T2_Optimized_Country_Weights.xlsx: 
  Turnover-optimized country weights with performance comparison
- T2_Optimized_Strategy_Analysis.pdf:
  Comprehensive performance analysis charts
- T2_Turnover_Analysis.pdf:
  Detailed turnover analysis charts
- T2_Weighted_Average_Factor_Exposure.pdf:
  Factor exposure analysis charts
- T2_Weighted_Average_Factor_Rolling_Analysis.pdf:
  Rolling factor exposure analysis charts

VERSION: 2.0 - CVXPY Enhanced
LAST UPDATED: 2025-06-16
AUTHOR: Claude Code

DESCRIPTION:
Optimizes monthly country portfolio weights to balance three objectives:
1. Maximize portfolio alpha (weighted average of country alphas)
2. Minimize drift from rolled-forward previous weights (lambda penalty)
3. Minimize transaction costs from turnover (tcost penalty)

The optimization rolls forward previous month's weights using monthly returns,
then solves for optimal weights that maximize:
Alpha_Mult * Portfolio Alpha - Lambda * (Drift Penalty)² - Turnover * Transaction Cost

DEPENDENCIES:
- pandas
- numpy
- cvxpy (replaces scipy.optimize for faster convex optimization)
- matplotlib
- openpyxl

USAGE:
python "Step Fourteen Target Optimization.py"

NOTES:
- Uses CVXPY with OSQP solver for fast convex optimization
- Long-only constraints with maximum position size limits
- All weights must sum to 1.0
- Uses warm-start for repeated monthly optimizations
- First month uses original target weights as starting point
=============================================================================
"""

import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import os

# =============================================================================
# CONFIGURABLE PARAMETERS - MODIFY THESE AS NEEDED
# =============================================================================

# Alpha multiplier parameter - higher values increase emphasis on alpha maximization
ALPHA_MULT = 2.0

# Drift penalty parameter - higher values reduce deviation from rolled-forward weights
LAMBDA_DRIFT_PENALTY = 75.0

# Transaction cost parameter - higher values reduce turnover
TRANSACTION_COST = 6

# Maximum weight constraint - prevents concentration in single country
MAX_WEIGHT = 1.00

# =============================================================================
# SETUP LOGGING
# =============================================================================

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('T2_processing.log', mode='a'),
            logging.StreamHandler()
        ]
    )

def load_data():
    """
    Load all required data files
    
    Returns:
        tuple: (target_weights_df, returns_df, alphas_df, exposure_df)
    """
    logging.info("Loading input data files...")
    
    # Load target weights (All Periods sheet)
    target_weights_file = 'T2_Final_Country_Weights.xlsx'
    target_weights_df = pd.read_excel(target_weights_file, sheet_name='All Periods', index_col=0)
    target_weights_df.index = pd.to_datetime(target_weights_df.index)
    logging.info(f"Loaded target weights: {target_weights_df.shape[0]} periods, {target_weights_df.shape[1]} countries")
    
    # Load returns data
    returns_file = 'Portfolio_Data.xlsx'
    returns_df = pd.read_excel(returns_file, sheet_name='Returns', index_col=0)
    returns_df.index = pd.to_datetime(returns_df.index)
    logging.info(f"Loaded returns data: {returns_df.shape[0]} periods, {returns_df.shape[1]} countries")
    
    # Load country alphas - this file has dates as rows and countries as columns
    alphas_file = 'T2_Country_Alphas.xlsx'
    alphas_df = pd.read_excel(alphas_file, sheet_name='Country_Scores', index_col=0)
    alphas_df.index = pd.to_datetime(alphas_df.index)
    
    logging.info(f"Loaded country alphas: {alphas_df.shape[0]} periods, {alphas_df.shape[1]} countries")
    
    # Load factor exposure data
    exposure_file = 'T2_Top_20_Exposure.csv'
    exposure_df = pd.read_csv(exposure_file)
    exposure_df['Date'] = pd.to_datetime(exposure_df['Date'])
    logging.info(f"Loaded factor exposures: {exposure_df.shape[0]} observations, {len(exposure_df.columns)-2} factors")
    
    return target_weights_df, returns_df, alphas_df, exposure_df

def align_data(target_weights_df, returns_df, alphas_df):
    """
    Align all datasets to common countries and dates
    
    Args:
        target_weights_df: Target weights DataFrame
        returns_df: Returns DataFrame  
        alphas_df: Country alphas DataFrame (time-varying)
        
    Returns:
        tuple: (aligned_target_weights, aligned_returns, aligned_alphas)
    """
    logging.info("Aligning datasets...")
    
    # Find common countries across all datasets
    target_countries = set(target_weights_df.columns)
    returns_countries = set(returns_df.columns)
    alpha_countries = set(alphas_df.columns)  # alphas_df has countries as columns
    
    common_countries_set = target_countries.intersection(returns_countries).intersection(alpha_countries)
    
    # Preserve original order from alphas file instead of sorting alphabetically
    common_countries = [country for country in alphas_df.columns if country in common_countries_set]
    
    logging.info(f"Found {len(common_countries)} common countries")
    
    # Align to common countries
    target_weights_aligned = target_weights_df[common_countries].copy()
    returns_aligned = returns_df[common_countries].copy()
    alphas_aligned = alphas_df[common_countries].copy()  # Keep all dates for time-varying alphas
    
    # Find common date range (intersection of all three datasets)
    target_dates = set(target_weights_aligned.index)
    returns_dates = set(returns_aligned.index)
    alpha_dates = set(alphas_aligned.index)
    common_dates = target_dates.intersection(returns_dates).intersection(alpha_dates)
    common_dates = sorted(list(common_dates))
    
    logging.info(f"Found {len(common_dates)} common dates")
    
    # Align to common dates
    target_weights_aligned = target_weights_aligned.loc[common_dates]
    returns_aligned = returns_aligned.loc[common_dates]
    alphas_aligned = alphas_aligned.loc[common_dates]
    
    return target_weights_aligned, returns_aligned, alphas_aligned

def roll_forward_weights(prev_weights, returns):
    """
    Roll forward previous weights using monthly returns
    
    Args:
        prev_weights: Previous month's weights (pandas Series)
        returns: Monthly returns for current period (pandas Series)
        
    Returns:
        pandas Series: Rolled forward weights (normalized to sum to 1)
    """
    # Apply returns to get new values, handle NaN returns
    returns_clean = returns.fillna(0)  # Replace NaN returns with 0
    new_values = prev_weights * (1 + returns_clean)
    
    # Handle case where all values become zero or negative
    if new_values.sum() <= 0:
        logging.warning("All portfolio values are zero or negative after applying returns. Using equal weights.")
        rolled_weights = pd.Series(1.0 / len(prev_weights), index=prev_weights.index)
    else:
        # Normalize to sum to 1
        rolled_weights = new_values / new_values.sum()
    
    return rolled_weights

def create_cvxpy_model(n_countries, current_alphas, rolled_forward_weights, target_weights=None):
    """
    Create CVXPY model for optimization
    
    Args:
        n_countries: Number of countries
        current_alphas: Country alpha estimates for current period (pandas Series)
        rolled_forward_weights: Previous month's weights rolled forward with returns
        target_weights: Target weights for the current month (optional)
        
    Returns:
        tuple: (weights_var, problem)
    """
    # Define weights variable
    weights_var = cp.Variable(n_countries)
    
    # Define objective function
    portfolio_alpha = ALPHA_MULT * cp.sum(cp.multiply(current_alphas, weights_var))
    
    # Calculate drift penalty based on target_weights if provided, otherwise use rolled_forward_weights
    weights_for_drift = target_weights if target_weights is not None else rolled_forward_weights
    drift_penalty = LAMBDA_DRIFT_PENALTY * cp.sum_squares(weights_var - weights_for_drift)
    
    # Calculate turnover penalty
    turnover_penalty = TRANSACTION_COST * cp.sum(cp.abs(weights_var - rolled_forward_weights))
    
    # Define objective function (maximize alpha, minimize penalties)
    objective = cp.Maximize(portfolio_alpha - drift_penalty - turnover_penalty)
    
    # Define constraints
    constraints = [
        cp.sum(weights_var) == 1,  # Weights sum to 1
        weights_var >= 0,  # Long-only constraint
        weights_var <= MAX_WEIGHT  # Maximum weight constraint
    ]
    
    # Create problem
    problem = cp.Problem(objective, constraints)
    
    return weights_var, problem

def optimize_monthly_weights(target_weights, rolled_forward_weights, current_alphas):
    """
    Optimize portfolio weights for a given month using CVXPY
    
    Args:
        target_weights: Target weights for the current month
        rolled_forward_weights: Previous month's weights rolled forward with returns
        current_alphas: Alpha estimates for each country for the current month
        
    Returns:
        pandas.Series: Optimized weights that maximize expected alpha subject to constraints
    """
    # Handle NaN values in alphas
    current_alphas_clean = current_alphas.fillna(0).copy()
    
    # Number of countries
    n_countries = len(rolled_forward_weights)
    
    # Create CVXPY model
    weights_var, problem = create_cvxpy_model(n_countries, current_alphas_clean, rolled_forward_weights, target_weights)
    
    # Solve problem
    problem.solve(warm_start=True)
    
    # Get optimized weights
    optimized_weights = weights_var.value
    
    # Check if optimization was successful
    if optimized_weights is None:
        logging.warning("Optimization failed - weights_var.value is None, using target weights")
        optimized_weights = target_weights.values
    
    # Clip to bounds and normalize
    optimized_weights = np.clip(optimized_weights, 0.0, MAX_WEIGHT)
    optimized_weights = optimized_weights / np.sum(optimized_weights)
    
    return pd.Series(optimized_weights, index=target_weights.index)

def run_optimization(target_weights_df, returns_df, alphas_df):
    """
    Run the complete optimization process using CVXPY
    
    Args:
        target_weights_df: Target weights DataFrame
        returns_df: Returns DataFrame
        alphas_df: Country alphas DataFrame (time-varying)
        
    Returns:
        tuple: (optimized_weights_df, metrics_df)
    """
    logging.info("Starting optimization process...")
    
    # Get common dates and countries
    common_dates = target_weights_df.index.intersection(returns_df.index).intersection(alphas_df.index)
    common_countries = target_weights_df.columns.intersection(returns_df.columns).intersection(alphas_df.columns)
    
    if len(common_dates) == 0:
        raise ValueError("No common dates found between datasets")
    if len(common_countries) == 0:
        raise ValueError("No common countries found between datasets")
    
    # Align all data to common dates and countries
    target_weights_aligned = target_weights_df.loc[common_dates, common_countries]
    returns_aligned = returns_df.loc[common_dates, common_countries]
    alphas_aligned = alphas_df.loc[common_dates, common_countries]
    
    # Fill missing values with column means
    target_weights_aligned = target_weights_aligned.fillna(target_weights_aligned.mean())
    returns_aligned = returns_aligned.fillna(returns_aligned.mean())
    alphas_aligned = alphas_aligned.fillna(alphas_aligned.mean())
    
    logging.info(f"Aligned data: {len(common_dates)} periods, {len(common_countries)} countries")
    
    # Convert to numpy arrays for CVXPY
    n_periods = len(common_dates)
    n_countries = len(common_countries)
    
    # Create CVXPY model with parameters (once only)
    weights_var = cp.Variable(n_countries)
    alpha_param = cp.Parameter(n_countries)
    target_param = cp.Parameter(n_countries)
    rolled_param = cp.Parameter(n_countries)
    
    # Objective function
    portfolio_alpha = ALPHA_MULT * cp.sum(cp.multiply(alpha_param, weights_var))
    drift_penalty = LAMBDA_DRIFT_PENALTY * cp.sum_squares(weights_var - target_param)
    turnover_penalty = TRANSACTION_COST * cp.sum(cp.abs(weights_var - rolled_param))
    objective = cp.Maximize(portfolio_alpha - drift_penalty - turnover_penalty)
    
    # Constraints
    constraints = [
        cp.sum(weights_var) == 1,
        weights_var >= 0,
        weights_var <= MAX_WEIGHT
    ]
    
    # Create problem
    problem = cp.Problem(objective, constraints)
    
    # Initialize tracking variables
    optimized_weights_list = []
    metrics_list = []
    previous_weights = target_weights_aligned.iloc[0]  # Use first month target weights as initial
    
    # Progress tracking
    logging.info(f"Starting monthly optimization for {n_periods} periods...")
    
    for t, date in enumerate(common_dates):
        if t % 30 == 0 or t == n_periods - 1:
            progress = (t + 1) / n_periods * 100
            logging.info(f"... {t+1:3d}/{n_periods} months ({progress:5.1f}%)")
        
        # Roll forward previous weights using current returns
        if t > 0:
            current_returns = returns_aligned.iloc[t]
            rolled_forward_weights = roll_forward_weights(previous_weights, current_returns)
        else:
            rolled_forward_weights = previous_weights.copy()
        
        # Get current data
        current_target = target_weights_aligned.iloc[t]
        current_alphas = alphas_aligned.iloc[t]
        
        # Update CVXPY parameters
        alpha_param.value = current_alphas.values
        target_param.value = current_target.values
        rolled_param.value = rolled_forward_weights.values
        
        # Solve optimization
        problem.solve(warm_start=True)
        
        if problem.status in ["optimal", "optimal_inaccurate"]:
            # Get optimized weights
            optimized_weights = weights_var.value
            
            # Check if optimization was successful
            if optimized_weights is None:
                logging.warning(f"Optimization failed for {date} - weights_var.value is None, using target weights")
                optimized_weights = current_target.values
            else:
                optimized_weights = np.clip(optimized_weights, 0.0, MAX_WEIGHT)
                optimized_weights = optimized_weights / np.sum(optimized_weights)
            
            # Store results
            weights_series = pd.Series(optimized_weights, index=common_countries)
            optimized_weights_list.append(weights_series)
            
            # Calculate metrics
            portfolio_alpha_value = ALPHA_MULT * np.dot(optimized_weights, current_alphas)
            original_portfolio_alpha_value = ALPHA_MULT * np.dot(current_target, current_alphas)
            drift_penalty_value = LAMBDA_DRIFT_PENALTY * np.sum((optimized_weights - current_target) ** 2)
            turnover_penalty_value = TRANSACTION_COST * np.sum(np.abs(optimized_weights - rolled_forward_weights))
            weight_diff_pct = np.sum(np.abs(optimized_weights - current_target)) * 100
            
            metrics_list.append({
                'Date': date,
                'Portfolio_Alpha': portfolio_alpha_value,
                'Original_Portfolio_Alpha': original_portfolio_alpha_value,
                'Optimized_Portfolio_Alpha': portfolio_alpha_value,
                'Weight_Diff_Pct': weight_diff_pct,
                'Drift_Penalty': drift_penalty_value,
                'Turnover_Penalty': turnover_penalty_value,
                'Total_Objective': portfolio_alpha_value - drift_penalty_value - turnover_penalty_value,
                'Solver_Status': problem.status
            })
            
            # Update previous weights for next iteration
            previous_weights = weights_series
            
        else:
            # Fallback if optimization fails
            logging.warning(f"Optimization failed for {date}, using target weights")
            weights_series = current_target.copy()
            optimized_weights_list.append(weights_series)
            
            # Calculate what we can even in fallback case
            original_portfolio_alpha_value = ALPHA_MULT * np.dot(current_target, current_alphas)
            
            metrics_list.append({
                'Date': date,
                'Portfolio_Alpha': original_portfolio_alpha_value,  # Same as original since we're using target weights
                'Original_Portfolio_Alpha': original_portfolio_alpha_value,
                'Optimized_Portfolio_Alpha': original_portfolio_alpha_value,  # Same since fallback to target
                'Weight_Diff_Pct': 0.0,  # No difference since using target weights
                'Drift_Penalty': 0.0,  # No drift penalty since using target weights
                'Turnover_Penalty': TRANSACTION_COST * np.sum(np.abs(current_target - rolled_forward_weights)),
                'Total_Objective': np.nan,  # Can't calculate since optimization failed
                'Solver_Status': problem.status
            })
            
            previous_weights = weights_series
    
    # Create result DataFrames
    optimized_weights_df = pd.DataFrame(optimized_weights_list, index=common_dates)
    metrics_df = pd.DataFrame(metrics_list)
    
    logging.info("Optimization completed successfully")
    
    return optimized_weights_df, metrics_df

def calculate_portfolio_returns(weights_df, returns_df):
    """
    Calculate portfolio returns given weights and country returns
    
    Args:
        weights_df: Portfolio weights DataFrame
        returns_df: Country returns DataFrame
        
    Returns:
        pandas Series: Portfolio returns
    """
    # Find common dates
    common_dates = sorted(list(set(weights_df.index).intersection(set(returns_df.index))))
    
    portfolio_returns = []
    
    for date in common_dates:
        weights = weights_df.loc[date]
        returns = returns_df.loc[date]
        
        # Find common countries
        common_countries = set(weights.index).intersection(set(returns.index))
        
        if len(common_countries) == 0:
            portfolio_returns.append(np.nan)
            continue
        
        # Calculate weighted return
        weighted_return = 0
        total_weight = 0
        
        for country in common_countries:
            if not np.isnan(returns[country]) and weights[country] > 0:
                weighted_return += weights[country] * returns[country]
                total_weight += weights[country]
        
        # Normalize by total weight
        if total_weight > 0:
            portfolio_returns.append(weighted_return / total_weight)
        else:
            portfolio_returns.append(np.nan)
    
    return pd.Series(portfolio_returns, index=common_dates)

def calculate_turnover_series(weights_df):
    """
    Calculate turnover series for a strategy
    
    Args:
        weights_df: Portfolio weights DataFrame
        
    Returns:
        pandas Series: Turnover series
    """
    turnover_data = []
    dates = weights_df.index
    
    for i, date in enumerate(dates):
        if i == 0:
            turnover_data.append(np.nan)
            continue
        
        current_weights = weights_df.loc[date]
        previous_weights = weights_df.loc[dates[i-1]]
        
        # Get all countries
        all_countries = set(current_weights.index).union(set(previous_weights.index))
        
        turnover = 0
        for country in all_countries:
            current_weight = current_weights.get(country, 0)
            previous_weight = previous_weights.get(country, 0)
            
            if pd.isna(current_weight):
                current_weight = 0
            if pd.isna(previous_weight):
                previous_weight = 0
            
            turnover += abs(current_weight - previous_weight)
        
        # One-way turnover
        turnover_data.append(turnover / 2)
    
    return pd.Series(turnover_data, index=dates)

def calculate_performance_stats(returns, turnover=None):
    """
    Calculate comprehensive performance statistics
    
    Args:
        returns: Return series
        turnover: Optional turnover series
        
    Returns:
        pandas Series: Performance statistics
    """
    stats = {}
    
    # Calculate total cumulative return - fix for NaN issue
    if len(returns) == 0 or returns.isna().all():
        stats['Total Return'] = 0.0
        stats['Annual Return'] = 0.0
    else:
        # Remove any NaN values before calculation
        clean_returns = returns.dropna()
        if len(clean_returns) == 0:
            stats['Total Return'] = 0.0
            stats['Annual Return'] = 0.0
        else:
            total_return = (1 + clean_returns).cumprod().iloc[-1] - 1
            stats['Total Return'] = total_return * 100  # Convert to percentage
            
            # Also calculate annualized return for reference
            years = len(clean_returns) / 12
            if years > 0:
                annualized_return = (1 + total_return) ** (1/years) - 1
                stats['Annual Return'] = annualized_return * 100
            else:
                stats['Annual Return'] = 0.0
    
    stats['Annual Vol'] = returns.std() * np.sqrt(12) * 100
    stats['Sharpe Ratio'] = (returns.mean() * 12) / (returns.std() * np.sqrt(12))
    stats['Max Drawdown'] = ((1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1).min() * 100
    stats['Hit Rate'] = (returns > 0).mean() * 100
    stats['Skewness'] = returns.skew()
    stats['Kurtosis'] = returns.kurtosis()
    
    if turnover is not None:
        stats['Avg Monthly Turnover'] = turnover.mean() * 100
        stats['Annual Turnover'] = turnover.mean() * 12 * 100
        stats['Max Monthly Turnover'] = turnover.max() * 100
        stats['Min Monthly Turnover'] = turnover.min() * 100
        stats['Turnover Volatility'] = turnover.std() * 100
    
    return pd.Series(stats)

def calculate_portfolio_factor_exposures(weights_df, exposure_df, portfolio_name):
    """
    Calculate portfolio-level factor exposures by weighting country exposures
    
    Args:
        weights_df: Portfolio weights DataFrame (dates x countries)
        exposure_df: Factor exposure DataFrame (Date, Country, factors...)
        portfolio_name: Name for the portfolio (used in column naming)
        
    Returns:
        pandas DataFrame: Factor exposures for the portfolio over time
    """
    logging.info(f"Calculating factor exposures for {portfolio_name}...")
    
    # Get factor columns (exclude Date and Country)
    factor_columns = [col for col in exposure_df.columns if col not in ['Date', 'Country']]
    
    # Initialize results list
    portfolio_exposures = []
    
    for date in weights_df.index:
        # Get weights for this date
        current_weights = weights_df.loc[date]
        
        # Get exposures for this date
        date_exposures = exposure_df[exposure_df['Date'] == date].copy()
        
        if len(date_exposures) == 0:
            # No exposure data for this date, skip
            continue
        
        # Set country as index for easy lookup
        date_exposures = date_exposures.set_index('Country')
        
        # Calculate weighted exposures for each factor
        factor_exposures = {}
        factor_exposures['Date'] = date
        
        for factor in factor_columns:
            weighted_exposure = 0.0
            total_weight = 0.0
            
            for country in current_weights.index:
                if country in date_exposures.index and not pd.isna(current_weights[country]):
                    country_weight = current_weights[country]
                    country_exposure = date_exposures.loc[country, factor]
                    
                    if not pd.isna(country_exposure) and country_weight > 0:
                        weighted_exposure += country_weight * country_exposure
                        total_weight += country_weight
            
            # Normalize by total weight if we have any valid data
            if total_weight > 0:
                factor_exposures[factor] = weighted_exposure / total_weight
            else:
                factor_exposures[factor] = 0.0
        
        portfolio_exposures.append(factor_exposures)
    
    # Convert to DataFrame
    portfolio_exposures_df = pd.DataFrame(portfolio_exposures)
    if len(portfolio_exposures_df) > 0:
        portfolio_exposures_df = portfolio_exposures_df.set_index('Date')
        portfolio_exposures_df.index = pd.to_datetime(portfolio_exposures_df.index)
    
    logging.info(f"Calculated {len(portfolio_exposures_df)} periods of factor exposures for {portfolio_name}")
    
    return portfolio_exposures_df

def compare_factor_exposures(original_weights_df, optimized_weights_df, exposure_df):
    """
    Compare factor exposures between original and optimized portfolios
    
    Args:
        original_weights_df: Original portfolio weights
        optimized_weights_df: Optimized portfolio weights  
        exposure_df: Factor exposure data
        
    Returns:
        tuple: (exposure_comparison_df, average_exposures_df)
    """
    logging.info("Comparing factor exposures between portfolios...")
    
    # Calculate exposures for both portfolios
    original_exposures = calculate_portfolio_factor_exposures(
        original_weights_df, exposure_df, "Original Strategy"
    )
    optimized_exposures = calculate_portfolio_factor_exposures(
        optimized_weights_df, exposure_df, "Optimized Strategy"
    )
    
    # Find common dates and factors
    if len(original_exposures) == 0 or len(optimized_exposures) == 0:
        logging.warning("No common exposure data found")
        return pd.DataFrame(), pd.DataFrame()
    
    common_dates = sorted(list(set(original_exposures.index).intersection(set(optimized_exposures.index))))
    common_factors = sorted(list(set(original_exposures.columns).intersection(set(optimized_exposures.columns))))
    
    if len(common_dates) == 0 or len(common_factors) == 0:
        logging.warning("No common dates or factors found")
        return pd.DataFrame(), pd.DataFrame()
    
    # Create comparison DataFrames for each type of column
    original_cols = {f'{factor}_Original': original_exposures.loc[common_dates, factor] 
                    for factor in common_factors}
    optimized_cols = {f'{factor}_Optimized': optimized_exposures.loc[common_dates, factor] 
                     for factor in common_factors}
    diff_cols = {f'{factor}_Difference': (optimized_exposures.loc[common_dates, factor] - 
                                        original_exposures.loc[common_dates, factor])
               for factor in common_factors}
    
    # Combine all columns at once using concat
    exposure_comparison = pd.concat([
        pd.DataFrame(original_cols, index=pd.Index(common_dates)),
        pd.DataFrame(optimized_cols, index=pd.Index(common_dates)),
        pd.DataFrame(diff_cols, index=pd.Index(common_dates))
    ], axis=1)
    
    # Calculate average exposures
    average_exposures = pd.DataFrame({
        'Factor': common_factors,
        'Original_Strategy': [original_exposures[factor].mean() for factor in common_factors],
        'Optimized_Strategy': [optimized_exposures[factor].mean() for factor in common_factors]
    })
    
    # Add difference column
    average_exposures['Difference'] = (
        average_exposures['Optimized_Strategy'] - average_exposures['Original_Strategy']
    )
    
    # Add absolute difference for sorting
    average_exposures['Abs_Difference'] = average_exposures['Difference'].abs()
    
    # Sort by absolute difference (largest changes first)
    average_exposures = average_exposures.sort_values('Abs_Difference', ascending=False)
    average_exposures = average_exposures.drop('Abs_Difference', axis=1)
    
    logging.info(f"Calculated factor exposure comparisons for {len(common_factors)} factors over {len(common_dates)} periods")
    
    return exposure_comparison, average_exposures

def calculate_weighted_average_factor_exposure(weights_df, exposure_df, portfolio_name):
    """
    Calculate the weighted average of all factor exposures for a portfolio over time
    
    Args:
        weights_df: Portfolio weights DataFrame (dates x countries)
        exposure_df: Factor exposure DataFrame (Date, Country, factors...)
        portfolio_name: Name for the portfolio (used in column naming)
        
    Returns:
        pandas Series: Weighted average factor exposure for each date
    """
    logging.info(f"Calculating weighted average factor exposure for {portfolio_name}...")
    
    # Get factor columns (exclude Date and Country)
    factor_columns = [col for col in exposure_df.columns if col not in ['Date', 'Country']]
    
    # Initialize results list
    weighted_avg_exposures = []
    
    for date in weights_df.index:
        # Get weights for this date
        current_weights = weights_df.loc[date]
        
        # Get exposures for this date
        date_exposures = exposure_df[exposure_df['Date'] == date].copy()
        
        if len(date_exposures) == 0:
            # No exposure data for this date, skip
            continue
        
        # Set country as index for easy lookup
        date_exposures = date_exposures.set_index('Country')
        
        # Calculate the weighted average across all factors for this date
        total_weighted_exposure = 0.0
        total_weight = 0.0
        
        for country in current_weights.index:
            if country in date_exposures.index and not pd.isna(current_weights[country]):
                country_weight = current_weights[country]
                
                if country_weight > 0:
                    # Sum all factor exposures for this country
                    country_total_exposure = 0.0
                    valid_factors = 0
                    
                    for factor in factor_columns:
                        country_exposure = date_exposures.loc[country, factor]
                        if not pd.isna(country_exposure):
                            country_total_exposure += country_exposure
                            valid_factors += 1
                    
                    # Average exposure across all factors for this country
                    if valid_factors > 0:
                        country_avg_exposure = country_total_exposure / valid_factors
                        total_weighted_exposure += country_weight * country_avg_exposure
                        total_weight += country_weight
        
        # Calculate portfolio-level weighted average
        if total_weight > 0:
            portfolio_weighted_avg = total_weighted_exposure / total_weight
        else:
            portfolio_weighted_avg = 0.0
        
        weighted_avg_exposures.append({
            'Date': date,
            'Weighted_Avg_Exposure': portfolio_weighted_avg
        })
    
    # Convert to DataFrame and then Series
    weighted_avg_df = pd.DataFrame(weighted_avg_exposures)
    if len(weighted_avg_df) > 0:
        weighted_avg_df = weighted_avg_df.set_index('Date')
        weighted_avg_df.index = pd.to_datetime(weighted_avg_df.index)
        result = weighted_avg_df['Weighted_Avg_Exposure']
    else:
        result = pd.Series(dtype=float)
    
    logging.info(f"Calculated {len(result)} periods of weighted average factor exposure for {portfolio_name}")
    
    return result

def compare_weighted_average_exposures(original_weights_df, optimized_weights_df, exposure_df):
    """
    Compare weighted average factor exposures between portfolios over time
    
    Args:
        original_weights_df: Original portfolio weights
        optimized_weights_df: Optimized portfolio weights  
        exposure_df: Factor exposure data
        
    Returns:
        pandas DataFrame: Time series comparison of weighted average exposures
    """
    logging.info("Comparing weighted average factor exposures between portfolios...")
    
    # Calculate weighted average exposures for both portfolios
    original_weighted_avg = calculate_weighted_average_factor_exposure(
        original_weights_df, exposure_df, "Original Strategy"
    )
    optimized_weighted_avg = calculate_weighted_average_factor_exposure(
        optimized_weights_df, exposure_df, "Optimized Strategy"
    )
    
    # Find common dates
    if len(original_weighted_avg) == 0 or len(optimized_weighted_avg) == 0:
        logging.warning("No weighted average exposure data found")
        return pd.DataFrame()
    
    common_dates = sorted(list(set(original_weighted_avg.index).intersection(set(optimized_weighted_avg.index))))
    
    if len(common_dates) == 0:
        logging.warning("No common dates found for weighted average exposures")
        return pd.DataFrame()
    
    # Create comparison DataFrame
    weighted_avg_comparison = pd.DataFrame({
        'Original_Weighted_Avg': original_weighted_avg.loc[common_dates],
        'Optimized_Weighted_Avg': optimized_weighted_avg.loc[common_dates]
    }, index=pd.Index(common_dates))
    
    # Add difference
    weighted_avg_comparison['Difference'] = (
        weighted_avg_comparison['Optimized_Weighted_Avg'] - 
        weighted_avg_comparison['Original_Weighted_Avg']
    )
    
    # Calculate summary statistics
    original_mean = weighted_avg_comparison['Original_Weighted_Avg'].mean()
    optimized_mean = weighted_avg_comparison['Optimized_Weighted_Avg'].mean()
    mean_difference = optimized_mean - original_mean
    
    logging.info(f"Original portfolio mean weighted avg exposure: {original_mean:.4f}")
    logging.info(f"Optimized portfolio mean weighted avg exposure: {optimized_mean:.4f}")
    logging.info(f"Mean difference: {mean_difference:+.4f}")
    
    return weighted_avg_comparison

def compare_performance(original_weights_df, optimized_weights_df, returns_df, metrics_df=None):
    """
    Comprehensive performance comparison between original and optimized strategies
    
    Args:
        original_weights_df: Original target weights
        optimized_weights_df: Optimized weights
        returns_df: Monthly returns
        metrics_df: Optimization metrics DataFrame (optional)
        
    Returns:
        tuple: (performance_results_df, comprehensive_stats_df, summary_stats)
    """
    logging.info("Calculating comprehensive strategy performance...")
    
    # Calculate portfolio returns for both strategies
    original_returns = calculate_portfolio_returns(original_weights_df, returns_df)
    optimized_returns = calculate_portfolio_returns(optimized_weights_df, returns_df)
    
    # Calculate turnover for both strategies
    original_turnover = calculate_turnover_series(original_weights_df)
    optimized_turnover = calculate_turnover_series(optimized_weights_df)
    
    # Load benchmark returns (equal weight)
    try:
        benchmark_file = 'Portfolio_Data.xlsx'
        benchmark_df = pd.read_excel(benchmark_file, sheet_name='Benchmarks', index_col=0)
        benchmark_df.index = pd.to_datetime(benchmark_df.index)
        equal_weight_returns = benchmark_df['equal_weight']
    except:
        logging.warning("Could not load benchmark data. Creating equal weight benchmark.")
        # Create simple equal weight benchmark
        equal_weight_returns = returns_df.mean(axis=1)
    
    # Align all data to common dates
    common_dates = sorted(list(set(original_returns.index).intersection(set(optimized_returns.index)).intersection(set(equal_weight_returns.index))))
    
    # Create comprehensive results DataFrame
    performance_results = pd.DataFrame({
        'Original_Strategy': original_returns.loc[common_dates],
        'Optimized_Strategy': optimized_returns.loc[common_dates],
        'Equal_Weight_Benchmark': equal_weight_returns.loc[common_dates],
        'Original_Turnover': original_turnover.loc[common_dates],
        'Optimized_Turnover': optimized_turnover.loc[common_dates]
    })
    
    # Add optimization metrics if available
    if metrics_df is not None:
        # Ensure metrics_df is aligned with common dates
        metrics_common_dates = sorted(list(set(common_dates).intersection(set(metrics_df.index))))
        
        # Add portfolio alphas if available
        if 'Original_Portfolio_Alpha' in metrics_df.columns and 'Optimized_Portfolio_Alpha' in metrics_df.columns:
            performance_results['Original_Portfolio_Alpha'] = metrics_df.loc[metrics_common_dates, 'Original_Portfolio_Alpha']
            performance_results['Optimized_Portfolio_Alpha'] = metrics_df.loc[metrics_common_dates, 'Optimized_Portfolio_Alpha']
        
        # Add weight difference if available
        if 'Weight_Diff_Pct' in metrics_df.columns:
            performance_results['Weight_Diff_Pct'] = metrics_df.loc[metrics_common_dates, 'Weight_Diff_Pct']
    
    # Calculate net returns (active returns vs benchmark)
    performance_results['Original_Net'] = performance_results['Original_Strategy'] - performance_results['Equal_Weight_Benchmark']
    performance_results['Optimized_Net'] = performance_results['Optimized_Strategy'] - performance_results['Equal_Weight_Benchmark']
    performance_results['Strategy_Difference'] = performance_results['Optimized_Strategy'] - performance_results['Original_Strategy']
    
    # Calculate cumulative returns
    performance_results['Original_Cumulative'] = (1 + performance_results['Original_Strategy']).cumprod()
    performance_results['Optimized_Cumulative'] = (1 + performance_results['Optimized_Strategy']).cumprod()
    performance_results['Benchmark_Cumulative'] = (1 + performance_results['Equal_Weight_Benchmark']).cumprod()
    performance_results['Original_Net_Cumulative'] = (1 + performance_results['Original_Net']).cumprod()
    performance_results['Optimized_Net_Cumulative'] = (1 + performance_results['Optimized_Net']).cumprod()
    
    # Calculate comprehensive statistics
    original_stats = calculate_performance_stats(performance_results['Original_Strategy'], performance_results['Original_Turnover'])
    optimized_stats = calculate_performance_stats(performance_results['Optimized_Strategy'], performance_results['Optimized_Turnover'])
    benchmark_stats = calculate_performance_stats(performance_results['Equal_Weight_Benchmark'])
    original_net_stats = calculate_performance_stats(performance_results['Original_Net'])
    optimized_net_stats = calculate_performance_stats(performance_results['Optimized_Net'])
    
    # Add zero turnover for benchmark and net returns
    turnover_fields = ['Avg Monthly Turnover', 'Annual Turnover', 'Max Monthly Turnover', 'Min Monthly Turnover', 'Turnover Volatility']
    for field in turnover_fields:
        if field in original_stats:
            benchmark_stats[field] = 0.0
            original_net_stats[field] = 0.0
            optimized_net_stats[field] = 0.0
    
    comprehensive_stats = pd.DataFrame({
        'Original_Strategy': original_stats,
        'Optimized_Strategy': optimized_stats,
        'Equal_Weight_Benchmark': benchmark_stats,
        'Original_Net_Return': original_net_stats,
        'Optimized_Net_Return': optimized_net_stats
    })
    
    # Summary statistics
    summary_stats = {
        'Original_Avg_Turnover': performance_results['Original_Turnover'].mean(),
        'Optimized_Avg_Turnover': performance_results['Optimized_Turnover'].mean(),
        'Turnover_Reduction_Pct': (performance_results['Original_Turnover'].mean() - performance_results['Optimized_Turnover'].mean()) / performance_results['Original_Turnover'].mean() * 100,
        'Original_Annual_Return': comprehensive_stats.loc['Annual Return', 'Original_Strategy'],
        'Optimized_Annual_Return': comprehensive_stats.loc['Annual Return', 'Optimized_Strategy'],
        'Original_Sharpe_Ratio': comprehensive_stats.loc['Sharpe Ratio', 'Original_Strategy'],
        'Optimized_Sharpe_Ratio': comprehensive_stats.loc['Sharpe Ratio', 'Optimized_Strategy'],
        'Return_Difference': comprehensive_stats.loc['Annual Return', 'Optimized_Strategy'] - comprehensive_stats.loc['Annual Return', 'Original_Strategy']
    }
    
    return performance_results, comprehensive_stats, summary_stats

def save_results(optimized_weights_df, metrics_df, performance_results_df, comprehensive_stats_df, summary_stats, alphas_df, 
                 exposure_comparison_df=None, average_exposures_df=None, weighted_avg_comparison_df=None):
    """
    Save all results to Excel file with comprehensive performance analysis
    
    Args:
        optimized_weights_df: Optimized weights
        metrics_df: Optimization metrics
        performance_results_df: Comprehensive performance comparison
        comprehensive_stats_df: Detailed performance statistics
        summary_stats: Summary statistics
        alphas_df: Country alphas DataFrame (time-varying)
        exposure_comparison_df: Factor exposure comparison over time (optional)
        average_exposures_df: Average factor exposures comparison (optional)
        weighted_avg_comparison_df: Weighted average factor exposure comparison (optional)
    """
    output_file = 'T2_Optimized_Country_Weights.xlsx'
    
    logging.info(f"Saving results to {output_file}")
    
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Main optimized weights (preserve original column order)
        optimized_weights_df.to_excel(writer, sheet_name='Optimized_Weights')
        
        # Latest weights in column format (countries in original order)
        latest_date = optimized_weights_df.index[-1]
        latest_weights = optimized_weights_df.loc[latest_date]
        latest_weights_df = pd.DataFrame({
            'Country': latest_weights.index,
            'Weight': latest_weights.values
        })
        latest_weights_df.to_excel(writer, sheet_name='Latest_Weights', index=False)
        
        # Monthly returns for all strategies
        performance_results_df[['Original_Strategy', 'Optimized_Strategy', 'Equal_Weight_Benchmark', 
                               'Original_Net', 'Optimized_Net', 'Strategy_Difference']].to_excel(
            writer, sheet_name='Monthly_Returns')
        
        # Cumulative returns
        performance_results_df[['Original_Cumulative', 'Optimized_Cumulative', 'Benchmark_Cumulative',
                               'Original_Net_Cumulative', 'Optimized_Net_Cumulative']].to_excel(
            writer, sheet_name='Cumulative_Returns')
        
        # Comprehensive performance statistics
        comprehensive_stats_df.to_excel(writer, sheet_name='Performance_Statistics')
        
        # Turnover analysis
        performance_results_df[['Original_Turnover', 'Optimized_Turnover']].to_excel(
            writer, sheet_name='Turnover_Analysis')
        
        # Optimization metrics (preserve original order)
        metrics_df.to_excel(writer, sheet_name='Optimization_Metrics')
        
        # Summary statistics
        summary_df = pd.DataFrame([summary_stats]).T
        summary_df.columns = ['Value']
        summary_df.to_excel(writer, sheet_name='Summary_Statistics')
        
        # Country alphas for reference (preserve original column order)
        alphas_df.to_excel(writer, sheet_name='Country_Alphas')
        
        # Configuration parameters
        config_df = pd.DataFrame({
            'Parameter': ['Alpha_Mult', 'Lambda_Drift_Penalty', 'Transaction_Cost', 'Max_Weight'],
            'Value': [ALPHA_MULT, LAMBDA_DRIFT_PENALTY, TRANSACTION_COST, MAX_WEIGHT]
        })
        config_df.to_excel(writer, sheet_name='Configuration', index=False)
        
        # Factor exposure analysis (if available)
        if average_exposures_df is not None and len(average_exposures_df) > 0:
            average_exposures_df.to_excel(writer, sheet_name='Factor_Exposures', index=False)
            logging.info(f"Saved factor exposure analysis: {len(average_exposures_df)} factors")
        
        if exposure_comparison_df is not None and len(exposure_comparison_df) > 0:
            exposure_comparison_df.to_excel(writer, sheet_name='Factor_Exposures_Time_Series')
            logging.info(f"Saved factor exposure time series: {len(exposure_comparison_df)} periods")
        
        # Weighted average factor exposure analysis (if available)
        if weighted_avg_comparison_df is not None and len(weighted_avg_comparison_df) > 0:
            weighted_avg_comparison_df.to_excel(writer, sheet_name='Weighted_Avg_Factor_Exposure')
            logging.info(f"Saved weighted average factor exposure analysis: {len(weighted_avg_comparison_df)} periods")
        
        # Apply proper date formatting to all sheets with date indices
        workbook = writer.book
        date_format = workbook.add_format({'num_format': 'dd-mmm-yyyy'})
        
        # List of sheets that have date indices (need date formatting)
        date_index_sheets = [
            'Optimized_Weights', 'Monthly_Returns', 'Cumulative_Returns', 
            'Turnover_Analysis', 'Optimization_Metrics', 'Country_Alphas'
        ]
        
        # Add time series sheets if they exist
        if exposure_comparison_df is not None and len(exposure_comparison_df) > 0:
            date_index_sheets.append('Factor_Exposures_Time_Series')
        if weighted_avg_comparison_df is not None and len(weighted_avg_comparison_df) > 0:
            date_index_sheets.append('Weighted_Avg_Factor_Exposure')
        
        # Apply date formatting to each sheet with date index
        for sheet_name in date_index_sheets:
            if sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                # Format the index column (column A) as dates
                worksheet.set_column(0, 0, 12, date_format)
        
        # Set column widths for better readability
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            # Set appropriate widths for all columns
            if sheet_name == 'Latest_Weights' or sheet_name == 'Configuration' or sheet_name == 'Factor_Exposures':
                # These sheets don't have date indices, just set general formatting
                worksheet.set_column(0, 0, 15)  # First column (labels/names)
                worksheet.set_column(1, 10, 10)  # Data columns
            else:
                # Sheets with date indices - first column is dates, already formatted above
                worksheet.set_column(1, 20, 10)  # Data columns
    
    logging.info("Results saved successfully")

def create_visualization(performance_results_df, comprehensive_stats_df):
    """
    Create comprehensive performance visualization charts similar to Step Nine
    
    Args:
        performance_results_df: Performance comparison data
        comprehensive_stats_df: Performance statistics
    """
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle('T2 Strategy Optimization: Comprehensive Performance Analysis', fontsize=16)
    
    # Plot 1: Cumulative Total Returns - Compare all strategies
    axes[0, 0].plot(performance_results_df.index, performance_results_df['Original_Cumulative'], 
                   label='Original Strategy', color='blue', linewidth=2)
    axes[0, 0].plot(performance_results_df.index, performance_results_df['Optimized_Cumulative'], 
                   label='Optimized Strategy', color='green', linewidth=2)
    axes[0, 0].plot(performance_results_df.index, performance_results_df['Benchmark_Cumulative'], 
                   label='Equal Weight Benchmark', color='red', alpha=0.7, linewidth=2)
    axes[0, 0].set_title('Cumulative Total Returns')
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Cumulative Net Returns (vs Benchmark)
    axes[0, 1].plot(performance_results_df.index, performance_results_df['Original_Net_Cumulative'], 
                   label='Original vs Benchmark', color='blue', linewidth=2)
    axes[0, 1].plot(performance_results_df.index, performance_results_df['Optimized_Net_Cumulative'], 
                   label='Optimized vs Benchmark', color='green', linewidth=2)
    axes[0, 1].axhline(y=1, color='r', linestyle='--', alpha=0.3)
    axes[0, 1].set_title('Cumulative Net Returns (vs Equal Weight)')
    axes[0, 1].set_ylabel('Cumulative Return')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Monthly Turnover Comparison
    axes[1, 0].plot(performance_results_df.index, performance_results_df['Original_Turnover'] * 100, 
                   label='Original Strategy', color='blue', alpha=0.7, linewidth=2)
    axes[1, 0].plot(performance_results_df.index, performance_results_df['Optimized_Turnover'] * 100, 
                   label='Optimized Strategy', color='green', alpha=0.7, linewidth=2)
    axes[1, 0].set_title('Monthly Portfolio Turnover')
    axes[1, 0].set_ylabel('Turnover (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Portfolio Alphas Over Time - Show both original and optimized alphas
    if 'Original_Portfolio_Alpha' in performance_results_df.columns and 'Optimized_Portfolio_Alpha' in performance_results_df.columns:
        axes[1, 1].plot(performance_results_df.index, performance_results_df['Original_Portfolio_Alpha'], 
                      label='Original Alpha', color='blue', linewidth=2)
        axes[1, 1].plot(performance_results_df.index, performance_results_df['Optimized_Portfolio_Alpha'], 
                      label='Optimized Alpha', color='green', linewidth=2)
        axes[1, 1].set_title('Portfolio Alpha Over Time')
        axes[1, 1].set_ylabel('Alpha (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Force axis limits to ensure data is visible
        alpha_min = min(performance_results_df['Original_Portfolio_Alpha'].min(), 
                       performance_results_df['Optimized_Portfolio_Alpha'].min())
        alpha_max = max(performance_results_df['Original_Portfolio_Alpha'].max(), 
                       performance_results_df['Optimized_Portfolio_Alpha'].max())
        
        # Only set limits if they are valid numbers
        if not (np.isnan(alpha_min) or np.isnan(alpha_max) or np.isinf(alpha_min) or np.isinf(alpha_max)):
            axes[1, 1].set_ylim(alpha_min * 1.1, alpha_max * 1.1)
    else:
        # Fallback to return difference if alpha data not available
        axes[1, 1].plot(performance_results_df.index, performance_results_df['Strategy_Difference'] * 100, 
                      color='purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[1, 1].set_title('Monthly Return Difference (Optimized - Original)')
        axes[1, 1].set_ylabel('Return Difference (%)')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Performance Statistics Bar Chart
    metrics_to_plot = ['Annual Return', 'Annual Vol', 'Sharpe Ratio', 'Max Drawdown']
    strategies = ['Original_Strategy', 'Optimized_Strategy', 'Equal_Weight_Benchmark']
    strategy_labels = ['Original', 'Optimized', 'Benchmark']
    colors = ['blue', 'green', 'red']
    
    x_pos = np.arange(len(metrics_to_plot))
    width = 0.25
    
    for i, (strategy, label, color) in enumerate(zip(strategies, strategy_labels, colors)):
        values = [comprehensive_stats_df.loc[metric, strategy] for metric in metrics_to_plot]
        axes[2, 0].bar(x_pos + i * width, values, width, label=label, color=color, alpha=0.7)
    
    axes[2, 0].set_title('Key Performance Metrics Comparison')
    axes[2, 0].set_ylabel('Value')
    axes[2, 0].set_xticks(x_pos + width)
    axes[2, 0].set_xticklabels(metrics_to_plot, rotation=45)
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Absolute Portfolio Weight Difference
    if 'Weight_Diff_Pct' in performance_results_df.columns:
        axes[2, 1].plot(performance_results_df.index, performance_results_df['Weight_Diff_Pct'], 
                      color='purple', linewidth=2)
        axes[2, 1].set_title('Absolute Portfolio Weight Difference')
        axes[2, 1].set_ylabel('Weight Difference (%)')
        axes[2, 1].set_xlabel('Date')
        axes[2, 1].grid(True, alpha=0.3)
        
        # Ensure proper axis scaling
        weight_min = performance_results_df['Weight_Diff_Pct'].min()
        weight_max = performance_results_df['Weight_Diff_Pct'].max()
        
        # Only set limits if they are valid numbers
        if not (np.isnan(weight_min) or np.isnan(weight_max) or np.isinf(weight_min) or np.isinf(weight_max)):
            axes[2, 1].set_ylim(weight_min * 0.9, weight_max * 1.1)
    else:
        # Fallback to rolling Sharpe if weight difference data not available
        window = 12
        original_rolling_sharpe = performance_results_df['Original_Strategy'].rolling(window).mean() * 12 / (performance_results_df['Original_Strategy'].rolling(window).std() * np.sqrt(12))
        optimized_rolling_sharpe = performance_results_df['Optimized_Strategy'].rolling(window).mean() * 12 / (performance_results_df['Optimized_Strategy'].rolling(window).std() * np.sqrt(12))
        
        axes[2, 1].plot(performance_results_df.index, original_rolling_sharpe, 
                      label='Original Strategy', color='blue', linewidth=2)
        axes[2, 1].plot(performance_results_df.index, optimized_rolling_sharpe, 
                      label='Optimized Strategy', color='green', linewidth=2)
        axes[2, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[2, 1].set_title(f'{window}-Month Rolling Sharpe Ratio')
        axes[2, 1].set_ylabel('Sharpe Ratio')
        axes[2, 1].set_xlabel('Date')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comprehensive analysis plot
    output_file = 'T2_Optimized_Strategy_Analysis.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Comprehensive visualization saved to {output_file}")
    
    # Create a separate turnover-focused plot
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(performance_results_df.index, performance_results_df['Original_Turnover'] * 100, 
             label='Original Strategy', color='blue', linewidth=2)
    plt.plot(performance_results_df.index, performance_results_df['Optimized_Turnover'] * 100, 
             label='Optimized Strategy', color='green', linewidth=2)
    plt.title('Monthly Turnover Comparison')
    plt.ylabel('Turnover (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    cumulative_original_turnover = (performance_results_df['Original_Turnover'] * 100).cumsum()
    cumulative_optimized_turnover = (performance_results_df['Optimized_Turnover'] * 100).cumsum()
    plt.plot(performance_results_df.index, cumulative_original_turnover, 
             label='Original Strategy', color='blue', linewidth=2)
    plt.plot(performance_results_df.index, cumulative_optimized_turnover, 
             label='Optimized Strategy', color='green', linewidth=2)
    plt.title('Cumulative Turnover')
    plt.ylabel('Cumulative Turnover (%)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save turnover analysis plot
    turnover_file = 'T2_Turnover_Analysis.pdf'
    plt.savefig(turnover_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Turnover analysis saved to {turnover_file}")

def create_weighted_average_factor_visualization(weighted_avg_comparison_df):
    """
    Create visualization for weighted average factor exposures comparison
    
    Args:
        weighted_avg_comparison_df: DataFrame with weighted average factor exposures over time
    """
    if len(weighted_avg_comparison_df) == 0:
        logging.warning("No weighted average factor data to visualize")
        return
    
    logging.info("Creating weighted average factor exposure visualization...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle('Portfolio Weighted Average Factor Exposure Analysis', fontsize=16)
    
    # Plot 1: Time series of weighted average exposures
    axes[0].plot(weighted_avg_comparison_df.index, weighted_avg_comparison_df['Original_Weighted_Avg'], 
                label='Original Strategy', color='blue', linewidth=2, alpha=0.8)
    axes[0].plot(weighted_avg_comparison_df.index, weighted_avg_comparison_df['Optimized_Weighted_Avg'], 
                label='Optimized Strategy', color='green', linewidth=2, alpha=0.8)
    
    axes[0].set_title('Weighted Average Factor Exposure Over Time')
    axes[0].set_ylabel('Weighted Average Factor Exposure')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add summary statistics as text box
    original_mean = weighted_avg_comparison_df['Original_Weighted_Avg'].mean()
    optimized_mean = weighted_avg_comparison_df['Optimized_Weighted_Avg'].mean()
    mean_diff = optimized_mean - original_mean
    original_std = weighted_avg_comparison_df['Original_Weighted_Avg'].std()
    optimized_std = weighted_avg_comparison_df['Optimized_Weighted_Avg'].std()
    
    stats_text = f"""Summary Statistics:
Original Mean: {original_mean:.4f}
Optimized Mean: {optimized_mean:.4f}
Mean Difference: {mean_diff:+.4f}
Original Std: {original_std:.4f}
Optimized Std: {optimized_std:.4f}"""
    
    axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                verticalalignment='top', fontsize=10, fontfamily='monospace')
    
    # Plot 2: Difference over time
    axes[1].plot(weighted_avg_comparison_df.index, weighted_avg_comparison_df['Difference'], 
                color='purple', linewidth=2, alpha=0.8)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].fill_between(weighted_avg_comparison_df.index, weighted_avg_comparison_df['Difference'], 
                        0, alpha=0.3, color='purple')
    
    axes[1].set_title('Difference in Weighted Average Factor Exposure (Optimized - Original)')
    axes[1].set_ylabel('Exposure Difference')
    axes[1].set_xlabel('Date')
    axes[1].grid(True, alpha=0.3)
    
    # Add statistics for difference
    diff_mean = weighted_avg_comparison_df['Difference'].mean()
    diff_std = weighted_avg_comparison_df['Difference'].std()
    diff_min = weighted_avg_comparison_df['Difference'].min()
    diff_max = weighted_avg_comparison_df['Difference'].max()
    
    diff_stats_text = f"""Difference Statistics:
Mean: {diff_mean:+.4f}
Std Dev: {diff_std:.4f}
Min: {diff_min:+.4f}
Max: {diff_max:+.4f}"""
    
    axes[1].text(0.02, 0.98, diff_stats_text, transform=axes[1].transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                verticalalignment='top', fontsize=10, fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'T2_Weighted_Average_Factor_Exposure.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Weighted average factor exposure visualization saved to {output_file}")
    
    # Create a secondary plot showing rolling statistics
    plt.figure(figsize=(15, 10))
    
    # Calculate rolling means and standard deviations
    window = 12  # 12-month rolling window
    
    original_rolling_mean = weighted_avg_comparison_df['Original_Weighted_Avg'].rolling(window).mean()
    optimized_rolling_mean = weighted_avg_comparison_df['Optimized_Weighted_Avg'].rolling(window).mean()
    original_rolling_std = weighted_avg_comparison_df['Original_Weighted_Avg'].rolling(window).std()
    optimized_rolling_std = weighted_avg_comparison_df['Optimized_Weighted_Avg'].rolling(window).std()
    
    # Create subplots for rolling analysis
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Rolling Analysis of Weighted Average Factor Exposures (12-Month Window)', fontsize=16)
    
    # Rolling means
    axes[0, 0].plot(original_rolling_mean.index, original_rolling_mean, 
                   label='Original Strategy', color='blue', linewidth=2)
    axes[0, 0].plot(optimized_rolling_mean.index, optimized_rolling_mean, 
                   label='Optimized Strategy', color='green', linewidth=2)
    axes[0, 0].set_title('12-Month Rolling Mean')
    axes[0, 0].set_ylabel('Rolling Mean Exposure')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rolling standard deviations
    axes[0, 1].plot(original_rolling_std.index, original_rolling_std, 
                   label='Original Strategy', color='blue', linewidth=2)
    axes[0, 1].plot(optimized_rolling_std.index, optimized_rolling_std, 
                   label='Optimized Strategy', color='green', linewidth=2)
    axes[0, 1].set_title('12-Month Rolling Standard Deviation')
    axes[0, 1].set_ylabel('Rolling Std Dev')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distribution comparison (histograms)
    axes[1, 0].hist(weighted_avg_comparison_df['Original_Weighted_Avg'], bins=30, alpha=0.7, 
                   color='blue', label='Original Strategy', density=True)
    axes[1, 0].hist(weighted_avg_comparison_df['Optimized_Weighted_Avg'], bins=30, alpha=0.7, 
                   color='green', label='Optimized Strategy', density=True)
    axes[1, 0].set_title('Distribution of Weighted Average Exposures')
    axes[1, 0].set_xlabel('Exposure Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation scatter plot
    axes[1, 1].scatter(weighted_avg_comparison_df['Original_Weighted_Avg'], 
                      weighted_avg_comparison_df['Optimized_Weighted_Avg'], 
                      alpha=0.6, color='purple')
    
    # Add diagonal line
    min_val = min(weighted_avg_comparison_df['Original_Weighted_Avg'].min(), 
                  weighted_avg_comparison_df['Optimized_Weighted_Avg'].min())
    max_val = max(weighted_avg_comparison_df['Original_Weighted_Avg'].max(), 
                  weighted_avg_comparison_df['Optimized_Weighted_Avg'].max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    
    # Calculate and display correlation
    correlation = weighted_avg_comparison_df['Original_Weighted_Avg'].corr(
        weighted_avg_comparison_df['Optimized_Weighted_Avg'])
    
    axes[1, 1].set_title(f'Correlation Analysis (r = {correlation:.3f})')
    axes[1, 1].set_xlabel('Original Strategy Exposure')
    axes[1, 1].set_ylabel('Optimized Strategy Exposure')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the rolling analysis plot
    rolling_output_file = 'T2_Weighted_Average_Factor_Rolling_Analysis.pdf'
    plt.savefig(rolling_output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Rolling analysis visualization saved to {rolling_output_file}")

def main():
    """Main execution function"""
    setup_logging()
    
    logging.info("="*80)
    logging.info("T2 FACTOR TIMING - STEP FOURTEEN: TARGET OPTIMIZATION")
    logging.info("="*80)
    logging.info(f"Alpha Multiplier: {ALPHA_MULT}")
    logging.info(f"Lambda Drift Penalty: {LAMBDA_DRIFT_PENALTY}")
    logging.info(f"Transaction Cost: {TRANSACTION_COST}")
    logging.info(f"Maximum Weight: {MAX_WEIGHT:.1%}")
    logging.info("="*80)
    
    try:
        # Load data
        target_weights_df, returns_df, alphas_df, exposure_df = load_data()
        
        # Align datasets
        target_weights_aligned, returns_aligned, alphas_aligned = align_data(
            target_weights_df, returns_df, alphas_df
        )
        
        # Run optimization
        optimized_weights_df, metrics_df = run_optimization(
            target_weights_aligned, returns_aligned, alphas_aligned
        )
        
        # Set proper date index for metrics_df
        if 'Date' in metrics_df.columns:
            metrics_df = metrics_df.set_index('Date')
            metrics_df.index = pd.to_datetime(metrics_df.index)
        
        # Compare performance with comprehensive analysis
        performance_results_df, comprehensive_stats_df, summary_stats = compare_performance(
            target_weights_aligned, optimized_weights_df, returns_aligned, metrics_df
        )
        
        # Calculate factor exposure comparison
        exposure_comparison_df, average_exposures_df = compare_factor_exposures(
            target_weights_aligned, optimized_weights_df, exposure_df
        )
        
        # Calculate weighted average factor exposure comparison
        weighted_avg_comparison_df = compare_weighted_average_exposures(
            target_weights_aligned, optimized_weights_df, exposure_df
        )
        
        # Display key results
        logging.info("="*80)
        logging.info("COMPREHENSIVE PERFORMANCE RESULTS")
        logging.info("="*80)
        logging.info(f"Average Original Turnover: {summary_stats['Original_Avg_Turnover']:.3f}")
        logging.info(f"Average Optimized Turnover: {summary_stats['Optimized_Avg_Turnover']:.3f}")
        logging.info(f"Turnover Reduction: {summary_stats['Turnover_Reduction_Pct']:.1f}%")
        logging.info(f"Original Annual Return: {summary_stats['Original_Annual_Return']:.2f}%")
        logging.info(f"Optimized Annual Return: {summary_stats['Optimized_Annual_Return']:.2f}%")
        logging.info(f"Return Difference: {summary_stats['Return_Difference']:.2f}%")
        logging.info(f"Original Sharpe Ratio: {summary_stats['Original_Sharpe_Ratio']:.3f}")
        logging.info(f"Optimized Sharpe Ratio: {summary_stats['Optimized_Sharpe_Ratio']:.3f}")
        
        # Display factor exposure results
        if len(average_exposures_df) > 0:
            logging.info("="*80)
            logging.info("FACTOR EXPOSURE COMPARISON")
            logging.info("="*80)
            logging.info("Top 10 factors with largest exposure differences:")
            for i, row in average_exposures_df.head(10).iterrows():
                logging.info(f"{row['Factor']:30s}: Original={row['Original_Strategy']:6.3f}, "
                           f"Optimized={row['Optimized_Strategy']:6.3f}, "
                           f"Diff={row['Difference']:+6.3f}")
        
        # Save comprehensive results
        save_results(optimized_weights_df, metrics_df, performance_results_df, 
                    comprehensive_stats_df, summary_stats, alphas_aligned,
                    exposure_comparison_df, average_exposures_df, weighted_avg_comparison_df)
        
        # Create comprehensive visualization
        create_visualization(performance_results_df, comprehensive_stats_df)
        
        # Create weighted average factor exposure visualization
        create_weighted_average_factor_visualization(weighted_avg_comparison_df)
        
        # Display comprehensive statistics table
        logging.info("="*80)
        logging.info("COMPREHENSIVE PERFORMANCE STATISTICS")
        logging.info("="*80)
        
        # Temporarily set pandas display options to show ALL columns
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        logging.info("\n" + str(comprehensive_stats_df))
        
        # Reset pandas display options to defaults
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width') 
        pd.reset_option('display.max_colwidth')
        
        logging.info("="*80)
        logging.info("STEP FOURTEEN COMPLETED SUCCESSFULLY")
        logging.info("="*80)
        
    except Exception as e:
        logging.error(f"Error in Step Fourteen: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()