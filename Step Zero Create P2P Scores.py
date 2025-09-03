###############################################################################
# P2PCountryFinal.py
# 
# DESCRIPTION:
# This program implements a Price-to-Peak (P2P) momentum strategy for country ETFs.
# It calculates P2P scores based on price trends and generates historical P2P scores
# for use in other parts of the Country Factor Momentum Strategy pipeline.
#
# VERSION: 1.1
# LAST UPDATED: 2025-05-28
#
# INPUT FILES:
# - AssetList.xlsx
#   Location: /Users/macbook2024/Dropbox/AAA Backup/A Complete/T2 Factor Timing/AssetList.xlsx
#   Description: List of country ETF tickers to analyze
#   Format: Excel file with tickers in the first column
#
# OUTPUT FILES:
# - P2P_Country_Historical_Scores.xlsx
#   Location: /Users/macbook2024/Dropbox/AAA Backup/A Complete/T2 Factor Timing/P2P_Country_Historical_Scores.xlsx
#   Description: Historical P2P scores for all countries over time
#   Format: Excel file with dates and scores by country
#
# MISSING DATA HANDLING:
# - For countries with missing price data, they are excluded from that month's calculation
# - If a country ticker has incomplete data, calculations proceed with available data
###############################################################################

import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import linregress
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

###############################################################################
# CORE CALCULATION FUNCTIONS
###############################################################################

def calculate_p2p_score(prices):
    """
    Calculate Price-to-Peak (P2P) score for a window of prices.
    
    The P2P score combines two momentum factors:
    1. Price relative to its 12-month rolling maximum (price/peak ratio)
    2. Trend strength and direction (r-squared * sign of slope)
    
    Parameters:
    -----------
    prices : pandas.Series
        Time series of prices (typically 12+ months)
    
    Returns:
    --------
    float or None
        P2P score between -1 and 1, where:
        - Positive values indicate positive momentum
        - Negative values indicate negative momentum
        - Higher absolute values indicate stronger trends
        - Returns None if insufficient data
    """
    # Check for sufficient data points
    if len(prices) < 2:
        return None
    
    # Calculate rolling 12-month maximum price
    max_price = prices.rolling(window=12, min_periods=1).max()
    price_to_peak = prices / max_price  # Current price relative to peak
    
    # Calculate trend statistics using linear regression
    x = np.arange(len(prices))
    slope, _, r_value, _, _ = linregress(x, prices)
    r_squared = r_value**2  # Strength of trend (0 to 1)
    sign = 1 if slope > 0 else -1  # Direction of trend
    
    # Final score is most recent P2P ratio adjusted by trend strength and direction
    return price_to_peak.iloc[-1] * r_squared * sign

def calculate_performance_metrics(returns):
    """Calculate key performance metrics"""
    # Annualized return
    ann_return = (1 + returns.mean()) ** 12 - 1
    
    # Annualized volatility
    ann_vol = returns.std() * np.sqrt(12)
    
    # Sharpe ratio (assuming 0% risk-free rate for simplicity)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    
    # Maximum drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    return {
        'Annualized Return': round(ann_return * 100, 2),
        'Annualized Volatility': round(ann_vol * 100, 2),
        'Sharpe Ratio': round(sharpe, 2),
        'Maximum Drawdown': round(max_drawdown * 100, 2)
    }

def run_strategy(tickers, start_date='2000-01-01'):
    """Run P2P strategy"""
    # Convert start_date to datetime if it's string
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
        
    # Get data starting 12 months before the requested start date
    data_start_date = start_date - relativedelta(months=12)
    
    # First download SPY to get the complete date range
    spy_data = yf.download('SPY', start=data_start_date, interval='1mo', progress=False)['Close']
    spy_data.index = spy_data.index.tz_localize(None)
    
    # Initialize DataFrame with SPY's index
    all_prices = pd.DataFrame(index=spy_data.index)
    all_prices['SPY'] = spy_data
    
    # Download data for all other tickers
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=data_start_date, interval='1mo', progress=False)['Close']
            data.index = data.index.tz_localize(None)
            if not data.empty:
                all_prices[ticker] = data
        except Exception as e:
            continue
    
    # Calculate monthly returns
    monthly_returns = all_prices.pct_change()
    
    # Initialize results
    strategy_returns = []
    equal_weight_returns = []
    spy_returns = []
    dates = []
    
    # Find the index position corresponding to start_date
    start_idx = all_prices.index.searchsorted(start_date)
    
    # For each month after start_date
    for i in range(start_idx, len(all_prices)):
        current_date = all_prices.index[i]
        
        # Calculate P2P scores for all stocks using previous 12 months
        scores = {}
        for ticker in tickers:
            if ticker in all_prices.columns:
                price_window = all_prices[ticker].iloc[i-12:i+1]
                if not price_window.isna().any():
                    score = calculate_p2p_score(price_window)
                    if score is not None:
                        scores[ticker] = score
        
        if len(scores) < 2:
            continue
            
        # Convert scores to Series
        score_series = pd.Series(scores)
        
        # Get next month's return if available
        if i + 1 < len(all_prices):
            next_date = all_prices.index[i + 1]
            
            try:
                # Select top 20% of tickers
                num_select = max(1, int(len(scores) * 0.2))
                top_tickers = score_series.nlargest(num_select).index
                
                # Calculate next month returns
                next_returns = monthly_returns.loc[next_date]
                strategy_return = next_returns[top_tickers].mean()
                equal_weight_return = next_returns[score_series.index].mean()
                spy_return = next_returns['SPY']
                
                # Only include if we have valid returns
                if not (np.isnan(strategy_return) or np.isnan(equal_weight_return) or np.isnan(spy_return)):
                    strategy_returns.append(strategy_return)
                    equal_weight_returns.append(equal_weight_return)
                    spy_returns.append(spy_return)
                    dates.append(next_date)
                
            except Exception as e:
                continue
    
    if len(dates) == 0:
        raise ValueError("No valid trading periods found")
    
    # Create return series
    returns_df = pd.DataFrame({
        'P2P Strategy': strategy_returns,
        'Equal Weight': equal_weight_returns,
        'S&P 500': spy_returns
    }, index=dates)
    
    # Calculate cumulative returns
    cum_returns = (1 + returns_df).cumprod()
    
    # Calculate performance metrics
    metrics = {
        column: calculate_performance_metrics(returns_df[column]) 
        for column in returns_df.columns
    }
    
    return returns_df, cum_returns, metrics, all_prices

def plot_performance_and_active(cum_returns, returns_df):
    """Plot both cumulative returns and active returns"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot cumulative returns
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
    for i, column in enumerate(cum_returns.columns):
        ax1.plot(cum_returns.index, cum_returns[column], label=column, 
                color=colors[i], linewidth=2)
    
    ax1.set_title('Cumulative Performance', fontsize=14, pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Cumulative Return (Log Scale)', fontsize=12)
    ax1.legend(fontsize=12, framealpha=0.8)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Calculate and plot active returns vs equal weight
    active_returns = returns_df['P2P Strategy'] - returns_df['Equal Weight']
    active_cum = (1 + active_returns).cumprod()
    
    ax2.plot(active_cum.index, active_cum, label='Active Return', 
             color='#9b59b6', linewidth=2)
    ax2.set_title('Cumulative Active Return vs Equal Weight', fontsize=14, pad=20)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Cumulative Active Return', fontsize=12)
    ax2.legend(fontsize=12, framealpha=0.8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def calculate_active_metrics(returns_df):
    """Calculate metrics for active returns"""
    active_returns = returns_df['P2P Strategy'] - returns_df['Equal Weight']
    
    # Calculate information ratio and tracking error
    tracking_error = active_returns.std() * np.sqrt(12)
    info_ratio = active_returns.mean() * 12 / tracking_error if tracking_error != 0 else 0
    
    # Calculate hit ratio
    hit_ratio = (active_returns > 0).mean() * 100
    
    # Calculate maximum drawdown
    active_cum = (1 + active_returns).cumprod()
    rolling_max = active_cum.cummax()
    drawdowns = (active_cum - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    return {
        'Active Return': round(active_returns.mean() * 12 * 100, 2),
        'Tracking Error': round(tracking_error * 100, 2),
        'Information Ratio': round(info_ratio, 2),
        'Hit Ratio': round(hit_ratio, 2),
        'Max Active Drawdown': round(max_drawdown * 100, 2)
    }

def calculate_annual_returns(returns_df):
    """Calculate annual returns and active returns"""
    # Convert to annual returns
    annual_returns = returns_df.groupby(returns_df.index.year).apply(
        lambda x: (1 + x).prod() - 1
    )
    
    # Calculate annual active returns
    annual_active = annual_returns['P2P Strategy'] - annual_returns['Equal Weight']
    
    # Create formatted DataFrame
    results = pd.DataFrame({
        'P2P Strategy': annual_returns['P2P Strategy'] * 100,
        'Equal Weight': annual_returns['Equal Weight'] * 100,
        'S&P 500': annual_returns['S&P 500'] * 100,
        'Active Return': annual_active * 100
    }).round(2)
    
    return results

def get_current_signals(all_prices, tickers):
    """Calculate current P2P scores for all tickers"""
    details = []
    last_date = all_prices.index[-1]
    
    for ticker in tickers:  # Iterate through tickers in original order
        if ticker in all_prices.columns:
            # Get last 12 months of prices
            price_window = all_prices[ticker].iloc[-13:]  # Extra month for current
            if not price_window.isna().any() and len(price_window) >= 12:
                # Calculate components
                max_price = price_window.rolling(window=12, min_periods=1).max()
                price_to_peak = price_window / max_price
                
                # Calculate trend statistics
                x = np.arange(len(price_window))
                slope, _, r_value, _, _ = linregress(x, price_window)
                r_squared = r_value**2
                score = price_to_peak.iloc[-1] * r_squared * (1 if slope > 0 else -1)
                
                details.append({
                    'Ticker': ticker,
                    'Current Price': price_window.iloc[-1],
                    'Rolling Max': max_price.iloc[-1],
                    'P2P Ratio': price_to_peak.iloc[-1],
                    'R-Squared': r_squared,
                    'Trend Slope': slope,
                    'Adjusted P2P': score
                })
    
    details_df = pd.DataFrame(details)
    return details_df, last_date

def format_current_signals(details_df, as_of_date):
    """Format current signals table"""
    # Get current date and time
    current_date = datetime.now()
    
    output = "\n" + "="*100 + "\n"
    output += " "*35 + "CURRENT SIGNALS AND RANKINGS\n"
    output += " "*35 + f"(as of {current_date.strftime('%Y-%m-%d')})\n"
    output += "="*100 + "\n\n"
    
    # Format the detailed DataFrame
    formatted_df = details_df.copy()
    formatted_df['Current Price'] = formatted_df['Current Price'].map(lambda x: f"${x:.2f}")
    formatted_df['Rolling Max'] = formatted_df['Rolling Max'].map(lambda x: f"${x:.2f}")
    formatted_df['P2P Ratio'] = formatted_df['P2P Ratio'].map(lambda x: f"{x:.3f}")
    formatted_df['R-Squared'] = formatted_df['R-Squared'].map(lambda x: f"{x:.3f}")
    formatted_df['Trend Slope'] = formatted_df['Trend Slope'].map(lambda x: f"{x:+.3f}")
    formatted_df['Adjusted P2P'] = formatted_df['Adjusted P2P'].map(lambda x: f"{x:+.3f}")
    
    output += formatted_df.to_string(index=False)
    return output

def format_enhanced_results(metrics, cum_returns, returns_df, start_date):
    """Format all results including active and annual returns"""
    # Base metrics
    df = pd.DataFrame(metrics)
    latest_date = cum_returns.index[-1].strftime('%Y-%m-%d')
    total_returns = (cum_returns.iloc[-1] - 1) * 100
    df.loc['Total Return'] = total_returns
    
    # Get active metrics
    active_metrics = calculate_active_metrics(returns_df)
    
    # Get annual returns
    annual_rets = calculate_annual_returns(returns_df)
    
    # Format main metrics table
    output = "\n" + "="*100 + "\n"
    output += " "*35 + "STRATEGY PERFORMANCE METRICS\n"
    output += " "*35 + f"({start_date} to {latest_date})\n"
    output += "="*100 + "\n\n"
    
    # Format metrics
    formatted_df = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df.columns:
        for idx in df.index:
            value = df.loc[idx, col]
            if "Ratio" in idx:
                formatted_df.loc[idx, col] = f"{value:>8.2f}"
            else:
                formatted_df.loc[idx, col] = f"{value:>8.2f}%"
    
    output += formatted_df.to_string(justify='right')
    
    # Add active metrics
    output += "\n\n" + "="*100 + "\n"
    output += " "*35 + "ACTIVE RETURN METRICS\n"
    output += "="*100 + "\n\n"

    active_df = pd.Series(active_metrics).to_frame('Value')
    active_df['Value'] = active_df['Value'].apply(
        lambda x: f"{x:>8.2f}%" if "Ratio" not in active_df.index[active_df['Value'] == x][0] else f"{x:>8.2f}"
    )
    output += active_df.to_string(justify='right')
    
    # Add annual returns
    output += "\n\n" + "="*100 + "\n"
    output += " "*35 + "ANNUAL RETURNS\n"
    output += "="*100 + "\n\n"
    
    # Format annual returns
    annual_formatted = annual_rets.applymap(lambda x: f"{x:>8.2f}%")
    output += annual_formatted.to_string(justify='right')
    
    return output + "\n\n" + "="*100

def save_historical_p2p_scores(all_prices, tickers, start_date='2000-01-01'):
    """Save historical P2P scores in the format of the reference file"""
    # Convert start_date to datetime if it's string
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
        
    # Get data starting 12 months before the requested start date
    data_start_date = start_date - relativedelta(months=12)
    
    # Initialize DataFrame for P2P scores
    p2p_scores = pd.DataFrame(index=all_prices.index)
    
    # For each month
    for i in range(12, len(all_prices)):
        current_date = all_prices.index[i]
        
        # Calculate P2P scores for all stocks using previous 12 months
        for ticker in tickers:
            if ticker in all_prices.columns:
                price_window = all_prices[ticker].iloc[i-12:i+1]
                if not price_window.isna().any() and len(price_window) > 1:
                    score = calculate_p2p_score(price_window)
                    if score is not None:
                        p2p_scores.loc[current_date, ticker] = score
    
    # Filter dates from start_date onwards
    p2p_scores = p2p_scores[p2p_scores.index >= start_date]
    
    # Reset index to make date a column
    p2p_scores.reset_index(inplace=True)
    p2p_scores.rename(columns={'index': 'Unnamed: 0'}, inplace=True)
    
    return p2p_scores

# Read tickers and run strategy
asset_list = pd.read_excel('AssetList.xlsx')
tickers = [str(ticker).strip().upper() for ticker in asset_list.iloc[:, 0] if pd.notna(ticker)]

# Run strategy
returns_df, cum_returns, metrics, all_prices = run_strategy(tickers)

# Make sure all datetime indexes are timezone-naive
returns_df.index = returns_df.index.tz_localize(None)
cum_returns.index = cum_returns.index.tz_localize(None)

# Format and display results
metrics_start_date = returns_df.index[0].strftime('%Y-%m-%d')
print(format_enhanced_results(metrics, cum_returns, returns_df, metrics_start_date))

# Get and display current signals
details_df, as_of_date = get_current_signals(all_prices, tickers)
print(format_current_signals(details_df, as_of_date))

# Calculate historical P2P scores
historical_p2p_scores = save_historical_p2p_scores(all_prices, tickers)

# Save only the historical P2P scores file
historical_file = 'P2P_Country_Historical_Scores.xlsx'
print(f"\nSaving historical P2P scores to {historical_file}...")

# Create Excel writer with date formatting
with pd.ExcelWriter(historical_file, engine='xlsxwriter') as writer:
    # Write the DataFrame
    historical_p2p_scores.to_excel(writer, sheet_name='Sheet1', index=False)
    
    # Access workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    
    # Create date format
    date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
    
    # Apply format to date column
    date_col_idx = 0  # First column is the date
    for row in range(len(historical_p2p_scores)):
        try:
            date_value = historical_p2p_scores.iloc[row, date_col_idx]
            worksheet.write(row+1, date_col_idx, date_value, date_format)
        except Exception as e:
            print(f"Warning: Could not format date at row {row+1}: {str(e)}")
    
    # Adjust column width for better visibility
    worksheet.set_column(date_col_idx, date_col_idx, 12)  # Width for date column
    worksheet.set_column(1, len(historical_p2p_scores.columns)-1, 8)  # Width for score columns

print(f"Successfully saved historical P2P scores to {historical_file}")
print("Note: Other Excel files and charts were not generated as requested.")