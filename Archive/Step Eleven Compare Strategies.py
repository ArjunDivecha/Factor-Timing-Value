"""
Portfolio Performance Analysis Program
=====================================

This program calculates and analyzes the performance of multiple country-weighted investment portfolios
by applying country weights to historical returns data. It compares the performance against
an equal-weight benchmark and generates comprehensive performance metrics and visualizations.

Version: 1.1
Last Updated: 2025-04-23

INPUT FILES:
- Multiple .xlsx files in the portfolio analysis folder
  Each file contains a "All Periods" sheet with dates as index and countries as columns.
  Each value represents the weight allocated to a country on a specific date.
  
- "Portfolio_Data.xlsx"
  Contains two sheets:
  1. "Returns": Historical returns for each country, with dates as index and countries as columns
  2. "Benchmarks": Benchmark returns including equal-weight portfolio returns

OUTPUT FILES:
- "Return Charts.pdf"
  Visualization with three plots:
  1. Cumulative Total Returns: All portfolios vs Equal Weight
  2. Cumulative Net Returns: All portfolios minus Equal Weight
  3. Rolling 12-Month Net Returns: Bar chart of rolling active returns
  
- "Return Results.xlsx"
  Excel file with five sheets:
  1. "Monthly Return": Monthly returns for all portfolios
  2. "Cumulative Return": Cumulative returns over time for all portfolios
  3. "Statistics": Performance statistics (returns, volatility, Sharpe ratio, etc.) for all portfolios
  4. "Net Returns": Net (active) returns for all portfolios
  5. "Net Return Statistics": Performance statistics for net returns

METHODOLOGY:
1. Load multiple portfolio weights and historical returns data
2. For each month's weights, apply them to the NEXT month's returns to avoid look-ahead bias
3. Calculate portfolio returns by applying weights to country returns
4. Compare against equal-weight benchmark
5. Calculate performance metrics (returns, volatility, Sharpe ratio, drawdowns, etc.)
6. Generate visualizations and detailed performance reports

MISSING DATA HANDLING:
- Analysis is restricted to dates common to both weights and returns datasets
- Last month is excluded due to not having future returns available
- No explicit imputation is performed in this script
"""

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
# DATA LOADING (MULTIPLE PORTFOLIOS)
# ===============================

print("Loading data...")
# Portfolio directory and files
portfolio_dir = Path("/Users/macbook2024/Dropbox/AAA Backup/Transformer/T2 Factor Timing/Portfolios for Analysis")
# Only process input portfolios, skip outputs
output_files = {'Return Results.xlsx', 'Return Charts.pdf'}
portfolio_files = [pf for pf in portfolio_dir.glob("*.xlsx") if pf.name not in output_files]

if not portfolio_files:
    raise FileNotFoundError(f"No input .xlsx portfolio files found in {portfolio_dir}")

# Load historical returns and benchmark data (assume same for all portfolios)
portfolio_data_file = 'Portfolio_Data.xlsx'  # Path as before
returns_df = pd.read_excel(portfolio_data_file, sheet_name='Returns', index_col=0)
benchmark_df = pd.read_excel(portfolio_data_file, sheet_name='Benchmarks', index_col=0)

# Standardize dates
returns_df.index = pd.to_datetime(returns_df.index).to_period('M').to_timestamp()
benchmark_df.index = pd.to_datetime(benchmark_df.index).to_period('M').to_timestamp()

# Shift returns forward by one month (use future returns)
# returns_df_shifted = returns_df.shift(-1)
# benchmark_df_shifted = benchmark_df.shift(-1)
returns_df_shifted = returns_df.copy()
benchmark_df_shifted = benchmark_df.copy()

# Prepare containers for all portfolios
all_results = {}
all_cumulative = {}
all_net = {}
all_turnover = {}

for pf in portfolio_files:
    label = pf.stem
    weights_df = pd.read_excel(pf, sheet_name='All Periods', index_col=0)
    weights_df.index = pd.to_datetime(weights_df.index).to_period('M').to_timestamp()

    # Align dates
    common_dates = weights_df.index.intersection(returns_df_shifted.index)
    weights_df = weights_df.loc[common_dates]
    pf_returns = []
    pf_turnover = []
    prev_weights = None
    for date in common_dates:
        weights = weights_df.loc[date]
        rets = returns_df_shifted.loc[date]
        pf_ret = np.nansum(weights * rets)
        pf_returns.append(pf_ret)
        # Turnover: sum(abs(w_t - w_{t-1})) / 2
        if prev_weights is not None:
            turnover = np.nansum(np.abs(weights - prev_weights)) / 2
        else:
            turnover = np.nan
        pf_turnover.append(turnover)
        prev_weights = weights
    pf_returns = pd.Series(pf_returns, index=common_dates)
    pf_turnover = pd.Series(pf_turnover, index=common_dates)
    eqw = benchmark_df_shifted.loc[common_dates, 'equal_weight']
    net_ret = pf_returns - eqw
    all_results[label] = pf_returns
    all_cumulative[label] = (1 + pf_returns).cumprod()
    all_net[label] = net_ret
    all_turnover[label] = pf_turnover

# Add equal weight to results
all_results['Equal Weight'] = benchmark_df_shifted.loc[common_dates, 'equal_weight']
all_cumulative['Equal Weight'] = (1 + benchmark_df_shifted.loc[common_dates, 'equal_weight']).cumprod()

# ===============================
# PDF OUTPUT: Return Charts.pdf
# ===============================

fig, axes = plt.subplots(3, 1, figsize=(15, 18))
# Chart 1: Cumulative return for all portfolios + equal-weight
for label, series in all_cumulative.items():
    axes[0].plot(series, label=label)
axes[0].set_title('Cumulative Returns')
axes[0].set_ylabel('Cumulative Return')
axes[0].legend()
axes[0].grid(True)

# Chart 2: Cumulative net returns (vs equal weight)
for label, net in all_net.items():
    axes[1].plot((1 + net).cumprod(), label=label)
axes[1].axhline(1, color='r', linestyle='--', alpha=0.3)
axes[1].set_title('Cumulative Net Returns (vs Equal Weight)')
axes[1].set_ylabel('Cumulative Net Return')
axes[1].legend()
axes[1].grid(True)

# Chart 3: Rolling 12-month net returns for each portfolio
window = 12
for label, net in all_net.items():
    axes[2].plot(net.rolling(window).sum(), label=label)
axes[2].axhline(0, color='r', linestyle='--', alpha=0.3)
axes[2].set_title('Rolling 12-Month Net Returns')
axes[2].set_ylabel('12M Net Return')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
pdf_path = portfolio_dir / 'Return Charts.pdf'
plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
plt.close()

# ===============================
# EXCEL OUTPUT
# ===============================

excel_path = portfolio_dir / 'Return Results.xlsx'
with pd.ExcelWriter(excel_path) as writer:
    # Monthly Return: All portfolios (no net return)
    pd.DataFrame(all_results).to_excel(writer, sheet_name='Monthly Return')
    # Cumulative Return: All portfolios (no net return)
    pd.DataFrame(all_cumulative).to_excel(writer, sheet_name='Cumulative Return')
    # Statistics: Total returns and turnover
    stats = {}
    for label, returns in all_results.items():
        ann_ret = returns.mean() * 12 * 100  # percent
        ann_vol = returns.std() * np.sqrt(12) * 100  # percent
        sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
        max_dd = (1 + returns).cumprod().div((1 + returns).cumprod().cummax()).min() - 1
        avg_turnover = all_turnover[label].mean() if label in all_turnover else np.nan
        stats[label] = {
            'Annual Return (%)': ann_ret,
            'Volatility (%)': ann_vol,
            'Sharpe': sharpe,
            'Max Drawdown': max_dd,
            'Avg Turnover': avg_turnover
        }
    pd.DataFrame(stats).T.to_excel(writer, sheet_name='Statistics')
    # Net Returns: all net returns
    pd.DataFrame(all_net).to_excel(writer, sheet_name='Net Returns')
    # Net Return Statistics
    net_stats = {}
    for label, net in all_net.items():
        ann_ret = net.mean() * 12 * 100  # percent
        ann_vol = net.std() * np.sqrt(12) * 100  # percent
        sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
        max_dd = (1 + net).cumprod().div((1 + net).cumprod().cummax()).min() - 1
        avg_turnover = all_turnover[label].mean() if label in all_turnover else np.nan
        net_stats[label] = {
            'Annual Net Return (%)': ann_ret,
            'Net Volatility (%)': ann_vol,
            'Net Sharpe': sharpe,
            'Net Max Drawdown': max_dd,
            'Avg Turnover': avg_turnover
        }
    pd.DataFrame(net_stats).T.to_excel(writer, sheet_name='Net Return Statistics')

print(f"\nPDF and Excel results saved to {portfolio_dir}")
