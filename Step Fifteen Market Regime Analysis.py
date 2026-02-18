"""
=============================================================================
SCRIPT NAME: Step Fifteen Market Regime Analysis.py
=============================================================================

INPUT FILES:
- /Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Complete/T2 Factor Timing/T2_Optimized_Country_Weights.xlsx:
  Optimized strategy returns and performance data (Monthly_Returns sheet)
- /Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Complete/T2 Factor Timing/T2_Optimized_Country_Weights.xlsx:
  Weighted average factor exposures (Weighted_Avg_Factor_Exposure sheet)
- /Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Complete/T2 Factor Timing/T2_Top_20_Exposure.csv:
  Individual factor exposures for detailed regime analysis

OUTPUT FILES:
- /Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Complete/T2 Factor Timing/T2_Market_Regime_Analysis.xlsx:
  Comprehensive market regime performance analysis
- /Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Complete/T2 Factor Timing/T2_Market_Regime_Performance.pdf:
  Performance visualization by market regime
- /Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Complete/T2 Factor Timing/T2_Regime_Factor_Analysis.pdf:
  Factor exposure analysis by performance regime

VERSION: 1.0
LAST UPDATED: 2025-06-16
AUTHOR: Claude Code

DESCRIPTION:
Analyzes the optimized T2 strategy performance across different MARKET CONDITION regimes,
identifying how the strategy performs in various market environments and which
factor exposures drive performance in each regime. The analysis includes:

1. Bull/Bear market regime classification (based on market drawdowns)
2. High/Low volatility regime classification (based on market volatility)
3. Economic expansion/contraction classification (based on market proxies)
4. Strategy performance analysis in each market condition regime
5. Factor attribution analysis showing which factors drive performance in each regime

DEPENDENCIES:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- openpyxl

USAGE:
python "Step Fifteen Market Regime Analysis.py"

NOTES:
- Focuses exclusively on optimized strategy performance
- Uses multiple regime classification methods
- Provides both statistical and visual analysis
- Identifies factor patterns associated with performance regimes
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import logging
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================

# Market regime classification parameters
BEAR_MARKET_THRESHOLD = 0.20  # 20% drawdown defines bear market
CORRECTION_THRESHOLD = 0.10   # 10% drawdown defines correction
VOLATILITY_WINDOW = 60        # Rolling window for volatility calculation (days)
VOLATILITY_PERCENTILES = [0.25, 0.75]  # Bottom 25% = Low Vol, Top 25% = High Vol
ECONOMIC_PROXY_WINDOW = 252   # Rolling window for economic regime (1 year)
MIN_REGIME_DURATION = 5       # Minimum days for regime persistence

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
    Load all required data files for market regime analysis
    
    Returns:
        tuple: (strategy_returns, market_returns, factor_exposures)
    """
    logging.info("Loading data for market regime analysis...")
    
    # Load optimized strategy returns and equal weight benchmark
    returns_file = 'T2_Optimized_Country_Weights.xlsx'
    returns_df = pd.read_excel(returns_file, sheet_name='Monthly_Returns', index_col=0)
    returns_df.index = pd.to_datetime(returns_df.index)
    
    # Calculate net returns (optimized strategy minus equal weight benchmark)
    strategy_returns = returns_df['Optimized_Strategy'] - returns_df['Equal_Weight_Benchmark']
    
    # Use equal weight benchmark as market proxy for regime classification
    market_returns = returns_df['Equal_Weight_Benchmark']
    
    logging.info(f"Loaded strategy net returns: {len(strategy_returns)} periods")
    logging.info(f"Loaded market returns for regime classification: {len(market_returns)} periods")
    
    # Load individual factor exposures for attribution analysis
    exposure_file = 'T2_Top_20_Exposure.csv'
    factor_exposures = pd.read_csv(exposure_file)
    factor_exposures['Date'] = pd.to_datetime(factor_exposures['Date'])
    logging.info(f"Loaded factor exposures: {factor_exposures.shape[0]} observations, {len(factor_exposures.columns)-2} factors")
    
    return strategy_returns, market_returns, factor_exposures

def classify_market_regimes(strategy_returns, market_returns):
    """
    Classify periods into market regimes based on MARKET CONDITIONS:
    1. Bull/Bear markets (based on drawdowns)
    2. High/Low volatility (based on market volatility)
    3. Economic expansion/contraction (based on market performance)
    
    Args:
        strategy_returns: Monthly returns of the optimized strategy (net of benchmark)
        market_returns: Monthly returns of market benchmark for regime classification
        
    Returns:
        pandas DataFrame: Market condition regime classifications
    """
    logging.info("Classifying market regimes based on market conditions...")
    
    # Create regime classification DataFrame
    regime_df = pd.DataFrame(index=strategy_returns.index)
    regime_df['Strategy_Returns'] = strategy_returns
    regime_df['Market_Returns'] = market_returns
    
    # =============================================================================
    # 1. BULL/BEAR MARKET CLASSIFICATION (based on drawdowns)
    # =============================================================================
    
    # Calculate cumulative market returns and rolling maximum (peak)
    market_cumulative = (1 + market_returns).cumprod()
    market_peak = market_cumulative.expanding().max()
    
    # Calculate drawdown from peak
    drawdown = (market_cumulative - market_peak) / market_peak
    regime_df['Market_Drawdown'] = drawdown
    
    # Classify market trend regimes
    regime_df['Market_Trend'] = 'Bull'  # Default
    regime_df.loc[drawdown <= -BEAR_MARKET_THRESHOLD, 'Market_Trend'] = 'Bear'
    regime_df.loc[(drawdown <= -CORRECTION_THRESHOLD) & (drawdown > -BEAR_MARKET_THRESHOLD), 'Market_Trend'] = 'Correction'
    
    # =============================================================================
    # 2. VOLATILITY REGIME CLASSIFICATION
    # =============================================================================
    
    # Calculate rolling volatility (convert monthly to annualized)
    market_vol = market_returns.rolling(window=min(12, len(market_returns))).std() * np.sqrt(12)
    regime_df['Market_Volatility'] = market_vol
    
    # Create volatility regime classification using expanding window approach
    regime_df['Volatility_Regime'] = 'Normal'  # Default
    
    for i, date in enumerate(regime_df.index):
        current_vol = market_vol.loc[date]
        if pd.notna(current_vol) and i >= 12:  # Need at least 12 periods of data
            # Get expanding window up to this date
            vol_history = market_vol.loc[:date].dropna()
            
            if len(vol_history) >= 12:
                low_threshold = vol_history.quantile(0.25)
                high_threshold = vol_history.quantile(0.75)
                
                if current_vol <= low_threshold:
                    regime_df.loc[date, 'Volatility_Regime'] = 'Low Vol'
                elif current_vol >= high_threshold:
                    regime_df.loc[date, 'Volatility_Regime'] = 'High Vol'
    
    # =============================================================================
    # 3. ECONOMIC REGIME CLASSIFICATION (based on market performance)
    # =============================================================================
    
    # Use rolling 12-month market performance as economic proxy
    market_12m_return = market_returns.rolling(window=12).sum()
    regime_df['Market_12M_Return'] = market_12m_return
    
    # Classify economic regimes based on market performance
    regime_df['Economic_Regime'] = 'Normal'  # Default
    
    for i, date in enumerate(regime_df.index):
        current_12m = market_12m_return.loc[date]
        if pd.notna(current_12m) and i >= 24:  # Need at least 24 periods of data
            # Get expanding window of 12-month returns up to this date
            returns_history = market_12m_return.loc[:date].dropna()
            
            if len(returns_history) >= 24:
                expansion_threshold = returns_history.quantile(0.6)  # Top 40% = Expansion
                contraction_threshold = returns_history.quantile(0.3)  # Bottom 30% = Contraction
                
                if current_12m >= expansion_threshold:
                    regime_df.loc[date, 'Economic_Regime'] = 'Expansion'
                elif current_12m <= contraction_threshold:
                    regime_df.loc[date, 'Economic_Regime'] = 'Contraction'
    
    # =============================================================================
    # 4. COMBINED REGIME CLASSIFICATION
    # =============================================================================
    
    # Create combined regime (most informative combination)
    regime_df['Combined_Regime'] = (regime_df['Market_Trend'] + '_' + 
                                   regime_df['Volatility_Regime'] + '_' + 
                                   regime_df['Economic_Regime'])
    
    # Create simplified combined regime (Bull/Bear + High/Low Vol)
    regime_df['Simple_Combined'] = regime_df['Market_Trend'] + '_' + regime_df['Volatility_Regime']
    
    # Summary statistics by regime
    logging.info("Market condition regime summary:")
    
    regime_types = ['Market_Trend', 'Volatility_Regime', 'Economic_Regime', 'Simple_Combined']
    for regime_type in regime_types:
        regime_counts = regime_df[regime_type].value_counts()
        total_periods = len(regime_df[regime_type].dropna())
        logging.info(f"\n{regime_type}:")
        for regime, count in regime_counts.items():
            pct = count / total_periods * 100
            logging.info(f"  {regime}: {count} periods ({pct:.1f}%)")
    
    # Log regime transition statistics
    logging.info("\nRegime persistence analysis:")
    for regime_type in ['Market_Trend', 'Volatility_Regime', 'Economic_Regime']:
        regime_series = regime_df[regime_type].dropna()
        if len(regime_series) > 1:
            # Calculate how often regimes persist month-to-month
            regime_changes = (regime_series != regime_series.shift(1)).sum()
            persistence_rate = (len(regime_series) - regime_changes) / len(regime_series) * 100
            logging.info(f"  {regime_type}: {persistence_rate:.1f}% month-to-month persistence")
    
    return regime_df

def analyze_strategy_performance_by_regime(regime_df):
    """
    Analyze strategy performance across different market condition regimes
    
    Args:
        regime_df: Market condition regime classifications
        
    Returns:
        dict: Performance statistics by regime type
    """
    logging.info("Analyzing strategy performance by market condition regime...")
    
    performance_stats = {}
    
    # Analyze performance for each regime type
    regime_types = ['Market_Trend', 'Volatility_Regime', 'Economic_Regime', 'Simple_Combined']
    
    for regime_type in regime_types:
        logging.info(f"\n=== {regime_type.upper()} PERFORMANCE ANALYSIS ===")
        
        regime_stats = regime_df.groupby(regime_type)['Strategy_Returns'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(6)
        
        # Calculate additional performance metrics
        regime_perf = []
        for regime in regime_stats.index:
            regime_data = regime_df[regime_df[regime_type] == regime]['Strategy_Returns'].dropna()
            
            if len(regime_data) > 0:
                annual_return = regime_data.mean() * 12
                annual_vol = regime_data.std() * np.sqrt(12)
                sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else np.nan
                
                # Win rate
                win_rate = (regime_data > 0).mean()
                
                # Max drawdown within regime periods
                regime_cumrets = (1 + regime_data).cumprod()
                regime_peak = regime_cumrets.expanding().max()
                regime_dd = (regime_cumrets - regime_peak) / regime_peak
                max_drawdown = regime_dd.min()
                
                perf_dict = {
                    'Regime': regime,
                    'Periods': len(regime_data),
                    'Avg_Monthly_Return': regime_data.mean(),
                    'Monthly_Volatility': regime_data.std(),
                    'Annual_Return': annual_return,
                    'Annual_Volatility': annual_vol,
                    'Sharpe_Ratio': sharpe_ratio,
                    'Win_Rate': win_rate,
                    'Max_Drawdown': max_drawdown,
                    'Min_Monthly': regime_data.min(),
                    'Max_Monthly': regime_data.max()
                }
                
                regime_perf.append(perf_dict)
                
                # Log key metrics
                logging.info(f"  {regime:15s}: {len(regime_data):3d} periods | "
                           f"Annual: {annual_return:7.2%} | Vol: {annual_vol:6.2%} | "
                           f"Sharpe: {sharpe_ratio:5.2f} | Win%: {win_rate:5.1%}")
        
        performance_stats[regime_type] = pd.DataFrame(regime_perf)
    
    return performance_stats

def analyze_factor_performance_by_regime(regime_df, factor_exposures):
    """
    Analyze which factors drive performance in different market condition regimes
    
    Args:
        regime_df: Market condition regime classifications  
        factor_exposures: Individual factor exposure data
        
    Returns:
        dict: Factor analysis results by regime type
    """
    logging.info("Analyzing factor performance by market condition regime...")
    
    # Get factor columns
    factor_columns = [col for col in factor_exposures.columns if col not in ['Date', 'Country']]
    
    factor_analysis = {}
    
    # Focus on the most interpretable regime types
    key_regimes = ['Market_Trend', 'Volatility_Regime', 'Economic_Regime']
    
    for regime_type in key_regimes:
        logging.info(f"\n=== FACTOR ANALYSIS: {regime_type.upper()} ===")
        
        # Calculate average factor exposures by regime
        regime_factor_analysis = []
        
        for regime in regime_df[regime_type].unique():
            if pd.notna(regime):
                regime_dates = regime_df[regime_df[regime_type] == regime].index
                
                # Get factor exposures for this regime's dates
                regime_exposures = factor_exposures[factor_exposures['Date'].isin(regime_dates)]
                
                if len(regime_exposures) > 0:
                    # Calculate mean exposure for each factor during this regime
                    regime_factor_means = {'Regime': regime, 'Period_Count': len(regime_dates)}
                    
                    for factor in factor_columns:
                        factor_mean = regime_exposures[factor].mean()
                        regime_factor_means[factor] = factor_mean
                    
                    regime_factor_analysis.append(regime_factor_means)
        
        if len(regime_factor_analysis) >= 2:
            factor_regime_df = pd.DataFrame(regime_factor_analysis)
            factor_regime_df = factor_regime_df.set_index('Regime')
            
            # Create factor variation analysis
            factor_data = factor_regime_df.drop('Period_Count', axis=1)
            
            # Calculate which factors vary most across regimes
            factor_stats = pd.DataFrame({
                'Factor': factor_columns,
                'Mean_Exposure': factor_data.mean(axis=0),
                'Regime_Volatility': factor_data.std(axis=0),
                'Regime_Range': factor_data.max(axis=0) - factor_data.min(axis=0)
            })
            
            # Add regime-specific exposures
            for regime in factor_data.index:
                factor_stats[f'{regime}_Exposure'] = factor_data.loc[regime]
            
            # Sort by regime volatility (factors that vary most across regimes)
            factor_stats = factor_stats.sort_values('Regime_Volatility', ascending=False)
            
            # Log top factors
            logging.info(f"Top 10 factors with largest variation across {regime_type}:")
            for i, row in factor_stats.head(10).iterrows():
                regime_exposures = [f"{col.replace('_Exposure', '')}={row[col]:.3f}" 
                                 for col in factor_stats.columns if col.endswith('_Exposure')]
                logging.info(f"  {row['Factor']:30s}: Vol={row['Regime_Volatility']:.3f} | {' | '.join(regime_exposures)}")
            
            factor_analysis[regime_type] = factor_stats
        else:
            logging.warning(f"Insufficient regimes for {regime_type} factor analysis")
            factor_analysis[regime_type] = pd.DataFrame()
    
    return factor_analysis

def create_market_regime_visualizations(regime_df, performance_stats, factor_analysis):
    """
    Create visualizations for market regime analysis
    
    Args:
        regime_df: Market condition regime classifications
        performance_stats: Performance statistics by regime
        factor_analysis: Factor analysis by regime
    """
    logging.info("Creating market regime visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create main analysis plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('T2 Strategy: Market Condition Regime Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Performance by Market Trend (Bull/Bear/Correction)
    if 'Market_Trend' in performance_stats and len(performance_stats['Market_Trend']) > 0:
        trend_perf = performance_stats['Market_Trend']
        bars = axes[0, 0].bar(trend_perf['Regime'], trend_perf['Annual_Return'], 
                             alpha=0.7, color=['green', 'red', 'orange'])
        axes[0, 0].set_title('Annual Returns by Market Trend')
        axes[0, 0].set_ylabel('Annual Return')
        axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, trend_perf['Annual_Return']):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.2%}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Performance by Volatility Regime
    if 'Volatility_Regime' in performance_stats and len(performance_stats['Volatility_Regime']) > 0:
        vol_perf = performance_stats['Volatility_Regime']
        bars = axes[0, 1].bar(vol_perf['Regime'], vol_perf['Sharpe_Ratio'], 
                             alpha=0.7, color=['blue', 'gray', 'purple'])
        axes[0, 1].set_title('Sharpe Ratio by Volatility Regime')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, vol_perf['Sharpe_Ratio']):
            if pd.notna(value):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Performance by Economic Regime
    if 'Economic_Regime' in performance_stats and len(performance_stats['Economic_Regime']) > 0:
        econ_perf = performance_stats['Economic_Regime']
        bars = axes[0, 2].bar(econ_perf['Regime'], econ_perf['Win_Rate'], 
                             alpha=0.7, color=['darkgreen', 'gray', 'darkred'])
        axes[0, 2].set_title('Win Rate by Economic Regime')
        axes[0, 2].set_ylabel('Win Rate')
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, econ_perf['Win_Rate']):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Strategy Returns Over Time by Market Trend
    trend_colors = {'Bull': 'green', 'Bear': 'red', 'Correction': 'orange'}
    for trend, color in trend_colors.items():
        trend_data = regime_df[regime_df['Market_Trend'] == trend]
        if len(trend_data) > 0:
            axes[1, 0].scatter(trend_data.index, trend_data['Strategy_Returns'], 
                              c=color, alpha=0.6, s=20, label=trend)
    
    axes[1, 0].set_title('Strategy Returns Over Time by Market Trend')
    axes[1, 0].set_ylabel('Monthly Net Return')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Regime Distribution (pie chart)
    if len(regime_df['Market_Trend'].dropna()) > 0:
        trend_counts = regime_df['Market_Trend'].value_counts()
        axes[1, 1].pie(trend_counts.values, labels=trend_counts.index, autopct='%1.1f%%',
                      colors=['green', 'red', 'orange'])
        axes[1, 1].set_title('Market Trend Distribution')
    
    # Plot 6: Combined Regime Performance
    if 'Simple_Combined' in performance_stats and len(performance_stats['Simple_Combined']) > 0:
        combined_perf = performance_stats['Simple_Combined']
        # Show only regimes with sufficient data
        sufficient_data = combined_perf[combined_perf['Periods'] >= 5]
        
        if len(sufficient_data) > 0:
            bars = axes[1, 2].bar(range(len(sufficient_data)), sufficient_data['Annual_Return'], 
                                 alpha=0.7)
            axes[1, 2].set_title('Annual Returns by Combined Regime')
            axes[1, 2].set_ylabel('Annual Return')
            axes[1, 2].set_xticks(range(len(sufficient_data)))
            axes[1, 2].set_xticklabels(sufficient_data['Regime'], rotation=45, ha='right')
            axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    output_file = 'T2_Market_Condition_Analysis.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Market regime visualization saved to {output_file}")

def save_market_regime_results(regime_df, performance_stats, factor_analysis):
    """
    Save market regime analysis results to Excel
    
    Args:
        regime_df: Market condition regime classifications
        performance_stats: Performance statistics by regime
        factor_analysis: Factor analysis by regime
    """
    output_file = 'T2_Market_Condition_Analysis.xlsx'
    
    logging.info(f"Saving market regime analysis results to {output_file}")
    
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Main regime classifications
        regime_df.to_excel(writer, sheet_name='Regime_Classifications')
        
        # Performance statistics by regime type
        for regime_type, perf_df in performance_stats.items():
            if len(perf_df) > 0:
                sheet_name = f'{regime_type}_Performance'[:31]
                perf_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Factor analysis by regime type
        for regime_type, factor_df in factor_analysis.items():
            if len(factor_df) > 0:
                sheet_name = f'{regime_type}_Factors'[:31]
                factor_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Summary statistics
        summary_data = []
        for regime_type, perf_df in performance_stats.items():
            if len(perf_df) > 0:
                best_regime = perf_df.loc[perf_df['Sharpe_Ratio'].idxmax()]
                worst_regime = perf_df.loc[perf_df['Sharpe_Ratio'].idxmin()]
                
                summary_data.extend([
                    {
                        'Regime_Type': regime_type,
                        'Best_Regime': best_regime['Regime'],
                        'Best_Annual_Return': best_regime['Annual_Return'],
                        'Best_Sharpe_Ratio': best_regime['Sharpe_Ratio'],
                        'Worst_Regime': worst_regime['Regime'],
                        'Worst_Annual_Return': worst_regime['Annual_Return'],
                        'Worst_Sharpe_Ratio': worst_regime['Sharpe_Ratio']
                    }
                ])
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Performance_Summary', index=False)
        
        # Apply proper date formatting to sheets with date indices
        workbook = writer.book
        date_format = workbook.add_format({'num_format': 'dd-mmm-yyyy'})
        
        # Format date column in Regime_Classifications sheet
        if 'Regime_Classifications' in writer.sheets:
            worksheet = writer.sheets['Regime_Classifications']
            worksheet.set_column(0, 0, 12, date_format)
    
    logging.info("Market regime analysis results saved successfully")

# Removed old detailed factor plot function - replaced with market condition analysis

def save_regime_analysis_results(regime_df, regime_factor_stats, factor_summary, temporal_analysis):
    """
    Save comprehensive market regime analysis results to Excel
    
    Args:
        regime_df: Performance regime classifications
        regime_factor_stats: Regime-based factor statistics  
        factor_summary: Individual factor analysis by regime
        temporal_analysis: Temporal pattern analysis
    """
    output_file = 'T2_Market_Regime_Analysis.xlsx'
    
    logging.info(f"Saving market regime analysis results to {output_file}")
    
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Main regime classifications
        regime_df.to_excel(writer, sheet_name='Regime_Classifications')
        
        # Regime-based performance statistics
        for regime_type, stats_df in regime_factor_stats.items():
            sheet_name = f'{regime_type}_Stats'[:31]  # Excel sheet name limit
            stats_df.to_excel(writer, sheet_name=sheet_name)
        
        # Individual factor analysis
        if len(factor_summary) > 0:
            factor_summary.to_excel(writer, sheet_name='Factor_Regime_Analysis', index=False)
        
        # Temporal analysis results
        if 'monthly_distribution' in temporal_analysis:
            temporal_analysis['monthly_distribution'].to_excel(writer, sheet_name='Monthly_Seasonality')
        
        if 'yearly_distribution' in temporal_analysis:
            temporal_analysis['yearly_distribution'].to_excel(writer, sheet_name='Yearly_Distribution')
        
        # Transition probabilities
        for key, transitions in temporal_analysis.items():
            if 'transitions' in key and isinstance(transitions, pd.DataFrame):
                sheet_name = key.replace('_transitions', '_Transitions')[:31]
                transitions.to_excel(writer, sheet_name=sheet_name)
        
        # Summary statistics by market regime
        summary_stats = []
        for regime in regime_df['Market_Regime'].unique():
            regime_data = regime_df[regime_df['Market_Regime'] == regime]
            if len(regime_data) > 0:
                stats = {
                    'Regime': regime,
                    'Period_Count': len(regime_data),
                    'Avg_Return': regime_data['Returns'].mean(),
                    'Return_Std': regime_data['Returns'].std(),
                    'Min_Return': regime_data['Returns'].min(),
                    'Max_Return': regime_data['Returns'].max(),
                    'Sharpe_Ratio': (regime_data['Returns'].mean() * 12) / (regime_data['Returns'].std() * np.sqrt(12)) if regime_data['Returns'].std() > 0 else np.nan
                }
                if 'Weighted_Avg_Exposure' in regime_data.columns:
                    stats['Avg_Factor_Exposure'] = regime_data['Weighted_Avg_Exposure'].mean()
                    stats['Factor_Exposure_Std'] = regime_data['Weighted_Avg_Exposure'].std()
                
                summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        # Apply proper date formatting to sheets with date indices
        workbook = writer.book
        date_format = workbook.add_format({'num_format': 'dd-mmm-yyyy'})
        
        # List of sheets that have date indices
        date_index_sheets = ['Regime_Classifications', 'Monthly_Seasonality', 'Yearly_Distribution']
        
        # Add regime stats sheets
        for regime_type in regime_factor_stats.keys():
            sheet_name = f'{regime_type}_Stats'[:31]
            date_index_sheets.append(sheet_name)
        
        # Apply date formatting to each sheet with date index
        for sheet_name in date_index_sheets:
            if sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                worksheet.set_column(0, 0, 12, date_format)
    
    logging.info("Market regime analysis results saved successfully")

def main():
    """Main execution function"""
    setup_logging()
    
    logging.info("="*80)
    logging.info("T2 FACTOR TIMING - STEP FIFTEEN: MARKET REGIME ANALYSIS")
    logging.info("="*80)
    logging.info(f"Bear Market Threshold: {BEAR_MARKET_THRESHOLD:.0%} drawdown")
    logging.info(f"Correction Threshold: {CORRECTION_THRESHOLD:.0%} drawdown")
    logging.info(f"Volatility Window: {VOLATILITY_WINDOW} days")
    logging.info(f"Economic Proxy Window: {ECONOMIC_PROXY_WINDOW} days")
    logging.info("="*80)
    
    try:
        # Load data
        strategy_returns, market_returns, factor_exposures = load_data()
        
        # Classify market regimes based on market conditions
        regime_df = classify_market_regimes(strategy_returns, market_returns)
        
        # Analyze strategy performance by regime
        performance_stats = analyze_strategy_performance_by_regime(regime_df)
        
        # Analyze factor performance by regime
        factor_analysis = analyze_factor_performance_by_regime(regime_df, factor_exposures)
        
        # No temporal analysis needed for market condition approach
        
        # Create visualizations
        create_market_regime_visualizations(
            regime_df, performance_stats, factor_analysis
        )
        
        # Save results
        save_market_regime_results(
            regime_df, performance_stats, factor_analysis
        )
        
        # Display key results
        logging.info("="*80)
        logging.info("MARKET REGIME ANALYSIS RESULTS")
        logging.info("="*80)
        
        # Display key results summary
        logging.info("\n" + "="*60)
        logging.info("STRATEGY PERFORMANCE SUMMARY BY MARKET REGIME")
        logging.info("="*60)
        
        # Show performance for each regime type
        for regime_type, perf_df in performance_stats.items():
            if len(perf_df) > 0:
                logging.info(f"\n{regime_type.replace('_', ' ').upper()}:")
                best_regime = perf_df.loc[perf_df['Sharpe_Ratio'].idxmax()]
                worst_regime = perf_df.loc[perf_df['Sharpe_Ratio'].idxmin()]
                
                logging.info(f"  BEST:  {best_regime['Regime']:15s} - {best_regime['Annual_Return']:7.2%} annual, {best_regime['Sharpe_Ratio']:5.2f} Sharpe")
                logging.info(f"  WORST: {worst_regime['Regime']:15s} - {worst_regime['Annual_Return']:7.2%} annual, {worst_regime['Sharpe_Ratio']:5.2f} Sharpe")
        
        # Show top factor insights
        logging.info("\n" + "="*60)
        logging.info("TOP FACTORS BY REGIME SENSITIVITY")
        logging.info("="*60)
        
        for regime_type, factor_df in factor_analysis.items():
            if len(factor_df) > 0:
                logging.info(f"\n{regime_type.replace('_', ' ').upper()} - Most Regime-Sensitive Factors:")
                top_factors = factor_df.head(5)
                for _, row in top_factors.iterrows():
                    logging.info(f"  {row['Factor']:25s}: Volatility={row['Regime_Volatility']:6.3f}, Range={row['Regime_Range']:6.3f}")
        
        logging.info("="*80)
        logging.info("STEP FIFTEEN COMPLETED SUCCESSFULLY")
        logging.info("="*80)
        
    except Exception as e:
        logging.error(f"Error in Step Fifteen: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()