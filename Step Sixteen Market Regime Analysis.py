"""
=============================================================================
SCRIPT NAME: Step Sixteen Market Regime Analysis.py
=============================================================================

NOTE: All file references in this script are now LOCAL FILE NAMES ONLY (e.g., 'T2_Factor_Performance_Heatmap.pdf'). Full directory paths have been removed for portability and clarity.


INPUT FILES:
- Normalized_T2_MasterCSV.csv:
  Normalized factor data and country returns (date, country, variable, value)
- T2_Top_20_Exposure.csv:
  Binary exposure matrix showing which countries are in each factor portfolio
- Portfolio_Data.xlsx:
  Equal weight benchmark returns for market regime classification
- T2 Top20.xlsx:
  Factor performance statistics (for reference)

OUTPUT FILES:
- T2_Factor_Market_Condition_Analysis.xlsx:
  Comprehensive factor performance analysis by market condition
- T2_Factor_Performance_Heatmap.pdf:
  Heatmap showing average factor returns under each market condition
- T2_Factor_Quilt_Chart.pdf:
  Factor quilt plot showing performance over time with market regime backgrounds
- T2_Factor_Correlation_Analysis.pdf:
  Conditional correlation analysis by market regime

VERSION: 1.0
LAST UPDATED: 2025-01-16
AUTHOR: Claude Code

DESCRIPTION:
Comprehensive factor performance analysis by market condition. This analysis provides
a robust and detailed examination of how individual factors perform under different,
well-defined market conditions. Key features include:

1. RICHER SET OF MARKET CONDITIONS:
   - Volatility: High vs. Low (based on rolling volatility percentiles)
   - Momentum: Strong vs. Weak (based on 12-month benchmark returns)
   - Dispersion: High vs. Low (cross-sectional standard deviation of country returns)
   - Economic Cycle: Expansion vs. Contraction (composite indicator)

2. FACTOR-LEVEL ANALYSIS:
   - Performance of each individual factor from T2 Top20 portfolios
   - Factor scorecard showing best/worst factors in each environment
   - Detailed attribution analysis

3. CONDITIONAL FACTOR CORRELATIONS:
   - How correlations between factors change in different market conditions
   - Identification of factors providing diversification benefits

4. COMPREHENSIVE VISUALIZATIONS:
   - Heatmaps of average factor returns by market condition
   - Factor quilt plots with market regime backgrounds
   - Correlation matrices by regime

DEPENDENCIES:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- openpyxl

USAGE:
python "Step Sixteen Market Regime Analysis.py"

NOTES:
- Focuses on individual factor performance, not combined strategy
- Uses multiple market condition dimensions for robust analysis
- Provides actionable insights for strategy refinement
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# Market condition thresholds
VOLATILITY_PERCENTILES = [0.33, 0.67]  # Low/Medium/High volatility
MOMENTUM_PERCENTILES = [0.33, 0.67]    # Weak/Neutral/Strong momentum
DISPERSION_PERCENTILES = [0.33, 0.67]  # Low/Medium/High dispersion
ECONOMIC_PERCENTILES = [0.4, 0.6]      # Contraction/Normal/Expansion

# Analysis parameters
MIN_PERIODS_PER_REGIME = 10           # Minimum periods for meaningful analysis
CORRELATION_MIN_PERIODS = 20          # Minimum periods for correlation calculation
TOP_N_FACTORS = 10                    # Number of top factors to highlight

# Skip these variables in factor analysis
SKIP_VARIABLES = [
    "1MRet", "3MRet", "6MRet", "9MRet", "12MRet",
    "120MA_CS", "129MA_TS", "Agriculture_TS", "Agriculture_CS",
    "Copper_TS", "Copper_CS", "Gold_CS", "Gold_TS",
    "Oil_CS", "Oil_TS", "MCAP Adj_CS", "MCAP Adj_TS",
    "MCAP_CS", "MCAP_TS", "PX_LAST_CS", "PX_LAST_TS",
    "Tot Return Index_CS", "Tot Return Index_TS",
    "Currency_CS", "Currency_TS", "BEST EPS_CS", "BEST EPS_TS",
    "Trailing EPS_CS", "Trailing EPS_TS"
]

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
    Load all required data files for factor-level market regime analysis
    
    Returns:
        tuple: (factor_returns, benchmark_returns, country_returns, factor_exposures)
    """
    logging.info("Loading data for factor-level market regime analysis...")
    
    # Load normalized data with factor values and returns
    data_file = 'Normalized_T2_MasterCSV.csv'
    data = pd.read_csv(data_file)
    data['date'] = pd.to_datetime(data['date'])
    
    # Load benchmark returns
    benchmark_file = 'Portfolio_Data.xlsx'
    benchmark_df = pd.read_excel(benchmark_file, sheet_name='Benchmarks', index_col=0)
    benchmark_df.index = pd.to_datetime(benchmark_df.index)
    benchmark_returns = benchmark_df['equal_weight']
    
    # Load factor exposures
    exposure_file = 'T2_Top_20_Exposure.csv'
    exposures = pd.read_csv(exposure_file)
    exposures['Date'] = pd.to_datetime(exposures['Date'])
    
    logging.info(f"Loaded normalized data: {data.shape[0]} rows")
    logging.info(f"Loaded benchmark returns: {len(benchmark_returns)} periods")
    logging.info(f"Loaded factor exposures: {exposures.shape}")
    
    return data, benchmark_returns, exposures

def calculate_factor_returns(data, exposures, benchmark_returns):
    """
    Calculate individual factor portfolio returns
    
    Args:
        data: Normalized data with country returns and factor values
        exposures: Binary exposure matrix from T2_Top_20_Exposure.csv
        benchmark_returns: Equal weight benchmark returns
        
    Returns:
        DataFrame: Factor returns with dates as index and factors as columns
    """
    logging.info("Calculating individual factor returns...")
    
    # Get unique dates and factors
    dates = sorted(exposures['Date'].unique())
    factors = [col for col in exposures.columns if col not in ['Date', 'Country']]
    
    # Filter out skip variables
    factors = [f for f in factors if f not in SKIP_VARIABLES]
    
    logging.info(f"Calculating returns for {len(factors)} factors across {len(dates)} dates")
    
    # Initialize results DataFrame
    factor_returns = pd.DataFrame(index=pd.DatetimeIndex(dates), columns=factors)
    
    # Get return data
    returns_data = data[data['variable'] == '1MRet'].copy()
    returns_data['date'] = pd.to_datetime(returns_data['date'])
    
    # Calculate returns for each factor and date
    for date in dates:
        # Get exposures for this date
        date_exposures = exposures[exposures['Date'] == date]
        
        # Get returns for this date
        date_returns = returns_data[returns_data['date'] == date]
        
        if len(date_returns) > 0:
            # Create a mapping of country to return
            country_returns_map = dict(zip(date_returns['country'], date_returns['value']))
            
            # Calculate return for each factor
            for factor in factors:
                # Get countries in the factor portfolio (where exposure = 1)
                factor_countries = date_exposures[date_exposures[factor] == 1]['Country'].tolist()
                
                if factor_countries:
                    # Calculate equal-weighted return
                    factor_return = np.mean([country_returns_map.get(country, np.nan) 
                                           for country in factor_countries 
                                           if country in country_returns_map])
                    factor_returns.loc[date, factor] = factor_return
    
    # Convert to numeric and align with benchmark dates
    factor_returns = factor_returns.apply(pd.to_numeric, errors='coerce')
    factor_returns = factor_returns.reindex(benchmark_returns.index)
    
    # Calculate excess returns
    factor_excess_returns = factor_returns.subtract(benchmark_returns, axis=0)
    
    logging.info(f"Calculated factor returns shape: {factor_returns.shape}")
    
    return factor_returns, factor_excess_returns

def calculate_market_conditions(benchmark_returns, country_returns_data):
    """
    Calculate comprehensive market conditions based on multiple criteria
    
    Args:
        benchmark_returns: Equal weight benchmark returns
        country_returns_data: Country-level return data for dispersion calculation
        
    Returns:
        DataFrame: Market conditions with multiple regime classifications
    """
    logging.info("Calculating comprehensive market conditions...")
    
    # Initialize market conditions DataFrame
    conditions = pd.DataFrame(index=benchmark_returns.index)
    conditions['Benchmark_Return'] = benchmark_returns
    
    # =============================================================================
    # 1. VOLATILITY REGIME (High/Medium/Low)
    # =============================================================================
    
    # Calculate rolling volatility
    volatility = benchmark_returns.rolling(window=12, min_periods=6).std() * np.sqrt(12)
    conditions['Volatility'] = volatility
    
    # Classify volatility regime using expanding percentiles
    conditions['Volatility_Regime'] = 'Medium'
    
    for i, date in enumerate(conditions.index):
        if i >= 12:  # Need sufficient history
            vol_history = volatility[:date].dropna()
            if len(vol_history) >= 12:
                low_threshold = vol_history.quantile(VOLATILITY_PERCENTILES[0])
                high_threshold = vol_history.quantile(VOLATILITY_PERCENTILES[1])
                
                current_vol = volatility.loc[date]
                if pd.notna(current_vol):
                    if current_vol <= low_threshold:
                        conditions.loc[date, 'Volatility_Regime'] = 'Low'
                    elif current_vol >= high_threshold:
                        conditions.loc[date, 'Volatility_Regime'] = 'High'
    
    # =============================================================================
    # 2. MOMENTUM REGIME (Strong/Neutral/Weak)
    # =============================================================================
    
    # Calculate 12-month momentum
    momentum = benchmark_returns.rolling(window=12, min_periods=6).sum()
    conditions['Momentum_12M'] = momentum
    
    # Classify momentum regime
    conditions['Momentum_Regime'] = 'Neutral'
    
    for i, date in enumerate(conditions.index):
        if i >= 24:  # Need sufficient history
            mom_history = momentum[:date].dropna()
            if len(mom_history) >= 24:
                weak_threshold = mom_history.quantile(MOMENTUM_PERCENTILES[0])
                strong_threshold = mom_history.quantile(MOMENTUM_PERCENTILES[1])
                
                current_mom = momentum.loc[date]
                if pd.notna(current_mom):
                    if current_mom <= weak_threshold:
                        conditions.loc[date, 'Momentum_Regime'] = 'Weak'
                    elif current_mom >= strong_threshold:
                        conditions.loc[date, 'Momentum_Regime'] = 'Strong'
    
    # =============================================================================
    # 3. DISPERSION REGIME (High/Medium/Low)
    # =============================================================================
    
    # Calculate cross-sectional dispersion of country returns
    dispersion_series = pd.Series(index=benchmark_returns.index, dtype=float)
    
    for date in benchmark_returns.index:
        # Get country returns for this date
        date_returns = country_returns_data[country_returns_data['date'] == date]
        if len(date_returns) > 5:  # Need sufficient countries
            dispersion = date_returns['value'].std()
            dispersion_series.loc[date] = dispersion
    
    conditions['Dispersion'] = dispersion_series
    
    # Classify dispersion regime
    conditions['Dispersion_Regime'] = 'Medium'
    
    for i, date in enumerate(conditions.index):
        if i >= 12:  # Need sufficient history
            disp_history = dispersion_series[:date].dropna()
            if len(disp_history) >= 12:
                low_threshold = disp_history.quantile(DISPERSION_PERCENTILES[0])
                high_threshold = disp_history.quantile(DISPERSION_PERCENTILES[1])
                
                current_disp = dispersion_series.loc[date]
                if pd.notna(current_disp):
                    if current_disp <= low_threshold:
                        conditions.loc[date, 'Dispersion_Regime'] = 'Low'
                    elif current_disp >= high_threshold:
                        conditions.loc[date, 'Dispersion_Regime'] = 'High'
    
    # =============================================================================
    # 4. ECONOMIC CYCLE (Expansion/Normal/Contraction)
    # =============================================================================
    
    # Use composite of momentum and volatility as economic proxy
    # High momentum + Low volatility = Expansion
    # Low momentum + High volatility = Contraction
    
    conditions['Economic_Score'] = pd.Series(index=conditions.index, dtype=float)
    
    # Create numeric scores
    vol_score = conditions['Volatility_Regime'].map({'Low': 1, 'Medium': 0, 'High': -1})
    mom_score = conditions['Momentum_Regime'].map({'Strong': 1, 'Neutral': 0, 'Weak': -1})
    
    conditions['Economic_Score'] = (vol_score + mom_score) / 2
    
    # Classify economic regime
    conditions['Economic_Regime'] = 'Normal'
    conditions.loc[conditions['Economic_Score'] >= 0.5, 'Economic_Regime'] = 'Expansion'
    conditions.loc[conditions['Economic_Score'] <= -0.5, 'Economic_Regime'] = 'Contraction'
    
    # =============================================================================
    # 5. COMBINED REGIMES
    # =============================================================================
    
    # Create informative combined regimes
    conditions['Vol_Mom'] = conditions['Volatility_Regime'] + '_' + conditions['Momentum_Regime']
    conditions['Full_Regime'] = (conditions['Volatility_Regime'] + '_' + 
                                conditions['Momentum_Regime'] + '_' + 
                                conditions['Dispersion_Regime'])
    
    # Log regime distribution
    logging.info("\nMarket Condition Distribution:")
    for regime_type in ['Volatility_Regime', 'Momentum_Regime', 'Dispersion_Regime', 'Economic_Regime']:
        regime_counts = conditions[regime_type].value_counts()
        logging.info(f"\n{regime_type}:")
        for regime, count in regime_counts.items():
            pct = count / len(conditions) * 100
            logging.info(f"  {regime}: {count} periods ({pct:.1f}%)")
    
    return conditions

def analyze_factor_performance_by_condition(factor_returns, factor_excess_returns, conditions):
    """
    Analyze individual factor performance under each market condition
    
    Args:
        factor_returns: Absolute factor returns
        factor_excess_returns: Factor excess returns vs benchmark
        conditions: Market condition classifications
        
    Returns:
        dict: Performance analysis results by condition type
    """
    logging.info("Analyzing factor performance by market condition...")
    
    performance_results = {}
    
    # Analyze each condition type
    condition_types = ['Volatility_Regime', 'Momentum_Regime', 'Dispersion_Regime', 
                      'Economic_Regime', 'Vol_Mom']
    
    for condition_type in condition_types:
        logging.info(f"\n=== ANALYZING {condition_type.upper()} ===")
        
        # Get unique regimes for this condition
        regimes = conditions[condition_type].dropna().unique()
        
        # Initialize results storage
        regime_results = {}
        
        for regime in regimes:
            # Get dates for this regime
            regime_dates = conditions[conditions[condition_type] == regime].index
            
            if len(regime_dates) >= MIN_PERIODS_PER_REGIME:
                # Calculate performance metrics for each factor
                factor_metrics = []
                
                for factor in factor_excess_returns.columns:
                    # Get factor returns for this regime
                    regime_returns = factor_excess_returns.loc[regime_dates, factor].dropna()
                    
                    if len(regime_returns) >= MIN_PERIODS_PER_REGIME:
                        metrics = {
                            'Factor': factor,
                            'Periods': len(regime_returns),
                            'Mean_Excess': regime_returns.mean(),
                            'Volatility': regime_returns.std(),
                            'Annual_Excess': regime_returns.mean() * 12,
                            'Annual_Vol': regime_returns.std() * np.sqrt(12),
                            'Info_Ratio': (regime_returns.mean() * 12) / (regime_returns.std() * np.sqrt(12)) if regime_returns.std() > 0 else np.nan,
                            'Win_Rate': (regime_returns > 0).mean(),
                            'Min_Return': regime_returns.min(),
                            'Max_Return': regime_returns.max(),
                            'Skewness': regime_returns.skew(),
                            'Kurtosis': regime_returns.kurtosis()
                        }
                        factor_metrics.append(metrics)
                
                # Convert to DataFrame and sort by Information Ratio
                if factor_metrics:
                    regime_df = pd.DataFrame(factor_metrics)
                    regime_df = regime_df.sort_values('Info_Ratio', ascending=False)
                    regime_results[regime] = regime_df
                    
                    # Log top and bottom performers
                    logging.info(f"\n{regime} - Top 5 Factors:")
                    for _, row in regime_df.head(5).iterrows():
                        logging.info(f"  {row['Factor']:30s}: IR={row['Info_Ratio']:6.2f}, Annual={row['Annual_Excess']:7.2%}")
                    
                    logging.info(f"\n{regime} - Bottom 5 Factors:")
                    for _, row in regime_df.tail(5).iterrows():
                        logging.info(f"  {row['Factor']:30s}: IR={row['Info_Ratio']:6.2f}, Annual={row['Annual_Excess']:7.2%}")
        
        performance_results[condition_type] = regime_results
    
    return performance_results

def calculate_conditional_correlations(factor_excess_returns, conditions):
    """
    Calculate factor correlations conditional on market regimes
    
    Args:
        factor_excess_returns: Factor excess returns
        conditions: Market condition classifications
        
    Returns:
        dict: Correlation matrices by regime
    """
    logging.info("Calculating conditional factor correlations...")
    
    correlation_results = {}
    
    # Focus on key condition types
    key_conditions = ['Volatility_Regime', 'Momentum_Regime', 'Economic_Regime']
    
    for condition_type in key_conditions:
        logging.info(f"\nAnalyzing correlations for {condition_type}")
        
        regime_correlations = {}
        regimes = conditions[condition_type].dropna().unique()
        
        for regime in regimes:
            # Get dates for this regime
            regime_dates = conditions[conditions[condition_type] == regime].index
            
            # Get factor returns for these dates
            regime_factor_returns = factor_excess_returns.loc[regime_dates].dropna()
            
            if len(regime_factor_returns) >= CORRELATION_MIN_PERIODS:
                # Calculate correlation matrix
                corr_matrix = regime_factor_returns.corr()
                regime_correlations[regime] = corr_matrix
                
                # Calculate average correlation (excluding diagonal)
                mask = np.ones_like(corr_matrix, dtype=bool)
                np.fill_diagonal(mask, 0)
                avg_corr = corr_matrix.values[mask].mean()
                
                logging.info(f"  {regime}: Average correlation = {avg_corr:.3f}")
        
        correlation_results[condition_type] = regime_correlations
    
    return correlation_results

def create_factor_performance_heatmap(performance_results, output_file):
    """
    Create heatmap showing average factor returns under each market condition
    """
    logging.info("Creating factor performance heatmap...")
    
    # Prepare data for heatmap - focus on Volatility and Momentum regimes
    vol_results = performance_results.get('Volatility_Regime', {})
    mom_results = performance_results.get('Momentum_Regime', {})
    
    # Get all unique factors
    all_factors = set()
    for regime_data in vol_results.values():
        all_factors.update(regime_data['Factor'].tolist())
    for regime_data in mom_results.values():
        all_factors.update(regime_data['Factor'].tolist())
    
    # Create matrix for heatmap
    conditions = []
    for vol_regime in ['Low', 'Medium', 'High']:
        for mom_regime in ['Weak', 'Neutral', 'Strong']:
            conditions.append(f"{vol_regime} Vol / {mom_regime} Mom")
    
    # Build heatmap data
    heatmap_data = pd.DataFrame(index=sorted(all_factors), columns=conditions)
    
    # Fill in data
    for vol_regime in ['Low', 'Medium', 'High']:
        for mom_regime in ['Weak', 'Neutral', 'Strong']:
            condition_name = f"{vol_regime} Vol / {mom_regime} Mom"
            
            # Get performance data for this combination
            if vol_regime in vol_results and mom_regime in mom_results:
                vol_data = vol_results[vol_regime].set_index('Factor')
                mom_data = mom_results[mom_regime].set_index('Factor')
                
                # Average the annual excess returns
                for factor in heatmap_data.index:
                    if factor in vol_data.index and factor in mom_data.index:
                        avg_return = (vol_data.loc[factor, 'Annual_Excess'] + 
                                    mom_data.loc[factor, 'Annual_Excess']) / 2
                        heatmap_data.loc[factor, condition_name] = avg_return * 100  # Convert to percentage
    
    # Create the heatmap
    plt.figure(figsize=(12, 20))
    
    # Convert to numeric and handle NaNs
    heatmap_numeric = heatmap_data.apply(pd.to_numeric, errors='coerce')
    
    # Create heatmap
    sns.heatmap(heatmap_numeric, cmap='RdYlGn', center=0, 
                fmt='.1f', annot=True, cbar_kws={'label': 'Annual Excess Return (%)'},
                linewidths=0.5, linecolor='gray')
    
    plt.title('Factor Performance Heatmap by Market Condition\n(Annual Excess Returns %)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Market Condition', fontsize=12)
    plt.ylabel('Factor', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Factor performance heatmap saved to {output_file}")

def create_factor_quilt_chart(factor_excess_returns, conditions, output_file):
    """
    Create factor quilt chart showing rolling performance with market regime backgrounds
    """
    logging.info("Creating factor quilt chart...")
    
    # Calculate rolling 12-month returns for each factor
    rolling_returns = factor_excess_returns.rolling(window=12).sum() * 100  # Convert to percentage
    
    # Get top N factors by overall performance
    factor_performance = factor_excess_returns.mean().sort_values(ascending=False)
    top_factors = factor_performance.head(TOP_N_FACTORS).index.tolist()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Define colors for factors
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_factors)))
    
    # Plot background shading for market regimes
    regime_colors = {
        'Low': 'lightgreen',
        'Medium': 'lightgray', 
        'High': 'lightcoral'
    }
    
    # Add background shading for volatility regimes
    for i in range(len(conditions) - 1):
        if i < len(conditions) - 1:
            regime = conditions.iloc[i]['Volatility_Regime']
            if regime in regime_colors:
                ax.axvspan(conditions.index[i], conditions.index[i+1], 
                          alpha=0.3, color=regime_colors[regime], 
                          edgecolor='none')
    
    # Plot factor returns
    for i, factor in enumerate(top_factors):
        factor_data = rolling_returns[factor].dropna()
        ax.plot(factor_data.index, factor_data.values, 
                label=factor[:20], color=colors[i], linewidth=2, alpha=0.8)
    
    # Formatting
    ax.set_title(f'Factor Performance Quilt Chart\nTop {TOP_N_FACTORS} Factors - Rolling 12-Month Excess Returns', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rolling 12-Month Excess Return (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add legend for factors
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    
    # Add legend for volatility regimes
    vol_legend_elements = [mpatches.Patch(color=regime_colors[regime], label=f'{regime} Volatility', alpha=0.3) 
                          for regime in ['Low', 'Medium', 'High']]
    vol_legend = ax.legend(handles=vol_legend_elements, loc='upper right', frameon=True)
    ax.add_artist(ax.get_legend())  # Keep the factor legend
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Factor quilt chart saved to {output_file}")

def create_correlation_analysis_plots(correlation_results, output_file):
    """
    Create visualization of conditional correlations
    """
    logging.info("Creating correlation analysis plots...")
    
    # Focus on volatility regimes
    vol_correlations = correlation_results.get('Volatility_Regime', {})
    
    if not vol_correlations:
        logging.warning("No correlation data available for visualization")
        return
    
    # Create subplots for each regime
    n_regimes = len(vol_correlations)
    fig, axes = plt.subplots(1, n_regimes, figsize=(6*n_regimes, 5))
    
    if n_regimes == 1:
        axes = [axes]
    
    # Sort regimes for consistent ordering
    regime_order = ['Low', 'Medium', 'High']
    regimes = [r for r in regime_order if r in vol_correlations]
    
    # Plot correlation matrices
    for i, regime in enumerate(regimes):
        corr_matrix = vol_correlations[regime]
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                   vmin=-1, vmax=1, square=True, ax=axes[i],
                   cbar_kws={'shrink': 0.8})
        
        axes[i].set_title(f'{regime} Volatility', fontsize=14, fontweight='bold')
        
        # Only show factor names on first plot
        if i > 0:
            axes[i].set_yticklabels([])
        
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
    
    fig.suptitle('Factor Correlations by Volatility Regime', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Correlation analysis saved to {output_file}")

def save_comprehensive_results(performance_results, correlation_results, conditions, 
                             factor_returns, factor_excess_returns):
    """
    Save all analysis results to Excel workbook
    """
    output_file = 'T2_Factor_Market_Condition_Analysis.xlsx'
    
    logging.info(f"Saving comprehensive results to {output_file}")
    
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Save market conditions
        conditions.to_excel(writer, sheet_name='Market_Conditions')
        
        # Save factor returns
        factor_returns.to_excel(writer, sheet_name='Factor_Returns')
        factor_excess_returns.to_excel(writer, sheet_name='Factor_Excess_Returns')
        
        # Save performance results by condition
        for condition_type, regime_results in performance_results.items():
            for regime, regime_df in regime_results.items():
                sheet_name = f'{condition_type}_{regime}'[:31]  # Excel limit
                regime_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Create summary sheet with best/worst factors by regime
        summary_data = []
        for condition_type, regime_results in performance_results.items():
            for regime, regime_df in regime_results.items():
                if len(regime_df) > 0:
                    # Get top and bottom performers
                    top_factor = regime_df.iloc[0]
                    bottom_factor = regime_df.iloc[-1]
                    
                    summary_data.append({
                        'Condition_Type': condition_type,
                        'Regime': regime,
                        'Periods': int(regime_df['Periods'].mean()),
                        'Best_Factor': top_factor['Factor'],
                        'Best_IR': top_factor['Info_Ratio'],
                        'Best_Annual_Return': top_factor['Annual_Excess'],
                        'Worst_Factor': bottom_factor['Factor'],
                        'Worst_IR': bottom_factor['Info_Ratio'],
                        'Worst_Annual_Return': bottom_factor['Annual_Excess']
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Performance_Summary', index=False)
        
        # Save correlation analysis summary
        corr_summary = []
        for condition_type, regime_corrs in correlation_results.items():
            for regime, corr_matrix in regime_corrs.items():
                # Calculate average correlation
                mask = np.ones_like(corr_matrix, dtype=bool)
                np.fill_diagonal(mask, 0)
                avg_corr = corr_matrix.values[mask].mean()
                
                # Find most and least correlated factor pairs
                corr_values = corr_matrix.values[mask]
                max_corr = np.max(corr_values)
                min_corr = np.min(corr_values)
                
                corr_summary.append({
                    'Condition_Type': condition_type,
                    'Regime': regime,
                    'Avg_Correlation': avg_corr,
                    'Max_Correlation': max_corr,
                    'Min_Correlation': min_corr
                })
        
        if corr_summary:
            corr_summary_df = pd.DataFrame(corr_summary)
            corr_summary_df.to_excel(writer, sheet_name='Correlation_Summary', index=False)
        
        # Apply date formatting
        workbook = writer.book
        date_format = workbook.add_format({'num_format': 'dd-mmm-yyyy'})
        
        # Format date columns
        for sheet_name in ['Market_Conditions', 'Factor_Returns', 'Factor_Excess_Returns']:
            if sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                worksheet.set_column(0, 0, 12, date_format)
    
    logging.info("Comprehensive results saved successfully")

def main():
    """Main execution function"""
    setup_logging()
    
    logging.info("="*80)
    logging.info("T2 FACTOR TIMING - STEP SIXTEEN: COMPREHENSIVE FACTOR PERFORMANCE ANALYSIS")
    logging.info("="*80)
    logging.info("MARKET CONDITION PARAMETERS:")
    logging.info(f"  Volatility Percentiles: {VOLATILITY_PERCENTILES}")
    logging.info(f"  Momentum Percentiles: {MOMENTUM_PERCENTILES}")
    logging.info(f"  Dispersion Percentiles: {DISPERSION_PERCENTILES}")
    logging.info(f"  Economic Percentiles: {ECONOMIC_PERCENTILES}")
    logging.info("="*80)
    
    try:
        # Load data
        data, benchmark_returns, exposures = load_data()
        
        # Get country returns for dispersion calculation
        country_returns_data = data[data['variable'] == '1MRet'].copy()
        
        # Calculate individual factor returns
        factor_returns, factor_excess_returns = calculate_factor_returns(
            data, exposures, benchmark_returns
        )
        
        # Calculate comprehensive market conditions
        conditions = calculate_market_conditions(benchmark_returns, country_returns_data)
        
        # Analyze factor performance by condition
        performance_results = analyze_factor_performance_by_condition(
            factor_returns, factor_excess_returns, conditions
        )
        
        # Calculate conditional correlations
        correlation_results = calculate_conditional_correlations(
            factor_excess_returns, conditions
        )
        
        # Create visualizations
        heatmap_file = 'T2_Factor_Performance_Heatmap.pdf'
        create_factor_performance_heatmap(performance_results, heatmap_file)
        
        quilt_file = 'T2_Factor_Quilt_Chart.pdf'
        create_factor_quilt_chart(factor_excess_returns, conditions, quilt_file)
        
        corr_file = 'T2_Factor_Correlation_Analysis.pdf'
        create_correlation_analysis_plots(correlation_results, corr_file)
        
        # Save comprehensive results
        save_comprehensive_results(
            performance_results, correlation_results, conditions,
            factor_returns, factor_excess_returns
        )
        
        # Display summary insights
        logging.info("\n" + "="*80)
        logging.info("KEY INSIGHTS FROM FACTOR ANALYSIS")
        logging.info("="*80)
        
        # Identify factors that perform consistently well across regimes
        logging.info("\nMOST ROBUST FACTORS (perform well across multiple conditions):")
        factor_scores = {}
        for condition_type, regime_results in performance_results.items():
            for regime, regime_df in regime_results.items():
                if len(regime_df) > 0:
                    # Score factors by their rank in each regime
                    for rank, (_, row) in enumerate(regime_df.iterrows()):
                        factor = row['Factor']
                        score = len(regime_df) - rank  # Higher score for better rank
                        if factor not in factor_scores:
                            factor_scores[factor] = 0
                        factor_scores[factor] += score
        
        # Sort and display top robust factors
        sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
        for factor, score in sorted_factors[:10]:
            logging.info(f"  {factor}: Robustness Score = {score}")
        
        # Identify regime-specific opportunities
        logging.info("\nREGIME-SPECIFIC OPPORTUNITIES:")
        for condition_type in ['Volatility_Regime', 'Momentum_Regime']:
            if condition_type in performance_results:
                logging.info(f"\n{condition_type}:")
                for regime, regime_df in performance_results[condition_type].items():
                    if len(regime_df) > 0:
                        top_factor = regime_df.iloc[0]
                        logging.info(f"  {regime}: Best = {top_factor['Factor']} (IR={top_factor['Info_Ratio']:.2f})")
        
        logging.info("\n" + "="*80)
        logging.info("STEP SIXTEEN COMPLETED SUCCESSFULLY")
        logging.info("="*80)
        
    except Exception as e:
        logging.error(f"Error in Step Sixteen: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()