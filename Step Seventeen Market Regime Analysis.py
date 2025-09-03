"""
=============================================================================
SCRIPT NAME: Step Seventeen Market Regime Analysis.py
=============================================================================

NOTE: All file references in this script are now LOCAL FILE NAMES ONLY (e.g., 'T2_GMM_Regime_Analysis.xlsx'). Full directory paths have been removed for portability and clarity.


INPUT FILES:
- T2_Optimized_Country_Weights.xlsx:
  - Contains 'Monthly_Returns' sheet with optimized strategy and benchmark returns.
- T2_Top_20_Exposure.csv:
  - Contains individual factor exposures for attribution analysis.

OUTPUT FILES:
- T2_GMM_Regime_Analysis.xlsx:
  - Excel report with GMM regime definitions, performance statistics, and factor attribution.
- T2_GMM_Regime_Performance.pdf:
  - PDF chart visualizing strategy performance across the identified regimes.
- T2_GMM_Factor_Analysis.pdf:
  - PDF charts analyzing factor characteristics and performance in each regime.
- optimization_log.txt:
  - Log file for monitoring script execution.

VERSION: 1.0
LAST UPDATED: 2025-06-25
AUTHOR: Cascade

DESCRIPTION:
This script implements an advanced market regime detection system using a Gaussian
Mixture Model (GMM) to identify distinct market environments from data rather than
pre-defined rules. It performs the following steps:

1.  FEATURE ENGINEERING: Creates a rich feature set from benchmark returns, including
    rolling volatility (3-month, 12-month), skewness, and kurtosis to capture
    the market's statistical properties.

2.  GMM CLUSTERING: Uses scikit-learn's GaussianMixture model to cluster the
    monthly feature sets. It automatically determines the optimal number of regimes
    (e.g., 3-5) by finding the model with the lowest Bayesian Information Criterion (BIC).

3.  REGIME INTERPRETATION: Analyzes the statistical characteristics of each
    discovered regime (e.g., high volatility, negative returns) and assigns a
    meaningful, descriptive label (e.g., 'High-Vol Bearish', 'Low-Vol Bullish').

4.  PERFORMANCE ATTRIBUTION: Conducts an in-depth analysis of how the investment
    strategy and its underlying factors perform within each identified regime,
    revealing what drives returns in different market conditions.

BENEFITS:
- Data-Driven: Regimes are discovered directly from data, not imposed by arbitrary rules.
- Granular: Identifies more subtle and complex market regimes than simple heuristics.
- Actionable Insights: Provides a deeper understanding of what drives strategy
  performance, enabling better risk management and strategy refinement.

=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import logging
import os
import warnings
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================
# Feature Engineering
VOL_WINDOW_SHORT = 3
VOL_WINDOW_LONG = 12
SKEW_KURT_WINDOW = 12

# GMM Configuration
MAX_REGIMES = 6 # Maximum number of regimes to test for the GMM
N_INIT = 10 # Number of initializations to perform for the GMM

# File Paths
INPUT_FILE_RETURNS = 'T2_Optimized_Country_Weights.xlsx'
INPUT_FILE_EXPOSURES = 'T2_Top_20_Exposure.csv'
OUTPUT_FILE_EXCEL = 'T2_GMM_Regime_Analysis.xlsx'
OUTPUT_PDF_PERFORMANCE = 'T2_GMM_Regime_Performance.pdf'
OUTPUT_PDF_FACTOR = 'T2_GMM_Factor_Analysis.pdf'
LOG_FILE = 'optimization_log.txt'

# =============================================================================
# SETUP LOGGING
# =============================================================================
def setup_logging():
    """Set up logging configuration to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'),
            logging.StreamHandler()
        ]
    )

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================
def load_data():
    """Load returns and factor exposure data."""
    logging.info("Loading data for GMM regime analysis...")
    try:
        returns_df = pd.read_excel(INPUT_FILE_RETURNS, sheet_name='Monthly_Returns', index_col=0)
        returns_df.index = pd.to_datetime(returns_df.index)
        
        factor_exposures = pd.read_csv(INPUT_FILE_EXPOSURES)
        factor_exposures['Date'] = pd.to_datetime(factor_exposures['Date'])
        
        logging.info(f"Successfully loaded {len(returns_df)} monthly returns.")
        logging.info(f"Successfully loaded {len(factor_exposures)} factor exposure records.")
        return returns_df, factor_exposures
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}. Please check file paths.")
        return None, None

def create_regime_features(market_returns):
    """Engineer features for the GMM model from market returns."""
    logging.info("Creating features for regime detection...")
    features = pd.DataFrame(index=market_returns.index)
    features['Returns'] = market_returns
    features[f'Volatility_{VOL_WINDOW_SHORT}M'] = market_returns.rolling(window=VOL_WINDOW_SHORT).std() * np.sqrt(12)
    features[f'Volatility_{VOL_WINDOW_LONG}M'] = market_returns.rolling(window=VOL_WINDOW_LONG).std() * np.sqrt(12)
    features[f'Skewness_{SKEW_KURT_WINDOW}M'] = market_returns.rolling(window=SKEW_KURT_WINDOW).skew()
    features[f'Kurtosis_{SKEW_KURT_WINDOW}M'] = market_returns.rolling(window=SKEW_KURT_WINDOW).kurt()
    
    features.dropna(inplace=True)
    logging.info(f"Created feature set with {len(features)} observations and {len(features.columns)} features.")
    return features

# =============================================================================
# GMM REGIME DETECTION
# =============================================================================
def find_optimal_regimes(features):
    """Find the optimal number of regimes using GMM and BIC."""
    logging.info("Finding optimal number of regimes using GMM and BIC...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    n_components = np.arange(2, MAX_REGIMES + 1)
    bics = []
    models = []

    for n in n_components:
        gmm = GaussianMixture(n_components=n, random_state=42, n_init=N_INIT)
        gmm.fit(scaled_features)
        bics.append(gmm.bic(scaled_features))
        models.append(gmm)
        logging.info(f"Tested {n} regimes, BIC: {bics[-1]:.2f}")

    optimal_n = n_components[np.argmin(bics)]
    optimal_gmm = models[np.argmin(bics)]
    logging.info(f"Optimal number of regimes found: {optimal_n}")
    
    regimes = optimal_gmm.predict(scaled_features)
    features['Regime'] = regimes
    
    return features, optimal_gmm, bics, n_components

def analyze_and_label_regimes(features):
    """Analyze regime characteristics and assign descriptive labels."""
    logging.info("Analyzing and labeling regimes...")
    regime_characteristics = features.groupby('Regime').mean()
    
    # Create labels based on volatility and returns
    vol_median = regime_characteristics[f'Volatility_{VOL_WINDOW_LONG}M'].median()
    ret_median = regime_characteristics['Returns'].median()
    
    labels = {}
    for i, row in regime_characteristics.iterrows():
        vol_label = 'High-Vol' if row[f'Volatility_{VOL_WINDOW_LONG}M'] > vol_median else 'Low-Vol'
        ret_label = 'Bullish' if row['Returns'] > 0 else 'Bearish'
        labels[i] = f"{vol_label} {ret_label}"
    
    regime_characteristics['Label'] = regime_characteristics.index.map(labels)
    features['Regime_Label'] = features['Regime'].map(labels)
    logging.info("Assigned descriptive labels to regimes.")
    logging.info(f"\nRegime Characteristics:\n{regime_characteristics}")
    
    return features, regime_characteristics

# =============================================================================
# PERFORMANCE AND FACTOR ATTRIBUTION
# =============================================================================
def analyze_performance_by_regime(regime_df, returns_df):
    """Analyze strategy performance within each identified regime."""
    logging.info("Analyzing strategy performance by regime...")
    data = returns_df.join(regime_df[['Regime_Label']])
    data.dropna(subset=['Regime_Label'], inplace=True)
    
    net_returns = data['Optimized_Strategy'] - data['Equal_Weight_Benchmark']
    
    performance_summary = []
    for regime in data['Regime_Label'].unique():
        regime_returns = net_returns[data['Regime_Label'] == regime]
        if len(regime_returns) > 0:
            annual_return = regime_returns.mean() * 12
            annual_vol = regime_returns.std() * np.sqrt(12)
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0
            win_rate = (regime_returns > 0).mean()
            performance_summary.append({
                'Regime': regime,
                'Periods': len(regime_returns),
                'Annual_Return': annual_return,
                'Annual_Volatility': annual_vol,
                'Sharpe_Ratio': sharpe,
                'Win_Rate': win_rate
            })
    
    perf_df = pd.DataFrame(performance_summary).set_index('Regime')
    logging.info(f"\nPerformance by Regime:\n{perf_df}")
    return perf_df

def analyze_factors_by_regime(regime_df, factor_exposures):
    """Analyze average factor exposures in each regime."""
    logging.info("Analyzing factor exposures by regime...")
    factor_cols = [c for c in factor_exposures.columns if c not in ['Date', 'Country']]
    
    # Aggregate factor exposures by date (average across countries)
    avg_daily_exposures = factor_exposures.groupby('Date')[factor_cols].mean()
    
    data = avg_daily_exposures.join(regime_df[['Regime_Label']])
    data.dropna(subset=['Regime_Label'], inplace=True)
    
    factor_analysis = data.groupby('Regime_Label')[factor_cols].mean().T
    logging.info(f"\nAverage Factor Exposures by Regime:\n{factor_analysis}")
    return factor_analysis

# =============================================================================
# VISUALIZATION
# =============================================================================
def create_visualizations(regime_df, perf_df, factor_analysis, bics, n_components, returns_df):
    """Create and save all visualizations to PDF files."""
    logging.info("Creating performance visualizations...")
    # --- Performance PDF ---
    with PdfPages(OUTPUT_PDF_PERFORMANCE) as pdf:
        # BIC Plot
        plt.figure(figsize=(10, 6))
        plt.plot(n_components, bics, marker='o')
        plt.title('GMM Bayesian Information Criterion (BIC)')
        plt.xlabel('Number of Regimes')
        plt.ylabel('BIC Score')
        plt.grid(True)
        pdf.savefig()
        plt.close()
        
        # Regime Plot
        fig, ax1 = plt.subplots(figsize=(15, 8))
        market_cumulative = (1 + returns_df['Equal_Weight_Benchmark']).cumprod()
        ax1.plot(market_cumulative.index, market_cumulative.values, color='black', label='Benchmark Cumulative Return')
        ax1.set_ylabel('Cumulative Return')
        ax1.set_yscale('log')
        
        colors = sns.color_palette('viridis', n_colors=len(regime_df['Regime_Label'].unique()))
        regime_map = {label: color for label, color in zip(regime_df['Regime_Label'].unique(), colors)}
        
        for i in range(len(regime_df) - 1):
            ax1.axvspan(regime_df.index[i], regime_df.index[i+1], 
                        color=regime_map[regime_df['Regime_Label'].iloc[i]], alpha=0.3)
        
        handles = [plt.Rectangle((0,0),1,1, color=color, alpha=0.3) for color in regime_map.values()]
        plt.legend(handles, regime_map.keys(), title='Regimes', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('Market Regimes Over Time')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Performance Bar Chart
        perf_df[['Annual_Return', 'Sharpe_Ratio']].plot(kind='bar', secondary_y='Sharpe_Ratio', figsize=(12, 7))
        plt.title('Strategy Performance by Market Regime')
        plt.ylabel('Annual Return')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    logging.info(f"Performance visualizations saved to {OUTPUT_PDF_PERFORMANCE}")

    logging.info("Creating factor analysis visualizations...")
    # --- Factor Analysis PDF ---
    with PdfPages(OUTPUT_PDF_FACTOR) as pdf:
        # Factor Exposure Heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(factor_analysis, cmap='viridis', annot=True, fmt='.2f')
        plt.title('Average Factor Exposure by Regime')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    logging.info(f"Factor analysis visualizations saved to {OUTPUT_PDF_FACTOR}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main execution function."""
    setup_logging()
    logging.info("--- Starting Step Seventeen: GMM Market Regime Analysis ---")
    
    returns_df, factor_exposures = load_data()
    if returns_df is None:
        logging.error("Halting execution due to data loading failure.")
        return

    market_returns = returns_df['Equal_Weight_Benchmark']
    
    # 1. Feature Engineering
    features = create_regime_features(market_returns)
    
    # 2. GMM Clustering
    regime_features, model, bics, n_components = find_optimal_regimes(features)
    
    # 3. Regime Interpretation
    final_regimes, regime_char = analyze_and_label_regimes(regime_features)
    
    # 4. Performance Attribution
    perf_by_regime = analyze_performance_by_regime(final_regimes, returns_df)
    factors_by_regime = analyze_factors_by_regime(final_regimes, factor_exposures)
    
    # 5. Save results to Excel
    logging.info(f"Saving analysis to {OUTPUT_FILE_EXCEL}...")
    with pd.ExcelWriter(OUTPUT_FILE_EXCEL, engine='xlsxwriter') as writer:
        final_regimes.to_excel(writer, sheet_name='Regime_Classification')
        regime_char.to_excel(writer, sheet_name='Regime_Characteristics')
        perf_by_regime.to_excel(writer, sheet_name='Performance_by_Regime')
        factors_by_regime.to_excel(writer, sheet_name='Factor_Exposures_by_Regime')
    logging.info("Excel file saved successfully.")
    
    # 6. Create Visualizations
    create_visualizations(final_regimes, perf_by_regime, factors_by_regime, bics, n_components, returns_df)
    
    logging.info("--- Step Seventeen: GMM Market Regime Analysis Completed ---")

if __name__ == "__main__":
    main()
