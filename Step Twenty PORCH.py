"""
Step Twenty PORCH.py - Portfolio Factor Exposure Analysis

INPUT FILES:
- T2_Final_Country_Weights.xlsx (Latest Weights sheet): Final optimized country weights
- T2_Optimized_Country_Weights.xlsx (Latest_Weights sheet): Optimized country weights
- Normalized_T2_MasterCSV.csv: Normalized factor data for all countries and factors

OUTPUT FILES:
- T2_Portfolio_Factor_Exposures.pdf: Factor exposure comparison table across three portfolios

DESCRIPTION:
This script calculates and compares factor exposures for three portfolios:
1. Final Country Weights Portfolio
2. Optimized Country Weights Portfolio  
3. Equal Weighted Benchmark Portfolio

The analysis shows how each portfolio's country allocation translates into 
factor exposures, providing insights into the risk characteristics and 
factor tilts of each strategy.

VERSION: 1.0
DATE: 2025-01-27
AUTHOR: T2 Factor Timing System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_portfolio_weights():
    """
    Load country weights from both portfolio files.
    
    Returns:
        tuple: (final_weights, optimized_weights) - DataFrames with country weights
    """
    logging.info("Loading portfolio weights...")
    
    # Load Final Country Weights (Latest Weights sheet)
    final_file = "T2_Final_Country_Weights.xlsx"
    final_weights = pd.read_excel(final_file, sheet_name="Latest Weights")
    
    # Clean up the final weights data
    final_weights = final_weights.iloc[1:].copy()  # Skip header row
    final_weights.columns = ['Country', 'Weight', 'Average_Weight', 'Days_with_Weight', 'Latest_Date']
    final_weights = final_weights[['Country', 'Weight']].copy()
    final_weights['Weight'] = pd.to_numeric(final_weights['Weight'], errors='coerce')
    final_weights = final_weights.dropna()
    
    # Load Optimized Country Weights (Latest_Weights sheet)
    optimized_file = "T2_Optimized_Country_Weights.xlsx"
    optimized_weights = pd.read_excel(optimized_file, sheet_name="Latest_Weights")
    optimized_weights['Weight'] = pd.to_numeric(optimized_weights['Weight'], errors='coerce')
    optimized_weights = optimized_weights.dropna()
    
    logging.info(f"Final weights: {len(final_weights)} countries")
    logging.info(f"Optimized weights: {len(optimized_weights)} countries")
    
    return final_weights, optimized_weights

def load_factor_exposures():
    """
    Load factor exposure data from Normalized_T2_MasterCSV.csv.
    
    Returns:
        DataFrame: Factor exposure data with Date, Country, and factor columns
    """
    logging.info("Loading factor exposure data from Normalized T2 Master...")
    
    # Load normalized data
    master_file = "Normalized_T2_MasterCSV.csv"
    master_df = pd.read_csv(master_file)
    master_df['date'] = pd.to_datetime(master_df['date'])
    
    # Get the latest date for exposure calculation
    latest_date = master_df['date'].max()
    latest_data = master_df[master_df['date'] == latest_date].copy()
    
    # Pivot to get factors as columns and countries as rows
    exposures_pivot = latest_data.pivot(index='country', columns='variable', values='value')
    exposures_pivot = exposures_pivot.reset_index()
    exposures_pivot = exposures_pivot.rename(columns={'country': 'Country'})
    
    # Fill NaN values with 0 for missing factor exposures
    exposures_pivot = exposures_pivot.fillna(0)
    
    logging.info(f"Loaded exposures for {len(exposures_pivot)} countries on {latest_date}")
    logging.info(f"Available factors: {len(exposures_pivot.columns) - 1}")
    
    return exposures_pivot

def create_equal_weighted_benchmark(countries):
    """
    Create equal weighted benchmark portfolio.
    
    Args:
        countries: List of country names
        
    Returns:
        DataFrame: Equal weighted portfolio with Country and Weight columns
    """
    logging.info("Creating equal weighted benchmark...")
    
    equal_weight = 1.0 / len(countries)
    benchmark_weights = pd.DataFrame({
        'Country': countries,
        'Weight': [equal_weight] * len(countries)
    })
    
    logging.info(f"Equal weighted benchmark: {len(countries)} countries, {equal_weight:.4f} weight each")
    
    return benchmark_weights

def calculate_portfolio_factor_exposures(portfolio_weights, factor_exposures):
    """
    Calculate factor exposures for a given portfolio.
    
    Args:
        portfolio_weights: DataFrame with Country and Weight columns
        factor_exposures: DataFrame with Country and factor exposure columns
        
    Returns:
        Series: Factor exposures for the portfolio
    """
    # Get factor columns (exclude Country column)
    factor_columns = [col for col in factor_exposures.columns if col != 'Country']
    
    # Merge portfolio weights with factor exposures
    merged = pd.merge(portfolio_weights, factor_exposures[['Country'] + factor_columns], 
                     on='Country', how='left')
    
    # Fill missing exposures with 0 (countries not in exposure data)
    merged[factor_columns] = merged[factor_columns].fillna(0)
    
    # Calculate weighted factor exposures
    portfolio_exposures = {}
    for factor in factor_columns:
        # Weight * Exposure for each country, then sum
        weighted_exposure = (merged['Weight'] * merged[factor]).sum()
        portfolio_exposures[factor] = weighted_exposure
    
    return pd.Series(portfolio_exposures)

def create_factor_exposure_comparison(final_exposures, optimized_exposures, benchmark_exposures):
    """
    Create comparison DataFrame of factor exposures across portfolios.
    
    Args:
        final_exposures: Series of factor exposures for final portfolio
        optimized_exposures: Series of factor exposures for optimized portfolio
        benchmark_exposures: Series of factor exposures for benchmark portfolio
        
    Returns:
        DataFrame: Comparison of factor exposures
    """
    comparison_df = pd.DataFrame({
        'Factor': final_exposures.index,
        'Final_Portfolio': final_exposures.values,
        'Optimized_Portfolio': optimized_exposures.values,
        'Equal_Weighted_Benchmark': benchmark_exposures.values
    })
    
    # Calculate relative exposures vs benchmark
    comparison_df['Final_vs_Benchmark'] = comparison_df['Final_Portfolio'] - comparison_df['Equal_Weighted_Benchmark']
    comparison_df['Optimized_vs_Benchmark'] = comparison_df['Optimized_Portfolio'] - comparison_df['Equal_Weighted_Benchmark']
    comparison_df['Final_vs_Optimized'] = comparison_df['Final_Portfolio'] - comparison_df['Optimized_Portfolio']
    
    # Sort by absolute exposure difference from benchmark
    comparison_df['Abs_Final_vs_Benchmark'] = comparison_df['Final_vs_Benchmark'].abs()
    comparison_df = comparison_df.sort_values('Abs_Final_vs_Benchmark', ascending=False)
    
    return comparison_df

def create_factor_exposure_tables(comparison_df, output_file):
    """
    Create PDF tables of factor exposure analysis.
    
    Args:
        comparison_df: DataFrame with factor exposure comparisons
        output_file: Path to output PDF file
    """
    logging.info(f"Creating factor exposure tables in {output_file}...")
    
    with PdfPages(output_file) as pdf:
        # All Factors - Absolute Exposures (Multiple Pages)
        factors_per_page = 45
        total_factors = len(comparison_df)
        num_pages = (total_factors + factors_per_page - 1) // factors_per_page
        
        for page_num in range(num_pages):
            start_idx = page_num * factors_per_page
            end_idx = min(start_idx + factors_per_page, total_factors)
            page_factors = comparison_df.iloc[start_idx:end_idx]
            
            fig, ax = plt.subplots(figsize=(11, 16))
            ax.axis('tight')
            ax.axis('off')
        
            # Prepare table data
            table_data = []
            for _, row in page_factors.iterrows():
                table_data.append([
                    row['Factor'],
                    f"{row['Final_Portfolio']:.4f}",
                    f"{row['Optimized_Portfolio']:.4f}",
                    f"{row['Equal_Weighted_Benchmark']:.4f}",
                    f"{row['Final_vs_Benchmark']:+.4f}",
                    f"{row['Optimized_vs_Benchmark']:+.4f}"
                ])
        
            # Create table
            table = ax.table(
                cellText=table_data,
                colLabels=['Factor', 'Final Portfolio', 'Optimized Portfolio', 'Equal Weighted', 'Final vs Benchmark', 'Optimized vs Benchmark'],
                cellLoc='center',
                loc='center',
                colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15]
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1.2, 1.3)
            
            # Style the table
            for i in range(6):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color code the difference columns
            for i in range(1, len(table_data) + 1):
                # Final vs Benchmark column (index 4)
                final_diff = float(table_data[i-1][4])
                if final_diff > 0.1:
                    table[(i, 4)].set_facecolor('#E8F5E8')  # Light green
                elif final_diff < -0.1:
                    table[(i, 4)].set_facecolor('#FFE8E8')  # Light red
                
                # Optimized vs Benchmark column (index 5)
                opt_diff = float(table_data[i-1][5])
                if opt_diff > 0.1:
                    table[(i, 5)].set_facecolor('#E8F5E8')  # Light green
                elif opt_diff < -0.1:
                    table[(i, 5)].set_facecolor('#FFE8E8')  # Light red
            
            ax.set_title(f'Portfolio Factor Exposures - All Factors (Page {page_num + 1} of {num_pages})\n(Sorted by |Final vs Benchmark| Difference)', 
                        fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Additional Pages: Top 50 Final vs Benchmark Differences
        fig, ax = plt.subplots(figsize=(11, 14))
        ax.axis('tight')
        ax.axis('off')
        
        top_final_diff = comparison_df.nlargest(50, 'Abs_Final_vs_Benchmark')
        
        table_data = []
        for _, row in top_final_diff.iterrows():
            table_data.append([
                row['Factor'],
                f"{row['Final_Portfolio']:.4f}",
                f"{row['Equal_Weighted_Benchmark']:.4f}",
                f"{row['Final_vs_Benchmark']:+.4f}",
                f"{abs(row['Final_vs_Benchmark']):.4f}"
            ])
        
        table = ax.table(
            cellText=table_data,
            colLabels=['Factor', 'Final Portfolio', 'Equal Weighted', 'Difference', 'Abs Difference'],
            cellLoc='center',
            loc='center',
            colWidths=[0.35, 0.18, 0.18, 0.15, 0.14]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(5):
            table[(0, i)].set_facecolor('#2196F3')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code differences
        for i in range(1, len(table_data) + 1):
            diff = float(table_data[i-1][3])
            if diff > 0.2:
                table[(i, 3)].set_facecolor('#C8E6C9')  # Strong green
            elif diff > 0.1:
                table[(i, 3)].set_facecolor('#E8F5E8')  # Light green
            elif diff < -0.2:
                table[(i, 3)].set_facecolor('#FFCDD2')  # Strong red
            elif diff < -0.1:
                table[(i, 3)].set_facecolor('#FFE8E8')  # Light red
        
        ax.set_title('Final Portfolio vs Equal Weighted Benchmark\nTop 50 Factor Exposure Differences', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Additional Page: Top 50 Optimized vs Benchmark Differences
        fig, ax = plt.subplots(figsize=(11, 14))
        ax.axis('tight')
        ax.axis('off')
        
        comparison_df['Abs_Optimized_vs_Benchmark'] = comparison_df['Optimized_vs_Benchmark'].abs()
        top_opt_diff = comparison_df.nlargest(50, 'Abs_Optimized_vs_Benchmark')
        
        table_data = []
        for _, row in top_opt_diff.iterrows():
            table_data.append([
                row['Factor'],
                f"{row['Optimized_Portfolio']:.4f}",
                f"{row['Equal_Weighted_Benchmark']:.4f}",
                f"{row['Optimized_vs_Benchmark']:+.4f}",
                f"{abs(row['Optimized_vs_Benchmark']):.4f}"
            ])
        
        table = ax.table(
            cellText=table_data,
            colLabels=['Factor', 'Optimized Portfolio', 'Equal Weighted', 'Difference', 'Abs Difference'],
            cellLoc='center',
            loc='center',
            colWidths=[0.35, 0.18, 0.18, 0.15, 0.14]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code differences
        for i in range(1, len(table_data) + 1):
            diff = float(table_data[i-1][3])
            if diff > 0.2:
                table[(i, 3)].set_facecolor('#C8E6C9')  # Strong green
            elif diff > 0.1:
                table[(i, 3)].set_facecolor('#E8F5E8')  # Light green
            elif diff < -0.2:
                table[(i, 3)].set_facecolor('#FFCDD2')  # Strong red
            elif diff < -0.1:
                table[(i, 3)].set_facecolor('#FFE8E8')  # Light red
        
        ax.set_title('Optimized Portfolio vs Equal Weighted Benchmark\nTop 50 Factor Exposure Differences', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Additional Page: Top 50 Final vs Optimized Comparison
        fig, ax = plt.subplots(figsize=(11, 14))
        ax.axis('tight')
        ax.axis('off')
        
        comparison_df['Abs_Final_vs_Optimized'] = comparison_df['Final_vs_Optimized'].abs()
        top_portfolio_diff = comparison_df.nlargest(50, 'Abs_Final_vs_Optimized')
        
        table_data = []
        for _, row in top_portfolio_diff.iterrows():
            table_data.append([
                row['Factor'],
                f"{row['Final_Portfolio']:.4f}",
                f"{row['Optimized_Portfolio']:.4f}",
                f"{row['Final_vs_Optimized']:+.4f}",
                f"{abs(row['Final_vs_Optimized']):.4f}"
            ])
        
        table = ax.table(
            cellText=table_data,
            colLabels=['Factor', 'Final Portfolio', 'Optimized Portfolio', 'Difference', 'Abs Difference'],
            cellLoc='center',
            loc='center',
            colWidths=[0.35, 0.18, 0.18, 0.15, 0.14]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(5):
            table[(0, i)].set_facecolor('#9C27B0')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code differences
        for i in range(1, len(table_data) + 1):
            diff = float(table_data[i-1][3])
            if diff > 0.1:
                table[(i, 3)].set_facecolor('#E1BEE7')  # Light purple
            elif diff < -0.1:
                table[(i, 3)].set_facecolor('#FFCDD2')  # Light red
        
        ax.set_title('Final Portfolio vs Optimized Portfolio\nTop 50 Factor Exposure Differences', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Final Page: Summary Statistics Table
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Calculate summary statistics
        summary_stats = {
            'Metric': [
                'Total Factors Analyzed',
                'Average Absolute Exposure - Final',
                'Average Absolute Exposure - Optimized', 
                'Average Absolute Exposure - Benchmark',
                'Max Positive Exposure - Final',
                'Max Negative Exposure - Final',
                'Max Positive Exposure - Optimized',
                'Max Negative Exposure - Optimized',
                'Factors with |Final - Benchmark| > 0.1',
                'Factors with |Optimized - Benchmark| > 0.1',
                'Factors with |Final - Optimized| > 0.1',
                'Correlation: Final vs Optimized',
                'Correlation: Final vs Benchmark',
                'Correlation: Optimized vs Benchmark'
            ],
            'Value': [
                len(comparison_df),
                f"{comparison_df['Final_Portfolio'].abs().mean():.4f}",
                f"{comparison_df['Optimized_Portfolio'].abs().mean():.4f}",
                f"{comparison_df['Equal_Weighted_Benchmark'].abs().mean():.4f}",
                f"{comparison_df['Final_Portfolio'].max():.4f}",
                f"{comparison_df['Final_Portfolio'].min():.4f}",
                f"{comparison_df['Optimized_Portfolio'].max():.4f}",
                f"{comparison_df['Optimized_Portfolio'].min():.4f}",
                len(comparison_df[comparison_df['Abs_Final_vs_Benchmark'] > 0.1]),
                len(comparison_df[comparison_df['Abs_Optimized_vs_Benchmark'] > 0.1]),
                len(comparison_df[comparison_df['Abs_Final_vs_Optimized'] > 0.1]),
                f"{comparison_df['Final_Portfolio'].corr(comparison_df['Optimized_Portfolio']):.4f}",
                f"{comparison_df['Final_Portfolio'].corr(comparison_df['Equal_Weighted_Benchmark']):.4f}",
                f"{comparison_df['Optimized_Portfolio'].corr(comparison_df['Equal_Weighted_Benchmark']):.4f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_stats)
        
        table = ax.table(cellText=summary_df.values.tolist(), colLabels=summary_df.columns.tolist(),
                        cellLoc='left', loc='center', colWidths=[0.7, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor('#FF9800')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Portfolio Factor Exposure Analysis - Summary Statistics', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    logging.info(f"Factor exposure tables saved to {output_file}")

def save_detailed_results(comparison_df, output_excel):
    """
    Save detailed results to Excel file.
    
    Args:
        comparison_df: DataFrame with factor exposure comparisons
        output_excel: Path to output Excel file
    """
    logging.info(f"Saving detailed results to {output_excel}...")
    
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        # Main comparison data
        comparison_df.to_excel(writer, sheet_name='Factor_Exposures', index=False)
        
        # Top differences
        top_final_vs_benchmark = comparison_df.nlargest(20, 'Abs_Final_vs_Benchmark')[
            ['Factor', 'Final_Portfolio', 'Equal_Weighted_Benchmark', 'Final_vs_Benchmark']
        ]
        top_final_vs_benchmark.to_excel(writer, sheet_name='Top_Final_vs_Benchmark', index=False)
        
        top_optimized_vs_benchmark = comparison_df.nlargest(20, 'Optimized_vs_Benchmark')[
            ['Factor', 'Optimized_Portfolio', 'Equal_Weighted_Benchmark', 'Optimized_vs_Benchmark']
        ]
        top_optimized_vs_benchmark.to_excel(writer, sheet_name='Top_Optimized_vs_Benchmark', index=False)
        
        top_final_vs_optimized = comparison_df.nlargest(20, 'Abs_Final_vs_Optimized')[
            ['Factor', 'Final_Portfolio', 'Optimized_Portfolio', 'Final_vs_Optimized']
        ]
        top_final_vs_optimized.to_excel(writer, sheet_name='Top_Final_vs_Optimized', index=False)
    
    logging.info(f"Detailed results saved to {output_excel}")

def main():
    """
    Main execution function for portfolio factor exposure analysis.
    """
    logging.info("Starting Portfolio Factor Exposure Analysis (PORCH)...")
    
    try:
        # Load data
        final_weights, optimized_weights = load_portfolio_weights()
        factor_exposures = load_factor_exposures()
        
        # Get all unique countries from exposure data
        all_countries = list(factor_exposures['Country'].unique())
        
        # Create equal weighted benchmark
        benchmark_weights = create_equal_weighted_benchmark(all_countries)
        
        # Calculate factor exposures for each portfolio
        logging.info("Calculating factor exposures...")
        
        final_exposures = calculate_portfolio_factor_exposures(final_weights, factor_exposures)
        optimized_exposures = calculate_portfolio_factor_exposures(optimized_weights, factor_exposures)
        benchmark_exposures = calculate_portfolio_factor_exposures(benchmark_weights, factor_exposures)
        
        # Create comparison DataFrame
        comparison_df = create_factor_exposure_comparison(final_exposures, optimized_exposures, benchmark_exposures)
        
        # Generate outputs
        output_pdf = "T2_Portfolio_Factor_Exposures.pdf"
        output_excel = "T2_Portfolio_Factor_Exposures.xlsx"
        
        create_factor_exposure_tables(comparison_df, output_pdf)
        save_detailed_results(comparison_df, output_excel)
        
        # Print summary
        logging.info("\n" + "="*60)
        logging.info("PORTFOLIO FACTOR EXPOSURE ANALYSIS COMPLETE")
        logging.info("="*60)
        logging.info(f"Analyzed {len(comparison_df)} factors across 3 portfolios")
        logging.info(f"Final Portfolio Countries: {len(final_weights)}")
        logging.info(f"Optimized Portfolio Countries: {len(optimized_weights)}")
        logging.info(f"Benchmark Portfolio Countries: {len(benchmark_weights)}")
        logging.info(f"")
        logging.info(f"Top 5 Factor Exposure Differences (Final vs Benchmark):")
        top_5 = comparison_df.head(5)
        for idx, row in top_5.iterrows():
            logging.info(f"  {row['Factor']}: {row['Final_vs_Benchmark']:+.4f}")
        logging.info(f"")
        logging.info(f"Output files created:")
        logging.info(f"  - {output_pdf}")
        logging.info(f"  - {output_excel}")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 