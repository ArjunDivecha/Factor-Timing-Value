"""
T2 Factor Timing - Step Seven: Visualize Factor Weights
======================================================

PURPOSE:
Generates professional visualizations of factor weight evolution over time from the
rolling window optimization strategy. These visualizations help analyze factor
allocation patterns, concentration, and stability in the investment strategy.

IMPORTANT NOTES:
- Only factors with non-trivial weights (≥1% allocation at any point) are visualized
- Uses a consistent color scheme across all visualizations
- Output is designed for high-resolution presentations and reports

INPUT FILES:
1. T2_rolling_window_weights.xlsx
   - Source: Output from portfolio optimization step
   - Format: Excel workbook with single sheet
   - Structure:
     * Index: Dates (datetime64[ns])
     * Columns: Factor names (str)
     * Values: Portfolio weights (float, 0-1)
   - Notes:
     * Weights should sum to 1.0 for each date
     * Missing values are not supported (should be 0)

OUTPUT FILES:
1. T2_latest_factor_allocation.pdf
   - Horizontal bar chart showing the most recent factor allocations
   - Format: PDF (vector graphics, 1200 DPI)
   - Layout:
     * Factors sorted by weight (highest at top)
     * Color gradient indicates magnitude
     * Percentage labels on bars
     * Professional styling with grid lines

2. T2_factor_allocation_grid.pdf
   - Small multiples visualization of weight evolution
   - Format: PDF (vector graphics, 1200 DPI)
   - Layout:
     * One subplot per significant factor
     * Time on x-axis, weight on y-axis
     * Consistent y-axis scale (0-100%)
     * Vertical grid lines at year boundaries
     * Factor name as subplot title

VISUALIZATION DETAILS:
- Color Scheme:
  * Uses cmocean.thermal (if available) or viridis colormap
  * Consistent coloring of factors across all charts
  * Color intensity scales with weight magnitude

- Typography:
  * Arial font family for consistent rendering
  * 10pt base font size
  * Bold titles and axis labels

- Layout:
  * 1-inch margins on all sides
  * Tight layout to minimize white space
  * High DPI (1200) for publication quality

VERSION HISTORY:
- 1.2 (2025-06-15): Added support for cmocean colormaps
- 1.1 (2025-05-20): Improved visualization quality and consistency
- 1.0 (2025-03-10): Initial version

AUTHOR: Quantitative Research Team
LAST UPDATED: 2025-06-17

NOTES:
- For best results, ensure all dependencies are up to date
- Uses vector graphics (PDF) for highest quality output
- Script is idempotent - can be safely rerun
- Runtime scales with number of factors and time periods

DEPENDENCIES:
- pandas>=2.0.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- openpyxl>=3.1.0 (for Excel file reading)
- cmocean>=3.0.0 (optional, for improved colormaps)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')

# Try to import additional color libraries
try:
    import cmocean
    has_cmocean = True
except ImportError:
    has_cmocean = False

def main():
    """
    Main function to execute the visualization process.
    """
    print("Starting Factor Weights Visualization...")
    
    # Load weights data
    print("Loading weight data from T2_rolling_window_weights.xlsx...")
    weights_df = pd.read_excel('T2_rolling_window_weights.xlsx', index_col=0)
    weights_df.index = pd.to_datetime(weights_df.index)
    weights_df = weights_df.sort_index()
    
    # --- Factor Selection: Include factors with max weight > 5% --- 
    max_weights = weights_df.max()
    threshold = 0.05 # 5% threshold
    significant_factors = max_weights[max_weights > threshold].index.tolist()

    if not significant_factors:
        print(f"Warning: No factors exceeded the {threshold:.0%} maximum weight threshold. Selecting top 5 by average weight as fallback.")
        significant_factors = weights_df.mean().nlargest(5).index.tolist()
        
    print(f"Identified {len(significant_factors)} factors with maximum weight > {threshold:.0%}")
    significant_weights = weights_df[significant_factors]
    
    # --- Sort factors by LATEST weight (descending) --- 
    latest_date = significant_weights.index.max()
    latest_weights_all_significant = significant_weights.loc[latest_date]
    # Sort factors based on latest weight
    sorted_factors_by_latest = latest_weights_all_significant.sort_values(ascending=False).index.tolist()
    # Reorder the DataFrame columns based on this new sorting
    significant_weights = significant_weights[sorted_factors_by_latest]

    # Define colors based on the sorted list of factors
    if has_cmocean:
        colors = cmocean.cm.deep(np.linspace(0, 1, len(significant_weights.columns)))
    else:
        colors = plt.cm.viridis(np.linspace(0, 0.85, len(significant_weights.columns)))
    color_map = {factor: colors[i] for i, factor in enumerate(significant_weights.columns)} 
 
    # --- Chart 1: Latest Factor Allocation --- 
    print("Creating latest factor allocation chart...")
    # Use the already sorted latest weights, filter negligible
    latest_weights_display = latest_weights_all_significant[sorted_factors_by_latest]
    latest_weights_display = latest_weights_display[latest_weights_display > 0.001]
 
    plt.figure(figsize=(12, 8))
        
    # Map colors based on the overall sorted factor list for consistency
    bar_colors = [color_map[factor] for factor in latest_weights_display.index]
         
    # Plot horizontal bars (already sorted descending by weight)
    bars = plt.barh(latest_weights_display.index, latest_weights_display.values, color=bar_colors, alpha=0.8)
    plt.xlabel('Portfolio Weight (%)', fontsize=14)
    plt.title(f'Factor Allocation for {latest_date.strftime("%Y-%m-%d")}', fontsize=18, pad=15, fontweight='bold')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    plt.xticks(fontsize=12)
    # Invert y-axis to show largest weight at the top
    plt.gca().invert_yaxis()
    plt.yticks(fontsize=10) # Adjust fontsize if needed for many factors
    plt.grid(True, alpha=0.3, linestyle='--', axis='x')
 
    # Add labels to bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2.,
                 f'{width:.1%}',
                 ha='left', va='center', fontsize=10)

    plt.tight_layout()
    output_file_latest = 'T2_latest_factor_allocation.pdf'
    print(f"Saving latest allocation chart to {output_file_latest}...")
    plt.savefig(output_file_latest, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    # --- Chart 2: Factor Allocation Through Time (Small Multiples Grid) --- 
    print("Creating factor allocation over time chart (small multiples grid)...")
    
    n_factors = len(significant_weights.columns)
    # Determine grid size (aim for roughly square)
    ncols = int(np.ceil(np.sqrt(n_factors)))
    nrows = int(np.ceil(n_factors / ncols))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4), sharex=True, sharey=True)
    axes = axes.flatten() # Flatten to 1D array for easy iteration
    
    # Use the same color map defined earlier
    # Colors are already mapped according to the new sort order
    # factor_colors = [color_map[factor] for factor in significant_weights.columns]
 
    # Iterate through factors in the order of LATEST weight (descending)
    for i, factor in enumerate(significant_weights.columns):
        ax = axes[i]
        ax.plot(significant_weights.index, significant_weights[factor], 
                color=color_map[factor], linewidth=2, label=factor)
        ax.set_title(factor, fontsize=12, pad=5)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Formatting for individual plots
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=10, rotation=0) 
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(4)) # Adjust locator as needed

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    # Overall Figure Formatting
    fig.suptitle('Factor Allocation Through Time (Individual Factors)', fontsize=20, y=1.02, fontweight='bold')
    fig.text(0.5, 0.01, 'Date', ha='center', va='center', fontsize=14)
    fig.text(0.01, 0.5, 'Portfolio Weight (%)', ha='center', va='center', rotation='vertical', fontsize=14)
    
    # Set shared y-axis limits (0 to max weight + buffer)
    max_weight = significant_weights.max().max()
    plt.ylim(0, max_weight * 1.1) # Add 10% buffer
    plt.xlim(significant_weights.index.min(), significant_weights.index.max())
    
    # Add dataset summary (similar to before, but adjusted for grid)
    start_date = significant_weights.index.min().strftime('%b %Y')
    end_date = significant_weights.index.max().strftime('%b %Y')
    factor_list_info = f"Factors with max weight > {threshold:.0%}"
    summary_text = (f"Dataset: {start_date} to {end_date}\n"
                   f"{factor_list_info}")
    fig.text(0.99, 0.01, summary_text, fontsize=10, ha='right', va='bottom',
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

    # Adjust layout
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97]) # Adjust rect to make space for suptitle and common labels

    # Save the figure as PDF
    output_file_grid = 'T2_factor_allocation_grid.pdf'
    print(f"Saving visualization to {output_file_grid}...")
    plt.savefig(output_file_grid, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print("Visualization complete!")

if __name__ == "__main__":
    main()