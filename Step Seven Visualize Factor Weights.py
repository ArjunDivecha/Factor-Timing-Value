"""
T2 Factor Timing — Step Seven (Long–Short): Visualize Factor Weights
===================================================================

PURPOSE:
Visualize net factor weights from the long–short optimizer (Step Five). Supports
negative weights. Produces a latest allocation bar chart (with diverging color by sign)
and a small-multiples grid of factor weight evolution.

INPUT FILES:
- T2_rolling_window_weights.xlsx
  - Reads Net_Weights sheet by default (falls back to first sheet)
  - Rows: dates (datetime), columns: factor names, values: net weights (− to +)

OUTPUT FILES:
- T2_latest_factor_allocation.pdf: Latest net weights (bars left/right of zero)
- T2_factor_allocation_grid.pdf: Time-series of net weights (zero line included)

NOTES:
- Factor selection is by maximum absolute weight (default threshold 5%)
- Symmetric y-limits around zero for small-multiples
- Colors: blue for positive, red for negative (latest sign)
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
    # Prefer Net_Weights sheet (long–short), fall back to default
    try:
        weights_df = pd.read_excel('T2_rolling_window_weights.xlsx', sheet_name='Net_Weights', index_col=0)
    except Exception:
        # Fallback: first sheet
        weights_df = pd.read_excel('T2_rolling_window_weights.xlsx', index_col=0)
    weights_df.index = pd.to_datetime(weights_df.index)
    weights_df = weights_df.sort_index()
    
    # --- Factor Selection: Include factors with max ABS weight > 5% (supports long–short) ---
    max_weights = weights_df.abs().max()
    threshold = 0.05 # 5% threshold
    significant_factors = max_weights[max_weights > threshold].index.tolist()

    if not significant_factors:
        print(f"Warning: No factors exceeded the {threshold:.0%} max abs weight threshold. Selecting top 5 by average ABS weight as fallback.")
        significant_factors = weights_df.abs().mean().nlargest(5).index.tolist()
        
    print(f"Identified {len(significant_factors)} factors with max ABS weight > {threshold:.0%}")
    significant_weights = weights_df[significant_factors]
    
    # --- Sort factors by LATEST weight (descending) --- 
    latest_date = significant_weights.index.max()
    latest_weights_all_significant = significant_weights.loc[latest_date]
    # Sort factors based on latest ABS weight (keep sign in values)
    sorted_factors_by_latest = latest_weights_all_significant.abs().sort_values(ascending=False).index.tolist()
    # Reorder the DataFrame columns based on this new sorting
    significant_weights = significant_weights[sorted_factors_by_latest]

    # Define colors based on the sorted list of factors
    # Use a simple diverging color scheme by sign for latest bar chart
    color_map = {factor: ('tab:blue' if latest_weights_all_significant[factor] >= 0 else 'tab:red')
                 for factor in significant_weights.columns}
 
    # --- Chart 1: Latest Factor Allocation --- 
    print("Creating latest factor allocation chart...")
    # Use the already sorted latest weights, filter negligible by ABS
    latest_weights_display = latest_weights_all_significant[sorted_factors_by_latest]
    latest_weights_display = latest_weights_display[latest_weights_display.abs() > 0.001]
 
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
    plt.axvline(0, color='gray', linewidth=1, alpha=0.6)
 
    # Add labels to bars
    for bar in bars:
        width = bar.get_width()
        x = width + (0.005 if width >= 0 else -0.005)
        ha = 'left' if width >= 0 else 'right'
        plt.text(x, bar.get_y() + bar.get_height()/2., f'{width:.1%}', ha=ha, va='center', fontsize=10)

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
 
    # Iterate through factors in the order of latest ABS weight (descending)
    for i, factor in enumerate(significant_weights.columns):
        ax = axes[i]
        series = significant_weights[factor]
        color = 'tab:blue' if series.iloc[-1] >= 0 else 'tab:red'
        ax.plot(significant_weights.index, series, color=color, linewidth=2, label=factor)
        ax.set_title(factor, fontsize=12, pad=5)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='gray', linewidth=1, alpha=0.6)
        
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
    
    # Set shared symmetric y-axis limits around zero (supports long–short)
    max_abs = significant_weights.abs().max().max()
    plt.ylim(-max_abs * 1.1, max_abs * 1.1)
    plt.xlim(significant_weights.index.min(), significant_weights.index.max())
    
    # Add dataset summary (similar to before, but adjusted for grid)
    start_date = significant_weights.index.min().strftime('%b %Y')
    end_date = significant_weights.index.max().strftime('%b %Y')
    factor_list_info = f"Factors with max ABS weight > {threshold:.0%}"
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
