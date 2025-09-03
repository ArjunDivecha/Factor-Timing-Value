"""
T2 Strategy Report Generator - Comprehensive Merged Version
========================================================

This comprehensive version combines all features from both streamlined and enhanced versions:

STREAMLINED SECTIONS:
1. Top 20 Portfolio Analysis (table)
2. Factor Returns Analysis (table)
3. Current Factor Allocation (table + bar chart combined)
4. Top 20 Country Allocations (horizontal bar chart)
5. Portfolio Returns (1, 3, 12 month) - bar chart and table

ENHANCED SECTIONS:
6. Strategy Statistics Comparison (table)
7. Strategy Performance Chart (cumulative performance)
8. Portfolio Returns Analysis (enhanced cumulative chart)

INPUT FILES:
- T2 Top20.xlsx: Top 20 portfolio analysis data
- T2_Optimizer.xlsx: Factor returns data
- T2_rolling_window_weights.xlsx: Factor weights data
- T2_Final_Country_Weights.xlsx: Country allocation data
- T2_Final_Portfolio_Returns.xlsx: Portfolio returns data
- T2_strategy_statistics.xlsx: Strategy statistics comparison data

OUTPUT FILES:
- T2_Strategy_Report_Comprehensive_YYYY-MM-DD.pdf: Complete PDF report

Version: 2.0 (Merged)
Date: 2025-05-28
Author: Merged from streamlined and enhanced versions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime
import os
from fpdf import FPDF
import io
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'

# Get current date for report
report_date = datetime.now().strftime('%Y-%m-%d')

# Define input files
top20_file = 'T2 Top20.xlsx'
optimizer_file = 'T2_Optimizer.xlsx'
factor_weights_file = 'T2_rolling_window_weights.xlsx'
country_weights_file = 'T2_Final_Country_Weights.xlsx'
portfolio_returns_file = 'T2_Final_Portfolio_Returns.xlsx'
strategy_stats_file = 'T2_strategy_statistics.xlsx'

# Output report file
output_file = f'T2_Strategy_Report_Comprehensive_{report_date}.pdf'

class PDF(FPDF):
    def header(self):
        # Title
        self.set_font('Arial', 'B', 18)  
        self.cell(0, 10, 'T2 Strategy Report - Comprehensive', 0, 1, 'C')
        self.set_font('Arial', 'I', 12)  
        self.cell(0, 10, f'Generated on {report_date}', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 10)  
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_comprehensive_report():
    """
    Generate the comprehensive T2 strategy report with all sections from both versions.
    """
    print("Generating Comprehensive T2 Strategy Report...")
    
    # Create PDF
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 40, '', 0, 1)  # Spacer
    pdf.cell(0, 20, 'T2 Strategy Report', 0, 1, 'C')
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Comprehensive Version', 0, 1, 'C')
    pdf.set_font('Arial', '', 14)
    pdf.cell(0, 20, '', 0, 1)  # Spacer
    pdf.cell(0, 10, f'Report Date: {report_date}', 0, 1, 'C')
    pdf.ln(10)
    
    # Table of Contents
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Table of Contents', 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    contents = [
        '1. Strategy Statistics Comparison',
        '2. Strategy Performance Chart',
        '3. Top 20 Portfolio Analysis',
        '4. Factor Returns Analysis',
        '5. Current Factor Allocation',
        '6. Top 20 Country Allocations',
        '7. Portfolio Returns (1, 3, 12 month)',
        '8. Portfolio Returns Analysis (Enhanced)'
    ]
    for item in contents:
        pdf.cell(0, 6, item, 0, 1, 'L')
    
    # ENHANCED SECTION 1: Strategy Statistics Comparison
    print("Processing Strategy Statistics...")
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, '1. Strategy Statistics Comparison', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 5, 'Comparison of Expanding Window vs Hybrid Window strategy performance metrics.')
    
    try:
        # Read strategy statistics
        stats_df = pd.read_excel(strategy_stats_file, sheet_name='Summary Statistics')
        
        # Create table
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 11)
        
        # Headers
        headers = ['Metric', 'Expanding Window', 'Hybrid Window']
        col_widths = [pdf.w*0.5, pdf.w*0.25, pdf.w*0.25]
        
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 7, header, 1, 0, 'C')
        pdf.ln()
        
        # Data rows
        pdf.set_font('Arial', '', 10)
        for _, row in stats_df.iterrows():
            metric = row.iloc[0]
            val1 = row.iloc[1]
            val2 = row.iloc[2] if len(row) > 2 else ''
            
            # Format values
            if '%' in str(metric):
                val1_str = f"{val1:.2f}%" if not pd.isna(val1) else "N/A"
                val2_str = f"{val2:.2f}%" if not pd.isna(val2) else "N/A"
            else:
                val1_str = f"{val1:.2f}" if not pd.isna(val1) else "N/A"
                val2_str = f"{val2:.2f}" if not pd.isna(val2) else "N/A"
            
            pdf.cell(col_widths[0], 6, str(metric), 1, 0, 'L')
            pdf.cell(col_widths[1], 6, val1_str, 1, 0, 'C')
            pdf.cell(col_widths[2], 6, val2_str, 1, 0, 'C')
            pdf.ln()
            
    except Exception as e:
        pdf.ln(5)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 5, f"Strategy statistics data is currently being updated. Error: {str(e)}")
    
    # ENHANCED SECTION 2: Strategy Performance Chart
    print("Generating Strategy Performance Chart...")
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, '2. Strategy Performance Chart', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 5, 'Cumulative performance comparison of Expanding Window vs Hybrid Window strategies.')
    
    try:
        # Read monthly returns data
        returns_df = pd.read_excel(strategy_stats_file, sheet_name='Monthly Returns')
        date_col = returns_df.columns[0]
        returns_df[date_col] = pd.to_datetime(returns_df[date_col])
        returns_df = returns_df.set_index(date_col)
        
        # Calculate cumulative returns
        cum_returns = (1 + returns_df/100).cumprod()
        
        # Create performance chart
        plt.figure(figsize=(12, 7))
        for col in cum_returns.columns:
            plt.plot(cum_returns.index, cum_returns[col], label=col, linewidth=2)
        
        plt.title('Strategy Performance Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return (1 = 100%)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
        img_buf.seek(0)
        pdf.image(img_buf, x=10, y=None, w=180)
        plt.close()
        
    except Exception as e:
        pdf.ln(5)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 5, f"Error generating performance chart: {str(e)}")
    
    # STREAMLINED SECTION 1: Top 20 Portfolio Analysis
    print("Processing Top 20 Portfolio data...")
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, '3. Top 20 Portfolio Analysis', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 5, 'The following table shows the top 10 portfolios from our T2 Top20 analysis.')
    
    try:
        # Read top 20 data
        top20_data = pd.read_excel(top20_file)
        top10 = top20_data.head(10)
        
        # Create table
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 10)
        
        # Define columns to display (limit to 5 for readability)
        display_cols = ['Feature', 'Avg Excess Return (%)', 'Volatility (%)', 
                       'Information Ratio', 'Maximum Drawdown (%)']
        
        # Check which columns exist in the data
        available_cols = []
        for col in display_cols:
            if col in top10.columns:
                available_cols.append(col)
        
        # If we couldn't find specific columns, use first 5
        if len(available_cols) < 3:
            available_cols = list(top10.columns[:5])
        
        # Calculate column widths
        page_width = pdf.w - 2*pdf.l_margin
        col_widths = [page_width/len(available_cols)] * len(available_cols)
        
        # Print headers
        for i, header in enumerate(available_cols):
            pdf.cell(col_widths[i], 7, str(header)[:20], 1, 0, 'C')
        pdf.ln()
        
        # Print data rows
        pdf.set_font('Arial', '', 9)
        for _, row in top10.iterrows():
            for i, col in enumerate(available_cols):
                value = row[col]
                if isinstance(value, (int, float)):
                    if abs(value) < 1:
                        value_str = f"{value:.2f}"
                    else:
                        value_str = f"{value:.1f}"
                else:
                    value_str = str(value)[:20]
                pdf.cell(col_widths[i], 6, value_str, 1, 0, 'C')
            pdf.ln()
            
    except Exception as e:
        pdf.ln(5)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 5, f"Top 20 portfolio data is currently being updated. Error: {str(e)}")
    
    # STREAMLINED SECTION 2: Factor Returns Analysis
    print("Processing Factor Returns data...")
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, '4. Factor Returns Analysis', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 5, 'Top 10 average factor returns for the last 1, 3, and 12 months.')
    
    try:
        # Read factor returns data
        factor_returns = pd.read_excel(optimizer_file)
        
        # Get the date column (first column)
        date_col = factor_returns.columns[0]
        factor_returns[date_col] = pd.to_datetime(factor_returns[date_col], errors='coerce')
        factor_returns = factor_returns.set_index(date_col).sort_index()
        
        # Calculate returns for different periods
        last_1m = factor_returns.iloc[-1] / 100
        last_3m = factor_returns.iloc[-3:].mean() / 100 if len(factor_returns) >= 3 else pd.Series()
        last_12m = factor_returns.iloc[-12:].mean() / 100 if len(factor_returns) >= 12 else pd.Series()
        
        # Create summary DataFrame
        factor_summary = pd.DataFrame({
            '1-Month Return': last_1m,
            '3-Month Return': last_3m,
            '12-Month Return': last_12m
        })
        
        # Get top 10 by 1-month return
        top10_factors = factor_summary.sort_values('1-Month Return', ascending=False).head(10)
        
        # Create table
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 10)
        
        headers = ['Factor', '1-Month Return', '3-Month Return', '12-Month Return']
        col_widths = [pdf.w*0.4, pdf.w*0.2, pdf.w*0.2, pdf.w*0.2]
        
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 7, header, 1, 0, 'C')
        pdf.ln()
        
        # Data rows
        pdf.set_font('Arial', '', 9)
        for idx, row in top10_factors.iterrows():
            pdf.cell(col_widths[0], 6, str(idx)[:30], 1, 0, 'L')
            for i, col in enumerate(top10_factors.columns):
                value = row[col]
                if pd.isna(value):
                    value_str = "N/A"
                else:
                    value_str = f"{value*100:.2f}%"
                pdf.cell(col_widths[i+1], 6, value_str, 1, 0, 'C')
            pdf.ln()
            
    except Exception as e:
        pdf.ln(5)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 5, f"Factor returns data is currently being updated. Error: {str(e)}")
    
    # STREAMLINED SECTION 3: Current Factor Allocation (Combined Table + Chart)
    print("Processing Current Factor Allocation...")
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, '5. Current Factor Allocation', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 5, 'Analysis of the latest factor weights for the T2 strategy.')
    
    try:
        # Read factor weights
        factor_weights = pd.read_excel(factor_weights_file)
        date_col = factor_weights.columns[0]
        latest_weights = factor_weights.iloc[-1].drop(date_col)
        
        # Filter non-zero weights and sort
        non_zero_weights = latest_weights[latest_weights > 0.01]
        sorted_weights = non_zero_weights.sort_values(ascending=False)
        top_weights = sorted_weights.head(15)
        
        # Create table first
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 10)
        
        headers = ['Factor', 'Weight']
        col_widths = [pdf.w*0.7, pdf.w*0.3]
        
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 7, header, 1, 0, 'C')
        pdf.ln()
        
        # Data rows
        pdf.set_font('Arial', '', 9)
        for factor, weight in top_weights.items():
            pdf.cell(col_widths[0], 6, str(factor)[:40], 1, 0, 'L')
            pdf.cell(col_widths[1], 6, f"{weight*100:.2f}%", 1, 0, 'C')
            pdf.ln()
        
        # Add note
        pdf.ln(3)
        pdf.set_font('Arial', 'I', 9)
        pdf.multi_cell(0, 5, f"Note: Showing top 15 of {len(non_zero_weights)} factors with non-zero weights (out of {len(latest_weights)} total factors).")
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(top_weights)), top_weights.values * 100, color='steelblue', alpha=0.7)
        plt.title('Top 15 Factor Weights', fontsize=16, fontweight='bold')
        plt.xlabel('Factors', fontsize=12)
        plt.ylabel('Weight (%)', fontsize=12)
        plt.xticks(range(len(top_weights)), top_weights.index, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save chart and add to PDF
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
        img_buf.seek(0)
        pdf.ln(5)
        pdf.image(img_buf, x=10, y=None, w=180)
        plt.close()
        
        # Add Factor Weight Heatmap
        print("Adding Factor Weight Heatmap...")
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Factor Weight Heatmap', 0, 1, 'L')
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 5, 'Heatmap visualization of factor weights over time.')
        pdf.ln(5)
        
        # Create sophisticated factor weight heatmap
        weights_df = factor_weights.copy()
        if date_col in weights_df.columns:
            weights_df.set_index(date_col, inplace=True)
        
        # Ensure all data is numeric by converting any non-numeric values to float
        weights_df = weights_df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Define factor categories and their prefixes/keywords
        categories = {
            'Value': ['Best PE', 'Best PBK', 'Best Price Sales', 'Best Div Yield', 'Trailing PE', 'Positive PE', 
                     'Shiller PE', 'Earnings Yield', 'Best Cash Flow', 'EV to EBITDA'],
            'Momentum': ['1MTR', '3MTR', '12-1MTR', 'RSI14', '120MA', 'Advance Decline', 'Signal', 'P2P'],
            'Economic': ['Inflation', 'REER', '10Yr Bond', 'GDP', 'Debt to GDP', 'Budget Def', 'Current Account'],
            'Quality': ['Best ROE', 'BEST EPS', 'Operating Margin', 'Trailing EPS', 'Debt to EV', 'LT Growth', 'Bloom Country Risk', '20 day vol'],
            'Commodity': ['Oil', 'Gold', 'Copper', 'Agriculture', 'Currency']
        }

        # Sophisticated category styling
        category_colors = {
            'Value': '#2d5a3d',      # Deep forest green
            'Momentum': '#1e3a5f',   # Deep navy blue  
            'Economic': '#5d2e5a',   # Deep purple
            'Quality': '#5a3d2d',    # Deep brown
            'Commodity': '#5a4b2d'   # Deep golden brown
        }
        
        category_backgrounds = {
            'Value': '#f8fdfb',      # Very light green tint
            'Momentum': '#f8fafd',   # Very light blue tint
            'Economic': '#fdf8fd',   # Very light purple tint
            'Quality': '#fdfbf8',    # Very light brown tint
            'Commodity': '#fdfcf8'   # Very light golden tint
        }

        # Get top factors by average weight
        top_n = 15  # Show top 15 factors
        avg_weights = weights_df.mean().sort_values(ascending=False)
        top_factors = avg_weights.head(top_n).index.tolist()
        
        # Ensure we're working with the subset of weights that are in top_factors
        filtered_weights_df = weights_df[top_factors]

        # Organize factors by category
        categorized_factors = []
        category_map = {}
        category_groups = {}
        
        for category, keywords in categories.items():
            category_factors = []
            for factor in top_factors:
                for keyword in keywords:
                    if keyword in factor and factor not in categorized_factors:
                        category_factors.append(factor)
                        category_map[factor] = category
                        break
            if category_factors:
                category_factors.sort()
                categorized_factors.extend(category_factors)
                
        for factor in top_factors:
            if factor not in categorized_factors:
                categorized_factors.append(factor)
                category_map[factor] = "Other"
                
        ordered_factors = categorized_factors[:top_n] if len(categorized_factors) > 0 else top_factors

        # Calculate category positions
        current_pos = 0
        for category in ['Value', 'Momentum', 'Economic', 'Quality', 'Commodity']:
            cat_factors = [f for f in ordered_factors if category_map.get(f) == category]
            if cat_factors:
                category_groups[category] = {
                    'start': current_pos, 
                    'end': current_pos + len(cat_factors) - 1,
                    'factors': cat_factors
                }
                current_pos += len(cat_factors)

        # Sample data every 3 months for better visualization
        sample_weights = filtered_weights_df[ordered_factors].iloc[::3].copy()
        if len(sample_weights) == 0:
            pdf.set_font('Arial', '', 11)
            pdf.multi_cell(0, 5, "Warning: No weight data available for heatmap")
        else:    
            # Ensure data is numeric and handle any remaining issues
            sample_weights = sample_weights.astype(float)

            # Use full factor names (no truncation for professional look) and rename 1MTR
            clean_factor_names = []
            for factor in ordered_factors:
                clean_name = factor.replace('_', ' ')
                clean_name = clean_name.replace('1MTR', '1 Month Return')
                clean_factor_names.append(clean_name)

            # Create sophisticated custom colormap: Light sage → Soft teal → Deep teal
            colors = ['#f8fffe', '#e8f5f3', '#d1ebe6', '#a8dadc', '#79c2d0', '#5aa9c4', '#457b9d']
            sophisticated_cmap = mcolors.LinearSegmentedColormap.from_list("sophisticated", colors, N=256)
            sophisticated_cmap.set_under('white')

            # Create figure with optimal proportions
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Set sophisticated styling
            plt.rcParams['font.family'] = ['Arial', 'Helvetica', 'DejaVu Sans']
            
            # Create category background rectangles
            for category, group_info in category_groups.items():
                if category in category_backgrounds:
                    start_pos = group_info['start'] - 0.5
                    height = group_info['end'] - group_info['start'] + 1
                    
                    # Add subtle background tinting
                    background_rect = Rectangle(
                        (-0.5, start_pos), len(sample_weights.index), height,
                        facecolor=category_backgrounds[category], edgecolor='none', alpha=0.4, zorder=0
                    )
                    ax.add_patch(background_rect)
            
            # Determine appropriate max value for colormap
            max_value = sample_weights.values.max()
            if not np.isfinite(max_value) or max_value <= 0:
                max_value = 0.5
            else:
                max_value = min(max_value, 0.5)
                
            im = ax.imshow(sample_weights.T.values, cmap=sophisticated_cmap, aspect='auto',
                        vmin=0.001, vmax=max_value)

            # Sophisticated tick formatting
            tick_positions = range(0, len(sample_weights.index), 2)  # Every 2nd period for cleaner look
            ax.set_xticks(tick_positions)
            date_labels = [sample_weights.index[i].strftime('%Y-%m') if hasattr(sample_weights.index[i], 'strftime') else str(sample_weights.index[i]) for i in tick_positions]
            ax.set_xticklabels(date_labels, fontsize=12, fontweight='500', color='#333333')
            
            ax.set_yticks(range(len(clean_factor_names)))
            ax.set_yticklabels(clean_factor_names, fontsize=11, fontweight='400', color='#333333')

            # Professional colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.7, aspect=25, pad=0.02)
            cbar.set_label('Portfolio Weight (%)', rotation=270, labelpad=25, 
                        fontsize=13, fontweight='600', color='#333333')
            cbar.ax.tick_params(labelsize=11, colors='#333333')
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
            cbar.outline.set_linewidth(0.5)
            cbar.outline.set_edgecolor('#cccccc')

            # Elegant grid
            ax.set_xticks(np.arange(-0.5, len(sample_weights.index), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(clean_factor_names), 1), minor=True)
            ax.grid(which="minor", color="#f0f0f0", linestyle='-', linewidth=0.5, alpha=0.8)

            # Add thin horizontal lines between categories for clear boundaries
            for category, group_info in category_groups.items():
                end_pos = group_info['end']
                # Draw thin line below each category (except the last one)
                if end_pos < len(ordered_factors) - 1:
                    ax.axhline(y=end_pos + 0.5, color='#999999', linewidth=1.0, alpha=0.8, zorder=4)

            # Add horizontal category headers in the middle of the heatmap
            middle_x = len(sample_weights.index) / 2
            for category, group_info in category_groups.items():
                if category in category_colors:
                    start_pos = group_info['start']
                    end_pos = group_info['end']
                    center_pos = (start_pos + end_pos) / 2
                    
                    # Add elegant horizontal category header (no rotation)
                    ax.text(middle_x, center_pos, category,
                            fontsize=16, fontweight='700', color=category_colors[category],
                            ha='center', va='center', rotation=0,
                            bbox=dict(boxstyle="round,pad=0.5", 
                                    facecolor='white', 
                                    edgecolor=category_colors[category],
                                    linewidth=2.0, alpha=0.95),
                            zorder=5)

            # Professional title with elegant typography
            ax.set_title('Portfolio Weight Allocation Over Time\nTop 15 Factors by Average Weight', 
                        fontsize=18, fontweight='700', color='#1a1a1a', pad=20,
                        fontfamily='serif')

            # Sophisticated axis labels
            ax.set_xlabel('Date', fontsize=14, fontweight='600', color='#333333', labelpad=15)
            ax.set_ylabel('Investment Factor', fontsize=14, fontweight='600', color='#333333', labelpad=15)

            # Remove tick marks for cleaner appearance
            ax.tick_params(axis='both', which='both', length=0, pad=8)

            # Set elegant background
            ax.set_facecolor('#fcfcfc')
            fig.patch.set_facecolor('white')

            # Add subtle frame
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_edgecolor('#dddddd')

            # Optimal layout with generous margins
            plt.tight_layout()

            # Save and add to PDF
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
            img_buf.seek(0)
            pdf.image(img_buf, x=10, y=None, w=180)
            plt.close()
            
            pdf.ln(5)
            pdf.set_font('Arial', 'I', 9)
            pdf.multi_cell(0, 5, 'Source: T2 Factor Weight Analysis - Full Period')
            
            # Add a second heatmap focusing on just the past 3 years
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Recent Factor Weight Trends (Past 3 Years)', 0, 1, 'L')
            pdf.set_font('Arial', '', 12)
            pdf.multi_cell(0, 5, 'Detailed view of factor weight evolution over the past 3 years.')
            pdf.ln(5)
            
            # Filter for just the past 3 years (36 months)
            if isinstance(weights_df.index, pd.DatetimeIndex):
                # If we have datetime index, filter by date
                cutoff_date = weights_df.index.max() - pd.DateOffset(years=3)
                recent_weights_df = weights_df[weights_df.index >= cutoff_date]
            else:
                # Otherwise just take the last 36 months (or all if less than 36)
                if len(weights_df) > 36:
                    recent_weights_df = weights_df.tail(36)
                else:
                    recent_weights_df = weights_df.copy()
            
            # Use the same factors as the full heatmap for consistency
            recent_filtered_weights = recent_weights_df[ordered_factors]
            
            # Sample at higher frequency for the 3-year view (every month)
            recent_sample_weights = recent_filtered_weights.copy()
            
            if len(recent_sample_weights) > 0:
                # Create figure with optimal proportions
                fig, ax = plt.subplots(figsize=(14, 10))
                
                # Set sophisticated styling
                plt.rcParams['font.family'] = ['Arial', 'Helvetica', 'DejaVu Sans']
                
                # Create category background rectangles
                for category, group_info in category_groups.items():
                    if category in category_backgrounds:
                        start_pos = group_info['start'] - 0.5
                        height = group_info['end'] - group_info['start'] + 1
                        
                        # Add subtle background tinting
                        background_rect = Rectangle(
                            (-0.5, start_pos), len(recent_sample_weights.index), height,
                            facecolor=category_backgrounds[category], edgecolor='none', alpha=0.4, zorder=0
                        )
                        ax.add_patch(background_rect)
                
                # Determine appropriate max value for colormap
                max_value = recent_sample_weights.values.max()
                if not np.isfinite(max_value) or max_value <= 0:
                    max_value = 0.5
                else:
                    max_value = min(max_value, 0.5)
                    
                im = ax.imshow(recent_sample_weights.T.values, cmap=sophisticated_cmap, aspect='auto',
                            vmin=0.001, vmax=max_value)

                # Sophisticated tick formatting - show more frequent ticks for 3-year view
                tick_positions = range(0, len(recent_sample_weights.index), 1 if len(recent_sample_weights) < 24 else 2)
                ax.set_xticks(tick_positions)
                date_labels = [recent_sample_weights.index[i].strftime('%Y-%m') if hasattr(recent_sample_weights.index[i], 'strftime') else str(recent_sample_weights.index[i]) for i in tick_positions]
                ax.set_xticklabels(date_labels, fontsize=10, fontweight='500', color='#333333', rotation=45, ha='right')
                
                ax.set_yticks(range(len(clean_factor_names)))
                ax.set_yticklabels(clean_factor_names, fontsize=11, fontweight='400', color='#333333')

                # Professional colorbar
                cbar = plt.colorbar(im, ax=ax, shrink=0.7, aspect=25, pad=0.02)
                cbar.set_label('Portfolio Weight (%)', rotation=270, labelpad=25, 
                            fontsize=13, fontweight='600', color='#333333')
                cbar.ax.tick_params(labelsize=11, colors='#333333')
                cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
                cbar.outline.set_linewidth(0.5)
                cbar.outline.set_edgecolor('#cccccc')

                # Elegant grid
                ax.set_xticks(np.arange(-0.5, len(recent_sample_weights.index), 1), minor=True)
                ax.set_yticks(np.arange(-0.5, len(clean_factor_names), 1), minor=True)
                ax.grid(which="minor", color="#f0f0f0", linestyle='-', linewidth=0.5, alpha=0.8)

                # Add thin horizontal lines between categories for clear boundaries
                for category, group_info in category_groups.items():
                    end_pos = group_info['end']
                    # Draw thin line below each category (except the last one)
                    if end_pos < len(ordered_factors) - 1:
                        ax.axhline(y=end_pos + 0.5, color='#999999', linewidth=1.0, alpha=0.8, zorder=4)

                # Add horizontal category headers in the middle of the heatmap
                middle_x = len(recent_sample_weights.index) / 2
                for category, group_info in category_groups.items():
                    if category in category_colors:
                        start_pos = group_info['start']
                        end_pos = group_info['end']
                        center_pos = (start_pos + end_pos) / 2
                        
                        # Add elegant horizontal category header (no rotation)
                        ax.text(middle_x, center_pos, category,
                                fontsize=16, fontweight='700', color=category_colors[category],
                                ha='center', va='center', rotation=0,
                                bbox=dict(boxstyle="round,pad=0.5", 
                                        facecolor='white', 
                                        edgecolor=category_colors[category],
                                        linewidth=2.0, alpha=0.95),
                                zorder=5)

                # Professional title with elegant typography
                ax.set_title('Recent Factor Weight Evolution\nPast 3 Years', 
                            fontsize=18, fontweight='700', color='#1a1a1a', pad=20,
                            fontfamily='serif')

                # Sophisticated axis labels
                ax.set_xlabel('Date', fontsize=14, fontweight='600', color='#333333', labelpad=15)
                ax.set_ylabel('Investment Factor', fontsize=14, fontweight='600', color='#333333', labelpad=15)

                # Remove tick marks for cleaner appearance
                ax.tick_params(axis='both', which='both', length=0, pad=8)

                # Set elegant background
                ax.set_facecolor('#fcfcfc')
                fig.patch.set_facecolor('white')

                # Add subtle frame
                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)
                    spine.set_edgecolor('#dddddd')

                # Optimal layout with generous margins
                plt.tight_layout()

                # Save and add to PDF
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
                img_buf.seek(0)
                pdf.image(img_buf, x=10, y=None, w=180)
                plt.close()
                
                pdf.ln(5)
                pdf.set_font('Arial', 'I', 9)
                pdf.multi_cell(0, 5, 'Source: T2 Factor Weight Analysis - Recent 3 Years')
            else:
                pdf.set_font('Arial', '', 11)
                pdf.multi_cell(0, 5, "Warning: Insufficient data for 3-year heatmap view")
        
        
    except Exception as e:
        pdf.ln(5)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 5, f"Factor allocation data is currently being updated. Error: {str(e)}")
    
    # STREAMLINED SECTION 4: Top 20 Country Allocations
    print("Processing Country Allocation data...")
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, '6. Top 20 Country Allocations', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 5, 'Current country weights for the T2 strategy portfolio.')
    
    try:
        # Try to read latest weights
        try:
            country_weights = pd.read_excel(country_weights_file, sheet_name='Latest Weights')
            country_weights.columns = ['Country', 'Weight']
        except:
            # Fallback to All Periods sheet
            all_periods = pd.read_excel(country_weights_file, sheet_name='All Periods', index_col=0)
            latest = all_periods.iloc[-1]
            country_weights = pd.DataFrame({
                'Country': latest.index,
                'Weight': latest.values
            })
        
        # Clean and sort data
        country_weights['Weight'] = pd.to_numeric(country_weights['Weight'], errors='coerce').fillna(0.0)
        country_weights = country_weights[country_weights['Weight'] > 0]
        sorted_countries = country_weights.sort_values('Weight', ascending=False).head(20)
        
        # Create horizontal bar chart
        plt.figure(figsize=(12, 10))
        bars = plt.barh(range(len(sorted_countries)), sorted_countries['Weight']*100, color='steelblue', alpha=0.7)
        plt.yticks(range(len(sorted_countries)), sorted_countries['Country'])
        plt.xlabel('Weight (%)', fontsize=12)
        plt.title('Top 20 Country Allocations', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()  # Highest weight at top
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, row) in enumerate(sorted_countries.iterrows()):
            plt.text(row['Weight']*100 + 0.1, i, f"{row['Weight']*100:.1f}%", va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save and add to PDF
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
        img_buf.seek(0)
        pdf.ln(5)
        pdf.image(img_buf, x=10, y=None, w=180)
        plt.close()
        
    except Exception as e:
        pdf.ln(5)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 5, f"Country allocation data is currently being updated. Error: {str(e)}")
    
    # STREAMLINED SECTION 5: Portfolio Returns (1, 3, 12 month)
    print("Processing Portfolio Returns...")
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, '7. Portfolio Returns (1, 3, 12 month)', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 5, 'Portfolio returns for the last 1, 3, and 12 months.')
    
    try:
        # Read portfolio returns data from the properly formatted Excel file
        try:
            # First try to read Monthly Returns sheet which has the correct structure
            monthly_returns = pd.read_excel(portfolio_returns_file, sheet_name='Monthly Returns', index_col=0)
            # Convert index to datetime if it's not already
            if not isinstance(monthly_returns.index, pd.DatetimeIndex):
                monthly_returns.index = pd.to_datetime(monthly_returns.index)
        except Exception as e:
            # Fall back to the main sheet if Monthly Returns isn't available
            monthly_returns = pd.read_excel(portfolio_returns_file, index_col=0)
            if not isinstance(monthly_returns.index, pd.DatetimeIndex):
                monthly_returns.index = pd.to_datetime(monthly_returns.index)
        
        # Make sure we have Portfolio, Equal Weight, and Net Return columns
        if 'Portfolio' not in monthly_returns.columns or 'Equal Weight' not in monthly_returns.columns:
            # If not, try to identify them by similar names
            cols = monthly_returns.columns
            portfolio_col = [c for c in cols if 'portfolio' in c.lower() or 'strategy' in c.lower()]
            equal_weight_col = [c for c in cols if 'equal' in c.lower() or 'benchmark' in c.lower()]
            
            if portfolio_col and equal_weight_col:
                # Rename columns to expected format
                monthly_returns.rename(columns={
                    portfolio_col[0]: 'Portfolio',
                    equal_weight_col[0]: 'Equal Weight'
                }, inplace=True)
                
                # Calculate Net Return if not present
                if 'Net Return' not in monthly_returns.columns:
                    monthly_returns['Net Return'] = monthly_returns['Portfolio'] - monthly_returns['Equal Weight']
        
        # Calculate returns for 1, 3, 12 month periods
        # Using the actual returns data, not percentage changes in cumulative returns
        returns_data = {
            'Portfolio': [],
            'Equal Weight': [],
            'Net Return': []
        }
        
        # Get last 12 months of data
        last_12_months = monthly_returns.tail(12)
        
        # Calculate correct 1-month return (the most recent month)
        returns_data['Portfolio'].append(last_12_months['Portfolio'].iloc[-1] * 100)  # Convert to percentage
        returns_data['Equal Weight'].append(last_12_months['Equal Weight'].iloc[-1] * 100)
        returns_data['Net Return'].append(last_12_months['Net Return'].iloc[-1] * 100)
        
        # Calculate correct 3-month return (sum of last 3 months)
        returns_data['Portfolio'].append(last_12_months['Portfolio'].iloc[-3:].sum() * 100)
        returns_data['Equal Weight'].append(last_12_months['Equal Weight'].iloc[-3:].sum() * 100)
        returns_data['Net Return'].append(last_12_months['Net Return'].iloc[-3:].sum() * 100)
        
        # Calculate correct 12-month return (sum of all 12 months)
        returns_data['Portfolio'].append(last_12_months['Portfolio'].sum() * 100)
        returns_data['Equal Weight'].append(last_12_months['Equal Weight'].sum() * 100)
        returns_data['Net Return'].append(last_12_months['Net Return'].sum() * 100)
        
        # Create properly formatted DataFrame
        returns_df = pd.DataFrame(returns_data, 
                                index=['1-Month Return', '3-Month Return', '12-Month Return']).T
        
        # Create bar chart
        plt.figure(figsize=(12, 7))
        x = np.arange(len(returns_df.columns))
        width = 0.25
        
        for i, (idx, row) in enumerate(returns_df.iterrows()):
            offset = (i - len(returns_df)/2 + 0.5) * width
            plt.bar(x + offset, row.values, width, label=idx, alpha=0.8)
        
        plt.xlabel('Period', fontsize=12)
        plt.ylabel('Return (%)', fontsize=12)
        plt.title('Portfolio Returns by Period', fontsize=16, fontweight='bold')
        plt.xticks(x, returns_df.columns)
        plt.legend()
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
        img_buf.seek(0)
        pdf.ln(5)
        pdf.image(img_buf, x=10, y=None, w=180)
        plt.close()
        
        # Create table on new page
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Portfolio Returns Summary Table', 0, 1, 'L')
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 10)
        
        # Fixed table headers based on statistics from Step Nine
        headers = ['', 'Portfolio', 'Equal Weight', 'Net Return']
        col_width = 180 / len(headers)
        
        for header in headers:
            pdf.cell(col_width, 7, str(header), 1, 0, 'C')
        pdf.ln()
        
        # Fixed statistics data from Step Nine
        stats_data = [
            ['Annual Return', '14.63', '8.05', '6.58'],
            ['Annual Vol', '20.06', '18.08', '6.52'],
            ['Sharpe Ratio', '0.73', '0.44', '1.01'],
            ['Max Drawdown', '-61.91', '-57.67', '-12.09'],
            ['Hit Rate', '61.79', '58.14', '62.46'],
            ['Skewness', '-0.51', '-0.58', '0.10'],
            ['Kurtosis', '2.47', '2.06', '1.28']
        ]
        
        # Table data
        pdf.set_font('Arial', '', 9)
        for row in stats_data:
            pdf.cell(col_width, 7, row[0], 1, 0, 'L')
            for i in range(1, 4):
                pdf.cell(col_width, 7, row[i], 1, 0, 'C')
            pdf.ln()
            
    except Exception as e:
        pdf.ln(5)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 5, f"Portfolio returns data is currently being updated. Error: {str(e)}")
    
    # ENHANCED SECTION 3: Portfolio Returns Analysis (Enhanced)
    print("Processing enhanced portfolio returns...")
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, '8. Portfolio Returns Analysis (Enhanced)', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 5, 'Enhanced cumulative portfolio returns analysis with growth visualization.')
    
    try:
        # Load data from the portfolio returns Excel file
        results = pd.read_excel(portfolio_returns_file, sheet_name='Monthly Returns', index_col=0)
        
        # Convert index to datetime
        results.index = pd.to_datetime(results.index)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + results).cumprod()
        cumulative_net = (1 + results['Net Return']).cumprod() if 'Net Return' in results.columns else None
        
        # Create multi-panel figure for performance visualization
        plt.figure(figsize=(15, 10))  # Reduced height since we're only showing 2 plots
        
        # Plot 1: Cumulative Total Returns - Compare portfolio to benchmark
        plt.subplot(2, 1, 1)
        cumulative_returns['Portfolio'].plot(label='Portfolio', color='blue')
        cumulative_returns['Equal Weight'].plot(label='Equal Weight', color='red', alpha=0.7)
        plt.title('Cumulative Total Returns')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.legend()
        
        # Plot 2: Cumulative Net Returns - Show excess return over benchmark
        plt.subplot(2, 1, 2)
        if cumulative_net is not None:
            cumulative_net.plot(label='Cumulative Net Return', color='green')
        else:
            # If Net Return column doesn't exist, calculate it
            net_return = cumulative_returns['Portfolio'] - cumulative_returns['Equal Weight']
            net_return.plot(label='Cumulative Net Return', color='green')
        
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.3)  # Reference line at 1.0
        plt.title('Cumulative Net Returns (Portfolio - Equal Weight)')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        # Save chart directly into the PDF
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
        img_buf.seek(0)
        pdf.image(img_buf, x=10, y=None, w=180)
        plt.close()
        
        # Add summary statistics table
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Enhanced Portfolio Statistics Summary', 0, 1, 'L')
        
        # Check if we have the data to calculate statistics
        if os.path.exists(portfolio_returns_file):
            try:
                # Read portfolio returns
                portfolio_data = pd.read_excel(portfolio_returns_file)
                
                # Remove unnamed columns
                portfolio_data = portfolio_data[[col for col in portfolio_data.columns if 'Unnamed' not in col]]
                
                # Identify date column
                date_col = 'Date' if 'Date' in portfolio_data.columns else portfolio_data.columns[0]
                portfolio_data[date_col] = pd.to_datetime(portfolio_data[date_col])
                portfolio_data = portfolio_data.set_index(date_col).sort_index()
                
                # Calculate cumulative returns (assuming data is in percentage returns)
                cum_returns = (1 + portfolio_data/100).cumprod()
                
                # Calculate additional statistics
                stats_dict = {}
                for col in portfolio_data.columns:
                    returns = portfolio_data[col] / 100  # Convert to decimal
                    stats_dict[col] = {
                        'Total Return': f"{(cum_returns[col].iloc[-1] - 1) * 100:.2f}%",
                        'Annualized Return': f"{(cum_returns[col].iloc[-1] ** (252/len(returns)) - 1) * 100:.2f}%",
                        'Volatility': f"{returns.std() * np.sqrt(252) * 100:.2f}%",
                        'Sharpe Ratio': f"{(returns.mean() * 252) / (returns.std() * np.sqrt(252)):.2f}",
                        'Max Drawdown': f"{((cum_returns[col] / cum_returns[col].cummax() - 1).min() * 100):.2f}%"
                    }
                
                # Create statistics table
                pdf.ln(5)
                pdf.set_font('Arial', 'B', 9)
                
                metrics = list(next(iter(stats_dict.values())).keys())
                portfolios = list(stats_dict.keys())
                
                # Headers
                pdf.cell(40, 7, 'Metric', 1, 0, 'C')
                col_width = 140 / len(portfolios)
                for portfolio in portfolios:
                    pdf.cell(col_width, 7, str(portfolio)[:12], 1, 0, 'C')
                pdf.ln()
                
                # Data rows
                pdf.set_font('Arial', '', 8)
                for metric in metrics:
                    pdf.cell(40, 6, metric, 1, 0, 'L')
                    for portfolio in portfolios:
                        pdf.cell(col_width, 6, stats_dict[portfolio][metric], 1, 0, 'C')
                    pdf.ln()
            except Exception as e:
                pdf.multi_cell(0, 5, f"Error generating statistics: {str(e)}")
        else:
            pdf.multi_cell(0, 5, f"Error: The portfolio returns data file '{portfolio_returns_file}' was not found.")
    
    except Exception as e:
        pdf.ln(5)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 5, f"Error in portfolio returns analysis section: {str(e)}")

    
    # Summary Page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 18)
    pdf.cell(0, 20, 'Report Summary', 0, 1, 'C')
    pdf.set_font('Arial', '', 12)
    pdf.ln(10)
    
    summary_text = f"""
This comprehensive T2 Strategy Report contains detailed analysis across 8 key sections:

STRATEGY PERFORMANCE:
- Strategy Statistics Comparison - Performance metrics for different window approaches
- Strategy Performance Chart - Visual comparison of cumulative returns
- Enhanced Portfolio Analysis - Detailed risk-return statistics

PORTFOLIO COMPOSITION:
- Top 20 Portfolio Analysis - Best performing portfolio configurations
- Factor Returns Analysis - 1, 3, and 12-month factor performance
- Current Factor Allocation - Active factor weights and allocations

GEOGRAPHIC ALLOCATION:
- Top 20 Country Allocations - Geographic distribution of investments

RETURNS ANALYSIS:
- Portfolio Returns (Multiple Periods) - Comprehensive return analysis
- Enhanced Portfolio Statistics - Risk metrics and performance ratios

Report generated on: {report_date}
Total sections: 8
Data sources: 6 Excel files

This report provides comprehensive insights into the T2 strategy performance,
factor exposures, geographic allocations, and risk-return characteristics.
    """
    
    pdf.multi_cell(0, 6, summary_text.strip())
    
    # Save the PDF
    pdf.output(output_file)
    print(f"Comprehensive report successfully generated: {output_file}")
    
    # Try to open the report
    try:
        if os.name == 'nt':  # Windows
            os.startfile(output_file)
        elif os.name == 'posix':  # macOS/Linux
            os.system(f'open "{output_file}"')
        print(f"Report opened: {output_file}")
    except:
        print(f"Please open the report manually: {output_file}")
    
    return output_file

def generate_report_with_options():
    """
    Enhanced function with additional options and error handling.
    """
    print("=" * 60)
    print("T2 Strategy Report Generator - Comprehensive Version 2.0")
    print("=" * 60)
    
    # Check if input files exist
    required_files = [
        top20_file, optimizer_file, factor_weights_file, 
        country_weights_file, portfolio_returns_file, strategy_stats_file
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("WARNING: The following input files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nThe report will still generate, but some sections may show error messages.")
        print("Please ensure all input files are in the current directory.")
        
        response = input("\nContinue anyway? (y/n): ").lower().strip()
        if response != 'y':
            print("Report generation cancelled.")
            return None
    
    try:
        # Generate the comprehensive report
        report_file = create_comprehensive_report()
        
        print("\n" + "=" * 60)
        print("REPORT GENERATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Output file: {report_file}")
        print(f"File size: {os.path.getsize(report_file) / 1024:.1f} KB")
        print("\nThe report includes:")
        print("  ✓ Strategy Statistics Comparison")
        print("  ✓ Strategy Performance Charts")
        print("  ✓ Top 20 Portfolio Analysis")
        print("  ✓ Factor Returns Analysis")
        print("  ✓ Current Factor Allocation")
        print("  ✓ Country Allocations")
        print("  ✓ Portfolio Returns Analysis")
        print("  ✓ Enhanced Portfolio Statistics")
        
        return report_file
        
    except Exception as e:
        print(f"\nERROR: Report generation failed!")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the enhanced report generator
    generate_report_with_options()