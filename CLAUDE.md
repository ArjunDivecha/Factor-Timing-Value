# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running Individual Steps
```bash
python "Step Zero Create P2P Scores.py"
python "Step One Create T2Master.py"
python "Step Two Create Normalized Tidy.py"
python "Step Two Point Five Create Benchmark Rets.py"
python "Step Three Top20 Portfolios.py"
python "Step Four Create Monthly Top20 Returns.py"
python "Step Four Point Five.py"
python "Step Five 60 Month Optimal Portfolios.py"
python "Step Five Tcost.py"
python "Step Six Create Country alphas from Factor alphas.py"
python "Step Six Point Five.py"
python "Step Seven Visualize Factor Weights.py"
python "Step Eight Write Country Weights.py"
python "Step Nine Calculate Portfolio Returns.py"
python "Step Ten Create Final Report.py"
python "Step Fourteen Target Optimization.py"
python "Step Fifteen Market Regime Analysis.py"
```

### Archived Steps
The following steps have been moved to Archive/ and are not part of the main flow:
- Step Six T2 Factor Timing Top3.py
- Step Eleven Compare Strategies.py
- Step Twelve MultiPeriod Forecast.py
- Step Thirteen Create Country alphas from Factor Alphas.py
- Step Run All.py

### Testing and Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebooks for interactive analysis
jupyter lab
```

## Architecture

### Pipeline Overview
The T2 Factor Timing system is a sequential pipeline for momentum-based country selection and portfolio optimization:

1. **Data Preparation (Steps 0-2.5)**: 
   - Ingests Bloomberg financial data and P2P scores
   - Creates normalized data with quality enhancements (outlier detection, winsorization)
   - Generates benchmark returns for performance comparison

2. **Portfolio Construction (Steps 3-5)**:
   - Identifies top 20 countries by momentum across multiple lookback periods
   - Calculates monthly returns for each momentum portfolio
   - Step Four Point Five calculates weighted-average trading cost per factor portfolio
   - Performs 60-month rolling window optimization to find optimal factor weights

3. **Country Alpha Generation (Steps 6-6.5)**:
   - Step Six creates country alphas using ALL factors (exposure × T60 trailing returns)
   - Step Six Point Five creates country alphas using ONLY Step Five selected factors — output: `T2_Country_Top_Alphas.xlsx`
   - Step Six Point Five formula: `Country_Alpha = Σ I(Exposure ≠ 0) × Weight × T60` — exposure acts as binary gate only (not proportional)

4. **Portfolio Implementation (Steps 7-9)**:
   - Visualizes factor weights over time
   - Translates factor weights to country-level allocations
   - Calculates final portfolio returns with rebalancing

5. **Reporting and Optimization (Steps 10, 14-15)**:
   - Creates comprehensive performance reports with visualizations
   - Optimizes country weights using CVXPY with turnover constraints
   - Analyzes strategy performance across market regime conditions

### Key Data Flow
```
Bloomberg Data → T2 Master → Normalized Data → Top20 Portfolios → Factor Returns → 
Factor Weights → Country Weights → Portfolio Returns → Performance Reports → 
Target Optimization → Market Regime Analysis
```

### Critical Implementation Details

**Data Quality Pipeline**: Step One implements sophisticated data cleaning with forward-filling, local outlier detection (±3σ rolling windows), and global winsorization. Market cap data receives special handling to preserve large-cap influence.

**Portfolio Optimization**: The strategy uses rolling window optimization to find optimal factor weights that maximize risk-adjusted returns. Factor weights are constrained by maximum allocation limits defined in Step Factor Categories.xlsx.

**Target Optimization**: Step Fourteen implements CVXPY-based optimization that balances three objectives: maximizing portfolio alpha, minimizing drift from rolled-forward weights, and minimizing transaction costs from turnover.

**Market Regime Analysis**: Step Fifteen analyzes strategy performance across different market conditions (Bull/Bear, High/Low Volatility, Economic Expansion/Contraction) and identifies which factors drive performance in each regime.

### File Dependencies
- Input data must be in Excel format with specific sheet names
- All intermediate outputs are Excel files for compatibility
- Visualizations saved as PDFs in outputs/visualizations/
- Date columns must be datetime-indexed throughout the pipeline

### Development and Debugging Tools

**Archived Analysis Files** (in Archive/ directory):
- `Step Six T2 Factor Timing Top3.py` - Factor timing with adaptive rotation
- `Step Six Grid Search.ipynb` - Jupyter notebook for parameter optimization
- `Step Run All.py` - Sequential pipeline execution script
- Various experimental and comparison analysis files

**Logging**: All scripts generate detailed logs to console and `T2_processing.log`

### Data Quality and Error Handling
- Missing country data is filled with the mean of available countries
- Forward-filling handles temporal gaps in data
- Outlier detection uses ±3σ rolling windows with winsorization
- Market cap data receives special handling to preserve large-cap influence
- All processing steps include comprehensive error handling and validation

## Learned User Preferences

- Best Cash Flow is Price-to-Cash-Flow (not Cash-Flow-to-Price). Higher P/CF = expensive = BAD; lower P/CF = cheap = GOOD. The `invert_norm` treatment in Step Two is correct for this variable.
- When tracing unexpected country alpha scores, always check both the normalized data (`_CS` z-scores) and the raw Bloomberg source data to identify data quality issues upstream.

## Learned Workspace Facts

- NASDAQ Best Cash Flow (CCMP Index `BEST_CPS_RATIO`) has bimodal Bloomberg data errors — intermittently returns near-zero values (0.03–0.4) for months at a time (observed in 2006-2008, 2011-2016, 2021-2022, 2025-03 to 2025-08, 2026-04+).
- Step One's outlier detection has a known gap with regime breaks: (a) bimodal distributions inflate MAD, making global winsorization bounds useless; (b) sustained bad data contaminates local rolling windows. A month-over-month percentage change detector (>80% threshold) has been proposed but not yet implemented.
- Step Six Point Five.py was added to create country alphas using only Step Five's selected factors. It reads `T2_rolling_window_weights.xlsx`, `T60.xlsx`, and `T2_Top_20_Exposure.csv`, outputting `T2_Country_Top_Alphas.xlsx`.
- Step Four Point Five.py calculates weighted-average trading cost per factor portfolio over time, outputting `T2_Tcost.xlsx`.
- The `_CS` suffix on factor names means cross-sectional z-score (computed in Step Two normalization).
- T60.xlsx contains trailing 60-month average returns per factor (produced by Step Four), not optimized weights.
- T2_rolling_window_weights.xlsx contains optimized portfolio weights per factor (produced by Step Five) — these are independent from T60 values.