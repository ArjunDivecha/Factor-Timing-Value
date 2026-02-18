#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: Step Three Top20 Portfolios Fast.py  (Soft 15–25 % Linear Band)
=============================================================================

INPUT FILES:
- Normalized_T2_MasterCSV.csv  (long format: date, country, variable, value)
- Portfolio_Data.xlsx (sheet "Benchmarks": equal-weight benchmark returns)

OUTPUT FILES:
- T2 Top20.xlsx            # performance table sorted by IR
- T2 Top20.pdf             # cumulative excess-return charts
- T2_Top_20_Exposure.csv   # monthly country weights (0-1, not just binary)

VERSION: 3.2 – FAST optimized version (identical output to original)
LAST UPDATED: 2025-12-03
AUTHOR: Claude Code (optimized for speed)

OPTIMIZATIONS:
- Pre-indexed data lookups using dictionaries instead of repeated DataFrame filtering
- Fast returns lookup using dictionary
- Vectorized weight calculations using numpy after pandas sorting
- Removed seaborn dependency (uses pure matplotlib)
=============================================================================
"""

import os
import warnings
from typing import Dict, Tuple

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
warnings.filterwarnings("ignore")
plt.style.use("ggplot")

SOFT_BAND_TOP = 0.15     # 15 % ⇒ full weight
SOFT_BAND_CUTOFF = 0.25  # 25 % ⇒ zero weight


# ------------------------------------------------------------------
# Core Analysis - Optimized with pre-indexing
# ------------------------------------------------------------------
def analyze_portfolios(
    data: pd.DataFrame,
    features: list,
    benchmark_returns: pd.Series,
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Builds factor portfolios with a soft 15–25 % linear taper.

    Returns
    -------
    monthly_returns : dict[str, Series]
    monthly_holdings : dict[str, DataFrame]   (country weights per date)
    results_df : DataFrame                   (performance summary)
    """
    results = []
    monthly_returns, monthly_holdings = {}, {}

    # Unique axes
    dates = sorted(data["date"].unique())
    countries = sorted(data["country"].unique())
    country_to_idx = {c: i for i, c in enumerate(countries)}
    n_countries = len(countries)
    n_dates = len(dates)

    # Pre-process returns data ONCE and create lookup dictionary
    print("Pre-indexing data...")
    returns_data = (
        data[data["variable"] == "1MRet"]
        .assign(date=lambda df: pd.to_datetime(df["date"]))
        .sort_values("date")
    )
    # Create a dictionary for O(1) lookups: {(date, country): return_value}
    returns_lookup = {
        (row.date, row.country): row.value 
        for row in returns_data.itertuples(index=False)
    }
    
    # Pre-group all feature data by variable for faster access
    # Store as pre-sorted numpy arrays for each date
    feature_data_cache = {}
    for feature in features:
        feat_df = (
            data[data["variable"] == feature]
            .assign(date=lambda df: pd.to_datetime(df["date"]))
            .sort_values("date")
        )
        # Only keep dates where we have at least one valid value
        valid_dates_mask = feat_df.groupby('date')['value'].apply(lambda x: x.notna().any())
        valid_dates = valid_dates_mask[valid_dates_mask].index.tolist()
        feat_df = feat_df[feat_df['date'].isin(valid_dates)]
        feat_df = feat_df.sort_values('date')
        
        # Create date-indexed dictionary with pre-sorted data as numpy arrays
        feature_data_cache[feature] = {}
        for date, group in feat_df.groupby("date"):
            # Drop NaN values and sort by value descending (using pandas for exact match)
            factor_only = group[["country", "value"]].dropna(subset=["value"])
            if not factor_only.empty:
                factor_only = factor_only.sort_values("value", ascending=False).reset_index(drop=True)
                # Store as numpy arrays
                feature_data_cache[feature][date] = (
                    factor_only["country"].values,
                    factor_only["value"].values
                )
    
    print(f"Processing {len(features)} features...")
    for idx, feature in enumerate(features):
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(features)} features...")
        
        feat_by_date = feature_data_cache[feature]

        # Storage using numpy arrays
        port_rets_arr = np.full(n_dates, np.nan)
        holdings_arr = np.zeros((n_dates, n_countries))

        # Process each date
        for date_idx, date in enumerate(dates):
            # Fast lookup using pre-indexed data
            if date not in feat_by_date:
                continue
            
            countries_arr, values_arr = feat_by_date[date]
            n = len(countries_arr)
            
            if n == 0:
                continue

            # Compute rank percentile (data is already sorted descending by value)
            rank_pct = (np.arange(n) + 1) / n

            # Compute weights using vectorized numpy operations
            weights = np.zeros(n)
            
            # Full weight for top band
            top_mask = rank_pct < SOFT_BAND_TOP
            weights[top_mask] = 1.0
            
            # Linearly decreasing weight inside the grey band
            in_band = (rank_pct >= SOFT_BAND_TOP) & (rank_pct <= SOFT_BAND_CUTOFF)
            weights[in_band] = 1.0 - (rank_pct[in_band] - SOFT_BAND_TOP) / (SOFT_BAND_CUTOFF - SOFT_BAND_TOP)
            
            # Filter to non-zero weights
            nonzero_mask = weights > 0
            if not nonzero_mask.any():
                continue
                
            weights_filtered = weights[nonzero_mask]
            countries_filtered = countries_arr[nonzero_mask]
            
            # Normalize weights to sum to 1
            weights_filtered = weights_filtered / weights_filtered.sum()

            # Store holdings
            for c, w in zip(countries_filtered, weights_filtered):
                holdings_arr[date_idx, country_to_idx[c]] = w

            # Calculate portfolio return using fast lookup
            total_ret = 0.0
            for c, w in zip(countries_filtered, weights_filtered):
                ret_val = returns_lookup.get((date, c))
                if ret_val is not None and not np.isnan(ret_val):
                    total_ret += w * ret_val
            
            port_rets_arr[date_idx] = total_ret if total_ret != 0.0 else np.nan

        # Convert to pandas objects
        port_rets = pd.Series(port_rets_arr, index=pd.to_datetime(dates))
        port_rets = port_rets.reindex(benchmark_returns.index)
        
        holdings = pd.DataFrame(
            holdings_arr, 
            index=pd.Index(dates, name="date"), 
            columns=pd.Index(countries, name="country")
        )

        monthly_returns[feature] = port_rets
        monthly_holdings[feature] = holdings

        # Metrics & turnover
        metrics = calculate_performance_metrics(port_rets, benchmark_returns)
        turnover = calculate_turnover(holdings)

        results.append(
            {
                "Feature": feature,
                **metrics,
                "Average Turnover (%)": turnover,
            }
        )

    results_df = (
        pd.DataFrame(results)[
            [
                "Feature",
                "Avg Excess Return (%)",
                "Volatility (%)",
                "Information Ratio",
                "Maximum Drawdown (%)",
                "Hit Ratio (%)",
                "Skewness",
                "Kurtosis",
                "Beta",
                "Tracking Error (%)",
                "Calmar Ratio",
                "Average Turnover (%)",
            ]
        ]
        if results
        else pd.DataFrame()
    )
    return monthly_returns, monthly_holdings, results_df


# ------------------------------------------------------------------
# Performance Metrics (unchanged)
# ------------------------------------------------------------------
def calculate_performance_metrics(returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
    returns, benchmark_returns = map(
        lambda s: pd.to_numeric(s, errors="coerce"), (returns, benchmark_returns)
    )
    valid = returns.notna() & benchmark_returns.notna()
    returns, benchmark_returns = returns[valid], benchmark_returns[valid]

    if returns.empty:
        return dict.fromkeys(
            [
                "Avg Excess Return (%)",
                "Volatility (%)",
                "Information Ratio",
                "Maximum Drawdown (%)",
                "Hit Ratio (%)",
                "Skewness",
                "Kurtosis",
                "Beta",
                "Tracking Error (%)",
                "Calmar Ratio",
            ],
            0,
        )

    excess = returns - benchmark_returns
    avg_excess = excess.mean() * 12 * 100
    vol = returns.std() * np.sqrt(12) * 100
    te = excess.std() * np.sqrt(12) * 100
    ir = avg_excess / te if te else 0
    cum = (1 + excess).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax() * 100).min()
    hit = (excess > 0).mean() * 100
    skew, kurt = excess.skew(), excess.kurtosis()
    beta = returns.cov(benchmark_returns) / benchmark_returns.var()
    calmar = -avg_excess / dd if dd else 0

    return {
        "Avg Excess Return (%)": round(avg_excess, 2),
        "Volatility (%)": round(vol, 2),
        "Information Ratio": round(ir, 2),
        "Maximum Drawdown (%)": round(dd, 2),
        "Hit Ratio (%)": round(hit, 2),
        "Skewness": round(skew, 2),
        "Kurtosis": round(kurt, 2),
        "Beta": round(beta, 2),
        "Tracking Error (%)": round(te, 2),
        "Calmar Ratio": round(calmar, 2),
    }


def calculate_turnover(holdings_df: pd.DataFrame) -> float:
    if len(holdings_df) <= 1:
        return 0
    diffs = holdings_df.diff().abs().sum(axis=1) / 2  # buys + sells
    return round(diffs.iloc[1:].mean() * 100, 2)  # exclude first NaN row


# ------------------------------------------------------------------
# Visualisation (pure matplotlib, no seaborn)
# ------------------------------------------------------------------
def create_performance_charts(
    returns_dict: Dict[str, pd.Series], benchmark_returns: pd.Series, output_path: str
) -> None:
    n_features, n_cols = len(returns_dict), 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(15, n_rows * 4))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)

    # Use a nice color palette without seaborn
    colors = plt.cm.tab10.colors

    for i, (feature, rets) in enumerate(returns_dict.items()):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(gs[row, col])

        excess = rets - benchmark_returns
        first_valid = excess.first_valid_index()
        if first_valid is not None:
            cum = (excess[first_valid:].cumsum()) * 100
            ax.plot(cum.index, cum, label="Excess Return (bps)", linewidth=1.5, color=colors[0])

        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_title(feature, fontsize=12, weight="bold")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------
def run_portfolio_analysis(data_path: str, benchmark_path: str, output_dir: str) -> None:
    print("\nStarting portfolio analysis (FAST version)…")
    os.makedirs(output_dir, exist_ok=True)

    skip_variables = [
        "1MRet", "3MRet", "6MRet", "9MRet", "12MRet",
        "120MA_CS", "129MA_TS", "Agriculture_TS", "Agriculture_CS",
        "Copper_TS", "Copper_CS", "Gold_CS", "Gold_TS",
        "Oil_CS", "Oil_TS", "MCAP Adj_CS", "MCAP Adj_TS",
        "MCAP_CS", "MCAP_TS", "PX_LAST_CS", "PX_LAST_TS",
        "Tot Return Index_CS", "Tot Return Index_TS",
        "Currency_CS", "Currency_TS", "BEST EPS_CS", "BEST EPS_TS",
        "Trailing EPS_CS", "Trailing EPS_TS",
    ]

    try:
        # Load data
        print("Loading data...")
        data = pd.read_csv(data_path)
        data["date"] = pd.to_datetime(data["date"]).dt.to_period("M").dt.to_timestamp()

        bench = pd.read_excel(benchmark_path, sheet_name="Benchmarks")
        bench = bench.set_index("Unnamed: 0")
        bench.index = pd.to_datetime(bench.index).to_period("M").to_timestamp()
        benchmark_returns = bench["equal_weight"]

        features = sorted(v for v in data["variable"].unique() if v not in skip_variables)
        print(f"Analyzing {len(features)} features with soft 15–25 % band…")

        monthly_returns, monthly_holdings, results = analyze_portfolios(
            data, features, benchmark_returns
        )

        # Save Excel
        print("\nSaving results...")
        results = results.sort_values("Information Ratio", ascending=False)
        excel_path = os.path.join(output_dir, "T2 Top20.xlsx")
        results.to_excel(excel_path, index=False, float_format="%.2f")

        # Charts
        print("Creating charts...")
        pdf_path = os.path.join(output_dir, "T2 Top20.pdf")
        create_performance_charts(monthly_returns, benchmark_returns, pdf_path)

        # Exposure CSV - keep original logic for exact match
        print("Creating exposure matrix...")
        exposure_rows = []
        all_dates = sorted({d for df in monthly_holdings.values() for d in df.index})
        all_countries = sorted({c for df in monthly_holdings.values() for c in df.columns})

        for date in all_dates:
            for country in all_countries:
                row = [date.strftime("%Y-%m-%d"), country]
                for factor in features:
                    w = monthly_holdings[factor].get(country, pd.Series()).get(date, 0.0)
                    row.append(round(float(w), 6))
                exposure_rows.append(row)

        exposure_df = pd.DataFrame(exposure_rows, columns=["Date", "Country"] + features)
        exposure_path = os.path.join(output_dir, "T2_Top_20_Exposure.csv")
        exposure_df.to_csv(exposure_path, index=False)

        # ----------------------------------------------------------------
        print("\nAnalysis complete!")
        print(f"Results  → {excel_path}")
        print(f"Charts   → {pdf_path}")
        print(f"Exposure → {exposure_path}")

    except Exception as err:
        import traceback
        print(f"\nERROR: {err}")
        traceback.print_exc()


if __name__ == "__main__":
    DATA_PATH = "Normalized_T2_MasterCSV.csv"
    BENCHMARK_PATH = "Portfolio_Data.xlsx"
    OUTPUT_DIR = "."  # current directory
    run_portfolio_analysis(DATA_PATH, BENCHMARK_PATH, OUTPUT_DIR)
