"""
=============================================================================
SCRIPT NAME: Step Nine Compare Alpha Mcap Strategies.py
=============================================================================

PURPOSE:
Compare three country-weighted portfolio strategies using the same return
methodology as Step Nine:

  1. Current pipeline — Step Eight fuzzy country weights
     (T2_Final_Country_Weights.xlsx)
  2. Alpha × MCAP — Step Six country alphas (all factors)
     (T2_Country_Alphas.xlsx)
  3. Alpha × MCAP — Step Six Point Five country alphas (Step Five factors only)
     (T2_Country_Top_Alphas.xlsx)

Weight formula for strategies 2 and 3 (each month, long-only, cap-weighted):
    raw_weight[country] = max(alpha[country], 0) × MCAP[country]
    portfolio_weight = raw_weight / sum(raw_weight)   (if sum > 0)

Set USE_SQRT_MCAP = True in the script to use sqrt(MCAP) instead of full MCAP.

INPUT FILES:
- T2_Final_Country_Weights.xlsx — sheet "All Periods" (Step Eight output)
- T2_Country_Alphas.xlsx — sheet "Country_Scores" (Step Six output)
- T2_Country_Top_Alphas.xlsx — sheet "Country_Scores" (Step Six Point Five output)
- T2 Master.xlsx — sheet "MCAP" (market capitalization by country)
- Portfolio_Data.xlsx — sheets "Returns", "Benchmarks"

OUTPUT FILES:
- T2_Strategy_Comparison.xlsx — monthly returns, cumulative returns, statistics
- T2_Strategy_Comparison.pdf — cumulative return charts (light background)

VERSION: 1.1
LAST UPDATED: 2026-05-28
=============================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
WORKDIR = Path(__file__).resolve().parent

WEIGHTS_STEP8 = WORKDIR / "T2_Final_Country_Weights.xlsx"
ALPHAS_STEP6 = WORKDIR / "T2_Country_Alphas.xlsx"
ALPHAS_STEP65 = WORKDIR / "T2_Country_Top_Alphas.xlsx"
MCAP_FILE = WORKDIR / "T2 Master.xlsx"
RETURNS_FILE = WORKDIR / "Portfolio_Data.xlsx"

OUTPUT_XLSX = WORKDIR / "T2_Strategy_Comparison.xlsx"
OUTPUT_PDF = WORKDIR / "T2_Strategy_Comparison.pdf"

# False = full cap weighting (alpha × MCAP); True = alpha × sqrt(MCAP)
USE_SQRT_MCAP = False

_CAP_SUFFIX = "√MCAP" if USE_SQRT_MCAP else "MCAP"
STRATEGY_LABELS = {
    "step8_fuzzy": "Step 8 Fuzzy Weights (current Step 9)",
    "alpha6_mcap": f"Step 6 Alpha × {_CAP_SUFFIX}",
    "alpha65_mcap": f"Step 6.5 Top Alpha × {_CAP_SUFFIX}",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def month_end_index(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize index to month-start timestamps (matches Step Nine)."""
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out.index = out.index.to_period("M").to_timestamp()
    return out


def load_returns_and_benchmark() -> Tuple[pd.DataFrame, pd.Series]:
    returns_df = pd.read_excel(RETURNS_FILE, sheet_name="Returns", index_col=0)
    benchmark_df = pd.read_excel(RETURNS_FILE, sheet_name="Benchmarks", index_col=0)
    returns_df = month_end_index(returns_df)
    benchmark_df = month_end_index(benchmark_df)
    if "equal_weight" not in benchmark_df.columns:
        raise KeyError("Benchmarks sheet must include 'equal_weight' column")
    return returns_df, benchmark_df["equal_weight"]


def load_mcap() -> pd.DataFrame:
    mcap = pd.read_excel(MCAP_FILE, sheet_name="MCAP", index_col=0)
    mcap = month_end_index(mcap)
    return mcap.apply(pd.to_numeric, errors="coerce")


def load_country_scores(path: Path, sheet: str = "Country_Scores") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, index_col=0)
    return month_end_index(df.apply(pd.to_numeric, errors="coerce"))


def build_alpha_mcap_weights(
    alphas: pd.DataFrame,
    mcap: pd.DataFrame,
    use_sqrt_mcap: bool = USE_SQRT_MCAP,
) -> pd.DataFrame:
    """
    Build monthly weights: max(alpha, 0) × MCAP (or sqrt(MCAP)), normalized to sum to 1.
    """
    common_dates = alphas.index.intersection(mcap.index)
    common_countries = alphas.columns.intersection(mcap.columns)

    if len(common_dates) == 0 or len(common_countries) == 0:
        raise ValueError("No overlapping dates or countries between alphas and MCAP")

    alphas = alphas.loc[common_dates, common_countries]
    mcap = mcap.loc[common_dates, common_countries]

    weights = pd.DataFrame(0.0, index=common_dates, columns=common_countries)
    cap_label = "sqrt(MCAP)" if use_sqrt_mcap else "MCAP"

    for date in common_dates:
        alpha_row = alphas.loc[date].fillna(0.0)
        mcap_row = mcap.loc[date].clip(lower=0.0)

        positive_alpha = alpha_row.clip(lower=0.0)
        cap_factor = np.sqrt(mcap_row) if use_sqrt_mcap else mcap_row

        raw = positive_alpha * cap_factor
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        total = raw.sum()
        if total > 0:
            weights.loc[date] = raw / total
        else:
            logger.warning(
                "Date %s: all raw weights zero (no positive alpha × %s); "
                "leaving weights at 0 for that month.",
                date.strftime("%Y-%m"),
                cap_label,
            )

    return weights


def calculate_portfolio_returns(
    weights_df: pd.DataFrame,
    returns_df: pd.DataFrame,
) -> Tuple[pd.Series, pd.DatetimeIndex]:
    """
    Same logic as Step Nine: weights at date t applied to returns at date t.
    """
    weights_df = month_end_index(weights_df)
    returns_df = month_end_index(returns_df)

    common_dates = sorted(weights_df.index.intersection(returns_df.index))
    if len(common_dates) > 0:
        common_dates = common_dates[:-1]  # exclude last month (no forward return)

    portfolio_returns = []
    valid_dates = []

    for date in common_dates:
        weights = weights_df.loc[date]
        month_returns = returns_df.loc[date]

        common_countries = weights.index.intersection(month_returns.index)
        weighted_return = 0.0
        total_weight = 0.0

        for country in common_countries:
            w = weights[country]
            r = month_returns[country]
            if pd.notna(r) and w > 0:
                weighted_return += w * r
                total_weight += w

        if total_weight > 0:
            portfolio_returns.append(weighted_return / total_weight)
            valid_dates.append(date)
        else:
            portfolio_returns.append(np.nan)
            valid_dates.append(date)

    return pd.Series(portfolio_returns, index=pd.DatetimeIndex(valid_dates)), pd.DatetimeIndex(
        valid_dates
    )


def calculate_turnover(weights_df: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """One-way turnover: sum |w(t) - w(t-1)| / 2 (matches Step Nine convention)."""
    turnover = []
    for i, date in enumerate(dates):
        if i == 0:
            turnover.append(np.nan)
            continue
        prev_date = dates[i - 1]
        if prev_date not in weights_df.index or date not in weights_df.index:
            turnover.append(np.nan)
            continue
        current = weights_df.loc[date].fillna(0.0)
        previous = weights_df.loc[prev_date].fillna(0.0)
        all_countries = current.index.union(previous.index)
        t = sum(abs(current.get(c, 0.0) - previous.get(c, 0.0)) for c in all_countries)
        turnover.append(t / 2.0)
    return pd.Series(turnover, index=dates)


def calculate_stats(returns: pd.Series, turnover: Optional[pd.Series] = None) -> pd.Series:
    stats: Dict[str, float] = {}
    stats["Annual Return"] = returns.mean() * 12 * 100
    stats["Annual Vol"] = returns.std() * np.sqrt(12) * 100
    stats["Sharpe Ratio"] = (returns.mean() * 12) / (returns.std() * np.sqrt(12))
    cum = (1 + returns).cumprod()
    stats["Max Drawdown"] = (cum / cum.cummax() - 1).min() * 100
    stats["Hit Rate"] = (returns > 0).mean() * 100
    stats["Skewness"] = returns.skew()
    stats["Kurtosis"] = returns.kurtosis()
    if turnover is not None:
        stats["Avg Monthly Turnover"] = turnover.mean() * 100
        stats["Annual Turnover"] = turnover.mean() * 12 * 100
    return pd.Series(stats)


def plot_comparison(
    cumulative: pd.DataFrame,
    output_path: Path,
) -> None:
    """Cumulative returns for three strategies vs equal weight (light mode)."""
    plt.style.use("default")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), facecolor="white")

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    ax1 = axes[0]
    for i, col in enumerate(cumulative.columns):
        if col == "Equal Weight":
            ax1.plot(
                cumulative.index,
                cumulative[col],
                label=col,
                color=colors[-1],
                alpha=0.75,
                linewidth=1.5,
                linestyle="--",
            )
        else:
            ax1.plot(
                cumulative.index,
                cumulative[col],
                label=col,
                color=colors[i % 3],
                linewidth=2,
            )
    ax1.set_title("Cumulative Returns — Three Strategies vs Equal Weight", fontsize=13)
    ax1.set_ylabel("Growth of $1")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("white")

    # Active return vs equal weight for each strategy
    ax2 = axes[1]
    if "Equal Weight" in cumulative.columns:
        for i, col in enumerate(cumulative.columns):
            if col == "Equal Weight":
                continue
            active = cumulative[col] / cumulative["Equal Weight"]
            ax2.plot(
                active.index,
                active,
                label=f"{col} / EW",
                color=colors[i % 3],
                linewidth=2,
            )
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.4)
    ax2.set_title("Cumulative Active Return (Strategy ÷ Equal Weight)", fontsize=13)
    ax2.set_ylabel("Relative wealth")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("white")

    plt.tight_layout()
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    logger.info("Loading returns and benchmark ...")
    returns_df, equal_weight = load_returns_and_benchmark()

    logger.info("Loading Step Eight weights ...")
    weights_step8 = pd.read_excel(WEIGHTS_STEP8, sheet_name="All Periods", index_col=0)
    weights_step8 = month_end_index(weights_step8.apply(pd.to_numeric, errors="coerce"))

    logger.info("Loading alphas and MCAP ...")
    alphas_step6 = load_country_scores(ALPHAS_STEP6)
    alphas_step65 = load_country_scores(ALPHAS_STEP65)
    mcap = load_mcap()

    cap_desc = "sqrt(MCAP)" if USE_SQRT_MCAP else "MCAP"
    logger.info("Building alpha × %s weight matrices ...", cap_desc)
    weights_alpha6 = build_alpha_mcap_weights(alphas_step6, mcap)
    weights_alpha65 = build_alpha_mcap_weights(alphas_step65, mcap)

    weight_sets = {
        "step8_fuzzy": weights_step8,
        "alpha6_mcap": weights_alpha6,
        "alpha65_mcap": weights_alpha65,
    }

    monthly: Dict[str, pd.Series] = {}
    turnovers: Dict[str, pd.Series] = {}

    for key, weights in weight_sets.items():
        label = STRATEGY_LABELS[key]
        logger.info("Calculating returns for: %s", label)
        port_rets, dates = calculate_portfolio_returns(weights, returns_df)
        monthly[label] = port_rets
        turnovers[label] = calculate_turnover(weights, dates)

    # Align all series on common dates
    results = pd.DataFrame(monthly)
    results["Equal Weight"] = equal_weight.reindex(results.index)
    results = results.dropna(how="all")

    # Statistics table
    stats_rows = {}
    for col in [c for c in results.columns if c != "Equal Weight"]:
        stats_rows[col] = calculate_stats(results[col].dropna(), turnovers.get(col))
    stats_rows["Equal Weight"] = calculate_stats(results["Equal Weight"].dropna())
    stats_df = pd.DataFrame(stats_rows)

    cumulative = (1 + results).cumprod()

    logger.info("Writing %s", OUTPUT_XLSX)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
        monthly_out = results.reset_index().rename(columns={"index": "Date"})
        monthly_out.to_excel(writer, sheet_name="Monthly Returns", index=False)
        cumulative.reset_index().rename(columns={"index": "Date"}).to_excel(
            writer, sheet_name="Cumulative Returns", index=False
        )
        stats_df.to_excel(writer, sheet_name="Statistics")

        for key, weights in weight_sets.items():
            short = {"step8_fuzzy": "Step8", "alpha6_mcap": "Alpha6", "alpha65_mcap": "Alpha65"}[key]
            w_aligned = weights.reindex(results.index).fillna(0.0)
            w_aligned.reset_index().rename(columns={"index": "Date"}).to_excel(
                writer, sheet_name=f"Weights {short}", index=False
            )

        workbook = writer.book
        date_fmt = workbook.add_format({"num_format": "dd-mmm-yyyy"})
        for sheet_name in ("Monthly Returns", "Cumulative Returns"):
            ws = writer.sheets[sheet_name]
            for row in range(len(monthly_out if sheet_name == "Monthly Returns" else cumulative)):
                val = (
                    monthly_out["Date"].iloc[row]
                    if sheet_name == "Monthly Returns"
                    else cumulative.index[row]
                )
                if pd.notna(val):
                    ws.write(row + 1, 0, val, date_fmt)
            ws.set_column(0, 0, 15)

    logger.info("Writing %s", OUTPUT_PDF)
    plot_comparison(cumulative, OUTPUT_PDF)

    # Console summary
    print("\n" + "=" * 72)
    print("STRATEGY COMPARISON SUMMARY (annualized, % where noted)")
    print("=" * 72)
    display_cols = ["Annual Return", "Annual Vol", "Sharpe Ratio", "Max Drawdown", "Hit Rate"]
    print(stats_df.loc[display_cols].round(2).to_string())
    print("=" * 72)
    print(f"\nResults saved to:\n  {OUTPUT_XLSX}\n  {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
