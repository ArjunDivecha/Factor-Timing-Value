"""
Multi-window performance summary for Step Five.

Builds the same metrics as ``calculate_strategy_performance`` for:
- Full sample
- Last 12 months
- Last 3 years (36 months)
- Last 5 years (60 months)

Used for terminal logging (fixed-width table).

VERSION: 1.0
LAST UPDATED: 2026-05-28
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

METRIC_ORDER = (
    "Annualized Return (%)",
    "Annualized Volatility (%)",
    "Sharpe Ratio",
    "Maximum Drawdown (%)",
    "Average Monthly Turnover (%)",
    "Positive Months (%)",
    "Skewness",
    "Kurtosis",
)


def _stats_for_slice(
    portfolio_returns: pd.Series,
    turnover_slice: pd.Series,
) -> dict[str, float]:
    """Compute the eight Step Five metrics for one return slice and turnover slice."""
    pr = portfolio_returns.astype(float)
    if len(pr) == 0:
        return {m: float("nan") for m in METRIC_ORDER}

    ann_return = (1 + pr.mean()) ** 12 - 1
    ann_vol = pr.std() * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0.0

    cum_returns = (1 + pr).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    if len(turnover_slice) > 0:
        avg_turnover = float(turnover_slice.mean()) * 100.0
    else:
        avg_turnover = float("nan")

    return {
        "Annualized Return (%)": ann_return * 100.0,
        "Annualized Volatility (%)": ann_vol * 100.0,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown (%)": max_drawdown * 100.0,
        "Average Monthly Turnover (%)": avg_turnover,
        "Positive Months (%)": (pr > 0).mean() * 100.0,
        "Skewness": float(pr.skew()) if len(pr) > 2 else float("nan"),
        "Kurtosis": float(pr.kurtosis()) if len(pr) > 3 else float("nan"),
    }


def build_multiwindow_stats_table(
    portfolio_returns_series: pd.Series,
    monthly_turnover: pd.Series,
) -> pd.DataFrame:
    """
    Rows = metrics, columns = Full Period | 12 Months | 3 Years | 5 Years.
    """
    pr = portfolio_returns_series.sort_index()
    to = monthly_turnover.sort_index()

    windows: list[tuple[int | None, str]] = [
        (None, "Full Period"),
        (12, "12 Months"),
        (36, "3 Years"),
        (60, "5 Years"),
    ]

    cols: dict[str, dict[str, float]] = {}
    for n_months, col_name in windows:
        if n_months is None:
            pr_sub = pr
            to_sub = to
        else:
            k = min(n_months, len(pr))
            pr_sub = pr.iloc[-k:] if k > 0 else pr.iloc[:0]
            k_to = min(n_months, len(to))
            to_sub = to.iloc[-k_to:] if k_to > 0 else to.iloc[:0]
        cols[col_name] = _stats_for_slice(pr_sub, to_sub)

    data = {col: [cols[col][m] for m in METRIC_ORDER] for col in cols}
    return pd.DataFrame(data, index=METRIC_ORDER)


def log_step_five_multiwindow_table(
    portfolio_returns_series: pd.Series,
    monthly_turnover: pd.Series,
    log_fn: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """
    Print a fixed-width table via ``log_fn`` (default ``logging.info``).
    Returns the DataFrame for optional reuse.
    """
    log = log_fn or logging.info
    df = build_multiwindow_stats_table(portfolio_returns_series, monthly_turnover)

    metric_w = 34
    num_w = 14
    headers = ["Metric"] + list(df.columns)
    sep = " | "

    def fmt_cell(x: float) -> str:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return f"{'n/a':>{num_w}}"
        return f"{x:>{num_w}.2f}"

    header_line = f"{headers[0]:<{metric_w}}" + sep + sep.join(f"{h:>{num_w}}" for h in headers[1:])
    log(header_line)
    log("-" * len(header_line))
    for metric in METRIC_ORDER:
        row = metric.ljust(metric_w) + sep + sep.join(fmt_cell(df.loc[metric, c]) for c in df.columns)
        log(row)

    return df
