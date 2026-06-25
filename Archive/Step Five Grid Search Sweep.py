"""
=============================================================================
SCRIPT NAME: Step Five Grid Search Sweep.py
=============================================================================

INPUT FILES:
- T2_Optimizer.xlsx: Monthly factor returns from Step Four
- Step Factor Categories.xlsx: Factor max weight constraints (Max=0 excludes 46 of 82 factors)
- T60.xlsx: Factor alpha forecasts (Date + factor columns)

OUTPUT FILES:
- Step-Five-Grid-Search-Results.xlsx: Full 2D results with sheets:
  * Annualized Return (%), Sharpe Ratio, Volatility (%), Max Drawdown (%),
    Turnover (%), Positive Months (%), Average HHI
  * Each sheet: lookback rows × lambda columns

DESCRIPTION:
2D grid search over lambda (risk aversion) and lookback period (rolling window
size). HHI penalty is fixed at 0.0 (no concentration penalty). Uses CVXPY
with OSQP solver (50-100x faster than scipy SLSQP) with a convex quadratic
formulation.

Optimization objective:
  Maximize: w'μ - λ * w'Σw - γ * ||w||²
  Subject to: sum(w) = 1, 0 ≤ w ≤ max_weights (from Step Factor Categories.xlsx)

Expected returns μ = 8 × window arithmetic mean (matching original scaling).
Covariance Σ = np.cov(..., ddof=0) × 12 (annualized).

Sweep dimensions:
- Lambda:         [5.0, 50.0, 500.0]
- HHI Penalty:    fixed at 0.0
- Lookback Period: [36, 60, 72, 90, 120] months

VERSION: 3.1 — CVXPY/OSQP + factor max-weight constraints
=============================================================================
"""

import pandas as pd
import numpy as np
import cvxpy as cp
import warnings
import time

warnings.filterwarnings("ignore")


# =============================================================================
# Fast CVXPY Portfolio Optimizer (from Step Five FAST.py)
# =============================================================================

class FastPortfolioOptimizer:
    """
    High-performance portfolio optimizer using CVXPY with warm-start.

    Converts utility function to quadratic form:
        Maximize: w'μ - λ*w'Σw - γ*||w||²
        Subject to: sum(w) = 1, 0 ≤ w ≤ max_weights
    """

    def __init__(
        self,
        n_assets: int,
        factor_names: list,
        lambda_param: float = 1.0,
        hhi_penalty: float = 0.0,
        max_weights: dict = None,
    ):
        self.n_assets = n_assets
        self.factor_names = factor_names
        self.lambda_param = lambda_param
        self.hhi_penalty = hhi_penalty

        # Pre-process max weights into array for vectorized operations
        if max_weights is None:
            self.max_weights_array = np.ones(n_assets)
        else:
            self.max_weights_array = np.array(
                [max_weights.get(name, 1.0) for name in factor_names]
            )

        # Pre-allocate CVXPY variables and constraints (reused)
        self.weights_var = cp.Variable(n_assets)
        self.constraints = [
            self.weights_var >= 0,
            self.weights_var <= self.max_weights_array,
            cp.sum(self.weights_var) == 1,
        ]
        self.prev_weights = None

    def optimize_weights(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
        P = np.asarray(covariance_matrix, dtype=np.float64)
        P = (P + P.T) * 0.5  # ensure symmetric
        mu = np.asarray(expected_returns, dtype=np.float64).ravel()

        portfolio_return = self.weights_var.T @ mu
        risk_penalty = self.lambda_param * cp.quad_form(self.weights_var, P)
        concentration_penalty = self.hhi_penalty * cp.sum_squares(self.weights_var)

        objective = cp.Maximize(portfolio_return - risk_penalty - concentration_penalty)
        problem = cp.Problem(objective, self.constraints)

        # Warm-start from previous solution
        if self.prev_weights is not None:
            self.weights_var.value = self.prev_weights

        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if problem.status == cp.OPTIMAL:
                self.prev_weights = self.weights_var.value.copy()
                return self.weights_var.value
            else:
                # Fall back to equal weights
                return np.ones(self.n_assets) / self.n_assets
        except Exception:
            return np.ones(self.n_assets) / self.n_assets


# =============================================================================
# Core rolling optimization (CVXPY version)
# =============================================================================

def run_rolling_optimization(
    returns: pd.DataFrame,
    lambda_param: float = 1.0,
    hhi_penalty: float = 0.0,
    lookback_period: int = 60,
    max_weights: dict = None,
):
    """
    Run expand-then-roll optimization using CVXPY/OSQP.

    Expected returns μ = 8 × window arithmetic mean.
    Covariance Σ = np.cov(..., ddof=0) × 12 (annualized).
    """
    factor_names = list(returns.columns)
    optimizer = FastPortfolioOptimizer(
        n_assets=len(factor_names),
        factor_names=factor_names,
        lambda_param=lambda_param,
        hhi_penalty=hhi_penalty,
        max_weights=max_weights,
    )

    dates = returns.index
    next_month_date = dates[-1] + pd.DateOffset(months=1)

    rolling_weights = pd.DataFrame(
        index=list(dates[1:]) + [next_month_date], columns=returns.columns
    )

    for i, current_date in enumerate(dates[1:], 1):
        # Determine window bounds
        if i <= lookback_period:
            start_idx, end_idx = 0, max(0, i - 1)
        else:
            start_idx, end_idx = max(0, i - lookback_period), max(0, i - 1)

        if end_idx >= start_idx:
            window_data = returns.loc[dates[start_idx : end_idx + 1]]

            # μ = 8 × arithmetic mean   (matching original scaling)
            mu = 8.0 * window_data.mean(axis=0).values

            # Σ = annualized covariance (ddof=0 matches original np.std behaviour)
            cov = np.cov(window_data.values.T, ddof=0) * 12

            # Ensure positive semi-definite
            try:
                evals, evecs = np.linalg.eigh(cov)
                evals = np.maximum(evals, 1e-6)
                cov = evecs @ np.diag(evals) @ evecs.T
            except np.linalg.LinAlgError:
                cov = np.eye(len(returns.columns)) * np.diag(cov).mean()

            rolling_weights.loc[current_date] = optimizer.optimize_weights(mu, cov)

    # Extra month
    start_idx = max(0, len(dates) - lookback_period)
    extra_data = returns.loc[dates[start_idx:]]
    mu_extra = 8.0 * extra_data.mean(axis=0).values
    cov_extra = np.cov(extra_data.values.T, ddof=0) * 12
    try:
        evals, evecs = np.linalg.eigh(cov_extra)
        evals = np.maximum(evals, 1e-6)
        cov_extra = evecs @ np.diag(evals) @ evecs.T
    except np.linalg.LinAlgError:
        cov_extra = np.eye(len(returns.columns)) * np.diag(cov_extra).mean()
    rolling_weights.loc[next_month_date] = optimizer.optimize_weights(mu_extra, cov_extra)

    # Strategy returns
    rolling_returns = pd.Series(index=dates[2:], dtype=float)
    for date in dates[2:]:
        prev_date = dates[dates < date][-1]
        rolling_returns[date] = np.sum(
            rolling_weights.loc[prev_date] * returns.loc[date]
        )

    return rolling_returns, rolling_weights.loc[dates[1:]]


# =============================================================================
# Metrics helper
# =============================================================================

def compute_metrics(portfolio_returns: pd.Series, weights_df: pd.DataFrame) -> dict:
    ann_ret = (1 + portfolio_returns.mean()) ** 12 - 1
    ann_vol = portfolio_returns.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0

    cum = (1 + portfolio_returns).cumprod()
    rolling_max = cum.expanding().max()
    dd = ((cum - rolling_max) / rolling_max).min()

    weight_changes = weights_df.diff().abs().sum(axis=1)
    avg_turnover = weight_changes.mean()

    pos_months = (portfolio_returns > 0).mean()

    hhi_series = (weights_df ** 2).sum(axis=1)
    avg_hhi = hhi_series.mean()
    n = len(weights_df.columns)
    norm_hhi = (avg_hhi - 1 / n) / (1 - 1 / n)

    return {
        "Annualized Return (%)": ann_ret * 100,
        "Sharpe Ratio": sharpe,
        "Volatility (%)": ann_vol * 100,
        "Max Drawdown (%)": dd * 100,
        "Turnover (%)": avg_turnover * 100,
        "Positive Months (%)": pos_months * 100,
        "Average HHI": avg_hhi,
        "Normalized HHI": norm_hhi,
    }


# =============================================================================
# Main — 2D Sweep (HHI=0, CVXPY)
# =============================================================================

def run_sweep():
    print("=" * 80)
    print("STEP FIVE — 2D GRID SEARCH: LAMBDA × LOOKBACK PERIOD  (HHI=0, CVXPY)")
    print("=" * 80)

    # ---- Load data ----
    print("\nLoading data...")
    t0 = time.time()
    returns = pd.read_excel("T2_Optimizer.xlsx", index_col=0)
    returns.index = pd.to_datetime(returns.index)
    if returns.abs().mean().mean() > 1:
        returns = returns / 100
    if "Monthly Return_CS" in returns.columns:
        returns = returns.drop(columns=["Monthly Return_CS"])

    # Load factor max weight constraints (matches Step Five FAST.py)
    factor_categories = pd.read_excel("Step Factor Categories.xlsx")
    max_weights = dict(zip(factor_categories["Factor Name"], factor_categories["Max"]))
    n_excluded = sum(1 for v in max_weights.values() if v == 0)
    n_active = sum(1 for v in max_weights.values() if v > 0)

    print(f"Returns data: {returns.shape[0]} periods × {returns.shape[1]} factors")
    print(f"Factor constraints: {n_active} active, {n_excluded} excluded (Max=0)")

    # ---- Sweep dimensions ----
    lambda_params = [5.0, 50.0, 500.0]
    hhi_fixed = 0.0
    lookback_periods = [36, 60, 72, 90, 120]

    total = len(lambda_params) * len(lookback_periods)
    print(
        f"\nSweep: {len(lambda_params)} λ × {len(lookback_periods)} L = {total} combinations  (γ={hhi_fixed})"
    )

    # ---- Storage ----
    results = {lb: {} for lb in lookback_periods}

    # ---- Sweep ----
    combo = 0
    for lb in lookback_periods:
        for lam in lambda_params:
            combo += 1
            t_cell = time.time()
            print(f"\n[{combo}/{total}] λ={lam:>5.1f}  L={lb:>3d}m  (γ=0)  ...", end=" ")

            port_rets, weights_df = run_rolling_optimization(
                returns, lam, hhi_fixed, lb, max_weights
            )
            metrics = compute_metrics(port_rets, weights_df)
            results[lb][lam] = metrics

            elapsed = time.time() - t_cell
            print(
                f"done ({elapsed:.1f}s)  "
                f"Ret={metrics['Annualized Return (%)']:>6.2f}%  "
                f"Sharpe={metrics['Sharpe Ratio']:>6.3f}  "
                f"Vol={metrics['Volatility (%)']:>6.2f}%  "
                f"DD={metrics['Max Drawdown (%)']:>6.2f}%  "
                f"TO={metrics['Turnover (%)']:>6.2f}%"
            )

    total_elapsed = time.time() - t0
    avg_cell = total_elapsed / total

    # ---- Build summary tables ----
    metric_names = [
        "Annualized Return (%)",
        "Sharpe Ratio",
        "Volatility (%)",
        "Max Drawdown (%)",
        "Turnover (%)",
        "Positive Months (%)",
        "Average HHI",
        "Normalized HHI",
    ]

    summary_grids = {}
    for metric in metric_names:
        df = pd.DataFrame(index=lookback_periods, columns=lambda_params, dtype=float)
        df.index.name = "Lookback"
        df.columns.name = "Lambda"
        for lb in lookback_periods:
            for lam in lambda_params:
                df.loc[lb, lam] = results[lb].get(lam, {}).get(metric, np.nan)
        summary_grids[metric] = df

    # ---- Display ----
    for metric in metric_names:
        print(f"\n{'=' * 80}")
        print(f"  {metric}")
        print(f"{'=' * 80}")
        print(
            summary_grids[metric].to_string(
                float_format=lambda v: f"{v:>8.3f}" if pd.notna(v) else "     NaN"
            )
        )

    # ---- Find optimal ----
    print(f"\n{'=' * 80}")
    print("  OVERALL OPTIMAL COMBINATIONS")
    print(f"{'=' * 80}")

    all_rows = []
    for lb in lookback_periods:
        for lam in lambda_params:
            m = results[lb].get(lam)
            if m:
                all_rows.append(
                    {
                        "Lookback": lb,
                        "Lambda": lam,
                        **{k: m[k] for k in metric_names},
                    }
                )
    flat = pd.DataFrame(all_rows)

    best = flat.loc[flat["Sharpe Ratio"].idxmax()]
    print(f"\n  Best Sharpe Ratio:       {best['Sharpe Ratio']:.3f}  "
          f"(λ={best['Lambda']}, L={best['Lookback']}m)")

    best_ret = flat.loc[flat["Annualized Return (%)"].idxmax()]
    print(f"  Best Return:             {best_ret['Annualized Return (%)']:.2f}%  "
          f"(λ={best_ret['Lambda']}, L={best_ret['Lookback']}m)")

    print(f"\n  Best Sharpe By Lookback:")
    for lb in lookback_periods:
        sub = flat[flat["Lookback"] == lb]
        row = sub.loc[sub["Sharpe Ratio"].idxmax()]
        print(f"    L={lb:>3d}m:  Sharpe={row['Sharpe Ratio']:.3f}  "
              f"(λ={row['Lambda']})  "
              f"Return={row['Annualized Return (%)']:.2f}%  "
              f"TO={row['Turnover (%)']:.1f}%")

    # ---- Save to Excel ----
    excel_path = "Step-Five-Grid-Search-Results.xlsx"
    print(f"\n  Saving to {excel_path} ...")
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        workbook = writer.book
        num_fmt = workbook.add_format({"num_format": "0.00"})

        for metric in metric_names:
            df = summary_grids[metric]
            safe_name = metric[:31]
            df.to_excel(writer, sheet_name=safe_name)
            ws = writer.sheets[safe_name]
            ws.set_column(0, 0, 10)
            ws.set_column(1, len(df.columns), 10, num_fmt)

        flat.to_excel(writer, sheet_name="All_Results", index=False)
        ws = writer.sheets["All_Results"]
        ws.set_column(0, len(flat.columns) - 1, 14)

        # Timing summary sheet
        timing_df = pd.DataFrame(
            {"Total (s)": [total_elapsed], "Avg per Cell (s)": [avg_cell]}
        )
        timing_df.to_excel(writer, sheet_name="Timing", index=False)

    print(f"  Done → {excel_path}")
    print(f"  Total: {total_elapsed:.1f}s  |  Avg per cell: {avg_cell:.2f}s")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    run_sweep()
