"""
=============================================================================
SCRIPT NAME: Step Five Top3 Tcost.py
Top-3 Factor Momentum with Hysteresis Band + Trading-Cost Hurdle
=============================================================================

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value/T2_Optimizer.xlsx
    Monthly factor net returns (percentage points) from Step Four.
    Rows = months, columns = 82 factor portfolios.
- /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value/Step Factor Categories.xlsx
    Factor Name / Category / Max columns. Only factors with Max > 0
    (the 36 Value + Quality factors) are eligible for selection.
- /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value/T2_Trading_Cost.xlsx
    Sheet "Trading_Costs": per-factor, per-month one-way trading cost in
    PERCENT (0.07 = 7 basis points). Produced by Step Four.

OUTPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value/T2_rolling_window_weights.xlsx
    Optimized factor weights. IDENTICAL format to Step Five FAST.py output:
    index = monthly dates plus one extra "next month" row, columns = ALL 82
    factors (zeros for factors not held). Consumed by Steps 6, 6.5, 7, 8, 10.
- /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value/T2_strategy_statistics.xlsx
    Strategy performance statistics, monthly returns, monthly turnover
    (same three sheets as Step Five FAST.py).
- /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value/T2_factor_weight_heatmap.pdf
    Heatmap of factor weights over time (top 20 factors by average weight).
- /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value/T2_strategy_performance.pdf
    Cumulative performance chart of the strategy's net returns.

VERSION: 1.0
LAST UPDATED: 2026-06-11
AUTHOR: Claude Code (deep-dive cost-hurdle variant, adopted by Arjun)

DESCRIPTION (plain English):
This is "Step Five Top3.py" plus ONE extra rule: a trading-cost hurdle on
factor swaps. Step by step, each month:

  1. SIGNAL (unchanged): score every eligible factor by its average net
     return over the trailing window (expanding until 60 months exist,
     then rolling 60 months) - 60-month factor momentum.
  2. HYSTERESIS BAND (unchanged): a held factor stays as long as its rank
     is better than N_TOP + EXIT_BAND (stays while ranked in the top 5).
  3. COST HURDLE (new): when a held factor DOES fall out of the band, the
     challenger only replaces it if the challenger's expected monthly
     return edge over the incumbent exceeds KAPPA times the round-trip
     trading cost of the swap (cost to sell the incumbent + cost to buy
     the challenger, from T2_Trading_Cost.xlsx for that month). If the
     edge is too small to pay for the trade, we simply keep the incumbent.
  4. Hold the resulting 3 factors equal-weighted at 1/3 each.

Why: in backtests 2000-2026 (all variants scored net of the same costs)
this adds +0.20%/yr over plain Top3 hysteresis (3.96% -> 4.16% net), cuts
factor turnover from 4.5%/mo to 3.2%/mo, keeps the same Sharpe (0.84) and
max drawdown (-10.3%), and improves the 2018-2026 era (1.22% -> 2.03%/yr).
KAPPA = 1.0 is the sweet spot; higher values make the portfolio stale.
See: Experiments Deep Dive/exp_value_tcost_aware.py.

The output file format is byte-compatible with Step Five FAST.py, so every
downstream step (Six, Six Point Five, Seven, Eight, Eight Point Five, Nine,
Ten, Fourteen, ...) runs unchanged.

NOTE: "Step Five FAST.py" and "Step Five Top3.py" are untouched. Running
either will overwrite T2_rolling_window_weights.xlsx - run one engine only.

FAIL IS FAIL: if T2_Trading_Cost.xlsx is missing or malformed, this script
raises an error rather than silently falling back to cost-blind behavior.

DEPENDENCIES:
- pandas, numpy, matplotlib, xlsxwriter, openpyxl

USAGE:
python "Step Five Top3 Tcost.py"
=============================================================================
"""

import logging
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from step_five_multiwindow_stats import log_step_five_multiwindow_table

plt.style.use("default")

# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================
N_TOP = 3          # number of factors held, equal-weighted
EXIT_BAND = 2      # held factor exits band only when rank >= N_TOP + EXIT_BAND
WINDOW_SIZE = 60   # trailing window for factor momentum (months)
KAPPA = 1.0        # cost-hurdle multiplier: swap only if monthly mu edge
                   # > KAPPA * (cost out + cost in). 1.0 = sweet spot.

TRADING_COST_FILE = "T2_Trading_Cost.xlsx"

WEIGHTS_OUTPUT = "T2_rolling_window_weights.xlsx"
STATS_OUTPUT = "T2_strategy_statistics.xlsx"
HEATMAP_OUTPUT = "T2_factor_weight_heatmap.pdf"
PERFORMANCE_OUTPUT = "T2_strategy_performance.pdf"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("T2_processing.log", mode="a"),
            logging.StreamHandler(),
        ],
    )


# =============================================================================
# DATA LOADING (identical conventions to Step Five FAST.py / Top3.py)
# =============================================================================
def load_and_prepare_data():
    """Load factor returns and eligibility. Returns (returns_df, eligible_cols)."""
    logging.info("Loading input data...")

    returns = pd.read_excel("T2_Optimizer.xlsx", index_col=0)
    returns.index = pd.to_datetime(returns.index)

    # Step Four writes monthly net returns as percentage points -> decimals
    returns = returns.apply(pd.to_numeric, errors="coerce") / 100.0

    if "Monthly Return_CS" in returns.columns:
        returns = returns.drop(columns=["Monthly Return_CS"])

    all_nan_rows = returns.isna().all(axis=1)
    if all_nan_rows.any():
        bad = returns.index[all_nan_rows].strftime("%Y-%m").tolist()
        logging.warning("Dropping %d all-NaN month(s): %s", len(bad), bad[:5])
        returns = returns.loc[~all_nan_rows]

    factor_categories = pd.read_excel("Step Factor Categories.xlsx")
    max_weights = dict(zip(factor_categories["Factor Name"], factor_categories["Max"]))

    # Eligible = Max > 0. Factors missing from the file default to 0 (excluded).
    eligible = [c for c in returns.columns if max_weights.get(c, 0.0) > 0]
    excluded = [c for c in returns.columns if c not in eligible]

    logging.info("Loaded returns: %d months x %d factors", *returns.shape)
    logging.info("Eligible factors (Max>0): %d | Excluded: %d", len(eligible), len(excluded))
    return returns, eligible


def load_trading_costs(returns_df, eligible):
    """
    Load the per-factor monthly one-way trading cost matrix (decimal units),
    aligned to the return dates and eligible factors.

    Missing factor costs are filled with that month's cross-factor mean
    (and finally the global mean) so a missing cell never blocks a trade
    decision. FAIL IS FAIL: a missing/unreadable file raises.
    """
    tc = pd.read_excel(TRADING_COST_FILE, sheet_name="Trading_Costs")
    if "Date" not in tc.columns:
        raise ValueError(f"{TRADING_COST_FILE}: expected a 'Date' column "
                         f"in sheet 'Trading_Costs'")
    tc["Date"] = pd.to_datetime(tc["Date"])
    tc = tc.set_index("Date")

    global_mean = float(tc.stack().mean())
    tc = tc.reindex(index=returns_df.index, columns=eligible)
    row_mean = tc.mean(axis=1)
    tc = tc.apply(lambda col: col.fillna(row_mean)).fillna(global_mean)

    tc = tc / 100.0  # percent -> decimal (0.07% -> 0.0007)
    logging.info("Trading costs loaded: %d months x %d factors | "
                 "mean one-way cost %.1f bps",
                 tc.shape[0], tc.shape[1], tc.stack().mean() * 1e4)
    return tc


# =============================================================================
# TOP-3 SELECTION WITH HYSTERESIS + COST HURDLE
# =============================================================================
def select_top3_weights(returns_df, eligible, costs_df):
    """
    Build the monthly weight matrix.

    For each month t (starting at the second month, like Step Five FAST):
      - momentum mu = mean of rows < t (expanding until WINDOW_SIZE, then rolling)
      - keep held factors ranked better than N_TOP + EXIT_BAND
      - for each open slot, the best-ranked challenger replaces the dropped
        incumbent ONLY if (mu_challenger - mu_incumbent) [monthly units]
        exceeds KAPPA * (cost_in + cost_out) for that month; otherwise the
        incumbent is retained despite being out of the band
      - equal weight 1/N_TOP

    Also computes the extra "next month" row from the last WINDOW_SIZE rows
    INCLUDING the final month (using the last available cost row), exactly
    like Step Five FAST's extra-month optimization.

    Returns a DataFrame indexed by dates[1:] + next_month with ALL factor
    columns (zeros for factors not held).
    """
    E = returns_df[eligible]
    dates = returns_df.index
    n = len(eligible)

    held: list[int] | None = None
    rows = {}
    swaps_made = swaps_blocked = 0

    def pick(window_data, held_prev, tc_row):
        nonlocal swaps_made, swaps_blocked
        mu = window_data.mean(axis=0).to_numpy(dtype=float)
        mu = np.nan_to_num(mu, nan=-9e9)          # factors with no data rank last
        order = np.argsort(mu)[::-1]
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(n)

        if held_prev is None:
            return list(order[:N_TOP])

        keep = [f for f in held_prev if ranks[f] < N_TOP + EXIT_BAND]
        dropped = [f for f in held_prev if f not in keep]
        candidates = [f for f in order if f not in keep]

        for f in candidates:
            if len(keep) >= N_TOP:
                break
            if dropped:
                incumbent = dropped[0]
                edge = mu[f] - mu[incumbent]                  # monthly mu units
                hurdle = KAPPA * (tc_row[f] + tc_row[incumbent])
                if edge <= hurdle:
                    # Edge too small to pay for the trade: keep the incumbent.
                    keep.append(incumbent)
                    dropped.remove(incumbent)
                    swaps_blocked += 1
                    continue
                dropped.remove(incumbent)
                swaps_made += 1
            keep.append(f)
        return keep[:N_TOP]

    mean_cost = float(costs_df.stack().mean())

    for i in range(1, len(dates)):
        window = E.iloc[:i] if i <= WINDOW_SIZE else E.iloc[i - WINDOW_SIZE:i]
        tc_row = np.nan_to_num(costs_df.iloc[i].to_numpy(dtype=float), nan=mean_cost)
        held = pick(window, held, tc_row)
        w = np.zeros(n)
        w[held] = 1.0 / len(held)
        rows[dates[i]] = w

    # Extra month: last WINDOW_SIZE rows including the final one; costs for
    # next month are not known yet, so use the most recent cost row.
    next_month = dates[-1] + pd.DateOffset(months=1)
    extra_window = E.iloc[max(0, len(dates) - WINDOW_SIZE):]
    tc_row = np.nan_to_num(costs_df.iloc[-1].to_numpy(dtype=float), nan=mean_cost)
    held = pick(extra_window, held, tc_row)
    w = np.zeros(n)
    w[held] = 1.0 / len(held)
    rows[next_month] = w
    logging.info(
        "Extra month (%s) holdings: %s",
        next_month.strftime("%Y-%m"),
        ", ".join(eligible[f] for f in held),
    )
    logging.info("Cost hurdle (kappa=%.1f): %d swaps executed, %d swaps blocked",
                 KAPPA, swaps_made, swaps_blocked)

    weights_eligible = pd.DataFrame.from_dict(rows, orient="index", columns=eligible)

    # Pad to the full factor universe (zeros for excluded factors) so the
    # output format matches Step Five FAST exactly.
    weights_df = weights_eligible.reindex(columns=returns_df.columns, fill_value=0.0)
    weights_df.index.name = returns_df.index.name
    return weights_df


# =============================================================================
# PERFORMANCE (identical logic to Step Five FAST.py)
# =============================================================================
def calculate_strategy_performance(weights_df, returns_df):
    """Weights at month t earn the factor-return row at t+1 (no look-ahead)."""
    logging.info("Calculating strategy performance...")

    portfolio_returns, aligned_dates = [], []
    for date in weights_df.index:
        if date in returns_df.index:
            next_idx = returns_df.index.get_loc(date) + 1
            if next_idx < len(returns_df.index):
                next_date = returns_df.index[next_idx]
                w = weights_df.loc[date].to_numpy(dtype=float)
                r = returns_df.loc[next_date].to_numpy(dtype=float)
                portfolio_returns.append(float(np.nansum(w * np.nan_to_num(r))))
                aligned_dates.append(next_date)

    pr = pd.Series(portfolio_returns, index=aligned_dates)

    ann_return = (1 + pr.mean()) ** 12 - 1
    ann_vol = pr.std() * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0

    cum = (1 + pr).cumprod()
    drawdowns = (cum - cum.expanding().max()) / cum.expanding().max()

    monthly_turnover = weights_df.diff().abs().sum(axis=1) / 2

    stats = {
        "Annualized Return (%)": ann_return * 100,
        "Annualized Volatility (%)": ann_vol * 100,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown (%)": drawdowns.min() * 100,
        "Average Monthly Turnover (%)": monthly_turnover.mean() * 100,
        "Positive Months (%)": (pr > 0).mean() * 100,
        "Skewness": pr.skew(),
        "Kurtosis": pr.kurtosis(),
    }
    return {
        "statistics": stats,
        "monthly_returns": pr,
        "monthly_turnover": monthly_turnover,
    }


# =============================================================================
# VISUALIZATIONS (same output filenames as Step Five FAST.py)
# =============================================================================
def create_factor_weight_heatmap(weights_df, top_n=20):
    logging.info("Creating factor weight heatmap...")
    w = weights_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    top_factors = w.mean().sort_values(ascending=False).head(top_n).index.tolist()
    sample = w[top_factors].iloc[::4].astype(float)
    if sample.empty:
        logging.warning("No weight data for heatmap")
        return

    fig, ax = plt.subplots(figsize=(20, 12), facecolor="white")
    im = ax.imshow(sample.T.values, cmap="YlGnBu", aspect="auto",
                   vmin=0.0, vmax=max(0.34, float(sample.values.max())))
    tick_positions = range(0, len(sample.index), 3)
    ax.set_xticks(list(tick_positions))
    ax.set_xticklabels([sample.index[i].strftime("%Y") for i in tick_positions], fontsize=11)
    ax.set_yticks(range(len(top_factors)))
    ax.set_yticklabels(top_factors, fontsize=10)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Portfolio Weight", rotation=270, labelpad=20, fontsize=12)
    ax.set_title(
        f"Top-{N_TOP} Factor Momentum Weights "
        f"(hysteresis band {EXIT_BAND}, cost hurdle kappa={KAPPA})\n"
        f"Top {top_n} factors by average weight",
        fontsize=16, fontweight="bold", pad=20,
    )
    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("Factor", fontsize=13)
    plt.tight_layout()
    plt.savefig(HEATMAP_OUTPUT, bbox_inches="tight", dpi=300,
                facecolor="white", format="pdf")
    plt.close()
    logging.info("Heatmap saved to %s", HEATMAP_OUTPUT)


def create_strategy_performance_plot(performance_results):
    logging.info("Creating strategy performance plot...")
    monthly_returns = performance_results["monthly_returns"]
    cum = (1 + monthly_returns).cumprod()

    plt.figure(figsize=(15, 8), facecolor="white")
    plt.plot(cum.index, cum.values,
             label=f"Top{N_TOP} Hysteresis + Tcost Hurdle Strategy",
             linewidth=2, color="#1f77b4")
    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline")
    plt.title("T2 Strategy Cumulative Performance (Top-3 + Hysteresis + Tcost Hurdle)",
              fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return", fontsize=12)
    plt.gca().yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: "{:.1%}".format(y - 1)))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc="upper left")

    s = performance_results["statistics"]
    plt.figtext(0.02, 0.02,
                f"Annual Return: {s['Annualized Return (%)']:.1f}%\n"
                f"Sharpe Ratio: {s['Sharpe Ratio']:.2f}\n"
                f"Max Drawdown: {s['Maximum Drawdown (%)']:.1f}%\n"
                f"Avg Turnover: {s['Average Monthly Turnover (%)']:.1f}%/mo",
                fontsize=11,
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray", pad=10))
    plt.tight_layout()
    plt.savefig(PERFORMANCE_OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logging.info("Performance plot saved to %s", PERFORMANCE_OUTPUT)


# =============================================================================
# SAVE RESULTS (same files/sheets as Step Five FAST.py)
# =============================================================================
def save_results(weights_df, performance_results):
    logging.info("Saving results...")

    weights_df.to_excel(WEIGHTS_OUTPUT)
    logging.info("Weights saved to %s", WEIGHTS_OUTPUT)

    with pd.ExcelWriter(STATS_OUTPUT, engine="xlsxwriter") as writer:
        stats_df = pd.DataFrame(list(performance_results["statistics"].items()),
                                columns=["Metric", "Value"])
        stats_df.to_excel(writer, sheet_name="Summary Statistics", index=False)

        pd.DataFrame({
            "Hybrid Strategy Returns": performance_results["monthly_returns"]
        }).to_excel(writer, sheet_name="Monthly Returns")

        pd.DataFrame({
            "Monthly Turnover": performance_results["monthly_turnover"]
        }).to_excel(writer, sheet_name="Monthly Turnover")

        workbook = writer.book
        date_format = workbook.add_format({"num_format": "dd-mmm-yyyy"})
        for sheet_name in ["Monthly Returns", "Monthly Turnover"]:
            if sheet_name in writer.sheets:
                writer.sheets[sheet_name].set_column(0, 0, 12, date_format)

    logging.info("Strategy statistics saved to %s", STATS_OUTPUT)

    create_factor_weight_heatmap(weights_df)
    create_strategy_performance_plot(performance_results)


def main():
    setup_logging()
    start = time.time()

    logging.info("=" * 80)
    logging.info("T2 FACTOR TIMING - STEP FIVE TOP3 TCOST: %d-FACTOR MOMENTUM, "
                 "HYSTERESIS BAND %d, WINDOW %dM, COST HURDLE KAPPA %.1f",
                 N_TOP, EXIT_BAND, WINDOW_SIZE, KAPPA)
    logging.info("=" * 80)

    try:
        returns_df, eligible = load_and_prepare_data()
        costs_df = load_trading_costs(returns_df, eligible)
        weights_df = select_top3_weights(returns_df, eligible, costs_df)
        performance_results = calculate_strategy_performance(weights_df, returns_df)
        save_results(weights_df, performance_results)

        logging.info("=" * 80)
        logging.info("RESULTS SUMMARY (multi-window)")
        logging.info("=" * 80)
        log_step_five_multiwindow_table(
            performance_results["monthly_returns"],
            performance_results["monthly_turnover"],
            logging.info,
        )
        logging.info("=" * 80)
        logging.info("STEP FIVE TOP3 TCOST COMPLETED in %.1fs", time.time() - start)
        logging.info("=" * 80)

    except Exception as e:
        logging.error("Error in Step Five Top3 Tcost: %s", e)
        import traceback
        traceback.print_exc()
        raise

    return performance_results


if __name__ == "__main__":
    main()
