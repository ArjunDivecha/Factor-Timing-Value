# AGENTS.md

## Learned User Preferences

- The primary performance metric for this strategy is annualized return, NOT Sharpe ratio — with reasonable turnover as a secondary consideration.
- The strategy must always remain a value strategy; the user's core belief is that value wins in the long term. Improvement ideas may add quality/rotation but cannot abandon the value orientation.
- Improvement experiments must never modify the original pipeline scripts — put experimental code in separate sandbox files/folders (e.g. `Experiments Deep Dive/`) and report findings.
- Data gap filling must be forward-only: no data from the future may propagate backward (no backfill/lookahead).
- The user frequently compares this repo step-by-step against the sibling repo `/Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy` and ports features between them (e.g. multi-window stats); keep conventions consistent across the two.
- Per-factor trading costs must be modeled as a dynamic time series (vector per date), not a scalar — factor portfolio composition changes over time, so its cost changes too.
- For Bloomberg/market data pulls, always use the Bloomberg skill (OpusBloomberg pipe); IBKR API market data is NOT entitled (error 10089) and should not be relied on.

## Learned Workspace Facts

- This workspace is the value/quality variant of `/Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy`: instead of fixed value variables, it uses factor momentum (60-month trailing factor returns) as the predictor, automatically rotating into quality factors when value underperforms (limits big value drawdowns like post-GFC).
- `Step Factor Categories.xlsx` is the factor eligibility whitelist for Step Five: only factors with Max > 0 (Value + Quality categories) may enter the optimization; Max = 0 factors (Momentum, Economic, Commodity, MCAP) must be dropped from the universe, and factors missing from the file must default to max weight 0.0 (not 1.0). This was a recurring bug in both Step Five FAST.py and Step Five Grid Search Sweep.py — any Step Five variant must apply this filter to produce comparable results.
- `Step Five FAST.py` was OVERWRITTEN with the Top3 Tcost engine (the original CVXPY/OSQP QP optimizer code is no longer in this file, despite CLAUDE.md/older AGENTS.md notes describing it). Its header now reads "Step Five Top3 Tcost.py" — the standalone `Step Five Top3 Tcost.py` was deleted and its logic merged into `Step Five FAST.py`. The active production engine for this Value/Quality variant is Top-3 60-month momentum + hysteresis band (EXIT_BAND=2) + κ=1 trading-cost-hurdle on factor swaps, using the dynamic per-factor cost vector from `T2_Trading_Cost.xlsx`.
- The Top3+hysteresis construction beat the CVXPY QP on every metric in the 36-factor Value/Quality universe because the top factors are correlated siblings (averaging denoises; QP winner-take-all over-concentrates). The opposite holds in the sibling all-factor `T2 Factor Timing Fuzzy` repo (82 heterogeneous factors) where concentration wins — DO NOT port the Top3 fix to the Fuzzy repo (it costs ~2.6%/yr there).
- The CVXPY QP optimizer for the Value variant survives only in git history (`Step Five 60 Month Optimal Portfolios.py` scipy-SLSQP version and `Step Five Grid Search Sweep.py` were deleted/archived). If the QP is needed: μ = 8 × 60-month trailing mean, LAMBDA = 0 (pure return maximization), HHI penalty γ = 0.005; the cost-aware QP variant used CLARABEL (OSQP can't handle the |·| cost term) with κ=2 and HHI=0.001, and swept lambda × lookback (36/60/72/90/120 months) via the now-deleted Grid Search Sweep script.
- Downstream steps (6 → 6.5 → 7 → 8 → 8.5 → 9 → 10 → 14) auto-adapt to any Step Five variant because the weights file format is identical: `T2_rolling_window_weights.xlsx` has all 82 factor columns (zeros for excluded factors), an extra next-month row that Step Eight consumes for its one-month vintage shift, plus the same `T2_strategy_statistics.xlsx` and PDF outputs.
- `Step Eight Basket Smoothing Lab.py` and `Step Eight Turnover Decomposition.py` are durable additions: the smoothing lab experiments with weight smoothing, and turnover decomposition splits monthly country turnover into bet-change (wanted) vs basket-refresh (avoidable) via an exact counterfactual `Wbar = B_t @ w_{t-1}`, validating reconstructed weights against production before trusting the split.
- `Step Fourteen Target Optimization.py` has a hard `weights_var >= 0` long-only constraint that strips any short country weights produced by Step Eight. `Step Fourteen Target Optimization LongShort.py` is the variant that removes this constraint to express true 130/30 (long-short) country weights end-to-end. Step Eight itself produces negative country weights correctly (positive factor weight → top-ranked countries; negative → bottom-ranked countries); only Step Fourteen long-only drops them.
- `step_five_multiwindow_stats.py` is a shared helper (copied from the Fuzzy repo) that prints a Full Period / 12 Months / 3 Years / 5 Years performance table at the end of Step Five FAST runs.
- `Step Nine Compare Alpha Mcap Strategies.py` compares the current Step Nine portfolio (Step Eight weights) against alpha × MCAP weighted strategies built from Step Six and Step Six Point Five alphas; weights use `max(alpha, 0) × MCAP` (long-only), with MCAP from `T2 Master.xlsx` sheet `MCAP`.
- The Step Three script's actual filename is `Step Three Top20 Portfolios Fast.py` (not `Step Three Top20 Portfolios.py` as listed in CLAUDE.md's command list).
- `T60.xlsx` contains no `_PRED` columns — legacy alpha-forecast lookup code paths that search for `{Factor}_PRED` always fall back to the historical window mean.


## Liquidity (ADV) position cap — added 2026-06-24 (mirrored from T2 Factor Timing Fuzzy)

Step Eight now applies a LIQUIDITY (ADV) POSITION CAP via the shared `step_liquidity_cap.py`:
each country is capped so a full rotation is at most `LIQ_MAXPART` (0.20) of one day's dollar
volume, then water-filled back to sum=1. Config lives at the top of the Step Eight script
(`APPLY_LIQUIDITY_CAP=True`, `LIQ_MAXPART=0.20`, `LIQ_AUM=7_000_000`,
`LIQ_PATH="Experiments Deep Dive/IBKR_Liquidity.xlsx"`). Set `APPLY_LIQUIDITY_CAP=False` to
restore the pre-cap behavior (and the exact factor==country return identity, which the cap
deliberately breaks because the factor model is blind to ETF liquidity).

Why: at small AUM the dominant trading cost is MARKET IMPACT in thin single-country ETFs
(Denmark/EDEN ~$0.3M ADV above all), not the ~4 bps quoted half-spread. A square-root impact
model on IBKR ADV puts the real one-way cost at ~25-40 bps at $7M; the cap recovers ~+1.24%/yr
net (validated via the real Step Nine in the classic repo) and is basic risk management. Same
34-ETF universe across all three repos, so the same `IBKR_Liquidity.xlsx` ADV data is reused.
If AUM changes materially, update `LIQ_AUM` and refresh ADV via
`Experiments Deep Dive/Step Tcost Impact Model.py`. Full write-up: classic repo README note
(2026-06-24) and llmchat.md.
