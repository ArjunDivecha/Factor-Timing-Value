# CLAUDE.md — T2 Factor Timing Fuzzy **Value**

Operator's manual for a coding agent. Global rules (light mode, doc headers, `file://`
links, FAIL-IS-FAIL, Bloomberg backup) live in `~/CLAUDE.md` and `../../CLAUDE.md` — not
repeated here. Deeper prose docs are in `openwiki/` (start at `openwiki/quickstart.md`).

## Purpose
Value/Quality country-allocation research pipeline **with a live real-money Schwab trading
extension**. It builds factor signals, times factors via 60-month momentum, converts factor
weights to country weights, backtests/report, and (separately) rebalances Schwab account
**#167 "Equity Value"** via a TWAP engine. This is the value/quality sibling of
`../T2 Factor Timing Fuzzy` (the all-factor "Momentum" strategy, account #090); the two share
the trading engine and must stay in sync (see gotchas).

## Architecture map (load-bearing files, absolute paths)
- `.../Step One Create T2Master.py` — Bloomberg raw → `T2 Master.xlsx`; contains the
  **data-quality guard** `detect_regime_breaks_sheet()` (lines ~196, 826-977). Untested.
- `.../Step Two Create Normalized Tidy.py` — cross-sectional z-scores (`_CS` suffix).
- `.../Step Five FAST.py` — **the active factor engine. Despite the filename its header reads
  "Step Five Top3 Tcost.py": Top-3 60-month momentum + hysteresis band (EXIT_BAND=2) + κ=1
  trading-cost hurdle.** The old CVXPY/SLSQP QP optimizer is gone from this file (git history only).
- `.../Step Six Create Country alphas from Factor alphas.py` — factor weights → country alphas.
- `.../Step Eight Write Country Weights.py` — country weights; applies the ADV liquidity cap via
  `step_liquidity_cap.py`. Produces the extra "next-month" vintage row downstream steps consume.
- `.../Step Nine Calculate Portfolio Returns.py` — final portfolio returns.
- `.../Step Fourteen Target Optimization.py` — CVXPY country optimizer, **long-only**
  (`weights_var >= 0` strips shorts). `...LongShort.py` is the 130/30 variant.
- `.../Step Schwab Trading.py` + `.../step_schwab_dashboard.py` — **live-money TWAP executor.**
- `.../Run_All_Pipeline.py` — canonical step order (from scratch). `Run_Limited_Pipeline.py`
  reruns from Step Five onward.
- `.../Step Factor Categories.xlsx` — factor eligibility whitelist (Max>0 → the 36 Value+Quality
  factors eligible for Step Five; Max=0 dropped; **missing factors default to 0.0, not 1.0**).
- `.../tests/test_schwab_twap_engine.py` — 33 fake-broker safety tests (trading engine only).

## Commands that work
```bash
python3 -m pytest tests/ -q          # 33 Schwab-engine tests — VERIFIED collect+run
python3 "Run_All_Pipeline.py"        # full pipeline, Step Zero→FINALFINAL (see caveat below)
python3 "Run_Limited_Pipeline.py"    # rerun Step Five→FINALFINAL (skips data rebuild)
python3 "Step Schwab Trading.py"     # LIVE MONEY. Defaults to dry-run; read the file first.
```
- Pipeline step order in `Run_All_Pipeline.py` was verified — every one of its 21 listed
  scripts exists on disk. The end-to-end run itself was **not executed here (unverified)**;
  it is long and rewrites all root `.xlsx`/`.pdf` outputs.
- **Step Zero/One cannot run as-is**: their Bloomberg input `Country Bloomberg Data Master T.xlsx`
  is **not in the repo**. A fresh full run needs that dump first (Bloomberg skill / OpusBloomberg).
  Most work resumes from the committed `T2 Master.xlsx` via `Run_Limited_Pipeline.py`.

## Data locations (all in repo root unless noted)
- Input (missing, must be supplied): `Country Bloomberg Data Master T.xlsx`
- Master data: `T2 Master.xlsx`, `Normalized_T2_Master.xlsx` (+ `...CSV.csv`), `P2P_Country_Historical_Scores.xlsx`
- Factor engine I/O: `T2_Optimizer.xlsx`, `T2_Trading_Cost.xlsx`, `T60.xlsx` → `T2_rolling_window_weights.xlsx`
- Country outputs: `T2_Country_Alphas.xlsx`, `T2_Country_Weights.xlsx`, `T2_Final_Country_Weights.xlsx`
- Reports/logs: `T2_Strategy_Report_Comprehensive_*.pdf`, `T2_ALL_OUTPUTS_MERGED_*.pdf`,
  `T2_processing.log`, `T2_regime_break_log.xlsx` (data-quality forward-fills)
- Trading audit: `outputs/schwab_trade_plan_YYYYMMDD.xlsx`, `outputs/schwab_execution_log_*.xlsx`,
  `outputs/schwab_live_marker_*.json`
- Liquidity data: `Experiments Deep Dive/IBKR_Liquidity.xlsx`

## Conventions & gotchas (repo-specific)
- **Value orientation is non-negotiable.** Primary metric is **annualized return** (not Sharpe),
  turnover secondary. Improvements may add quality/rotation but may not abandon value.
- **Diversification wins here** (36 correlated Value/Quality factors): Top-3 equal-weight beats
  the QP. The opposite is true in the sibling `T2 Factor Timing Fuzzy` (82 heterogeneous factors,
  concentration wins) — **do NOT port the Top-3 fix there** (~-2.6%/yr).
- **No lookahead, ever.** All gap-filling is forward-only; no future data may propagate backward.
- **Schwab engine parity is a standing invariant.** `Step Schwab Trading.py` and
  `step_schwab_dashboard.py` must stay byte-for-byte identical to the sibling Momentum repo except
  documented account-specific values (paths, `DEFAULT_ACCOUNT_NAME`, `T2_FINAL_T60_VALUE.xlsx`,
  VTV/VBR `ETF_OVERRIDES`, account #167). Any other diff = one repo has a fix the other lacks.
- IOC order durations are REJECTED by Schwab (HTTP 400) for these accounts — keep DAY marketable-limit.
- Trading costs in `T2_Trading_Cost.xlsx` are PERCENT (0.07 = 7 bps); Step Five divides by 100.
- Experiments must not modify production step scripts — put them in `Experiments Deep Dive/`.
- `Step Three` real filename is `Step Three Top20 Portfolios Fast.py`; `Step Four` is
  `...FAST.py`. Older command lists referencing `Step Five 60 Month Optimal Portfolios.py`,
  `Step Five Tcost.py`, `Step Eleven/Twelve/Thirteen` are **stale** — those are archived/renamed.
- `T60.xlsx` = trailing 60-month factor returns (no `_PRED` columns; forecast lookups fall back to
  window mean). `T2_rolling_window_weights.xlsx` = optimized weights. They are independent.

## Current state
- **Active, live-trading.** Account #167 completed its first hardened-engine live TWAP rebalance
  on 2026-07-01 (zero MANUAL_REQUIRED, zero aborts). Latest research outputs regenerated 2026-07-05.
- Done: Top-3 Tcost engine, ADV liquidity cap, hardened Schwab engine (33 tests pass), OpenWiki docs.
- **Known-untested:** the entire research pipeline (Steps 0-21). Only the trading engine has tests.
- **CORRECTION to prior notes:** the month-over-month regime-break detector is **implemented and
  wired in** (`detect_regime_breaks_sheet`, writes `T2_regime_break_log.xlsx`) — earlier CLAUDE/AGENTS
  notes calling it "proposed but not yet implemented" are stale.


## Cross-session messaging

Claude Code sessions can message each other directly. `ListAgents` (or `/list-agents`, `/peers`)
lists reachable sessions; `SendMessage` delivers plain text to one by name. Same-machine delivery
uses a local socket; cross-machine is reply-only via Remote Control. Use it to hand off a finding
to a session working elsewhere instead of relaying it through the user. A message is text only —
never conversation history or files; to share full context, resume the session instead.
