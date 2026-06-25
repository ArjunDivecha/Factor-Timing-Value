# llmchat.md - Project Context Log

This file is the shared memory between project sessions and agents.
It is append-only. Do not edit existing entries unless explicitly asked.
Each session appends a timestamped block at the bottom.

---
SESSION START: 2026-06-10 12:00 PDT | Agent: Cursor (Claude)
---

### Session Summary
Two sequential deep-dives into the Value/Quality factor timing strategy. First: systematic experimentation (3 rounds) to improve returns over the original QP optimizer, culminating in a Top-3 equal-weight + hysteresis engine. Second (this session): adding a trading-cost hurdle (κ=1) on top of the hysteresis to further suppress unnecessary swaps. Net result: 4.23%/yr, 0.85 Sharpe, 3.3%/mo turnover — beating the original QP on every metric.

### Decisions Made
- **Primary metric is return, not Sharpe** — all experiments optimized for max return with "reasonable" turnover as a soft constraint. Value orientation must be preserved (strategy must remain fundamentally value-driven).
- **Diversification wins in the 36-factor Value/Quality universe** — Top-3 equal-weight significantly outperforms the concentrated QP. This is the opposite of the all-factor sibling repo where concentration wins.
- **Hysteresis band=2 is the right default** — a held factor stays as long as it ranks in the top 5 (N_TOP + EXIT_BAND = 3+2). Wider bands made the portfolio stale; narrower bands raised turnover without return benefit.
- **Cost-hurdle swap rule** — when a held factor falls out of the hysteresis band, the challenger replaces it only if the edge in expected monthly return exceeds κ × (cost to sell + cost to buy). κ=1 is the sweet spot: it blocked ~50% of band exits, cut turnover 27%, and added +0.16%/yr net. Higher κ values made the portfolio stale and degraded returns.
- **Cost-aware QP does NOT make sense for Value** — porting the Fuzzy-style cost-aware QP improved the old QP's return (3.09%→4.21%) but at 12% turnover and -18.6% drawdown. The Top3 hysteresis engine beat it on every risk metric even before the cost-hurdle addition.
- **Original Step Five FAST.py remains untouched** — all experiments created new scripts; the original QP is preserved.

### Step Five Architecture (Final)

**`Step Five Top3 Tcost.py`** — the active engine. For each month:
1. Score each eligible factor by trailing 60-month mean return.
2. Hysteresis: held factors stay if ranked in top 5.
3. Cost-hurdle: for each band-exit incumbent, swap only if `(mu(challenger) - mu(incumbent)) > κ * (tcost_out + tcost_in)`.
4. Equal-weight the resulting 3 factors at 1/3 each.

`T2_Trading_Cost.xlsx` provides the per-factor monthly one-way cost vector (percent units, mean ~0.07).

### New Files Created
- `Step Five Top3 Tcost.py` — the active drop-in replacement (Top-3 + hysteresis + κ=1 cost-hurdle).
- `step_five_multiwindow_stats.py` — shared utility for multi-window performance table.
- `llmchat.md` — this file.

### Deleted Files
- `Step Five Top3.py` — obsolete cost-blind predecessor. Replaced by Tcost variant.
- `Experiments Deep Dive/exp_factor_lab.py`, `round2.py`, `round3.py` — experiment scripts from Round 1-3 deep dive. Results snapshot in `exp_value_tcost_results.xlsx` (retained).
- `Experiments Deep Dive/exp_value_tcost_aware.py` — the cost-aware comparison test.

### Key Performance (Step Five Top3 Tcost)
| Metric | Value |
|---|---|
| Net Annual Return | 4.23% |
| Sharpe Ratio | 0.85 |
| Max Drawdown | -10.2% |
| Avg Monthly Turnover | 3.27% |
| Cost Hurdle κ | 1.0 |
| Swaps made / blocked | 31 / 34 |

### Data Flow (same as original pipeline)
```
T2_Optimizer.xlsx + T2_Trading_Cost.xlsx
  → Step Five Top3 Tcost.py
    → T2_rolling_window_weights.xlsx
      → Steps 6, 6.5, 7, 8, 8.5, 9, 10, 14
```

### Constraints & Gotchas
- Run `Step Five Top3 Tcost.py` OR `Step Five FAST.py` — not both. The last one to run overwrites `T2_rolling_window_weights.xlsx`.
- `T2_Trading_Cost.xlsx` must be present (it is produced by Step Four alongside `T2_Optimizer.xlsx`). The script will raise an error if missing.
- Trading costs in `T2_Trading_Cost.xlsx` are in PERCENT units (0.07 = 7 bps). The script divides by 100 internally.
- The Tcost variant uses the Tcost-aware swap logic for the "next month" extra row too (uses the most recent available cost row as a proxy).
- Experiment script `exp_value_tcost_results.xlsx` in `Experiments Deep Dive/` contains the full comparison table across all variants tested — keep it as documentation.

### Context for Next Session
This repo (`T2 Factor Timing Fuzzy Value`) is the VALUE/QUALITY strategy (36 factors restricted by Step Factor Categories.xlsx Max>0). The sibling repo `T2 Factor Timing Fuzzy` is the all-factor strategy. Both now have cost-aware Step Five engines but the approaches differ fundamentally: Top-3 + hysteresis + cost-hurdle here, cost-aware QP with κ=2 over there. The `Tcost` branch tracks the latest changes; `main` has the original pipeline.

---
SESSION END: 2026-06-11 01:08 PDT | Agent: Cursor (Claude)
---
