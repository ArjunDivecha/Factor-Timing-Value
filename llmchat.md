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

---
SESSION START: 2026-06-30 13:00 PDT | Agent: Cursor (Claude)
---

### Session Summary
Brought `Step Schwab Trading.py` (live Schwab TWAP executor for Equity Value, account #167) to
full parity with the sister `T2 Factor Timing Fuzzy` (Equity Momentum, #090) repo after an
adversarial code review there (GLM, Sakana Fugu Ultra, a 49-agent effectiveness workflow) found
several real-money execution bugs. This repo had silently fallen ~1 hour behind the Momentum
repo's safety fixes mid-review — confirmed by `git status`/line-count, not assumed. All fixes
were re-applied here, the 31-scenario fake-broker test harness was ported, and both repos now
verify identical via a full-file `diff`.

### Decisions Made
- **Full parity with the Momentum repo is mandatory** — both repos trade real accounts and must
  never diverge on execution-safety logic. Confirmed parity by diffing the entire file: every
  remaining difference (file paths, `DEFAULT_ACCOUNT_NAME`, `T2_FINAL_T60_VALUE.xlsx`, the
  VTV/VBR `ETF_OVERRIDES` block) is an intentional, account-specific value — not a missed fix.
- See the Momentum repo's `llmchat.md` (2026-06-30 13:00 PDT entry) for the full root-cause
  writeup of each bug; not duplicated in full here to avoid drift between the two logs.

### Architecture / Design — fixes ported into this repo's `Step Schwab Trading.py`
1. `_is_filled()` no longer misreads a canceled partial fill as fully filled.
2. Market-order cleanup fails CLOSED (skips) on a quote-fetch failure instead of submitting a
   blind, unprotected market order.
3. Carry-forward is gated on broker-CONFIRMED terminal order status (`manual_required` /
   `confirmed_terminal`), preventing double-execution on cancel/timeout races.
4. Carry-cap deferred excess is accumulated (`+=`) instead of overwritten — shares can no longer
   vanish after a string of failed/partial slices (B1).
5. A submit exception that may have hit AFTER Schwab accepted the order now attempts
   account-history recovery via the new `_recover_or_flag_unknown_order()` helper before ever
   resubmitting — closes the B2 double-submit gap (mirrors the existing 201-no-Location path).
6. `_find_recent_order_id` now takes a `since` timestamp to disambiguate repeated
   same-symbol/same-qty TWAP slices instead of giving up on any ambiguous match.
7. New `check_market_hours()` (wired into `main()`) blocks live runs that start too close to or
   outside market hours; warns only in dry run.
8. `find_account_hash()` now returns `(hash, account_number)`; `main()` echoes a masked account
   number (`...0167`) labeled "Equity Value" before Step 2 proceeds, so a human can visually
   confirm the correct account before any live order.
9. `get_live_quotes(..., raise_on_partial=False)` isolates per-symbol quote failures during TWAP
   execution so one bad/halted ETF no longer stalls every other symbol's slice.
10. `claim_live_marker()` creates the live marker file atomically (`os.O_EXCL`), closing a
    duplicate-run race between the existence check and the write.
11. `scale_buy_plan_to_cash()` now takes `total_equity` and reserves the buffer against it (same
    basis as `build_trade_plan`) instead of the looser `available_cash`-based buffer.
12. Added `time.sleep(0.15)` pacing between consecutive `order_details` polls (in-slice,
    cancel-confirm, final fill check) to stay under Schwab's rate limits.

### Files Created/Updated
- `Step Schwab Trading.py` — all 12 fixes above, ported line-for-line from the Momentum repo
  with only account-specific substitutions.
- `requirements.txt` — added `rich`, `schwabdev`, `pytest` (already had most other deps).
- `tests/test_schwab_twap_engine.py` (NEW) — straight copy of the Momentum repo's 31-scenario
  fake-broker harness (the engine is identical; only the docstring's file paths were updated).
  All 31 pass against this repo's own `Step Schwab Trading.py`.

### Constraints & Gotchas
- This repo's engine MUST be kept byte-for-byte identical to the Momentum repo's, modulo the
  documented account-specific constants. Before trusting either repo's safety fixes, re-run
  `diff "T2 Factor Timing Fuzzy/Step Schwab Trading.py" "T2 Factor Timing Fuzzy Value/Step Schwab
  Trading.py"` and confirm every hunk is one of: file paths, account name/number references,
  `ETF_OVERRIDES` (VTV/VBR), or docstring-only notes. Any other hunk means one repo has a fix the
  other doesn't.
- `IMMEDIATE_OR_CANCEL` (IOC) order durations are confirmed REJECTED by Schwab's API (HTTP 400)
  for these accounts — do not switch from `DAY` marketable-limit orders without new evidence.
- A real (read-only) dry run was completed successfully end-to-end against the live Schwab API
  for this account: correctly echoed account number `...0167` for "Equity Value", built a
  16-ETF target trade plan (VTV/VBR + 14 country ETFs), and wrote
  `outputs/schwab_trade_plan_20260630.xlsx`. No live orders were submitted.

### Context for Next Session
Engine parity with the Momentum repo is the standing invariant going forward — any future fix to
`execute_twap_leg()`, `get_live_quotes()`, `scale_buy_plan_to_cash()`, etc. in either repo must be
ported to the other in the same session, not deferred. See the Momentum repo's `llmchat.md` for
deferred/out-of-scope items (cleanup-path recovery helper, incremental per-slice persistence,
Sakana's execution-quality redesign ideas) that apply equally here.

---
SESSION END: 2026-06-30 14:00 PDT | Agent: Cursor (Claude)
---

---
SESSION START: 2026-07-01 09:30 PDT | Agent: Cursor (Claude)
---

### Session Summary
Ran this account's FIRST full live TWAP rebalance under the hardened engine (immediately after
the same run on the Momentum sister repo), monitored closely via a read-only order-status checker
independent of the trading process's own output, and ported the B3 (SNAXX) fix plus a dashboard
bug fix from the Momentum repo. See the Momentum repo's `llmchat.md` (2026-07-01 session) for full
detail — this entry covers only what's specific to this account.

### Decisions Made
- B3 (SNAXX) patched here identically to Momentum: `allocatable` in `build_trade_plan()` now
  excludes SNAXX's market value, since it's never sold. Confirmed $0 SNAXX in this account via a
  fresh dry run before patching (dormant but real if a balance ever appears).
- Dashboard blotter bug (BUY leg invisible during Momentum's live run) fixed identically here —
  see Momentum's session log for the full root-cause writeup. Files are byte-identical between
  repos except account-specific values (confirmed via `diff`).
- Live run was launched in a real Terminal.app window (attached tty) so the `rich` dashboard
  genuinely renders, not redirected to a log file.

### Live Run Result (real money, 2026-07-01)
Equity Value (#167): 51,555 shares sold, 62,642 shares bought, ~28 minutes, zero
`MANUAL_REQUIRED` flags, zero aborts, no cash-buffer scale-down needed (unlike Momentum's run,
which needed a 99.9% scale-down — this account had sufficient cash headroom).

### Files Updated (identical to Momentum repo, confirmed via diff)
- `Step Schwab Trading.py` — B3 fix + dashboard call-site wiring.
- `step_schwab_dashboard.py` — combined order table, measure-and-shrink `render()`, blotter.
- `tests/test_schwab_twap_engine.py` — same 2 new B3 regression tests. 33/33 passing.

### Constraints & Gotchas
- Same standing invariant as before: this repo's engine must stay byte-for-byte identical to the
  Momentum repo modulo documented account-specific constants (file paths, account name/number,
  VTV/VBR `ETF_OVERRIDES`). Re-verified via `diff` this session — clean.
- The dashboard fix was not applied retroactively to this run (already executing when found); it
  takes effect starting the next run in either repo.

### Context for Next Session
See Momentum repo's `llmchat.md` for the full session writeup, TWAP slice-sizing mechanics
explanation, and open questions (slice-count visibility in the dashboard, incremental per-slice
persistence still not implemented).

---
SESSION END: 2026-07-01 10:38 PDT | Agent: Cursor (Claude)
---
