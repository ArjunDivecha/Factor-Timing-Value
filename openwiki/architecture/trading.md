---
type: Reference
title: Trading and live execution
description: The Schwab live-trading path built around a TWAP engine and terminal dashboard. Documents the full CLI safety-flag surface, the fake-broker regression suite invariants, and the dry-run/live boundary.
---

# Trading and live execution

This repository includes a live Schwab execution path built around a TWAP engine and a terminal dashboard. This is distinct from the research pipeline ([Pipeline architecture](pipeline.md)): it consumes final target country weights and converts them into broker orders.

## Core components

### `Step Schwab Trading.py`
This is the production trading engine. It:
- reads target country weights from `T2_FINAL_T60_VALUE.xlsx`
- maps countries to ETF tickers through `AssetList.xlsx`
- loads liquidity information from `Experiments Deep Dive/IBKR_Liquidity.xlsx`
- reads Schwab credentials from external auth files/env vars
- computes rebalance trades for the selected account
- executes them with a homegrown TWAP routine
- writes dated audit artifacts to `outputs/`

The script is dry-run by default. Live trading requires `--live --confirm-live`.

Important safety behaviors documented in the script and the fake-broker regression suite in `tests/test_schwab_twap_engine.py`:
- sells execute before buys so cash is available
- market-order cleanup is guarded by spread checks (`--max-cleanup-spread-bps`); cleanup is skipped, not blindly submitted, when the quote fetch fails
- stale target weights are rejected for live runs (`--max-target-weights-age-days`, default 35.0); dry runs only warn
- live cash is refetched before buying; failures abort instead of falling back to a stale pre-sell cash number
- the engine tracks terminal order states (`FILLED`, `CANCELED`, `REJECTED`, `EXPIRED`, `REPLACED`) carefully to avoid double submission; a canceled order with a partial fill is carried forward as a true remainder, never misread as complete
- a submit exception raised after Schwab may have already accepted the order is reconciled via account history rather than blindly resubmitted (201-no-Location recovery path)
- SNAXX sweep value is excluded from allocatable equity and never traded, even when a SNAXX balance is swept in
- the buy plan is scaled against an equity-based buffer, not a cash-based one, so a SNAXX balance cannot silently under-invest the book
- carry-forward slice excess is accumulated, not overwritten, so a burst of failed slices does not silently lose shares; a single slice is capped at `--max-slice-carry-multiple` (default 3.0×) of its base size to prevent a giant catch-up order
- live runs are blocked outside US market hours (weekends, before open, after close, and too close to close); dry runs are exempt
- a per-day live marker is claimed before live trading; `--force-rerun` may overwrite it only after confirming via `account_orders` that no working orders remain on the account
- a preview-order rejection blocks the first slice submission rather than proceeding
- a quote failure for one symbol does not stall the other symbols in the leg

### `step_schwab_dashboard.py`
This module renders a Rich-based live terminal dashboard. It shows:
- aggregated per-symbol execution state
- a per-order blotter with slice-level details
- bid/ask/spread, limit price, VWAP, slippage, and status information

The dashboard exists to make live execution observable during TWAP slices and to preserve order-by-order detail that a blended summary would hide.

### `tests/test_schwab_twap_engine.py`
This pytest suite loads the real trading script directly and runs it against a fake broker (`FakeSchwabClient`) so the tests always exercise whatever is on disk, with no stale-copy risk. It verifies safety behavior under broker edge cases such as:
- partial fills on terminal orders, including a canceled-with-partial-fill that must not be misread as complete
- quote failures: cleanup is skipped when the quote fetch fails or the spread is too wide, and a bad quote for one symbol does not stall the others
- canceled orders with ambiguous final states are marked manual-required rather than resubmitted
- submit failures after acceptance are reconciled via account history (201-no-Location recovery); truly unrecoverable cases flag manual with no carry
- carried-forward slice accumulation is preserved (not overwritten) and capped by `--max-slice-carry-multiple`
- SNAXX equity handling: SNAXX value is excluded from allocatable equity and never traded; a zero balance is a no-op
- market-hours blocking for live runs (weekend, before open, after close, too close to close); dry runs are never blocked
- live-marker claim semantics: succeeds with no prior marker, refuses on conflict, and allows overwrite only under `--force-rerun` after confirming no working orders
- preview-order rejection blocks the first slice submission
- buy-plan scaling uses an equity-based buffer rather than a cash-based one

The suite's docstring enumerates seven specific bugs it guards against (filled/canceled misread, blind cleanup on quote failure, stale cash fallback, ambiguous-cancel resubmission, carry-cap excess loss, submit-exception-after-acceptance, SNAXX under-investment). Treat that list as the durable rationale for the guards above.

## CLI flags
`Step Schwab Trading.py` exposes its full configuration through argparse (see `parse_args()`). The dry-run/live boundary is controlled by `--live --confirm-live`; the remaining flags tune the safety parameters above.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--account-name` | `DEFAULT_ACCOUNT_NAME` | Schwab account to trade. |
| `--live` | off | Submit orders to Schwab (requires `--confirm-live`). |
| `--confirm-live` | off | Second safety confirmation for live trading. |
| `--twap-window` | 15 | TWAP window in minutes. |
| `--twap-slices` | 10 | Number of TWAP slices. |
| `--min-trade` | 1000.0 | Minimum trade size in dollars. |
| `--cash-buffer` | 0.03 | Cash buffer as a fraction of equity. |
| `--no-liquidity-cap` | off | Disable the ADV liquidity cap (see [Strategy and domain model](../domain/value-quality-strategy.md) for the shared `step_liquidity_cap.py` rule). |
| `--liq-maxpart` | 0.20 | Max fraction of daily ADV per position. |
| `--max-unfilled-sell-pct` | 0.05 | Abort the BUY phase if more than this fraction of planned sell notional remains unfilled. |
| `--max-cleanup-spread-bps` | 50.0 | Skip end-of-window market-order cleanup for any symbol whose spread exceeds this many bps. |
| `--max-slice-carry-multiple` | 3.0 | Cap any single TWAP slice at this multiple of its base per-slice size. |
| `--max-target-weights-age-days` | 35.0 | Refuse to trade live if `T2_FINAL_T60_VALUE.xlsx` is older than this; dry runs only warn. |
| `--force-rerun` | off | Allow a live re-run despite an existing today-marker, after confirming no working orders via `account_orders`. |
| `--notify` | off | Send iMessage notifications. |

## Why this matters
The trading path is a separate risk domain from the research pipeline. Bugs here can create real trades, so the code is intentionally defensive and the test suite is designed to simulate broker failures rather than happy-path trading only.

This area was hardened in phases: the TWAP execution engine was added first, followed by safety hardening with a fake-broker regression suite, a SNAXX sweep-value handling fix, and a dashboard overflow fix tied to the first live run. (The per-phase commit hashes no longer appear in the repository's squashed history; treat the behaviors above as the durable record.)

## Change guidance for future agents
- Read `tests/test_schwab_twap_engine.py` before editing the engine; it encodes the safety contract.
- Do not weaken live-run guards unless you can explain the broker failure mode being addressed.
- If you change the dashboard, keep the order blotter and summary table semantics distinct.
- When editing live execution, check whether the change affects the dry-run path, not just live mode.
- Keep all sensitive credentials outside the repo; this code expects external auth files and token persistence.

## Source references
- `Step Schwab Trading.py`
- `step_schwab_dashboard.py`
- `tests/test_schwab_twap_engine.py`
- `AGENTS.md`
