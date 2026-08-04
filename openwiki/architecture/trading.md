---
type: Reference
title: Trading and live execution
description: The Schwab live-trading path built around a TWAP engine and terminal dashboard. Documents safety behaviors, the fake-broker regression suite, and the dry-run/live boundary.
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

Important safety behaviors documented in the script and tests:
- sells execute before buys so cash is available
- market-order cleanup is guarded by spread checks
- stale target weights are rejected for live runs
- live cash is refetched before buying; failures abort instead of guessing
- the engine tracks terminal order states carefully to avoid double submission
- SNAXX sweep value is excluded from allocatable equity after the B3 fix

### `step_schwab_dashboard.py`
This module renders a Rich-based live terminal dashboard. It shows:
- aggregated per-symbol execution state
- a per-order blotter with slice-level details
- bid/ask/spread, limit price, VWAP, slippage, and status information

The dashboard exists to make live execution observable during TWAP slices and to preserve order-by-order detail that a blended summary would hide.

### `tests/test_schwab_twap_engine.py`
This pytest suite loads the real trading script directly and runs it against a fake broker. It verifies safety behavior under broker edge cases such as:
- partial fills on terminal orders
- quote failures
- canceled orders with ambiguous final states
- submit failures after acceptance
- carried-forward slice accumulation
- SNAXX equity handling

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
