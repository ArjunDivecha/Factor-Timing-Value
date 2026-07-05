# OpenWiki quickstart

This repository is a factor-timing and country-allocation research pipeline with a live Schwab trading extension. The main flow builds country/factor signals, converts them into country weights, evaluates performance, generates reports, and can optionally trade a target portfolio through a TWAP execution engine.

## Start here
- [Pipeline architecture](architecture/pipeline.md)
- [Strategy and domain model](domain/value-quality-strategy.md)
- [Trading and live execution](architecture/trading.md)
- [Operations runbook](operations/runbook.md)

## What this repo does
- Builds a sequential research pipeline from P2P scoring through final reporting.
- Uses factor momentum and factor/category constraints to construct a value/quality-oriented strategy.
- Converts factor weights into country weights and portfolio returns.
- Produces a final reporting layer with charts, PDFs, and Excel outputs.
- Includes a Schwab live-trading path with a TWAP safety layer, a terminal dashboard, and pytest coverage.

## Main entrypoints
- `Run_All_Pipeline.py` — runs the core pipeline scripts in order.
- `Run_Limited_Pipeline.py` — partial pipeline runner for later-stage reruns.
- `Step Zero Create P2P Scores.py` through `Step Ten Create Final Report.py` — core research chain.
- `Step Fourteen Target Optimization.py` and `Step Fourteen Target Optimization LongShort.py` — country optimization variants.
- `Step Schwab Trading.py` — live/dry-run Schwab execution engine.
- `tests/test_schwab_twap_engine.py` — safety and regression tests for the trading engine.

## Repository shape
- `Archive/` contains retired or superseded step scripts.
- `Experiments Deep Dive/` contains analysis and research experiments that should not alter production scripts.
- `ElasticNet/` appears to be a separate experimental model area.
- `outputs/` is used by the Schwab execution tooling for dated audit artifacts.
- Excel files in the repo root are the working data and outputs of the pipeline.

## Key source files to read next
- `CLAUDE.md` for command-level workflow guidance and current architecture notes.
- `AGENTS.md` for the most recent project-specific facts and caveats.
- `Run_All_Pipeline.py` to see the step order.
- `Step Five FAST.py` to understand the active factor-selection engine.
- `Step Schwab Trading.py` and `step_schwab_dashboard.py` for the live trading path.

## Documentation map
- [Pipeline architecture](architecture/pipeline.md): end-to-end flow and step responsibilities.
- [Strategy and domain model](domain/value-quality-strategy.md): why the strategy exists and how key rules work.
- [Trading and live execution](architecture/trading.md): Schwab TWAP behavior, safety rules, and test coverage.
- [Operations runbook](operations/runbook.md): how to run the pipeline and trading scripts safely.

## Notes for future agents
- Treat the factor pipeline and the Schwab trading path as separate domains with different risk profiles.
- When changing Step Five or Step Eight logic, check downstream file contracts before editing anything else.
- When changing live trading behavior, read the test suite first and preserve the safety assumptions documented there.
- Avoid editing archived scripts unless you are explicitly reviving an old experiment.
