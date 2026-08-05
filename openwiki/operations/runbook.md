---
type: Reference
title: Operations runbook
description: Practical commands, safety checks, output artifacts, and testing guidance for running the research pipeline and the Schwab trading engine in the Factor-Timing-Value repository.
---

# Operations runbook

This page captures the practical commands and checks that matter most when working in the repository.

## Common commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the main pipeline
```bash
python "Run_All_Pipeline.py"
```

### Run the limited pipeline
```bash
python "Run_Limited_Pipeline.py"
```

### Run individual steps
Use the command list in `CLAUDE.md`. The key scripts are:
- `Step Zero Create P2P Scores.py`
- `Step One Create T2Master.py`
- `Step Two Create Normalized Tidy.py`
- `Step Three Top20 Portfolios Fast.py`
- `Step Four Create Monthly Top20 Returns FAST.py`
- `Step Five FAST.py`
- `Step Six Point Five.py`
- `Step Eight Write Country Weights.py`
- `Step Nine Calculate Portfolio Returns.py`
- `Step Ten Create Final Report.py`
- `Step Fourteen Target Optimization.py`

### Run the Schwab trading engine
Dry run:
```bash
python "Step Schwab Trading.py"
```

Live mode:
```bash
python "Step Schwab Trading.py" --live --confirm-live
```

## Safety checks before changing or running live trading
- Confirm the target workbook is fresh enough; the script enforces staleness checks.
- Confirm the account name and account type are what you expect.
- Confirm Schwab credentials and token persistence are configured outside the repo.
- Confirm `outputs/` exists or will be created for audit artifacts.
- Verify the latest tests in `tests/test_schwab_twap_engine.py` still pass after changes.

## Output artifacts to watch
The pipeline and trading scripts create many root-level artifacts. The most important ones are:
- `T2_Top_20_Exposure.csv` — the Step Three cross-step exposure contract consumed by Step Six, Step Six Point Five, and the optimizers.
- `T2_rolling_window_weights.xlsx`
- `T2_strategy_statistics.xlsx`
- `T2_Country_Top_Alphas.xlsx`
- `T2_Final_Country_Weights.xlsx`
- `T2_Final_Portfolio_Returns.xlsx`
- `T2_Strategy_Report_Comprehensive_*.pdf`
- `outputs/schwab_trade_plan_*.xlsx`
- `outputs/schwab_execution_log_*.xlsx`
- `outputs/schwab_live_marker_*.json`

## Repository maintenance notes
- `Archive/` holds retired or superseded scripts and should usually be left alone.
- `Experiments Deep Dive/` is for sandbox work and model experiments; do not put exploratory logic back into production scripts unless it is intentionally promoted.
- Untracked root files such as `optimization_log.txt` and `T2_Asset_Class_MissingDataLog.txt` are runtime artifacts in this workspace snapshot and may not be intended for documentation changes.

## Testing guidance
For trading-related edits, run:
```bash
python -m pytest tests/test_schwab_twap_engine.py -v
```

If you change factor-selection or country-weight logic, a full pipeline rerun may be necessary to verify the downstream output contracts.

## Source references
- `Run_All_Pipeline.py`
- `Run_Limited_Pipeline.py`
- `CLAUDE.md`
- `Step Schwab Trading.py`
- `tests/test_schwab_twap_engine.py`
- `requirements.txt`
