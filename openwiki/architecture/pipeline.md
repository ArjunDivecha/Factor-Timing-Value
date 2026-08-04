---
type: Reference
title: Pipeline architecture
description: End-to-end research pipeline flow for the Factor-Timing-Value repository, covering step order, per-stage outputs, and the cross-step file contracts that downstream consumers depend on.
---

# Pipeline architecture

The research pipeline is a stepwise script flow. Most scripts are standalone executables that read and write Excel/CSV/PDF artifacts in the repository root, so the contract between steps matters more than any individual script implementation.

## End-to-end flow
The main sequence is reflected in `Run_All_Pipeline.py` and the command list in `CLAUDE.md`:

1. `Step Zero Create P2P Scores.py`
2. `Step One Create T2Master.py`
3. `Step Two Create Normalized Tidy.py`
4. `Step Two Point Five Create Benchmark Rets.py`
5. `Step Three Top20 Portfolios Fast.py`
6. `Step Four Create Monthly Top20 Returns FAST.py`
7. `Step Five FAST.py`
8. `Step Six Create Country alphas from Factor alphas.py`
9. `Step Seven Visualize Factor Weights.py`
10. `Step Eight Write Country Weights.py`
11. `Step Eight Point Five Write Country Weights US Adjustment.py`
12. `Step Nine Calculate Portfolio Returns.py`
13. `Step Ten Create Final Report.py`
14. `Step Fourteen Target Optimization.py`
15. `Step Fifteen Market Regime Analysis.py`
16. `Step Sixteen Market Regime Analysis.py`
17. `Step Seventeen Market Regime Analysis.py`
18. `Step Eighteen Asset Class Charts.py`
19. `Step Twenty PORCH.py`
20. `Step Twenty One Master Report.py`
21. `Step FINALFINAL.py`

`Run_Limited_Pipeline.py` is a shorter runner for later-stage reruns.

## What each stage produces

### Data preparation
- **Step Zero**: builds historical P2P scores from country ETF prices.
- **Step One**: creates `T2 Master.xlsx`, the central data workbook used by later steps.
- **Step Two**: normalizes and tidies the master data.
- **Step Two Point Five**: creates benchmark return series.

### Factor construction and optimization
- **Step Three**: forms top-20 factor portfolios. It writes three artifacts: `T2 Top20.xlsx` (performance tables sorted by IR with a `Full_Sample` sheet plus trailing `Trailing_1Y` / `Trailing_3Y` / `Trailing_5Y` sheets on the benchmark calendar), `T2 Top20.pdf` (cumulative excess-return charts), and `T2_Top_20_Exposure.csv` (monthly country weights, 0–1). The exposure CSV is the cross-step contract consumed by Step Six, Step Six Point Five, Step Fourteen, and the market-regime scripts (Fifteen/Sixteen/Seventeen).
- **Step Four**: turns those portfolios into monthly returns.
- **Step Five**: selects factor weights over time and writes `T2_rolling_window_weights.xlsx` plus summary stats.
- **Step Six**: converts factor weights into country alphas using the full factor set.
- **Step Six Point Five**: same idea, but gated to Step Five-selected factors only.

### Country allocation and return realization
- **Step Seven**: visualizes factor weights.
- **Step Eight**: converts factor weights into country weights.
- **Step Eight Point Five**: applies a US-specific adjustment variant.
- **Step Nine**: computes portfolio returns from country weights.
- **Step Ten**: assembles the final report artifacts.

### Downstream analysis and reporting
- **Step Fourteen**: optimizes country weights.
- **Steps Fifteen-Seventeen**: market regime analyses.
- **Step Eighteen**: asset class charts.
- **Step Twenty** and **Step Twenty One**: higher-level report assembly.
- **Step FINALFINAL**: final wrap-up artifact generation.

## The most important file contracts

### Step Five output contract
`Step Five FAST.py` is the key upstream contract for the rest of the pipeline. It writes:
- `T2_rolling_window_weights.xlsx`
- `T2_strategy_statistics.xlsx`
- `T2_factor_weight_heatmap.pdf`
- `T2_strategy_performance.pdf`

The output format must stay compatible with downstream scripts, especially Step Six, Step Eight, Step Nine, and Step Fourteen.

`AGENTS.md` records two important Step Five facts:
- The current active production engine is the Top-3 60-month momentum + hysteresis + trading-cost hurdle variant merged into `Step Five FAST.py`.
- Factor eligibility is driven by `Step Factor Categories.xlsx`; only factors with `Max > 0` are allowed into the optimization universe.

### Step Three output contract
`Step Three Top20 Portfolios Fast.py` produces `T2_Top_20_Exposure.csv`, the wide-format monthly country exposure matrix (0–1 weights, not binary) that the rest of the country chain depends on. Downstream readers load it directly:
- `Step Six Create Country alphas from Factor alphas.py` and `Step Six Point Five.py` multiply exposures by factor alphas.
- `Step Fourteen Target Optimization.py` (and the LongShort variant) read the same exposure file.
- The market-regime scripts (Fifteen/Sixteen/Seventeen) consume it for factor attribution.

`T2 Top20.xlsx` carries the IR-sorted performance tables; its `Full_Sample` sheet is the default read for Step Ten's report table, while the trailing `Trailing_1Y` / `Trailing_3Y` / `Trailing_5Y` sheets provide windowed views on the benchmark calendar. Changing the exposure CSV's column order, date format, or weight semantics breaks Step Six and the optimizers, so treat it as a stable contract.

### Step Eight and Step Nine contract
`step_fuzzy_bands.py` exists to keep factor-to-country band logic consistent across Step Four and Step Eight. That matters because Step Four produces factor-level returns while Step Eight consumes the same logic for country weights.

### Step Fourteen contract
`Step Fourteen Target Optimization.py` is the long-only optimizer. `Step Fourteen Target Optimization LongShort.py` removes the long-only constraint so the pipeline can express true long-short country weights.

## Change guidance for future agents
- If you change any step’s output file name, sheet name, or index shape, trace every downstream consumer before editing.
- Preserve date alignment and month-end/month-start conventions; several scripts standardize dates in different ways.
- When changing factor eligibility logic, check `Step Factor Categories.xlsx` handling first.
- When changing the Step Eight / Step Nine chain, use `step_fuzzy_bands.py` as the canonical rule source rather than duplicating logic.

## Source references
- `Run_All_Pipeline.py`
- `Run_Limited_Pipeline.py`
- `CLAUDE.md`
- `AGENTS.md`
- `Step Three Top20 Portfolios Fast.py`
- `Step Five FAST.py`
- `Step Six Point Five.py`
- `step_fuzzy_bands.py`
- `Step Fourteen Target Optimization.py`
