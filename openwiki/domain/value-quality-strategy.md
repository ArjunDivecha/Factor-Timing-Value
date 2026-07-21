---
type: "Reference"
title: "Strategy and domain model"
openwiki_generated: true
---

# Strategy and domain model

This repository implements a value/quality-oriented factor timing system. The learned project notes in `AGENTS.md` are important here because they describe the active production intent more precisely than the filenames alone.

## Strategy intent
The active repository variant is the value/quality branch of the broader T2 factor timing work. Its core behavior is:
- use factor momentum as the predictive signal
- rotate into quality factors when value underperforms
- preserve the value orientation rather than replacing it with a generic momentum strategy

The user preference notes also emphasize that annualized return is the primary objective, with turnover as a secondary concern.

## Factor universe rules
A crucial Step Five rule is the factor-eligibility whitelist in `Step Factor Categories.xlsx`:
- factors with `Max > 0` are eligible
- factors with `Max = 0` must be excluded from the optimization universe
- factors missing from the whitelist should default to `Max = 0.0`, not to an implicit include-all fallback

This rule exists because the strategy universe is intentionally narrower than the full factor list. It was a recurring bug source in Step Five variants, so any future Step Five change should verify the filter explicitly.

## Active Step Five engine
`Step Five FAST.py` is currently the production factor-selection engine. `AGENTS.md` explains that it now contains the Top-3 60-month momentum + hysteresis + trading-cost-hurdle logic that used to live in a separate script.

The active engine behavior is:
- compute 60-month trailing factor momentum
- hold the top 3 factors equal-weighted
- retain held factors using a hysteresis band
- only swap a held factor if the expected edge clears a trading-cost hurdle
- use the per-factor monthly cost vector from `T2_Trading_Cost.xlsx`

The intent is to reduce churn without giving up the dominant factor signal.

## Country mapping and band logic
The country allocation logic depends on a shared fuzzy-band module:
- `step_fuzzy_bands.py` is the canonical implementation of band eligibility and deterministic tie-breaking
- Step Four and Step Eight use the same band rules so factor-to-country conversion stays consistent

This module matters because Step Four builds factor returns, while Step Eight consumes similar logic to build investable country weights. If those rules diverge, Step Nine’s realized return chain can drift from the optimizer’s assumptions.

## Liquidity and trading cost controls
Two additional risk controls are part of the strategy domain:
- `step_liquidity_cap.py` caps country weights by ADV to avoid infeasible positions in thin ETFs
- `step_five_multiwindow_stats.py` provides multi-window performance logging for Step Five so recent performance can be compared with the full sample

`AGENTS.md` notes that liquidity control was added because market impact, not just spread, is the dominant risk at the live AUM level.

## Optimization variants
`Step Fourteen Target Optimization.py` is the long-only country optimizer. `Step Fourteen Target Optimization LongShort.py` removes the long-only constraint to express true 130/30 style weights.

The existence of both scripts is important:
- the long-only version strips negative weights from the Step Eight output
- the long-short version is the correct choice when the goal is to preserve the sign of country allocations end-to-end

## Comparison and analysis scripts
Useful supporting scripts in this domain include:
- `Step Nine Compare Alpha Mcap Strategies.py` — compares Step Eight country weights with alpha × MCAP alternatives
- `Step Eight Basket Smoothing Lab.py` — experiments with smoother country weights
- `Step Eight Turnover Decomposition.py` — splits turnover into wanted vs avoidable components
- `step_five_multiwindow_stats.py` — reports step-five performance over multiple horizons

## Change guidance for future agents
- Preserve the value-first strategy intent unless the user explicitly asks for a different strategy family.
- Treat whitelist behavior, liquidity caps, and trading cost vectors as core strategy rules rather than optional embellishments.
- If you change factor selection, check both Step Five and Step Six Point Five because the latter depends on the former’s output shape.
- If you change country-weight construction, verify Step Nine and the optimizer scripts still agree on month alignment and row/column conventions.

## Source references
- `AGENTS.md`
- `Step Five FAST.py`
- `Step Factor Categories.xlsx`
- `step_fuzzy_bands.py`
- `step_liquidity_cap.py`
- `step_five_multiwindow_stats.py`
- `Step Fourteen Target Optimization.py`
- `Step Fourteen Target Optimization LongShort.py`
- `Step Nine Compare Alpha Mcap Strategies.py`
- `Step Eight Basket Smoothing Lab.py`
- `Step Eight Turnover Decomposition.py`
