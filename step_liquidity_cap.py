"""
=============================================================================
SCRIPT NAME: step_liquidity_cap.py - Shared Liquidity (ADV) Position-Cap Utility
=============================================================================

INPUT FILES:  NONE directly (pure functions; load_adv reads an IBKR_Liquidity
              cache path you pass in).
OUTPUT FILES: NONE.

VERSION: 1.0   LAST UPDATED: 2026-06-24   AUTHOR: Devin (impact-cost study)

WHY THIS EXISTS (plain English):
At small AUM the dominant trading cost is MARKET IMPACT in thin single-country
ETFs (e.g. Denmark/EDEN ~ $0.3M ADV). The factor model has no idea those ETFs
are illiquid, so it will happily size a position you cannot trade. This module
caps each country's weight so that a FULL rotation of the position is at most
MAXPART of one day's dollar volume:

    cap_c   = MAXPART * ADV_c / AUM           (clipped to 1.0)
    w_c    <= cap_c                            (enforced by water-filling)

The cap is enforced by water-filling (fix names at their cap, redistribute the
freed weight only to UNCAPPED names, repeat) so the book stays long-only and
fully invested (sum = 1) with NO residual cap violations.

Validated in "T2 Factor Timing Fuzzy" (square-root impact model @ $7M, k=1):
MAXPART=0.20 lifted net active return ~+1.2%/yr vs uncapped (real Step Nine),
by raising gross return AND cutting impact cost. It is also basic risk
management (you should never hold what you cannot liquidate) independent of any
cost assumption.

This is THE single source of truth for the cap so Steps Eight in all sister
repos behave identically. Import it; do not re-implement.

USAGE:
    from step_liquidity_cap import load_adv, apply_liquidity_cap
    adv = load_adv("Experiments Deep Dive/IBKR_Liquidity.xlsx", list(W.columns))
    W_capped, report = apply_liquidity_cap(W, adv, aum=7_000_000, maxpart=0.20)

DEPENDENCIES: pandas, numpy
=============================================================================
"""

import os
import numpy as np
import pandas as pd


def load_adv(liq_path, countries):
    """Per-country dollar ADV from an IBKR_Liquidity cache (columns: Country, ADV_USD).
    Missing countries are filled with the cross-sectional median (logged by caller)."""
    if not os.path.exists(liq_path):
        raise FileNotFoundError(
            f"Liquidity cache not found: {liq_path}\n"
            "Generate it with 'Experiments Deep Dive/Step Tcost Impact Model.py' "
            "(pulls per-ETF ADV from IBKR), or disable the cap.")
    liq = pd.read_excel(liq_path)
    if "Country" not in liq.columns or "ADV_USD" not in liq.columns:
        raise ValueError(f"{liq_path} must have columns 'Country' and 'ADV_USD'.")
    adv = liq.set_index("Country")["ADV_USD"].reindex(countries)
    if adv.isna().any():
        adv = adv.fillna(adv.median())
    return adv


def _cap_row_long_only(w, cap, tol=1e-12, max_iter=200):
    """Water-fill one long-only weight row to satisfy w <= cap with sum = 1."""
    w = w.clip(lower=0.0).astype(float)
    s = w.sum()
    if s <= 0:
        return w
    w = w / s
    for _ in range(max_iter):
        over = w > cap + tol
        if not over.any():
            break
        w = w.where(~over, cap)
        free = ~over
        deficit = 1.0 - w.sum()
        if deficit <= tol or not free.any():
            break
        base = w.where(free, 0.0)
        if base.sum() > tol:
            w = w + base / base.sum() * deficit          # pro-rata to free names
        else:
            w = w + free.astype(float) / free.sum() * deficit
    return w


def apply_liquidity_cap(weights_df, adv, aum, maxpart, tol=1e-9):
    """Apply a per-country liquidity cap to a (dates x countries) weight frame.

    Parameters
    ----------
    weights_df : DataFrame  (index dates, columns country names, LONG-ONLY)
    adv        : Series      per-country dollar ADV (index = country names)
    aum        : float       portfolio value used to translate cap to weight units
    maxpart    : float       full rotation <= maxpart * one day's ADV

    Returns
    -------
    capped_df : DataFrame    same shape, each row long-only, sums to 1 (or 0 if the
                             input row was all-zero), with no cap violation.
    report    : DataFrame    per-country ADV, cap %, cap $, bind frequency, pre/post mean.
    """
    countries = list(weights_df.columns)
    adv = adv.reindex(countries)
    cap = (maxpart * adv / aum).clip(upper=1.0)

    neg_frac = float((weights_df.fillna(0.0) < -tol).to_numpy().mean())
    if neg_frac > 0.001:
        raise ValueError(
            f"apply_liquidity_cap: {neg_frac:.1%} of weights are negative; the "
            "long-only water-filling cap is not valid for a long-short book. "
            "Disable the cap or add a sign-aware variant.")

    capped = weights_df.copy().astype(float)
    for d, row in weights_df.iterrows():
        if row.abs().sum() <= tol:
            continue
        capped.loc[d] = _cap_row_long_only(row.fillna(0.0), cap).reindex(countries).values

    report = pd.DataFrame({
        "ADV_USD": adv,
        "Cap_%": cap * 100,
        "Cap_$": cap * aum,
        "Bind_Freq_%": weights_df.gt(cap, axis=1).mean() * 100,
        "Mean_Wt_Pre_%": weights_df.mean() * 100,
        "Mean_Wt_Post_%": capped.mean() * 100,
    }).sort_values("Cap_%")
    return capped, report
