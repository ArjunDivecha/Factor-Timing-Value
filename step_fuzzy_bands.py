"""
=============================================================================
SCRIPT NAME: step_fuzzy_bands.py - THE Shared Fuzzy-Band Utility
=============================================================================

INPUT FILES:  NONE (pure functions - no file I/O)
OUTPUT FILES: NONE (pure functions - no file I/O)

VERSION: 1.0
LAST UPDATED: 2026-06-11
AUTHOR: Arjun Divecha

DESCRIPTION (10th-grader version):
This module is the SINGLE SOURCE OF TRUTH for how the T2 pipeline turns
factor scores into country portfolio weights (the fuzzy "top 15-25%" band).

Two production scripts translate between factors and countries:
  - Step Four  (factor -> return):  builds each factor's country portfolio,
    earns its return, writes T2_Optimizer.xlsx for the optimizer.
  - Step Eight (factor -> weights): combines the optimizer's factor weights
    into one country portfolio for actual investment.

Historically each script had its OWN copy of the band logic, and tiny
differences (most importantly: Step Four only includes countries that have
a RETURN that month, Step Eight didn't check) made Step Nine's realized
returns drift from Step Five's optimizer returns. Putting the band math in
one shared module makes the two directions identical BY CONSTRUCTION.

THE RULES (encoded here, used by both directions):
1. ELIGIBILITY: a country enters a factor's band only if it has BOTH a
   factor score AND a 1-month return that month (no return = can't be
   traded = not in the portfolio). Use `eligible_scores()` to apply this.
2. RANKING: countries are ranked by score, descending for normal factors
   (positive weight), ascending for the short/negative direction. TIES are
   broken ALPHABETICALLY BY COUNTRY NAME - deterministic and independent of
   input row order, so both directions always agree (the old scripts broke
   ties by input row order, which differed between Step Four's CSV order
   and Step Eight's alphabetical pivot and silently changed band members).
3. BAND: rank percentile < 15% gets full weight 1.0; 15-25% tapers
   linearly to zero; > 25% gets nothing.
4. NORMALIZATION: each factor's band weights are scaled to sum to 1.0.

USAGE:
    from step_fuzzy_bands import eligible_scores, band_weights, band_matrix

    # one factor, one month (Step Four style)
    w = band_weights(eligible_scores(scores, returns))        # Series

    # all factors at once, one month (Step Eight style)
    B = band_matrix(pivot.where(returns.notna(), axis=0))     # DataFrame

DEPENDENCIES: pandas, numpy
=============================================================================
"""

import numpy as np
import pandas as pd

# Fuzzy band constants - same values as Steps Three/Four/Eight have always used
SOFT_BAND_TOP = 0.15     # rank percentile below this => full weight
SOFT_BAND_CUTOFF = 0.25  # rank percentile above this => zero weight


def fuzzy_taper(rank_pct: np.ndarray) -> np.ndarray:
    """
    Convert rank percentiles into raw band weights.

    < 15%  -> 1.0 (full weight)
    15-25% -> linear taper from 1.0 down to 0.0
    > 25%  -> 0.0
    NaN    -> 0.0 (a NaN percentile means the country is ineligible)
    """
    rank_pct = np.asarray(rank_pct, dtype=float)
    safe = np.nan_to_num(rank_pct, nan=1.0)   # NaN lands past the cutoff
    full = (safe < SOFT_BAND_TOP).astype(float)
    in_band = (safe >= SOFT_BAND_TOP) & (safe <= SOFT_BAND_CUTOFF)
    taper = 1.0 - (safe - SOFT_BAND_TOP) / (SOFT_BAND_CUTOFF - SOFT_BAND_TOP)
    return full + np.where(in_band, taper, 0.0)


def eligible_scores(scores: pd.Series, returns: pd.Series) -> pd.Series:
    """
    Apply THE eligibility rule: a country keeps its score only if it also
    has a return this month. Index-aligned; countries missing from
    `returns` are dropped too.
    """
    aligned = returns.reindex(scores.index)
    return scores.where(aligned.notna())


def _deterministic_ranks(scores: pd.Series, ascending: bool) -> np.ndarray:
    """
    Positional ranks 1..n with ORDER-INDEPENDENT tie-breaking:
    primary key = score (desc or asc), secondary key = country name (A->Z).
    Returns ranks aligned to scores' index order.
    """
    names = scores.index.astype(str).to_numpy()
    name_order = np.argsort(names, kind="stable")        # secondary key
    vals = scores.to_numpy(dtype=float)
    key = vals if ascending else -vals
    order = name_order[np.argsort(key[name_order], kind="stable")]
    ranks = np.empty(len(vals), dtype=float)
    ranks[order] = np.arange(1, len(vals) + 1)
    return ranks


def band_weights(scores: pd.Series, ascending: bool = False) -> pd.Series:
    """
    Band weights for ONE factor in ONE month.

    Args:
        scores: factor scores indexed by country (NaN = ineligible).
        ascending: False -> top-ranked countries (normal factors);
                   True  -> bottom-ranked countries (negative weights).

    Returns:
        Country weights summing to 1.0 over eligible countries.
        Empty Series if no country is eligible.
    """
    s = scores.dropna()
    n = len(s)
    if n == 0:
        return pd.Series(dtype=float)
    rank_pct = _deterministic_ranks(s, ascending) / n
    w = fuzzy_taper(rank_pct)
    total = w.sum()
    if total <= 0:
        return pd.Series(dtype=float)
    return pd.Series(w / total, index=s.index)


def band_matrix(pivot: pd.DataFrame, ascending_cols=None) -> pd.DataFrame:
    """
    Band weights for MANY factors in ONE month.

    Built column-by-column on top of band_weights so the tie-breaking is
    GUARANTEED identical to the single-factor path (one code path, no
    drift). 82 factors x 34 countries per month is trivial work.

    Args:
        pivot: Countries x Factors matrix of scores. Mask ineligible
               countries to NaN BEFORE calling (e.g. with
               `pivot.where(returns.notna(), axis=0)`).
        ascending_cols: optional iterable of column names to rank
               ascending (the short direction). All others descending.

    Returns:
        Countries x Factors DataFrame; each column with at least one
        eligible country sums to 1.0, columns with none are all zero.
        (Check column availability with `df.sum() > 0`.)
    """
    asc = set(ascending_cols) if ascending_cols is not None else set()
    out = pd.DataFrame(0.0, index=pivot.index, columns=pivot.columns)
    for col in pivot.columns:
        w = band_weights(pivot[col], ascending=(col in asc))
        if not w.empty:
            out.loc[w.index, col] = w.values
    return out
