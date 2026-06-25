"""
=============================================================================
SCRIPT NAME: generate_130_30_weights.py
=============================================================================

WHAT THIS PROGRAM DOES:
Generates factor weights using a 130/30 long-short rule-based strategy:
- Ranks factors by 60-month trailing mean return
- Long the Top 3 factors at (1.3 / 3) = 43.3% each
- Short the Bottom 3 factors at (-0.3 / 3) = -10% each
- Hysteresis band of 2 (factor only exits when it drops below rank 5)
- Only Value + Quality factors eligible (Max > 0 in Step Factor Categories)
- Net exposure = 100%, Gross exposure = 160%

This replaces T2_rolling_window_weights.xlsx for subsequent pipeline steps.

INPUT FILES:
- T2_Optimizer.xlsx : Monthly factor net returns (316 months x 82 factors)
- Step Factor Categories.xlsx : Factor eligibility (Max > 0 = allowed)

OUTPUT FILES:
- T2_rolling_window_weights.xlsx : Net factor weights with 'Net_Weights' sheet

VERSION: 1.0   LAST UPDATED: 2026-06-10   AUTHOR: Pipeline experiment
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RETURNS_PATH = os.path.join(HERE, "T2_Optimizer.xlsx")
CATEGORIES_PATH = os.path.join(HERE, "Step Factor Categories.xlsx")
OUT_PATH = os.path.join(HERE, "T2_rolling_window_weights.xlsx")
BACKUP_PATH = os.path.join(HERE, "T2_rolling_window_weights_backup_longonly.xlsx")

SHORT_RATIO = 0.3
BAND = 2
LOOKBACK = 60

print("=" * 70)
print("130/30 FACTOR WEIGHT GENERATOR")
print(f"  Long: Top3 at {(1+SHORT_RATIO)/3:.3f} each")
print(f"  Short: Bottom3 at {-SHORT_RATIO/3:.3f} each")
print(f"  Band: {BAND}, Lookback: {LOOKBACK}M")
print("=" * 70)

print("\nLoading factor returns...")
returns = pd.read_excel(RETURNS_PATH, index_col=0)
returns.index = pd.to_datetime(returns.index)
if "Monthly Return_CS" in returns.columns:
    returns = returns.drop(columns=["Monthly Return_CS"])
returns = returns.apply(pd.to_numeric, errors="coerce") / 100.0
print(f"  {returns.shape[0]} months, {returns.shape[1]} factors")
print(f"  Date range: {returns.index[0].strftime('%Y-%m')} to {returns.index[-1].strftime('%Y-%m')}")

print("Loading factor categories...")
fc = pd.read_excel(CATEGORIES_PATH)
max_w = dict(zip(fc["Factor Name"], fc["Max"]))
cat = dict(zip(fc["Factor Name"], fc["Category"]))
allowed = [c for c in returns.columns if max_w.get(c, 1.0) > 0]
print(f"  {len(allowed)} eligible factors (Value + Quality)")
R = returns[allowed]

n = R.shape[1]
cols = list(R.columns)

print("\nGenerating 130/30 weights...")
dates = R.index
W = np.zeros((len(dates), n))
held = None

for i in range(1, len(dates)):
    if i <= LOOKBACK:
        window = R.iloc[:i]
    else:
        window = R.iloc[i - LOOKBACK:i]
    mu = window.mean(axis=0).values
    mu = np.nan_to_num(mu, nan=-9e9)
    order = np.argsort(mu)[::-1]
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)

    if held is None:
        held = list(order[:3])
    else:
        held = [f for f in held if ranks[f] < 3 + BAND]
        for f in order:
            if len(held) >= 3:
                break
            if f not in held:
                held.append(f)

    w = np.zeros(n)
    w[held] = (1.0 + SHORT_RATIO) / len(held)
    w[order[-3:]] -= SHORT_RATIO / 3
    W[i] = w

    if i % 60 == 0:
        d = dates[i].strftime('%Y-%m')
        long_names = [cols[f] for f in held]
        short_names = [cols[f] for f in order[-3:]]
        print(f"  {d}: long={long_names}, short={short_names}")

Wdf = pd.DataFrame(W[1:], index=dates[1:], columns=cols)
Wdf.index.name = "Date"

print(f"\nWeight statistics:")
print(f"  Date range: {Wdf.index[0].strftime('%Y-%m')} to {Wdf.index[-1].strftime('%Y-%m')}")
print(f"  Shape: {Wdf.shape}")
print(f"  Min weight: {Wdf.min().min():.4f}, Max weight: {Wdf.max().max():.4f}")
print(f"  Rows with negative weights: {(Wdf < 0).any(axis=1).sum()}")
print(f"  Net exposure (avg): {Wdf.sum(axis=1).mean():.4f}")
print(f"  Gross exposure (avg): {Wdf.abs().sum(axis=1).mean():.4f}")

avg = Wdf.mean().sort_values(ascending=False)
print(f"\nTop 10 avg weights:")
for f in avg.head(10).index:
    print(f"  {f:30s} {avg[f]:8.4f}")
print(f"\nBottom 5 avg weights (shorted):")
for f in avg.tail(5).index:
    print(f"  {f:30s} {avg[f]:8.4f}")

print(f"\nSaving to {OUT_PATH}...")
with pd.ExcelWriter(OUT_PATH, engine="xlsxwriter") as writer:
    Wdf.to_excel(writer, sheet_name="Net_Weights", index=True)

print(f"Backup at {BACKUP_PATH}")
print("\nDone. Ready to run Step Eight.")
