"""
=============================================================================
SCRIPT NAME: generate_top3_longonly_weights.py
=============================================================================

WHAT THIS PROGRAM DOES:
Generates factor weights using Top3 equal-weight with hysteresis band:
- Ranks factors by 60-month trailing mean return
- Long Top 3 at 33.3% each (long-only, no short)
- Hysteresis band of 2 (factor only exits when it drops below rank 5)
- Only Value + Quality factors eligible (Max > 0 in Step Factor Categories)

INPUT FILES:
- T2_Optimizer.xlsx : Monthly factor net returns
- Step Factor Categories.xlsx : Factor eligibility

OUTPUT FILES:
- T2_rolling_window_weights.xlsx : Net_Weights sheet with factor weights

VERSION: 1.0   LAST UPDATED: 2026-06-10
=============================================================================
"""

import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RETURNS_PATH = os.path.join(HERE, "T2_Optimizer.xlsx")
CATEGORIES_PATH = os.path.join(HERE, "Step Factor Categories.xlsx")
OUT_PATH = os.path.join(HERE, "T2_rolling_window_weights.xlsx")

BAND = 2
LOOKBACK = 60

print("=" * 70)
print("TOP3 LONG-ONLY EQUAL-WEIGHT (band=2)")
print("=" * 70)

print("\nLoading factor returns...")
returns = pd.read_excel(RETURNS_PATH, index_col=0)
returns.index = pd.to_datetime(returns.index)
if "Monthly Return_CS" in returns.columns:
    returns = returns.drop(columns=["Monthly Return_CS"])
returns = returns.apply(pd.to_numeric, errors="coerce") / 100.0

print("Loading factor categories...")
fc = pd.read_excel(CATEGORIES_PATH)
max_w = dict(zip(fc["Factor Name"], fc["Max"]))
cat = dict(zip(fc["Factor Name"], fc["Category"]))
allowed = [c for c in returns.columns if max_w.get(c, 1.0) > 0]
print(f"  {len(allowed)} eligible factors")
R = returns[allowed]

n = R.shape[1]
cols = list(R.columns)
dates = R.index

W = np.zeros((len(dates), n))
held = None

for i in range(1, len(dates)):
    window = R.iloc[:i] if i <= LOOKBACK else R.iloc[i - LOOKBACK:i]
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
    w[held] = 1.0 / len(held)
    W[i] = w

    if i % 60 == 0:
        d = dates[i].strftime('%Y-%m')
        long_names = [cols[f] for f in held]
        print(f"  {d}: long={long_names}")

Wdf = pd.DataFrame(W[1:], index=dates[1:], columns=cols)
Wdf.index.name = "Date"

print(f"\nWeight statistics:")
print(f"  Date range: {Wdf.index[0].strftime('%Y-%m')} to {Wdf.index[-1].strftime('%Y-%m')}")
print(f"  Shape: {Wdf.shape}")
print(f"  Rows with any weight > 0: {(Wdf > 0.01).any(axis=1).sum()}")
print(f"  Avg active factors: {(Wdf > 0.005).sum(axis=1).mean():.1f}")

avg = Wdf.mean().sort_values(ascending=False)
print(f"\nTop 10 avg weights:")
for f in avg.head(10).index:
    print(f"  {f:30s} {avg[f]:8.4f}")

print(f"\nSaving to {OUT_PATH}...")
with pd.ExcelWriter(OUT_PATH, engine="xlsxwriter") as writer:
    Wdf.to_excel(writer, sheet_name="Net_Weights", index=True)

print("Done.")
