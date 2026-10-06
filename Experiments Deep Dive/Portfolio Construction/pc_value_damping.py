#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: pc_value_damping.py   (Value repo / Experiments Deep Dive / Portfolio Construction)
=============================================================================

QUESTION: The Momentum repo adopted EMA(0.30) on Step Five's factor-weight
vector with its country-book cost term set to 0 (2026-10-05). Does the same
idea help the VALUE engine (Top-3 equal-weight, hysteresis EXIT_BAND=2,
kappa=1 cost HURDLE on swaps)?

Differences that matter: Value's engine is already damped twice (hysteresis
band + cost hurdle), holds 3 equal-weighted factors (a swap moves 1/3 of the
book, not all of it), and turns over ~3%/mo at the factor level vs ~25%/mo in
the Momentum QP. The held SET depends only on ranks, never on weight
magnitudes, so EMA on the output panel == EMA inside the loop exactly; no
production code is touched.

ARMS: ctrl (production), k0 (hurdle off, kappa=0), ema_a{0.25,0.30,0.50}
(hurdle kept), ema_a0.30_k0 (hurdle off + EMA). Country books via the
Step Eight band path -> liquidity cap -> Step 8.5 US dial (production target),
validated against the live T2_Final_Country_Weights.xlsx. Gross, active vs
equal weight.

GUARDS: ctrl factor panel == T2_rolling_window_weights.xlsx; ctrl capped
US-adjusted book == live file.

OUTPUT: runs/<timestamp>/PC_value_damping_results.xlsx, run.log

VERSION: 1.0  2026-10-05
=============================================================================
"""

import os
import sys
import time
import logging
import datetime as dt
import importlib.util
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

VALUE_ROOT = "/Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value"
MOM_PC = ("/Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy/"
          "Experiments Deep Dive/Portfolio Construction")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, VALUE_ROOT)
sys.path.insert(0, MOM_PC)

import pc_mv_experiment as pcm        # noqa: E402  (Step Eight book builder; ROOT patched below)
from pc_slicing import us_adjust      # noqa: E402  (Value Step 8.5 US dial)
import pc_variants as pcv             # noqa: E402  (stats)

pcm.ROOT = VALUE_ROOT
HOLDOUT_MONTHS = 24
EMA_ARMS = [0.25, 0.30, 0.50]
RUN_DIR = None


def log(msg):
    logging.info(msg)


def load_value_step_five():
    spec = importlib.util.spec_from_file_location(
        "value_sf", os.path.join(VALUE_ROOT, "Step Five FAST.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ema_panel(w, alpha):
    """EMA on the output factor-weight panel, damped vector carried forward,
    renormalized each month (same semantics as Momentum Step Five v3.0)."""
    out = w.copy().astype(float)
    prev = None
    for d in out.index:
        row = out.loc[d].to_numpy()
        if prev is not None:
            row = alpha * row + (1 - alpha) * prev
            row = row / row.sum()
        out.loc[d] = row
        prev = row
    return out


def main():
    global RUN_DIR
    t0 = time.time()
    RUN_DIR = os.path.join(HERE, "runs", dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(RUN_DIR, exist_ok=True)
    pcm.RUN_DIR = RUN_DIR
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.FileHandler(os.path.join(RUN_DIR, "run.log")),
                                  logging.StreamHandler(sys.stdout)])
    os.chdir(VALUE_ROOT)

    vsf = load_value_step_five()
    returns_df, eligible = vsf.load_and_prepare_data()
    log(f"Value Step Five: N_TOP={vsf.N_TOP} EXIT_BAND={vsf.EXIT_BAND} KAPPA={vsf.KAPPA} "
        f"WINDOW={vsf.WINDOW_SIZE}; eligible factors={len(eligible)}")
    costs_df = vsf.load_trading_costs(returns_df, eligible)

    # ---- factor-weight panels ----------------------------------------------
    panels = {}
    panels["ctrl"] = vsf.select_top3_weights(returns_df, eligible, costs_df)
    kappa_prod = vsf.KAPPA
    vsf.KAPPA = 0.0
    panels["k0"] = vsf.select_top3_weights(returns_df, eligible, costs_df)
    vsf.KAPPA = kappa_prod
    for a in EMA_ARMS:
        panels[f"ema_a{a:g}"] = ema_panel(panels["ctrl"], a)
    panels["ema_a0.3_k0"] = ema_panel(panels["k0"], 0.30)
    churn = {n: float(p.diff().abs().sum(axis=1).mean()) for n, p in panels.items()}

    # GUARD 1
    prod_w = pd.read_excel(os.path.join(VALUE_ROOT, "T2_rolling_window_weights.xlsx"), index_col=0)
    prod_w.index = pd.to_datetime(prod_w.index)
    rep = panels["ctrl"].reindex(prod_w.index)[prod_w.columns].astype(float)
    maxdiff = float((rep - prod_w.astype(float)).abs().max().max())
    guard1 = maxdiff < 1e-8
    log(f"GUARD1 ctrl vs T2_rolling_window_weights.xlsx: max|dw|={maxdiff:.3e} "
        f"{'PASS' if guard1 else 'FAIL'}")

    # ---- country books (Step Eight band path) ------------------------------
    _, _, fdf, ret, bench, _ = pcm.load_inputs()
    countries = list(ret.columns)
    from step_liquidity_cap import load_adv, apply_liquidity_cap
    adv = load_adv(os.path.join(VALUE_ROOT, "Experiments Deep Dive", "IBKR_Liquidity.xlsx"), countries)
    alphas = pd.read_excel(os.path.join(VALUE_ROOT, "T2_Country_Alphas.xlsx"))
    alphas["date"] = pd.to_datetime(alphas["date"])
    alphas = alphas.set_index("date")

    strategies, raw_books = {}, {}
    for name, p in panels.items():
        log(f"building Step Eight books for {name} ...")
        w_stamped = p.astype(float).shift(1).iloc[1:]
        book = pcm.build_books(w_stamped, fdf, countries)
        raw_books[name] = book
        capped, _ = apply_liquidity_cap(book.copy(), adv, 7_000_000, 0.20)
        strategies[name] = us_adjust(capped, alphas)

    # GUARD 2: ctrl production target == live file
    live = pd.read_excel(os.path.join(VALUE_ROOT, "T2_Final_Country_Weights.xlsx"),
                         sheet_name="All Periods", index_col=0)
    live.index = pd.to_datetime(live.index)
    common = strategies["ctrl"].index.intersection(live.index)
    bdiff = (strategies["ctrl"].loc[common] -
             live.loc[common].reindex(columns=countries).fillna(0.0)).abs()
    guard2 = float(bdiff.max().max()) < 1e-6
    log(f"GUARD2 ctrl book (cap + US dial) vs live file: max diff={float(bdiff.max().max()):.3e} "
        f"{'PASS' if guard2 else 'FAIL'}")

    # ---- evaluate ----------------------------------------------------------
    stamped = strategies["ctrl"].index
    realized = ret.dropna(how="all").index
    bm = bench["equal_weight"]
    real_dates = [d for d in stamped if d in realized and pd.notna(bm.get(d, np.nan))
                  and strategies["ctrl"].loc[d].sum() > 1e-9]
    periods = {"Full": real_dates,
               f"Holdout (last {HOLDOUT_MONTHS}m)": real_dates[-HOLDOUT_MONTHS:],
               "2017+": [d for d in real_dates if d >= pd.Timestamp("2017-01-01")],
               "2000-2016": [d for d in real_dates if d < pd.Timestamp("2017-01-01")]}

    rows, monthly = [], {}
    for name, Wdf in strategies.items():
        port = (Wdf.reindex(real_dates).fillna(0.0) * ret.loc[real_dates, countries].fillna(0.0)).sum(axis=1)
        active = port - bm.loc[real_dates]
        monthly[name] = active
        to = 0.5 * Wdf.diff().abs().sum(axis=1)
        for pn, pdates in periods.items():
            s_ = pcv.stats(active.loc[pdates], port.loc[pdates], bm.loc[pdates])
            if s_:
                s_.update({"Strategy": name, "Period": pn,
                           "AvgTurnover_%mo": 100 * to.reindex(pdates).mean(),
                           "FactorChurn_%mo": 100 * churn[name]})
                rows.append(s_)
    summ = pd.DataFrame(rows)
    summ = summ[["Strategy", "Period"] + [c for c in summ.columns if c not in ("Strategy", "Period")]]

    # switchover vs live at latest vintage
    d_live = stamped[-1]
    live_row = live.loc[d_live].reindex(countries).fillna(0.0) if d_live in live.index \
        else live.iloc[-1].reindex(countries).fillna(0.0)
    sw = pd.DataFrame([{"arm": n, "oneway_turnover_vs_live_%":
                        100 * 0.5 * float((W.loc[d_live] - live_row).abs().sum())}
                       for n, W in strategies.items()])

    out_x = os.path.join(RUN_DIR, "PC_value_damping_results.xlsx")
    with pd.ExcelWriter(out_x, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as xw:
        summ.to_excel(xw, sheet_name="Summary", index=False)
        pd.DataFrame(monthly).to_excel(xw, sheet_name="Monthly_Active_Returns")
        sw.to_excel(xw, sheet_name="Switchover", index=False)
        pd.DataFrame({"guard": ["G1", "G2"], "value": [f"{maxdiff:.3e}", f"{float(bdiff.max().max()):.3e}"],
                      "pass": [guard1, guard2]}).to_excel(xw, sheet_name="Guards", index=False)
    pd.to_pickle(strategies, os.path.join(RUN_DIR, "value_damping_books.pkl"))

    pd.set_option("display.width", 250)
    for pn in periods:
        sub = summ[summ["Period"] == pn]
        log(f"\n== {pn}\n" + sub[["Strategy", "Months", "ActiveRet_%yr(arith)", "TE_%yr", "IR",
                                   "MaxDD_active_%", "AvgTurnover_%mo", "FactorChurn_%mo"]]
            .round(2).to_string(index=False))
    log("\n== Switchover vs live (%s) ==\n%s" % (d_live.date(), sw.round(2).to_string(index=False)))
    log(f"GUARDS g1={guard1} g2={guard2}. Done in {time.time()-t0:.0f}s. Run dir: {RUN_DIR}")


if __name__ == "__main__":
    main()
