# ARJUN.md — Product memo: T2 Factor Timing Fuzzy **Value**

*Fable 5, 2026-07-06. Blunt, ranked by value ÷ effort.*

## What this repo is worth

**Alive and load-bearing — this trades real money.** Account #167 ("Equity Value") ran its first
hardened-engine live TWAP rebalance on 2026-07-01, clean. It is the Value/Quality sibling of
`../T2 Factor Timing Fuzzy` (Momentum, #090); the two share the Schwab execution engine and are
kept byte-identical by hand. Not dormant, not superseded. The strategy itself is settled (Top-3
momentum + hysteresis + κ=1 cost hurdle, 4.23%/yr net, value orientation preserved) — you don't
need more signal research here. **The real exposure is operational**: a live-money pipeline whose
research/data layer has zero tests, and a trading engine duplicated across two repos by manual
diffing. That's where your hours should go, not another factor experiment.

## Extensions ranked by value ÷ effort

1. **Extract the Schwab engine into one shared, tested package** *(highest leverage).*
   `Step Schwab Trading.py` + `step_schwab_dashboard.py` must stay byte-for-byte identical across
   #167 and #090 — your own llmchat calls this a "standing invariant" enforced by re-diffing every
   session. That is a real-money bug waiting to happen (one repo gets a fix, the other silently
   doesn't). Package it (`t2_execution`), each repo imports it with an account config object
   (account name, `ETF_OVERRIDES`, paths); the 33-test harness moves into the package once.
   *Why now:* you're live on both accounts — every future safety fix currently doubles work and risk.
   *First step:* scaffold the package, move engine + tests, wire both repos to import it.
   *Reuse:* the existing pytest harness; Codex CLI for the mechanical move under your review.

2. **Run the P0 data-quality contract (it's ready to hand off).**
   `FABLE.md` contains a validated Divecha contract locking the forward-only + regime-neutralization
   invariants of `detect_regime_breaks_sheet`. This is the guard between corrupt Bloomberg data and
   your live trades, and it's untested today. *Why now:* trivial to start, high downside if it
   regresses. *First step:* save the embedded contract as `dq_regime_guard_tests.spec.md` and run the
   handoff prompt (delegate to Codex/Sonnet — the gates do the judging). *Reuse:* Divecha skill.

3. **Monthly pre-trade data-quality acknowledgement.**
   The guard already writes `T2_regime_break_log.xlsx`, but nothing forces you to look before you
   trade. Turn it into a one-page red/green summary (or an iMessage via the notification plumbing
   already in `Step Schwab Trading.py`) that you must acknowledge before the live rebalance fires.
   *Why now:* a silent bad-data month = a bad real trade. *First step:* read the regime log at the
   top of the trading run, emit a summary, block on a keypress in live mode. *Reuse:* existing
   iMessage/TCA notification path. ~half a day.

4. **Independent replication on the QuantConnect 34-ETF harness.**
   Your backtest is a bespoke Excel pipeline; a second, independent LEAN implementation on the same
   34-country universe cross-checks the 4.23%/yr headline and exposes pipeline bugs a single
   implementation can't see. *Why now:* cheap insurance on a live strategy. *Reuse:* the `backtest`
   skill directly. Medium effort.

5. **Regime-aware gross sizing from work you've already done.**
   You compute GMM regimes (Steps 15-17, `T2_GMM_Regime_Analysis.xlsx`) but don't feed them into
   live sizing. Scale gross exposure down in the identified bad regime. *Reuse:* existing GMM outputs
   + the Schwab engine's cash-buffer logic. Medium value, low-ish effort since the regime labels exist.

6. **Move the "learned facts" into the personal-knowledge MCP.**
   The hand-maintained "Learned Preferences/Facts" in CLAUDE/AGENTS drift (I found two stale claims
   this pass). Durable, queryable knowledge (value orientation, metric = return, no-lookahead,
   Top-3-here-not-in-Fuzzy) belongs in the personal-knowledge MCP so every agent gets it right
   without you re-editing markdown. *Reuse:* personal-knowledge MCP. Quick-ish.

## Quick wins (< 1 hour each)

- **Fix `Step Five FAST.py`'s identity crisis** — the active engine's header says "Step Five Top3
  Tcost.py". In a live-money repo an operator can run the wrong thing off a wrong name. Rename the
  file or fix the header so they agree. (10 min.)
- **Fix `Run_Limited_Pipeline.py:2`** — its docstring still says "Run_All_Pipeline.py". (5 min.)
- **Kick off P0** — the first pytest case (forward-only) is well under an hour and immediately
  protects real trades.

## What NOT to do

- **Don't unify the *signal* engine across Value and Momentum.** They differ on purpose: Top-3
  equal-weight wins in this 36-factor correlated universe; concentration wins in the 82-factor
  Fuzzy universe. Porting the Top-3 fix to Fuzzy costs ~2.6%/yr (documented). Unify the *execution*
  layer (#1 above); keep the signal layers separate.
- **Don't spin up another T2 variant fork.** The ecosystem already has Fuzzy, Fuzzy Value, Fuzzy
  Daily, Fuzzy Long Short, T2-Factor-Timing-Daily, Factor Timing Codex, T3. Each new fork multiplies
  the parity-maintenance burden that already threatens the shared live trading engine. If you need a
  new variant, make it a *config* of a consolidated engine, not a copy.
- **Don't revisit the CVXPY QP optimizer here.** It's been beaten on every metric in this universe;
  reviving it is negative-value time.
