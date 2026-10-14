# Pre-registration: Value's U.S. sizing rule (Step Eight Point Five)

**Status: DRAFT, ready to freeze on Arjun's go (written 2026-10-14, revised same day). Nothing below has been run except
the exploratory variants declared in §2.**

**Design constraint (Arjun, 2026-10-14, now in production):** the U.S. is a special case — the U.S. weight stays within
[25 %, 50 %] whatever the rule. Every arm below respects the band; "no adjustment" is therefore not an arm.

## 1. Question
Value sets its U.S. weight as `25% + 50% × U.S. country alpha`, clipped to [0, 1], and rescales the other countries.
Over 2000-2026 that averages 36 % of the book and ranges 0-100 %. It is the largest single decision in Value, it has
been in the repo since its first commit (header dated 2025-07-29), and it has no validation record. Its input is the
Step Six country alpha, an all-factor composite (every factor's 60-month alpha × the U.S. exposure to it), so factors
Value never holds — momentum, macro, commodity — choose Value's U.S. weight. Does this rule earn its place, and with
which input?

## 2. Already observed (exploratory, 2026-10-14; `us_adjustment_variants.xlsx` in this folder)
Gross active vs equal weight, %/yr, on the then-production code (commit 08fad78):
current 2.93 (2000-12 1.91, 2013-24 3.94); Value-factor alphas 2.82 (2.16, 3.35); no adjustment 3.21 (4.23, 1.55).
The formula cost ~2.3 pp/yr before 2013 and earned ~2.4 pp/yr 2013-24: one regime carries its record. These numbers
are seen; the decision rule below is written to resist exactly that.

## 3. Arms (identical pipeline; only Step Eight Point Five and its alpha input change)
1. **A — current:** 25 % + 50 % × all-factor U.S. alpha, clamped to [25 %, 50 %] (control = production).
2. **B — Value-factor alpha:** same formula, alpha from Value's eligible factors (Max > 0) only.
3. **C — holdings-weighted alpha:** same formula, alpha = Σ_f Step Five weight_f × exposure_US,f × alpha_f, i.e. only
   what Value actually holds this month.
4. **D — fixed U.S. weight:** constant 37.5 % (the middle of the band; sensitivity 25 % and 50 %, reported only).
(The former arm E, no adjustment, is dropped: it violates the 25 % floor.)

## 4. Metrics and tests
Active return (primary), IR, TE, max active drawdown, U.S.-weight mean/min/max/turnover, correlation of active returns
with the Momentum strategy (#090). Periods: Full, H1 (2000-02..2013-06), H2 (2013-07..2026-09), 2000-12, 2013-24, 5y,
3y, 1y. Paired stationary block bootstrap (mean block 12, 5,000 draws) vs A; Reality Check over B-D vs A.

## 5. Decision rule (fixed now)
- Replace A with a candidate only if the candidate (i) has Full IR ≥ A's IR − 0.02, (ii) is not worse than A by
  more than 0.5 pp/yr in EITHER half, and (iii) is simpler or more principled (B, C or D all qualify on (iii)).
  Among qualifying candidates prefer, in order: C (sizes by what Value holds), B, D.
- If no candidate qualifies, keep A and record it in Investment Learnings as tested-and-retained.
- 5y / 3y / 1y are reported only. No cost or turnover penalty (research is gross).

## 6. Deliverables
Run dir in this folder, results workbook, light-mode PDF charts (U.S. weight path per arm, cumulative active), FINDINGS.md,
and an Investment Learnings entry whatever the verdict. Production changes only on Arjun's sign-off.
