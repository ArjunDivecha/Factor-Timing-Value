# T2 Value leverage comparison

Constant 150% means $150 invested and $50 borrowed per $100 of NAV, reset each month.
All arms use the unchanged production Value/Quality country book. The variable arm uses the existing breadth dial.

Common sample: November 2000–September 2026, 311 monthly observations.

## Results

| Period | Strategy | CAGR | Volatility | Max drawdown | Worst month | $1 grows to |
|---|---|---:|---:|---:|---:|---:|
| Full common history | 100% long | 11.46% | 17.72% | -60.53% | -23.65% | 16.63 |
| Full common history | Constant 150% | 14.61% | 26.57% | -77.30% | -35.54% | 34.28 |
| Full common history | Breadth 0–200% | 16.25% | 20.14% | -42.52% | -24.31% | 49.51 |
| Since 2012 | 100% long | 12.08% | 14.19% | -23.97% | -15.28% | 5.38 |
| Since 2012 | Constant 150% | 16.19% | 21.28% | -35.11% | -22.97% | 9.15 |
| Since 2012 | Breadth 0–200% | 12.24% | 15.61% | -29.33% | -14.94% | 5.49 |
| Last 10 years | 100% long | 13.29% | 15.07% | -23.97% | -15.28% | 3.48 |
| Last 10 years | Constant 150% | 17.53% | 22.60% | -35.11% | -22.97% | 5.03 |
| Last 10 years | Breadth 0–200% | 14.32% | 16.85% | -29.33% | -12.09% | 3.81 |
| Last 5 years | 100% long | 14.36% | 14.89% | -19.82% | -9.96% | 1.96 |
| Last 5 years | Constant 150% | 18.49% | 22.30% | -29.62% | -15.10% | 2.34 |
| Last 5 years | Breadth 0–200% | 14.42% | 17.26% | -17.71% | -12.09% | 1.96 |
| Last 3 years | 100% long | 22.89% | 11.80% | -6.93% | -6.93% | 1.86 |
| Last 3 years | Constant 150% | 31.95% | 17.71% | -10.59% | -10.59% | 2.30 |
| Last 3 years | Breadth 0–200% | 30.19% | 17.52% | -12.09% | -12.09% | 2.21 |
| Last 12 months | 100% long | 24.05% | 15.06% | -6.93% | -6.93% | 1.24 |
| Last 12 months | Constant 150% | 33.90% | 22.59% | -10.59% | -10.59% | 1.34 |
| Last 12 months | Breadth 0–200% | 37.49% | 25.42% | -12.09% | -12.09% | 1.37 |
| 2026 YTD (Jan–Sep) | 100% long | 21.93% | 17.63% | -6.93% | -6.93% | 1.16 |
| 2026 YTD (Jan–Sep) | Constant 150% | 30.18% | 26.44% | -10.59% | -10.59% | 1.22 |
| 2026 YTD (Jan–Sep) | Breadth 0–200% | 31.77% | 29.72% | -12.09% | -12.09% | 1.23 |

## Accounting and interpretation

- Constant 150%: monthly return = 1.5 × unlevered return − 0.5 × (T-bill monthly return + 1% annual spread converted monthly).
- 0–200%: use the previous completed month-end breadth × 2; cash below 100% earns the T-bill proxy, borrowing above 100% pays that proxy plus 1%.
- All results are gross of trading costs and after financing. Financing on the borrowed part is not a research exclusion criterion.
- T-bill proxy is the exact historical ^IRX monthly series saved in the Oct-1 production report, derived from mean annual yields. This is a yield-based cash-return approximation. The fixed 1% spread is a scenario, not a verified broker tariff.
- Monthly reset means leverage drifts within each month. These are month-end returns/drawdowns; no daily margin-call liquidation, broker collateral simulation, taxes, FX repair, or new liquidity-cap resizing is modeled.
- This measures an exposure overlay on the saved underlying backtest. It does not independently certify the upstream backtest as point-in-time, USD-executable, or free of survivorship bias.
- Before each ETF exists, Step Ten splices country index history into ETF history. The long historical dial is therefore a proxy-history study; the 2012 subperiod still contains some pre-ETF proxy segments.
- The first seven unlevered months (April–October 2000) are omitted from the matched comparison because the historical dial needs its 200-day warm-up and 30 valid countries.
- No parameter sweep or historical tuning was performed. Recent-period CAGRs shorter than a year are annualized; calendar-year/YTD actual returns are in calendar_years.csv.

## Validation

20 checks passed. See validation.csv for all evidence.
Independently rebuilt every official monthly return from country weights, reconstructed historical breadth, reproduced the old saved dial at its old 0.42% spread, verified signal causality, and confirmed raw source-date alignment.

## Source timing caveat

The Step Nine / Step One comments call the rows forward returns. Actual source timestamps and numeric reconstruction show Portfolio_Data row YYYY-MM-01 matches the return during YYYY-MM. Step One standardize_date stamps each month-end level as the first day of the NEXT month; its subsequent forward one-month return therefore earns the return during that stamped month. This experiment follows numeric/source-date evidence, preserving the existing one-month exposure lag; it changes no production script.

## Files

- compare_leverage.py: rerunnable calculation, no network calls
- summary.csv: all period/strategy metrics
- monthly_returns.csv: base, exposure, rates, financing, and all three return paths
- calendar_years.csv: actual annual returns (2000/2026 partial)
- comparison.png: growth and drawdown
- validation.csv: executed checks
- provenance.json: sources and checksums
