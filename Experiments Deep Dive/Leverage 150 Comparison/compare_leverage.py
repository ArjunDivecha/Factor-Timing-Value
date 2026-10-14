"""Fixed, user-requested exposure comparison. Reads production; writes only here.

100%, monthly reset 150%, and existing previous-month-end breadth dial 0--200%.
Gross of trading costs, after margin financing at T-bill + 1% annual spread.
Uses the Oct-1 saved price/rate snapshot, ending September 2026. No downloads.
Run: python3 'Experiments Deep Dive/Leverage 150 Comparison/compare_leverage.py'
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/t2-leverage-matplotlib')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PRICE_PATH = ROOT / 'outputs/exposure_dial_prices_20261001.parquet'
FINAL_PATH = ROOT / 'T2_FINAL_T60_VALUE.xlsx'
LABELS = {'long_100': '100% long', 'constant_150': 'Constant 150%', 'dial_200': 'Breadth 0–200%'}
COLORS = {'long_100': '#718096', 'constant_150': '#c67c20', 'dial_200': '#236b92'}
SPREAD = .01


def net_return(r, exposure, rf, spread=SPREAD):
    # Same monthly convention as Step Ten, with the requested annual spread.
    # Spread converted to an effective monthly rate. Borrowing applies only above NAV.
    spread_monthly = np.expm1(np.log1p(spread) / 12)
    return exposure * r + (1 - exposure) * rf - np.maximum(exposure - 1, 0) * spread_monthly


def stats(r, rf):
    nav = (1 + r).cumprod()
    peak = nav.cummax().clip(lower=1)  # include the initial NAV of 1
    excess = r - rf
    return {'months': len(r), 'cagr': float(nav.iloc[-1] ** (12 / len(r)) - 1),
            'volatility': float(r.std(ddof=1) * np.sqrt(12)),
            'sharpe': float(excess.mean() / excess.std(ddof=1) * np.sqrt(12)),
            'max_drawdown': float((nav / peak - 1).min()),
            'worst_month': float(r.min()), 'growth_of_1': float(nav.iloc[-1]),
            'worst_12_months': float(((1 + r).rolling(12).apply(np.prod, raw=True) - 1).min())}


def module(path):
    spec = importlib.util.spec_from_file_location('exposure_dial', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run():
    checks = []

    def check(name, condition, detail=''):
        if not bool(condition):
            raise AssertionError(f'{name}: {detail}')
        checks.append({'check': name, 'passed': True, 'detail': detail})

    weights = pd.read_excel(ROOT / 'T2_Final_Country_Weights.xlsx', 'All Periods', index_col=0)
    returns = pd.read_excel(ROOT / 'Portfolio_Data.xlsx', 'Returns', index_col=0)
    official = pd.read_excel(ROOT / 'T2_Final_Portfolio_Returns.xlsx', 'Monthly Returns', index_col=0)
    saved = pd.read_excel(FINAL_PATH, 'Dial_Monthly', index_col=0)
    for d in (weights, returns, official, saved):
        d.index = pd.to_datetime(d.index).to_period('M').to_timestamp()
        check('Unique chronological months: ' + str(d.shape), d.index.is_unique and d.index.is_monotonic_increasing)
    rebuilt = (weights.reindex(official.index) * returns.reindex(official.index)).fillna(0).sum(axis=1)
    delta = (rebuilt - official['Portfolio']).abs().max()
    check('Rebuild all Step Nine returns from country weights', delta < 1e-12, f'max diff={delta:.3e}; n={len(rebuilt)}')
    check('Long-only fully invested underlying book', weights.min().min() >= -1e-12 and np.allclose(weights.sum(axis=1), 1))

    # Verify actual calendar mapping against the original monthly index levels.
    raw_path = Path('/Users/arjundivecha/Dropbox/AAA Backup/Master Database/Country Bloomberg Data Master T.xlsx')
    raw = pd.read_excel(raw_path, 'Tot Return Index ').iloc[1:]
    raw.index = pd.to_datetime(raw.iloc[:, 0]).dt.to_period('M').dt.to_timestamp()
    tri = raw.iloc[:, 1:35].apply(pd.to_numeric, errors='coerce')
    tri.columns = returns.columns
    actual = tri.pct_change(fill_method=None).reindex(returns.index)
    differences = (actual - returns).abs().stack()
    check('Portfolio_Data returns map to actual calendar month', differences.median() < 1e-8,
          f'median absolute country/month discrepancy={differences.median():.3e}; p95={differences.quantile(.95):.3e}')

    dial = module(ROOT / 'Step Ten Exposure Dial.py')
    px = pd.read_parquet(PRICE_PATH)
    mapping = dial.load_tickers()
    signal_px, _ = dial.splice_etf_and_bloomberg(px, dial.load_bloomberg_tri(), list(mapping.values()))
    hist = dial.exposure_history(signal_px, px.index.max())
    month_end = hist.groupby(hist.index.to_period('M')).last()['exposure']
    held = month_end.copy()
    held.index = (held.index + 1).to_timestamp()
    m = pd.DataFrame({'base_return': rebuilt, 'rf': saved['T-bill return'], 'dial_exposure': held}).dropna()
    check('Identical common-month sample to current dial report', m.index.equals(saved.index))
    check('No internal missing months', len(m) == len(pd.period_range(m.index[0], m.index[-1], freq='M')))
    exp_diff = (m['dial_exposure'] - saved['Dial exposure held']).abs().max()
    check('Independently rebuild current historical dial', exp_diff < 1e-12, f'max exposure diff={exp_diff:.3e}')
    legacy = net_return(m.base_return, m.dial_exposure, m.rf, spread=.0042)
    legacy_diff = (legacy - saved['Dial return']).abs().max()
    check('Reproduce old 0.42% spread accounting', legacy_diff < 1e-12, f'max return diff={legacy_diff:.3e}')

    # Causality: later price rows cannot alter past daily exposure signals.
    for cut in ['2008-06-13', '2020-03-16', '2025-06-13']:
        cut = pd.Timestamp(cut)
        truncated = dial.exposure_history(signal_px.loc[:cut], cut)
        check(f'Breadth prefix invariance {cut.date()}', np.allclose(truncated.exposure, hist.loc[truncated.index, 'exposure']))
    signal_dates = pd.Series(hist.index, index=hist.index).groupby(hist.index.to_period('M')).last()
    signal_dates.index = (signal_dates.index + 1).to_timestamp()
    m['signal_date'] = signal_dates.reindex(m.index)
    check('Exposure decided strictly before return month', (m.signal_date < m.index).all())
    check('100/150/200 cash-borrow accounting', np.allclose(
        net_return(.04, np.array([0, 1, 1.5, 2]), .003),
        [.003, .04, .06-.0015-.5*np.expm1(np.log1p(.01)/12), .08-.003-np.expm1(np.log1p(.01)/12)]))
    check('T-bill rates cover entire sample', np.isfinite(m.rf).all() and (m.rf > -.01).all())
    m['annual_tbill_yield_proxy'] = (1 + m.rf) ** 12 - 1
    for col, exposure in [('long_100', pd.Series(1., index=m.index)),
                          ('constant_150', pd.Series(1.5, index=m.index)),
                          ('dial_200', m.dial_exposure)]:
        m[col + '_exposure'] = exposure
        m[col + '_financing'] = np.maximum(exposure - 1, 0) * (m.rf + np.expm1(np.log1p(SPREAD)/12))
        m[col] = net_return(m.base_return, exposure, m.rf)
    check('100% arm reproduces unlevered returns exactly', np.allclose(m.long_100, m.base_return, atol=1e-15))
    check('No NAV insolvency', (m[list(LABELS)] > -1).all().all())

    periods = {'Full common history': m, 'Since 2012': m.loc['2012-01-01':], 'Last 10 years': m.iloc[-120:],
               'Last 5 years': m.iloc[-60:], 'Last 3 years': m.iloc[-36:], 'Last 12 months': m.iloc[-12:],
               '2026 YTD (Jan–Sep)': m.loc['2026-01-01':]}
    rows = []
    for period, sub in periods.items():
        for col, label in LABELS.items():
            rows.append({'period': period, 'strategy': label, 'start': str(sub.index[0].date()),
                         'end': str(sub.index[-1].date()), **stats(sub[col], sub.rf),
                         'total_return': float((1 + sub[col]).prod() - 1),
                         'average_exposure': float(sub[col + '_exposure'].mean()),
                         'average_financing_drag_per_year': float(sub[col + '_financing'].mean() * 12)})
    summary = pd.DataFrame(rows)
    yearly = (1 + m[list(LABELS)]).groupby(m.index.year).prod() - 1
    yearly = yearly.rename(columns=LABELS)
    m.to_csv(HERE / 'monthly_returns.csv', index_label='month')
    summary.to_csv(HERE / 'summary.csv', index=False)
    yearly.to_csv(HERE / 'calendar_years.csv', index_label='year')
    pd.DataFrame(checks).to_csv(HERE / 'validation.csv', index=False)

    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    start = m.index[0] - pd.offsets.MonthEnd(1)
    dates = m.index + pd.offsets.MonthEnd(0)
    for col, label in LABELS.items():
        nav = (1 + m[col]).cumprod()
        x = [start, *dates]
        values = np.r_[1., nav.to_numpy()]
        full = summary[(summary.period == 'Full common history') & (summary.strategy == label)].iloc[0]
        ax[0].plot(x, values, color=COLORS[col], lw=1.8, label=f'{label}: {full.cagr:.1%}/yr; $1 → ${nav.iloc[-1]:.1f}')
        ax[1].plot(x, values / np.maximum.accumulate(values) - 1, color=COLORS[col], lw=1.3)
    ax[0].set_yscale('log')
    ax[0].set_ylabel('Growth of $1 (log scale)')
    ax[0].legend(loc='upper left', frameon=False)
    ax[1].set_ylabel('Drawdown')
    ax[1].yaxis.set_major_formatter(PercentFormatter(1))
    for a in ax:
        a.grid(alpha=.2)
        a.spines[['top', 'right']].set_visible(False)
    fig.suptitle('T2 Value: 100% vs constant 150% vs breadth 0–200%\nNovember 2000–September 2026 · monthly rebalancing · T-bill + 1% margin spread', fontsize=14)
    fig.text(.08, .012, 'After financing; gross of trading costs. Month-end drawdowns. Historical underlying and breadth contain index proxies before ETF inception.', fontsize=9)
    fig.tight_layout(rect=[0, .035, 1, .94])
    fig.savefig(HERE / 'comparison.png', dpi=170)
    plt.close(fig)

    lines = ['# T2 Value leverage comparison', '', 'Constant 150% means $150 invested and $50 borrowed per $100 of NAV, reset each month.',
             'All arms use the unchanged production Value/Quality country book. The variable arm uses the existing breadth dial.', '',
             f'Common sample: {m.index[0]:%B %Y}–{m.index[-1]:%B %Y}, {len(m)} monthly observations.', '',
             '## Results', '', '| Period | Strategy | CAGR | Volatility | Max drawdown | Worst month | $1 grows to |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['period']} | {r['strategy']} | {r['cagr']:.2%} | {r['volatility']:.2%} | {r['max_drawdown']:.2%} | {r['worst_month']:.2%} | {r['growth_of_1']:.2f} |")
    lines += ['', '## Accounting and interpretation', '',
              '- Constant 150%: monthly return = 1.5 × unlevered return − 0.5 × (T-bill monthly return + 1% annual spread converted monthly).',
              '- 0–200%: use the previous completed month-end breadth × 2; cash below 100% earns the T-bill proxy, borrowing above 100% pays that proxy plus 1%.',
              '- All results are gross of trading costs and after financing. Financing on the borrowed part is not a research exclusion criterion.',
              '- T-bill proxy is the exact historical ^IRX monthly series saved in the Oct-1 production report, derived from mean annual yields. This is a yield-based cash-return approximation. The fixed 1% spread is a scenario, not a verified broker tariff.',
              '- Monthly reset means leverage drifts within each month. These are month-end returns/drawdowns; no daily margin-call liquidation, broker collateral simulation, taxes, FX repair, or new liquidity-cap resizing is modeled.',
              '- This measures an exposure overlay on the saved underlying backtest. It does not independently certify the upstream backtest as point-in-time, USD-executable, or free of survivorship bias.',
              '- Before each ETF exists, Step Ten splices country index history into ETF history. The long historical dial is therefore a proxy-history study; the 2012 subperiod still contains some pre-ETF proxy segments.',
              '- The first seven unlevered months (April–October 2000) are omitted from the matched comparison because the historical dial needs its 200-day warm-up and 30 valid countries.',
              '- No parameter sweep or historical tuning was performed. Recent-period CAGRs shorter than a year are annualized; calendar-year/YTD actual returns are in calendar_years.csv.',
              '', '## Validation', '', f'{len(checks)} checks passed. See validation.csv for all evidence.',
              'Independently rebuilt every official monthly return from country weights, reconstructed historical breadth, reproduced the old saved dial at its old 0.42% spread, verified signal causality, and confirmed raw source-date alignment.',
              '', '## Source timing caveat', '',
              'The Step Nine / Step One comments call the rows forward returns. Actual source timestamps and numeric reconstruction show Portfolio_Data row YYYY-MM-01 matches the return during YYYY-MM. Step One standardize_date stamps each month-end level as the first day of the NEXT month; its subsequent forward one-month return therefore earns the return during that stamped month. This experiment follows numeric/source-date evidence, preserving the existing one-month exposure lag; it changes no production script.',
              '', '## Files', '', '- compare_leverage.py: rerunnable calculation, no network calls', '- summary.csv: all period/strategy metrics',
              '- monthly_returns.csv: base, exposure, rates, financing, and all three return paths', '- calendar_years.csv: actual annual returns (2000/2026 partial)',
              '- comparison.png: growth and drawdown', '- validation.csv: executed checks', '- provenance.json: sources and checksums', '']
    (HERE / 'REPORT.md').write_text('\n'.join(lines))
    # Protect all source steps and material production artifacts from accidental mutation.
    before = json.loads((HERE / 'input_hashes_before.json').read_text())
    after = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in before}
    check('Production source and output hashes unchanged', before == after)
    pd.DataFrame(checks).to_csv(HERE / 'validation.csv', index=False)
    provenance = {'snapshot_date': '2026-10-01', 'return_end': '2026-09-30', 'margin_spread_annual': SPREAD,
                  'hashes': after, 'raw_monthly_index_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                  'bloomberg_daily_path': str(dial.BBG_DAILY_FILE),
                  'bloomberg_daily_sha256': hashlib.sha256(dial.BBG_DAILY_FILE.read_bytes()).hexdigest(),
                  'checks_passed': len(checks), 'production_unchanged': before == after}
    (HERE / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    report = (HERE / 'REPORT.md').read_text().replace(f'{len(checks)-1} checks passed.', f'{len(checks)} checks passed.')
    (HERE / 'REPORT.md').write_text(report)
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print(f'CHECKS_PASSED={len(checks)}; report={HERE / "REPORT.md"}')


if __name__ == '__main__':
    run()
