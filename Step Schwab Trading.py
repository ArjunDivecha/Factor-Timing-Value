#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: Step Schwab Trading.py
=============================================================================

DESCRIPTION:
    Reads target country weights from T2_FINAL_T60_VALUE.xlsx, maps countries to
    ETF tickers via AssetList.xlsx, pulls current holdings from a Schwab
    brokerage account (Equity Value by default), calculates the trades
    needed to rebalance to the target portfolio, and executes them using a
    homegrown TWAP (Time-Weighted Average Price) algorithm.

    TWAP splits each trade into 10 equal slices and drips them into the
    market over 15 minutes using marketable limit orders at the current
    NBBO touch. Unfilled slices roll forward into the next slice. Sells
    complete before buys begin so that cash is available.

    By default this script runs in DRY-RUN mode: it shows the trade plan
    but does NOT submit orders. Pass --live --confirm-live to actually
    trade.

INPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value/T2_FINAL_T60_VALUE.xlsx
        Target country weights (sheet: Latest_Country_Alpha_Weights).
        Columns: Country, Country Alpha, Country Weight.
    /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value/AssetList.xlsx
        Country-to-ETF-ticker mapping (sheet: Yahoo). Row order matches
        the country order in T2 Master.xlsx.
    /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value/Experiments Deep Dive/IBKR_Liquidity.xlsx
        Per-ETF average daily volume in USD for the liquidity cap.
    /Users/arjundivecha/Dropbox/AAA Backup/A Complete/AA/schwab_credentials.json  (or env vars)
        Schwab API OAuth credentials.
    ~/.schwabdev/tokens.db
        Schwab OAuth token persistence (SQLite).

OUTPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value/outputs/schwab_trade_plan_YYYYMMDD.xlsx
        Audit workbook with the trade plan, summary, and target weights.
    /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value/outputs/schwab_execution_log_YYYYMMDD.xlsx
        Per-slice execution log with TCA (Transaction Cost Analysis).
    /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value/outputs/schwab_live_marker_YYYYMMDD.json
        JSON marker preventing duplicate live runs on the same date.

VERSION: 2.0
LAST UPDATED: 2026-06-30
AUTHOR: Arjun Divecha

DEPENDENCIES:
    - pandas, numpy, openpyxl, xlsxwriter
    - schwabdev (for live trading / Schwab API)
    - step_liquidity_cap (local module for ADV position cap)

USAGE:
    python "Step Schwab Trading.py"                             # dry run
    python "Step Schwab Trading.py" --live --confirm-live        # live trading
    python "Step Schwab Trading.py" --live --confirm-live --account-name "Arjun IRA"
    python "Step Schwab Trading.py" --twap-window 30 --twap-slices 20
    python "Step Schwab Trading.py" --live --confirm-live --force-rerun  # recover
        # after a crash; only proceeds if Schwab confirms no open orders remain.

NOTES (2026-06-30 vetting pass — GPT + GLM audits):
    - place_order returning HTTP 201 with no Location header (schwabdev's
      own docstring: "if immediately filled then order number not
      returned") is recovered via an account_orders lookup rather than
      assumed to be a failure, to avoid double-submitting shares.
    - TWAP slice carry-forward is capped at --max-slice-carry-multiple x the
      base per-slice size, so a string of failures can't force one giant
      order at the end.
    - --force-rerun allows recovery from a crash after a live marker exists,
      but only after Schwab confirms there are no open/working orders.
    - T2_FINAL_T60_VALUE.xlsx staleness is checked (--max-target-weights-age-days);
      live trading refuses to run on a stale file.
    - Account type (margin vs cash) is checked and a warning is printed if
      not margin, since the 5s sell->buy gap assumes instant buying power.

NOTES (2026-06-30 strategic review — Sakana Fugu Ultra):
    - _is_filled() previously read remainingQuantity==0 as "fully filled"
      regardless of order status. A CANCELED order with a partial fill also
      reports remainingQuantity=0 (it means "nothing left working," not
      "everything filled") — this silently dropped the unfilled remainder
      with no carry-forward and no warning. Fixed: only status=="FILLED" or
      filledQuantity>=expected counts as fully filled; for every other
      terminal status, filledQuantity is trusted as the true amount.
    - Market-order cleanup had a fail-OPEN bug: if the quote fetch failed
      (mkt_quote is None), the spread guard's `if mkt_quote is not None and
      spread > limit` was False, so it fell through and submitted an
      unguarded market order. Fixed: no quote now means no cleanup order,
      same as a too-wide spread.
    - The post-sell live-cash refetch (before sizing buys) silently fell
      back to the stale PRE-sell cash estimate if the refetch failed. Fixed:
      retries 3x, then ABORTS the buy phase loudly (sells already done,
      rerun with --force-rerun after reconciling) instead of guessing.
    - Cancel-then-carry-forward did not require the cancelled order's
      status to be broker-CONFIRMED terminal before reusing the "unfilled"
      quantity in the next slice. If the order was still WORKING /
      PENDING_CANCELLATION (or its status simply couldn't be checked), the
      remainder could double-fill in the background while also being
      resubmitted. Fixed: a symbol whose final status isn't confirmed
      terminal now carries forward ZERO and is permanently excluded from
      further slices and cleanup for that leg ("MANUAL REQUIRED"), instead
      of guessing it's safe to resubmit.
    - IOC (IMMEDIATE_OR_CANCEL) order duration was investigated as a way to
      remove the cancel/poll race by construction. CONFIRMED via a live
      previewOrder call (non-binding) that Schwab's Trader API REJECTS it
      ("Invalid value 'IMMEDIATE_OR_CANCEL'", HTTP 400) despite some
      third-party libraries listing it. FILL_OR_KILL IS accepted but trades
      all-or-nothing fills per slice (no partials) for the race-elimination
      benefit. Decision: keep DAY orders + the terminal-confirmation fix
      above rather than switch to FOK's all-or-nothing semantics. Do not
      re-investigate IOC without new evidence that Schwab has added support.
    - A 16-scenario fake-broker test harness (tests/test_schwab_twap_engine.py
      in the sister 'T2 Factor Timing Fuzzy' repo, identical here) exercises
      execute_twap_leg() against a fully scripted fake Schwab client to lock
      in all of the above. One scenario is an intentional `xfail`: place_order()
      raising AFTER Schwab actually accepted the order can still cause a
      blind resubmission -- a real, currently-unfixed gap, not yet authorized
      to fix.
    - Ported from the 'T2 Factor Timing Fuzzy' sister repo (identical engine,
      different Schwab account + T2_FINAL_T60_VALUE.xlsx target file). The
      ETF_OVERRIDES (VTV/VBR) liquidity-cap ADV mismatch found by an earlier
      GLM audit was fixed by refreshing IBKR_Liquidity.xlsx's U.S./US
      SmallCap rows from Schwab's own price history.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import os
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from step_schwab_dashboard import TwapDashboard

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
T2_FINAL_PATH = SCRIPT_DIR / "T2_FINAL_T60_VALUE.xlsx"
ASSET_LIST_PATH = SCRIPT_DIR / "AssetList.xlsx"
LIQ_PATH = SCRIPT_DIR / "Experiments Deep Dive" / "IBKR_Liquidity.xlsx"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

AA_DIR = Path("/Users/arjundivecha/Dropbox/AAA Backup/A Complete/AA")
GLOBAL_ENV_FILE = Path("/Users/arjundivecha/Dropbox/AAA Backup/.env.txt")
DEFAULT_SCHWAB_CREDENTIALS = AA_DIR / "schwab_credentials.json"
DEFAULT_SCHWAB_TOKENS_DB = Path.home() / ".schwabdev" / "tokens.db"

# ---------------------------------------------------------------------------
# Account registry (same as AA repo)
# ---------------------------------------------------------------------------
ACCOUNT_NAME_TO_LAST3 = {
    "Equity Value": "167",
    "Joint": "966",
    "Arjun IRA": "970",
    "Schwab IRA": "696",
    "Dancing Elephant": "647",
    "TD Ameritrade": "476",
    "Equity Momentum": "090",
}

DEFAULT_ACCOUNT_NAME = "Equity Value"
SNAXX_SYMBOL = "SNAXX"

# ---------------------------------------------------------------------------
# TWAP defaults
# ---------------------------------------------------------------------------
DEFAULT_TWAP_WINDOW_MINUTES = 15
DEFAULT_TWAP_SLICES = 10
DEFAULT_MIN_TRADE_DOLLARS = 1000.0
DEFAULT_CASH_BUFFER_PCT = 0.03
DEFAULT_LIQ_MAXPART = 0.20
DEFAULT_MAX_UNFILLED_SELL_PCT = 0.05
DEFAULT_MAX_CLEANUP_SPREAD_BPS = 50.0
DEFAULT_MAX_SLICE_CARRY_MULTIPLE = 3.0
DEFAULT_MAX_TARGET_WEIGHTS_AGE_DAYS = 35.0

# Schwab API balance fields — use cashBalance for actual cash (NOT margin buying power)
CASH_BALANCE_FIELD = "cashBalance"
TOTAL_EQUITY_FIELD = "liquidationValue"

DEFAULT_IMESSAGE_RECIPIENT = "+15104212111"

# Schwab order terminal statuses
TERMINAL_ORDER_STATUSES = {"FILLED", "CANCELED", "REJECTED", "EXPIRED", "REPLACED"}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class TradingError(RuntimeError):
    """General trading error."""


class LiveQuoteError(TradingError):
    """Cannot obtain a usable live NBBO quote."""


class LiveOrderError(TradingError):
    """Live order submission or preflight failure."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    account_name: str
    live: bool
    confirm_live: bool
    twap_window_minutes: int
    twap_slices: int
    min_trade_dollars: float
    cash_buffer_pct: float
    apply_liquidity_cap: bool
    liq_maxpart: float
    notify: bool
    output_dir: Path
    max_unfilled_sell_pct: float
    max_cleanup_spread_bps: float
    max_slice_carry_multiple: float
    max_target_weights_age_days: float
    force_rerun: bool


# ============================================================================
# SECTION 1: LOAD TARGET WEIGHTS
# ============================================================================

# The Equity Value portfolio uses value-oriented ETFs for U.S. exposure:
#   U.S.        -> VTV  (Vanguard Value, replaces SPY)
#   US SmallCap -> VBR  (Vanguard Small-Cap Value, replaces IWM)
# IMPORTANT: the liquidity cap (step_liquidity_cap.py) is keyed by COUNTRY,
# so IBKR_Liquidity.xlsx's "U.S." / "US SmallCap" rows must hold VTV's/VBR's
# own ADV and Sigma_Daily -- NOT SPY's/IWM's. This was a real bug found by a
# GLM audit (2026-06-30): the liquidity file held SPY's $25.5B ADV and IWM's
# $5.0B ADV, ~40-100x overstating how liquid VTV/VBR actually are. Fixed by
# refreshing those two rows from Schwab's own price history.
ETF_OVERRIDES = {
    "U.S.": "VTV",
    "US SmallCap": "VBR",
}


def load_country_etf_mapping() -> dict[str, str]:
    """Build a {Country: ETF_Ticker} mapping from AssetList.xlsx + T2 Master country order.

    The Equity Value portfolio uses value-oriented ETFs for U.S. exposure:
      U.S.        -> VTV  (Vanguard Value, replaces SPY)
      US SmallCap -> VBR  (Vanguard Small-Cap Value, replaces IWM)
    """
    country_names = [
        'Singapore', 'Australia', 'Canada', 'Germany', 'Japan',
        'Switzerland', 'U.K.', 'NASDAQ', 'U.S.', 'France', 'Netherlands',
        'Sweden', 'Italy', 'ChinaA', 'Chile', 'Indonesia', 'Philippines',
        'Poland', 'US SmallCap', 'Malaysia', 'Taiwan', 'Mexico', 'Korea',
        'Brazil', 'South Africa', 'Denmark', 'India', 'ChinaH', 'Hong Kong',
        'Thailand', 'Turkey', 'Spain', 'Vietnam', 'Saudi Arabia',
    ]
    tickers = pd.read_excel(ASSET_LIST_PATH, sheet_name="Yahoo")["Ticker"].tolist()
    if len(tickers) != len(country_names):
        raise ValueError(
            f"AssetList.xlsx has {len(tickers)} tickers but expected {len(country_names)} countries."
        )
    mapping = dict(zip(country_names, tickers))
    mapping.update(ETF_OVERRIDES)
    return mapping


def load_target_weights() -> pd.Series:
    """Load country weights from T2_FINAL_T60_VALUE.xlsx and convert to ETF-ticker-indexed Series."""
    df = pd.read_excel(T2_FINAL_PATH, sheet_name="Latest_Country_Alpha_Weights")
    country_to_etf = load_country_etf_mapping()

    etf_weights = {}
    for _, row in df.iterrows():
        country = row["Country"]
        weight = float(row["Country Weight"])
        etf = country_to_etf.get(country)
        if etf is None:
            print(f"  WARNING: Country '{country}' not found in AssetList.xlsx mapping — skipping.")
            continue
        if weight > 1e-8:
            etf_weights[etf] = weight

    weights = pd.Series(etf_weights, dtype=float)
    if weights.sum() > 0:
        weights = weights / weights.sum()  # renormalize after dropping near-zeros
    print(f"  Target weights: {len(weights)} ETFs, sum = {weights.sum():.6f}")
    for etf, w in weights.sort_values(ascending=False).items():
        print(f"    {etf:6s}  {w:8.4%}")
    return weights


def check_target_weights_staleness(max_age_days: float, is_live: bool) -> float:
    """Check how old T2_FINAL_T60_VALUE.xlsx is and warn (or block live trading)
    if it looks stale.

    Returns the file age in days. In LIVE mode, an over-age file raises
    TradingError (FAIL IS FAIL — trading real money on a forgotten,
    un-regenerated target file is exactly the kind of silent mistake this
    pipeline must not allow). In DRY RUN mode it only prints a warning so
    you can still inspect a stale plan.
    """
    mtime = T2_FINAL_PATH.stat().st_mtime
    age_days = (datetime.now().timestamp() - mtime) / 86400.0
    if age_days > max_age_days:
        msg = (
            f"{T2_FINAL_PATH.name} is {age_days:.1f} days old "
            f"(limit: {max_age_days:.0f} days). Did you forget to "
            f"regenerate the monthly target weights before trading?"
        )
        if is_live:
            raise TradingError(f"STALE TARGET WEIGHTS: {msg} Refusing to trade live.")
        print(f"  WARNING: STALE TARGET WEIGHTS: {msg}")
    else:
        print(f"  Target weights file age: {age_days:.1f} days (OK, limit {max_age_days:.0f})")
    return age_days


def check_market_hours(twap_window_minutes: float, is_live: bool, now: datetime | None = None) -> str:
    """Check that the market is open AND there is enough time left before
    the close to run the full SELL leg, then the BUY leg (each
    `twap_window_minutes` long), plus cleanup/cancel overhead, before the
    close. Raises TradingError in LIVE mode if not — a partial rebalance
    forced into or past the close (sells done, buys cut off, or a blind
    market-cleanup sweep firing after-hours) is a real-money risk with no
    automated recovery. In DRY RUN mode this only prints a warning.

    Returns a human-readable status string.

    NOTE: this checks the regular 9:30-16:00 ET session and weekends only.
    It does NOT consult a market-holiday calendar (no such data source is
    wired into this script). A market holiday will still be caught
    downstream (Schwab will reject/leave orders unfilled), but not here.
    """
    eastern = ZoneInfo("America/New_York")
    now_et = (now or datetime.now(eastern)).astimezone(eastern)

    if now_et.weekday() >= 5:
        msg = f"today ({now_et:%A}) is a weekend — the market is closed"
    else:
        open_t = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        close_t = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        cleanup_overhead_min = 10.0
        required_min = 2 * twap_window_minutes + cleanup_overhead_min
        if now_et < open_t:
            msg = f"market has not opened yet (opens {open_t:%H:%M} ET, now {now_et:%H:%M} ET)"
        elif now_et >= close_t:
            msg = f"market is closed (closed {close_t:%H:%M} ET, now {now_et:%H:%M} ET)"
        else:
            minutes_to_close = (close_t - now_et).total_seconds() / 60.0
            if minutes_to_close < required_min:
                msg = (
                    f"only {minutes_to_close:.0f} minutes remain before the "
                    f"{close_t:%H:%M} ET close, but a full SELL+BUY rebalance needs "
                    f"~{required_min:.0f} minutes ({twap_window_minutes:.0f} min/leg "
                    f"x2 + {cleanup_overhead_min:.0f} min cleanup buffer). Starting now "
                    "risks the BUY leg (or its market-cleanup sweep) running into or "
                    "past the close."
                )
            else:
                status = (
                    f"market open, {minutes_to_close:.0f} minutes until close "
                    f"(need ~{required_min:.0f})"
                )
                print(f"  Market hours: {status}")
                return status

    if is_live:
        raise TradingError(f"MARKET HOURS CHECK FAILED: {msg}. Refusing to trade live.")
    print(f"  WARNING: MARKET HOURS CHECK FAILED (would block a live run): {msg}")
    return msg


def apply_liquidity_cap_to_weights(weights: pd.Series, aum: float, maxpart: float) -> pd.Series:
    """Apply the ADV-based liquidity cap from step_liquidity_cap.py."""
    from step_liquidity_cap import load_adv, apply_liquidity_cap

    country_to_etf = load_country_etf_mapping()
    etf_to_country = {v: k for k, v in country_to_etf.items()}

    all_countries = list(country_to_etf.keys())
    weights_row = pd.DataFrame(
        [{etf_to_country.get(etf, etf): w for etf, w in weights.items()}],
        columns=all_countries,
    ).fillna(0.0)

    adv = load_adv(str(LIQ_PATH), all_countries)
    capped_df, report = apply_liquidity_cap(weights_row, adv, aum=aum, maxpart=maxpart)

    capped_weights = {}
    for country in capped_df.columns:
        val = float(capped_df.iloc[0][country])
        etf = country_to_etf.get(country)
        if etf and val > 1e-8:
            capped_weights[etf] = val

    capped = pd.Series(capped_weights, dtype=float)

    binds = report[report["Bind_Freq_%"] > 0]
    if not binds.empty:
        print("\n  Liquidity cap binds:")
        for country, row in binds.iterrows():
            print(f"    {country:15s}  ADV=${row['ADV_USD']:>12,.0f}  cap={row['Cap_%']:5.1f}%")

    return capped


# ============================================================================
# SECTION 2: SCHWAB API CONNECTION
# ============================================================================

def get_schwab_client() -> Any:
    """Create a schwabdev Client, loading credentials from env or JSON file."""
    try:
        import schwabdev
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "schwabdev is not installed. Install with: pip install schwabdev"
        ) from exc

    client_id = os.environ.get("SCHWAB_CLIENT_ID")
    client_secret = os.environ.get("SCHWAB_CLIENT_SECRET")
    cred_path = Path(os.environ.get("SCHWAB_CREDENTIALS_PATH", str(DEFAULT_SCHWAB_CREDENTIALS)))

    if (not client_id or not client_secret) and cred_path.exists():
        data = json.loads(cred_path.read_text())
        client_id = client_id or data.get("client_id") or data.get("app_key")
        client_secret = client_secret or data.get("client_secret") or data.get("app_secret")

    if (not client_id or not client_secret) and GLOBAL_ENV_FILE.exists():
        for line in GLOBAL_ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if key == "SCHWAB_CLIENT_ID" and not client_id:
                client_id = val
            elif key == "SCHWAB_CLIENT_SECRET" and not client_secret:
                client_secret = val

    if not client_id or not client_secret:
        raise RuntimeError(
            "Set SCHWAB_CLIENT_ID and SCHWAB_CLIENT_SECRET env vars, or provide "
            f"{DEFAULT_SCHWAB_CREDENTIALS} or {GLOBAL_ENV_FILE} with the keys."
        )
    tokens_db = os.environ.get("SCHWAB_TOKENS_DB", str(DEFAULT_SCHWAB_TOKENS_DB))

    for sensitive_path in (cred_path, GLOBAL_ENV_FILE, Path(tokens_db)):
        _warn_if_world_readable(sensitive_path)

    return schwabdev.Client(client_id, client_secret, tokens_db=tokens_db)


def _warn_if_world_readable(path: Path) -> None:
    """Print a non-blocking warning if a credentials/token file is readable
    by group or other users. This only warns — it never changes permissions
    on a file automatically, since that file may be shared with other tools.
    """
    try:
        if not path.exists():
            return
        mode = path.stat().st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            print(
                f"  WARNING: '{path}' has overly-open permissions "
                f"({oct(mode & 0o777)}). Recommended: chmod 600 \"{path}\""
            )
    except OSError:
        pass


def find_account_hash(client: Any, account_name: str) -> tuple[str, str]:
    """Look up the Schwab API hash value for a named account.

    Returns (hash_value, account_number). The account_number (Schwab's
    human-readable account number) is returned alongside the opaque hash
    so the caller can ECHO a visually-verifiable confirmation before
    submitting live orders — the hash alone can't be sanity-checked by a
    human glancing at the console output. The actual name->account
    correctness is already enforced below (exactly one match required on
    the expected last-3 digits); this is purely for operator visibility.
    """
    linked = client.linked_accounts().json()
    wanted_last3 = ACCOUNT_NAME_TO_LAST3.get(account_name)
    if not wanted_last3:
        raise ValueError(
            f"Unknown account name '{account_name}'. Known: {sorted(ACCOUNT_NAME_TO_LAST3)}"
        )
    matches = [a for a in linked if str(a.get("accountNumber", "")).endswith(wanted_last3)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one '{account_name}' account ending {wanted_last3}; "
            f"found {len(matches)}."
        )
    return matches[0]["hashValue"], str(matches[0].get("accountNumber", ""))


def get_account_details(client: Any, account_hash: str) -> dict[str, Any]:
    return client.account_details(account_hash, fields="positions").json()


def check_margin_account(account_details: dict[str, Any]) -> str:
    """Warn (non-blocking) if the account is not margin-enabled.

    The sell->buy TWAP flow only waits 5 seconds for "settlement" before
    sizing buys off freshly-fetched cash. That is fine for a margin account
    (Schwab grants near-instant buying power from sale proceeds), but a
    cash/IRA account may not show the proceeds as usable buying power that
    quickly (T+1 settlement), which would cause buy rejections. This never
    blocks the run — it just makes the risk visible up front.
    """
    acct_type = str(account_details.get("securitiesAccount", {}).get("type", "")).upper()
    if acct_type and acct_type != "MARGIN":
        print(
            f"  WARNING: account type is '{acct_type}', not MARGIN. Sell proceeds may "
            "not be available as instant buying power — the 5s sell->buy gap in this "
            "script assumes margin-style instant settlement. Buy orders could be "
            "rejected for insufficient funds on a cash/IRA account."
        )
    elif acct_type:
        print(f"  Account type: {acct_type} (instant buying power expected after sells)")
    return acct_type


def parse_holdings(account_details: dict[str, Any]) -> tuple[pd.DataFrame, float, float]:
    """Parse Schwab positions into a holdings DataFrame plus cash and total equity.

    Uses cashBalance for actual cash (NOT margin buying power like availableFunds)
    and liquidationValue for total account equity.

    Returns (holdings_df, cash_balance, total_equity).
    """
    sec = account_details.get("securitiesAccount", {})
    positions = sec.get("positions", [])
    balances = sec.get("currentBalances", {}) or {}

    cash = float(balances.get(CASH_BALANCE_FIELD, 0) or 0)
    total_equity = float(balances.get(TOTAL_EQUITY_FIELD, 0) or 0)

    if total_equity <= 0:
        total_mv = sum(float(p.get("marketValue", 0) or 0) for p in positions)
        total_equity = total_mv + cash

    rows = []
    for pos in positions:
        symbol = str(pos.get("instrument", {}).get("symbol", "")).strip().upper()
        mv = float(pos.get("marketValue", 0) or 0)
        qty = float(pos.get("longQuantity", 0) or 0)
        rows.append({"Symbol": symbol, "Market Value": mv, "Long Quantity": qty})

    holdings = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Symbol", "Market Value", "Long Quantity"])

    return holdings, cash, total_equity


# ============================================================================
# SECTION 3: LIVE QUOTES
# ============================================================================

@dataclass
class LiveQuote:
    symbol: str
    bid: float
    ask: float
    last: float
    bid_size: float
    ask_size: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_bps(self) -> float:
        m = self.mid
        if m <= 0:
            return float("nan")
        return (self.ask - self.bid) / m * 1e4


def get_live_quotes(
    client: Any, symbols: list[str], raise_on_partial: bool = True,
) -> dict[str, LiveQuote]:
    """Fetch live NBBO quotes from Schwab. FAIL IS FAIL — no silent fallback
    for a fully-failed request.

    raise_on_partial: if True (default — used when EVERY symbol's price is
    required, e.g. building the trade plan), any single symbol with a
    missing/bad quote raises LiveQuoteError immediately for the whole call,
    exactly like before. If False (used during TWAP execution, where a
    per-symbol "no quote this slice" fallback already exists downstream),
    a bad quote for ONE symbol is simply excluded from the returned dict
    instead of failing the entire batch.

    This matters because Schwab returns quotes for a whole symbol list in
    ONE response: previously a single halted/illiquid/bad-data ETF made
    this function raise for the WHOLE batch, which coupled every OTHER
    symbol's carry-forward to that one bad name for the rest of the TWAP
    leg (found by an independent adversarial review, 2026-06-30).

    Still raises LiveQuoteError if the HTTP request itself fails (non-200),
    or if not even ONE symbol in the batch produced a usable quote
    (regardless of raise_on_partial) — a fully-failed quote service should
    still be treated as "no data for anyone this slice," not silently
    swallowed into an empty dict.
    """
    symbols = [s.upper() for s in symbols]
    if not symbols:
        return {}
    response = client.quotes(symbols, fields="quote")
    status = getattr(response, "status_code", None)
    if status != 200:
        raise LiveQuoteError(
            f"Schwab quotes request failed: HTTP {status} "
            f"{getattr(response, 'text', '')[:300]}"
        )
    payload = response.json()

    quotes: dict[str, LiveQuote] = {}
    errors: dict[str, str] = {}
    for sym in symbols:
        try:
            block = None
            if sym in payload and isinstance(payload[sym], dict):
                block = payload[sym]
            else:
                for k, v in payload.items():
                    if k.upper() == sym and isinstance(v, dict):
                        block = v
                        break
            if block is None:
                raise LiveQuoteError(f"No quote block returned for {sym}.")

            q = block.get("quote", block) if isinstance(block, dict) else {}
            if not isinstance(q, dict):
                q = {}

            def _num(*names: str) -> float:
                for n in names:
                    v = q.get(n)
                    if v is None:
                        v = block.get(n)
                    try:
                        if v is not None:
                            return float(v)
                    except (TypeError, ValueError):
                        continue
                return float("nan")

            bid = _num("bidPrice", "bid")
            ask = _num("askPrice", "ask")
            last_px = _num("lastPrice", "last", "mark", "closePrice")
            bid_size = _num("bidSize")
            ask_size = _num("askSize")

            if not (bid > 0) or not (ask > 0):
                raise LiveQuoteError(
                    f"{sym}: no usable NBBO (bid={bid}, ask={ask}). "
                    "Market may be closed."
                )
            if bid > ask:
                raise LiveQuoteError(f"{sym}: crossed market bid={bid} > ask={ask}.")

            quotes[sym] = LiveQuote(
                symbol=sym, bid=bid, ask=ask,
                last=last_px if last_px > 0 else (bid + ask) / 2.0,
                bid_size=bid_size if bid_size == bid_size else 0.0,
                ask_size=ask_size if ask_size == ask_size else 0.0,
            )
        except LiveQuoteError as e:
            if raise_on_partial:
                raise
            errors[sym] = str(e)
            continue

    if not quotes and errors:
        raise LiveQuoteError(f"No usable quote for any of {len(errors)} symbol(s): {errors}")
    return quotes


# ============================================================================
# SECTION 4: TRADE PLAN
# ============================================================================

def build_trade_plan(
    target_weights: pd.Series,
    holdings: pd.DataFrame,
    investable_cash: float,
    total_equity: float,
    reference_prices: pd.Series,
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate integer-share trades to rebalance from current to target.

    Returns (plan_df, summary_df).
    """
    holdings = holdings.copy()
    if not holdings.empty:
        holdings["Symbol"] = holdings["Symbol"].astype(str).str.upper()
    current_qty = holdings.groupby("Symbol")["Long Quantity"].sum() if not holdings.empty else pd.Series(dtype=float)
    current_mv = holdings.groupby("Symbol")["Market Value"].sum() if not holdings.empty else pd.Series(dtype=float)

    # NOTE: despite the name, these are NOT holdings the script leaves
    # untouched. The trade loop below unions in every current holding (see
    # the `for etf in sorted(...)` line), so anything here with a zero
    # target weight this period gets a full SELL order, and its value
    # legitimately flows into `allocatable`. This list is purely a
    # diagnostic: "ETFs being fully exited this rebalance because they're
    # not in this period's target weights." A truly unknown/foreign symbol
    # (outside the 33-ETF universe entirely) would ALSO show up here and
    # would ALSO get force-sold by the same loop — there is no separate
    # "leave this alone" carve-out, so double-check this list before going
    # live if the account could hold anything outside the strategy.
    zero_target_holdings = sorted(set(current_qty.index) - set(target_weights.index) - {SNAXX_SYMBOL})
    zero_target_value = float(current_mv.reindex(zero_target_holdings).fillna(0.0).sum()) if zero_target_holdings else 0.0

    cash_buffer = total_equity * config.cash_buffer_pct
    allocatable = max(0.0, total_equity - cash_buffer)

    rows = []
    for etf in sorted(set(target_weights.index) | (set(current_qty.index) - {SNAXX_SYMBOL})):
        price = float(reference_prices.get(etf, np.nan))
        if not np.isfinite(price) or price <= 0:
            continue
        target_w = float(target_weights.get(etf, 0.0))
        target_dollars = allocatable * target_w
        target_qty = math.floor(target_dollars / price)
        current = float(current_qty.get(etf, 0.0))
        shares_to_trade = target_qty - current
        trade_dollars = shares_to_trade * price

        if abs(trade_dollars) < config.min_trade_dollars:
            shares_to_trade = 0.0
            trade_dollars = 0.0

        action = "BUY" if shares_to_trade > 0 else ("SELL" if shares_to_trade < 0 else "HOLD")
        rows.append({
            "Symbol": etf,
            "Reference Price": price,
            "Current Shares": current,
            "Current Dollars": current * price,
            "Target Weight": target_w,
            "Target Dollars": target_dollars,
            "Target Shares": target_qty,
            "Shares to Trade": shares_to_trade,
            "Trade Dollars": trade_dollars,
            "Action": action,
        })

    plan = pd.DataFrame(rows).sort_values(["Action", "Symbol"])

    gross_buy = float(plan.loc[plan["Trade Dollars"] > 0, "Trade Dollars"].sum())
    gross_sell = float(-plan.loc[plan["Trade Dollars"] < 0, "Trade Dollars"].sum())
    summary = pd.DataFrame([
        {"Metric": "Account", "Value": config.account_name},
        {"Metric": "Total Equity (from Schwab)", "Value": f"${total_equity:,.2f}"},
        {"Metric": "Investable Cash", "Value": f"${investable_cash:,.2f}"},
        {"Metric": "Cash Buffer (3%)", "Value": f"${cash_buffer:,.2f}"},
        {"Metric": "Allocatable Equity", "Value": f"${allocatable:,.2f}"},
        {"Metric": "Gross Buy Dollars", "Value": f"${gross_buy:,.2f}"},
        {"Metric": "Gross Sell Dollars", "Value": f"${gross_sell:,.2f}"},
        {"Metric": "Net Trade Dollars", "Value": f"${float(plan['Trade Dollars'].sum()):,.2f}"},
        {"Metric": "Trades Count", "Value": int((plan["Shares to Trade"] != 0).sum())},
        {"Metric": "Zero-Target Holdings (full exit this rebalance)", "Value": ", ".join(zero_target_holdings) if zero_target_holdings else "None"},
        {"Metric": "Zero-Target Value (included in allocatable)", "Value": f"${zero_target_value:,.2f}"},
        {"Metric": "TWAP Window (min)", "Value": config.twap_window_minutes},
        {"Metric": "TWAP Slices", "Value": config.twap_slices},
    ])
    return plan, summary


def scale_buy_plan_to_cash(
    plan: pd.DataFrame,
    available_cash: float,
    cash_buffer_pct: float,
    reference_prices: pd.Series,
    total_equity: float,
) -> pd.DataFrame:
    """Scale down BUY-side share counts (if needed) so planned buy dollars
    never exceed cash that is genuinely available right now.

    This is called AFTER the sell leg completes, using a freshly re-fetched
    Schwab cash balance — never the pre-trade plan's assumed dollar amounts.
    Sell proceeds can differ from the original plan due to partial fills,
    price slippage, or unexpected fees, and blindly trusting the original
    buy-dollar targets risks overspending (margin usage / rejected orders).

    SELL and HOLD rows pass through unchanged. Returns a new DataFrame.

    `total_equity`: the cash buffer here is computed as
    `total_equity * cash_buffer_pct`, matching `build_trade_plan`'s
    original buffer basis exactly — NOT `available_cash * cash_buffer_pct`.
    Using available_cash here used to make this rescue path LESS
    conservative than the original plan whenever it actually triggered
    (available_cash is typically far smaller than total_equity once sell
    proceeds underperform expectations), which is backwards: this path
    only runs when something already didn't go as planned, which is
    exactly when at least as much caution is warranted, not less.
    (Independent adversarial review, 2026-06-30.)
    """
    plan = plan.copy()
    buy_mask = plan["Action"] == "BUY"
    buy_rows = plan[buy_mask]
    if buy_rows.empty:
        return plan

    planned_buy_dollars = float(buy_rows["Trade Dollars"].sum())
    cash_buffer = total_equity * cash_buffer_pct
    spendable = max(0.0, available_cash - cash_buffer)

    if planned_buy_dollars <= 0 or planned_buy_dollars <= spendable:
        return plan  # enough cash on hand — no scaling needed

    scale = spendable / planned_buy_dollars
    print(f"  WARNING: live cash (${spendable:,.2f} spendable) insufficient for "
          f"planned buys (${planned_buy_dollars:,.2f}). Scaling buy orders by {scale:.1%}.")

    for idx in buy_rows.index:
        symbol = plan.at[idx, "Symbol"]
        price = float(reference_prices.get(symbol, np.nan))
        if not np.isfinite(price) or price <= 0:
            price = float(plan.at[idx, "Reference Price"])
        orig_qty = float(plan.at[idx, "Shares to Trade"])
        new_qty = math.floor(orig_qty * scale)
        plan.at[idx, "Shares to Trade"] = new_qty
        plan.at[idx, "Trade Dollars"] = new_qty * price

    return plan


# ============================================================================
# SECTION 5: TWAP EXECUTION ENGINE
# ============================================================================

def _round_cent(price: float) -> float:
    return round(max(0.01, price), 2)


def _marketable_limit_price(quote: LiveQuote, action: str, buffer_bps: float = 5.0) -> float:
    """Compute a marketable limit at the touch + small buffer.

    SELL -> place at bid (take liquidity at the bid)
    BUY  -> place at ask (take liquidity at the ask)
    The buffer_bps gives a tiny amount of crossing to improve fill probability.
    """
    buffer_frac = buffer_bps / 1e4
    if action.upper() == "SELL":
        return _round_cent(quote.bid * (1.0 - buffer_frac))
    else:
        return _round_cent(quote.ask * (1.0 + buffer_frac))


def _build_limit_payload(symbol: str, quantity: int, action: str, price: float) -> dict[str, Any]:
    return {
        "orderType": "LIMIT",
        "session": "NORMAL",
        "duration": "DAY",
        "orderStrategyType": "SINGLE",
        "price": f"{_round_cent(price):.2f}",
        "orderLegCollection": [{
            "instruction": action.upper(),
            "quantity": int(abs(quantity)),
            "instrument": {"symbol": symbol, "assetType": "EQUITY"},
        }],
    }


def _build_market_payload(symbol: str, quantity: int, action: str) -> dict[str, Any]:
    return {
        "orderType": "MARKET",
        "session": "NORMAL",
        "duration": "DAY",
        "orderStrategyType": "SINGLE",
        "orderLegCollection": [{
            "instruction": action.upper(),
            "quantity": int(abs(quantity)),
            "instrument": {"symbol": symbol, "assetType": "EQUITY"},
        }],
    }


def _extract_order_id(location: str | None) -> str | None:
    if not location:
        return None
    return location.rstrip("/").split("/")[-1]


def _check_order_status(client: Any, account_hash: str, order_id: str) -> dict[str, Any]:
    resp = client.order_details(account_hash, order_id)
    if getattr(resp, "status_code", None) != 200:
        raise LiveOrderError(
            f"order_details failed for {order_id}: HTTP {getattr(resp, 'status_code', None)}"
        )
    return resp.json()


def _check_order_status_retrying(
    client: Any, account_hash: str, order_id: str, attempts: int = 4, backoff_sec: float = 1.0,
) -> dict[str, Any] | None:
    """Retry _check_order_status with exponential backoff.

    Returns None (instead of raising) only after all attempts are exhausted.
    Callers MUST treat None as "truth unknown" — never assume zero fill on
    failure, since that risks re-submitting shares that were actually
    already filled (a real double-execution risk with live money).
    """
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return _check_order_status(client, account_hash, order_id)
        except Exception as exc:
            last_exc = exc
            if i < attempts - 1:
                time.sleep(backoff_sec * (2 ** i))
    return None


def _find_recent_order_id(
    client: Any, account_hash: str, symbol: str, action: str, qty: int,
    lookback_sec: float = 90.0, since: datetime | None = None,
) -> str | None:
    """Best-effort recovery of an order ID when place_order returns HTTP 201
    with no Location header (or raises an exception that may have happened
    after Schwab already accepted the order).

    schwabdev's own place_order docstring warns: "if immediately filled then
    order number not returned." Our marketable limit orders are designed to
    fill instantly, so this is a realistic case, not a rare edge case. We
    must NOT assume the order failed (that risks re-submitting the same
    shares on the next slice — a real double-execution risk). Instead, look
    up recent account orders and match on symbol + side + quantity.

    `since`: if provided, candidates whose `enteredTime` is more than 10
    seconds before `since` are excluded. Because TWAP slices for the same
    symbol repeat the same qty every ~60-90s, symbol+side+qty alone is
    frequently ambiguous (2+ matches in the lookback window) — which used
    to make recovery silently give up and flag manual reconciliation even
    when the order WAS findable. Narrowing by submission time disambiguates
    this in the common case. Orders with a missing/unparseable
    `enteredTime` are kept as candidates (never excluded) since we can't
    disprove their recency.

    Returns None if no single unambiguous match is found — callers must
    treat None as "truth unknown", never as "safe to resubmit".
    """
    try:
        now = datetime.now(timezone.utc)
        resp = client.account_orders(
            account_hash,
            now - timedelta(seconds=lookback_sec + 30),
            now + timedelta(seconds=30),
            maxResults=50,
        )
        if getattr(resp, "status_code", None) != 200:
            return None
        orders = resp.json()
    except Exception:
        return None

    cutoff = (since - timedelta(seconds=10)) if since is not None else None

    matches = []
    for o in orders or []:
        legs = o.get("orderLegCollection") or []
        if not legs:
            continue
        leg = legs[0]
        leg_symbol = str(leg.get("instrument", {}).get("symbol", "")).upper()
        leg_instruction = str(leg.get("instruction", "")).upper()
        try:
            leg_qty = int(float(leg.get("quantity", 0) or 0))
        except (TypeError, ValueError):
            leg_qty = 0
        if not (leg_symbol == symbol.upper() and leg_instruction == action.upper() and leg_qty == int(qty)):
            continue
        if cutoff is not None:
            entered_raw = o.get("enteredTime")
            if entered_raw:
                try:
                    entered_dt = datetime.fromisoformat(str(entered_raw).replace("Z", "+00:00"))
                    if entered_dt.tzinfo is None:
                        entered_dt = entered_dt.replace(tzinfo=timezone.utc)
                    if entered_dt < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass  # unparseable timestamp: keep as a candidate
        matches.append(o)

    if len(matches) == 1:
        oid = matches[0].get("orderId")
        return str(oid) if oid is not None else None
    return None  # zero or ambiguous matches — cannot safely recover


def _recover_or_flag_unknown_order(
    client: Any, account_hash: str, sym: str, action: str, qty: int,
    reason: str, dashboard: Any, config: "RunConfig", manual_required: set[str],
    submitted_at: datetime,
) -> str | None:
    """Shared recovery path for ANY place_order outcome whose truth is
    unknown: a 201 with no Location header, OR an exception raised by the
    HTTP call itself (which may have happened AFTER Schwab already
    accepted the order -- e.g. a connection reset while reading the
    response). In both cases the order may already exist and may already
    be filling, so the caller must NEVER carry the quantity forward as a
    plain failure (that risks a duplicate submission).

    Looks up recent account activity for a single unambiguous match. If
    found, returns the recovered order_id so the caller can track it
    normally. If not found, adds `sym` to `manual_required` (stopping all
    further automated trading on this symbol for the rest of the leg) and
    returns None -- the caller must carry forward NOTHING in that case.
    """
    time.sleep(1.0)
    recovered_oid = _find_recent_order_id(client, account_hash, sym, action, qty, since=submitted_at)
    if recovered_oid:
        if dashboard:
            dashboard.add_log(
                f"[yellow]RECOVERED[/yellow] {sym}: order {recovered_oid} "
                f"found via account lookup ({reason})"
            )
            dashboard.refresh()
        return recovered_oid

    manual_required.add(sym)
    msg = (
        f"{sym}: {reason}. The order may have filled with an UNKNOWN ID — "
        f"{qty} shares NOT resubmitted to avoid double-execution. "
        "MANUAL RECONCILIATION REQUIRED — automated trading stopped for this symbol."
    )
    if dashboard:
        dashboard.add_log(f"[bold red]UNRECOVERABLE ORDER ID[/bold red] {msg}")
        dashboard.refresh()
    else:
        print(f"    UNRECOVERABLE ORDER ID: {msg}")
    notify_user(config, "T2 TWAP: UNRECOVERABLE ORDER ID — manual check required", msg)
    return None


def _preview_order(client: Any, account_hash: str, payload: dict[str, Any]) -> tuple[bool, str]:
    """Non-binding preflight check via Schwab's previewOrder endpoint.

    Returns (ok, reason). Only a CONFIRMED rejection from Schwab's own
    pre-trade validation returns ok=False — any error, non-200, or
    unparseable response returns ok=True so a flaky/unsupported preview
    path never blocks real trading (place_order remains authoritative).
    """
    try:
        resp = client.preview_order(account_hash, payload)
        if getattr(resp, "status_code", None) != 200:
            return True, ""
        data = resp.json()
        status = str(data.get("orderStrategy", {}).get("status", "")).upper()
        rejects = (data.get("orderValidationResult") or {}).get("rejects") or []
        if status == "REJECTED" or rejects:
            msgs = "; ".join(r.get("activityMessage", "") for r in rejects if r.get("activityMessage"))
            return False, msgs or "Order rejected by Schwab preview."
        return True, ""
    except Exception:
        return True, ""


def _get_open_orders(client: Any, account_hash: str, lookback_hours: float = 24.0) -> list[dict[str, Any]]:
    """Return any non-terminal (still potentially live) orders from today.

    Used as the safety gate for --force-rerun: if any order is still
    working at Schwab, recomputing and resubmitting a fresh trade plan
    could double up on shares an in-flight order is also trying to trade.
    Raises LiveOrderError on failure — we never treat "couldn't check" as
    "safe to proceed".
    """
    now = datetime.now(timezone.utc)
    resp = client.account_orders(
        account_hash, now - timedelta(hours=lookback_hours), now + timedelta(minutes=5), maxResults=200,
    )
    if getattr(resp, "status_code", None) != 200:
        raise LiveOrderError(
            f"Could not check for open orders before force-rerun: HTTP {getattr(resp, 'status_code', None)}"
        )
    orders = resp.json() or []
    return [o for o in orders if str(o.get("status", "")).upper() not in TERMINAL_ORDER_STATUSES]


def _is_filled(order: dict[str, Any], expected_qty: int) -> bool:
    """True only if the FULL expected quantity actually executed.

    IMPORTANT (Sakana Fugu Ultra audit, 2026-06-30): `remainingQuantity == 0`
    is NOT a reliable "fully filled" signal on its own. Schwab reports
    `remainingQuantity=0` for ANY terminal order — including a CANCELED
    order that only partially filled (e.g. filledQuantity=4,
    remainingQuantity=0, status=CANCELED means "4 filled, 6 canceled," not
    "10 filled"). The previous version of this function used
    `remaining <= 0` as a fully-filled proxy, which silently misread partial
    cancels as complete fills — the unfilled remainder was then dropped
    with no carry-forward, no warning, and no error: a silent position-drift
    bug. We now require either the broker's explicit FILLED status, or an
    observed filled quantity that already meets/exceeds what we expected.
    For every other case (including CANCELED/REJECTED/EXPIRED with a
    partial fill), the caller must trust `filledQuantity` as the actual
    executed amount, not this function's boolean.
    """
    status = str(order.get("status", "")).upper()
    if status == "FILLED":
        return True
    filled_qty = float(order.get("filledQuantity", 0) or 0)
    return filled_qty >= expected_qty


def _avg_fill_price(order: dict[str, Any]) -> float | None:
    total_qty = 0.0
    total_notional = 0.0
    for activity in order.get("orderActivityCollection", []) or []:
        for leg in activity.get("executionLegs", []) or []:
            try:
                qty = float(leg.get("quantity", 0) or 0)
                px = float(leg.get("price", 0) or 0)
            except (TypeError, ValueError):
                continue
            if qty > 0 and px > 0:
                total_qty += qty
                total_notional += qty * px
    if total_qty > 0:
        return total_notional / total_qty
    for key in ("price", "filledPrice", "averagePrice"):
        v = order.get(key)
        try:
            if v is not None and float(v) > 0:
                return float(v)
        except (TypeError, ValueError):
            continue
    return None


@dataclass
class SliceResult:
    """Result of executing one TWAP slice."""
    symbol: str
    action: str
    slice_num: int
    target_qty: int
    filled_qty: int
    limit_price: float
    fill_price: float | None
    arrival_mid: float
    filled: bool
    order_id: str | None
    timestamp: str
    notes: str


def _wait_remainder(t0: float, interval_sec: float, si: int, num_slices: int) -> None:
    """Sleep for the remaining time in the current slice interval."""
    if si < num_slices - 1:
        remaining = interval_sec - (time.monotonic() - t0)
        if remaining > 0:
            time.sleep(remaining)


def execute_twap_leg(
    client: Any,
    account_hash: str,
    plan: pd.DataFrame,
    action: str,
    config: RunConfig,
    dashboard: TwapDashboard | None = None,
) -> list[SliceResult]:
    """Execute all orders for one leg (SELL or BUY) via synchronized TWAP.

    All symbols in the leg execute their slices simultaneously within the
    same time window. At each slice interval every active symbol gets one
    slice submitted, fills are polled, unfilled orders are cancelled at
    interval end, and residual shares roll into the next slice.

    With 10 slices over 15 minutes, each slice interval is 90 seconds.
    A full leg (e.g. 25 sells) completes in ~15 minutes, not 25 × 15 min.
    """
    leg_plan = plan[
        (plan["Action"] == action.upper()) & (plan["Shares to Trade"].abs() >= 1)
    ].copy()

    if leg_plan.empty:
        if dashboard:
            dashboard.add_log(f"No {action} orders to execute.")
            dashboard.refresh()
        else:
            print(f"  No {action} orders to execute.")
        return []

    action = action.upper()
    num_slices = config.twap_slices
    interval_sec = (config.twap_window_minutes * 60.0) / num_slices

    if dashboard:
        dashboard.set_leg(action)
        dashboard.add_log(
            f"[bold]Starting {action} leg: {len(leg_plan)} orders × "
            f"{num_slices} slices over {config.twap_window_minutes} min[/bold]"
        )
        dashboard.refresh()
    else:
        print(f"\n  Starting {action} leg: {len(leg_plan)} orders × "
              f"{num_slices} slices over {config.twap_window_minutes} min")

    # Pre-compute per-symbol slice quantities
    sym_state: dict[str, dict[str, Any]] = {}
    for _, row in leg_plan.iterrows():
        sym = row["Symbol"]
        total_qty = int(abs(row["Shares to Trade"]))
        base = total_qty // num_slices
        remainder = total_qty % num_slices
        slices = [base + (1 if i < remainder else 0) for i in range(num_slices)]
        sym_state[sym] = {
            "total_qty": total_qty, "slices_qty": slices, "carry": 0,
            "base_slice_qty": max(1, base + (1 if remainder else 0)),
        }

    all_syms = sorted(sym_state)
    all_results: list[SliceResult] = []

    # Symbols whose true order state could NOT be confirmed terminal after
    # cancel + retries. Sakana Fugu Ultra audit (2026-06-30): the previous
    # version would still compute `carry = qty - last_known_fill` for these
    # and resubmit the "remainder" in the next slice — but if the prior
    # order is actually still WORKING/PENDING_CANCELLATION at the broker
    # (or its status couldn't be checked at all), that remainder may not
    # exist, and resubmitting risks a duplicate fill. Once a symbol lands
    # here, it is PERMANENTLY excluded from further slices and the market
    # cleanup sweep for this leg — automation stops, a human reconciles.
    manual_required: set[str] = set()

    for si in range(num_slices):
        is_last = (si == num_slices - 1)
        t0 = time.monotonic()

        if dashboard:
            dashboard.add_log(f"[bold]--- Slice {si+1}/{num_slices} ---[/bold]")
            dashboard.refresh()
        else:
            print(f"\n    --- Slice {si+1}/{num_slices} ---")

        # -- Batch-fetch quotes for all symbols --
        #
        # raise_on_partial=False: a single halted/illiquid/bad-data ETF
        # must NOT stall every OTHER symbol's slice. get_live_quotes only
        # raises here if NOT ONE symbol in the whole batch got a usable
        # quote (a genuinely dead quote service) -- in that catastrophic
        # case carrying everyone forward below is correct. Any partial
        # failure instead falls through with that symbol simply missing
        # from `quotes`, handled by the existing per-symbol "no quote"
        # check in the submission loop just below (independent adversarial
        # review, 2026-06-30).
        try:
            quotes = get_live_quotes(client, all_syms, raise_on_partial=False)
        except LiveQuoteError as e:
            for sym in all_syms:
                if sym in manual_required:
                    continue
                sq = sym_state[sym]["slices_qty"][si] + sym_state[sym]["carry"]
                sym_state[sym]["carry"] = sq
                all_results.append(SliceResult(
                    symbol=sym, action=action, slice_num=si + 1,
                    target_qty=sq, filled_qty=0, limit_price=0.0,
                    fill_price=None, arrival_mid=0.0, filled=False,
                    order_id=None, timestamp=datetime.now().isoformat(),
                    notes=f"Quote error: {e}",
                ))
            if dashboard:
                dashboard.add_log(f"[red]QUOTE ERROR: {e}[/red]")
                dashboard.refresh()
            _wait_remainder(t0, interval_sec, si, num_slices)
            continue

        # -- Submit one slice per symbol --
        in_flight: dict[str, tuple[str, int, float, float]] = {}
        submitted = 0
        for sym in all_syms:
            if sym in manual_required:
                continue
            st = sym_state[sym]
            qty_due = st["slices_qty"][si] + st["carry"]
            st["carry"] = 0
            if qty_due <= 0:
                continue

            # Cap how much a single slice attempts, even if prior failures
            # have accumulated a large carry. Without this, a string of
            # failed slices forces the last slice to attempt the ENTIRE
            # original order at once. Any excess simply rolls forward again
            # (and the leg's existing cleanup-sweep / unfilled-sell checks
            # still catch whatever never gets filled by the end).
            carry_cap = max(1, int(math.ceil(config.max_slice_carry_multiple * st["base_slice_qty"])))
            if qty_due > carry_cap:
                qty = carry_cap
                st["carry"] = qty_due - carry_cap
            else:
                qty = qty_due

            q = quotes.get(sym)
            if q is None:
                st["carry"] += qty  # += : a carry-cap excess may already be queued above
                all_results.append(SliceResult(
                    symbol=sym, action=action, slice_num=si + 1,
                    target_qty=qty, filled_qty=0, limit_price=0.0,
                    fill_price=None, arrival_mid=0.0, filled=False,
                    order_id=None, timestamp=datetime.now().isoformat(),
                    notes=f"No quote for {sym}",
                ))
                continue

            arrival = q.mid
            limit_px = _marketable_limit_price(q, action, buffer_bps=10.0 if is_last else 5.0)

            if dashboard:
                dashboard.update_quote(sym, action, q.bid, q.ask)
                dashboard.update_slice_start(sym, action, si + 1, limit_px, arrival)

            payload = _build_limit_payload(sym, qty, action, limit_px)

            # One-time preflight on the first slice of each symbol: a
            # confirmed Schwab pre-trade rejection (insufficient funds,
            # restricted symbol, etc.) is caught here instead of being
            # discovered only after a failed live submission. Never blocks
            # on a preview that itself errors or is unsupported — only a
            # confirmed REJECT skips the real submission.
            if si == 0:
                preview_ok, preview_reason = _preview_order(client, account_hash, payload)
                if not preview_ok:
                    st["carry"] += qty
                    msg = f"{sym}: preflight rejected — {preview_reason}"
                    if dashboard:
                        dashboard.add_log(f"[bold red]PREFLIGHT REJECTED[/bold red] {msg}")
                        dashboard.refresh()
                    else:
                        print(f"    PREFLIGHT REJECTED: {msg}")
                    notify_user(config, "T2 TWAP: order preflight rejected", msg)
                    all_results.append(SliceResult(
                        symbol=sym, action=action, slice_num=si + 1,
                        target_qty=qty, filled_qty=0, limit_price=limit_px,
                        fill_price=None, arrival_mid=arrival, filled=False,
                        order_id=None, timestamp=datetime.now().isoformat(),
                        notes=msg,
                    ))
                    time.sleep(0.05)
                    continue

            submit_attempted_at = datetime.now(timezone.utc)
            try:
                resp = client.place_order(account_hash, payload)
                sc = getattr(resp, "status_code", None)
                loc = resp.headers.get("Location") if hasattr(resp, "headers") else None
                oid = _extract_order_id(loc)
                if sc == 201 and oid:
                    in_flight[sym] = (oid, qty, arrival, limit_px)
                    submitted += 1
                elif sc == 201 and oid is None:
                    # Schwab returned success but no Location header. Per
                    # schwabdev's own docstring this commonly happens when
                    # the order fills immediately — exactly what our
                    # marketable limit orders are designed to do. We must
                    # NOT assume failure and carry the quantity forward
                    # blindly (that would double-submit shares that may
                    # already be filled). Try to recover the real order ID
                    # from recent account activity instead.
                    recovered_oid = _recover_or_flag_unknown_order(
                        client, account_hash, sym, action, qty,
                        "place_order returned HTTP 201 with no Location header "
                        "and no matching order found in account history",
                        dashboard, config, manual_required, submit_attempted_at,
                    )
                    if recovered_oid:
                        in_flight[sym] = (recovered_oid, qty, arrival, limit_px)
                        submitted += 1
                    else:
                        all_results.append(SliceResult(
                            symbol=sym, action=action, slice_num=si + 1,
                            target_qty=qty, filled_qty=0, limit_price=limit_px,
                            fill_price=None, arrival_mid=arrival, filled=False,
                            order_id=None, timestamp=datetime.now().isoformat(),
                            notes=(
                                f"UNRECOVERABLE ORDER ID after HTTP 201 with no Location "
                                f"header — MANUAL RECONCILIATION REQUIRED, {qty} shares "
                                f"NOT resubmitted"
                            ),
                        ))
                else:
                    st["carry"] += qty
                    all_results.append(SliceResult(
                        symbol=sym, action=action, slice_num=si + 1,
                        target_qty=qty, filled_qty=0, limit_price=limit_px,
                        fill_price=None, arrival_mid=arrival, filled=False,
                        order_id=oid, timestamp=datetime.now().isoformat(),
                        notes=f"Submit failed: HTTP {sc}",
                    ))
            except Exception as exc:
                # The HTTP call itself raised (timeout, connection reset,
                # etc.) -- but Schwab may have already received and
                # accepted the order before the failure occurred while
                # reading the response. Blindly carrying `qty` forward (the
                # old behavior) risks a duplicate fill if that's what
                # happened. Mirror the 201-no-Location recovery path: look
                # for the order in account history before assuming nothing
                # happened (independent adversarial review, 2026-06-30).
                recovered_oid = _recover_or_flag_unknown_order(
                    client, account_hash, sym, action, qty,
                    f"place_order raised an exception ({exc}) — outcome unknown, "
                    "and no matching order found in account history",
                    dashboard, config, manual_required, submit_attempted_at,
                )
                if recovered_oid:
                    in_flight[sym] = (recovered_oid, qty, arrival, limit_px)
                    submitted += 1
                else:
                    all_results.append(SliceResult(
                        symbol=sym, action=action, slice_num=si + 1,
                        target_qty=qty, filled_qty=0, limit_price=limit_px,
                        fill_price=None, arrival_mid=arrival, filled=False,
                        order_id=None, timestamp=datetime.now().isoformat(),
                        notes=(
                            f"Submit error: {exc} — MANUAL RECONCILIATION REQUIRED, "
                            f"{qty} shares NOT resubmitted"
                        ),
                    ))
            time.sleep(0.05)

        if dashboard:
            dashboard.add_log(f"Submitted {submitted} orders")
            dashboard.refresh()

        if not in_flight:
            _wait_remainder(t0, interval_sec, si, num_slices)
            continue

        # -- Poll fills until interval deadline --
        # slice_fills tracks the LAST KNOWN GOOD filled quantity per symbol.
        # It is only ever updated on a successful status check, so a
        # transient API failure during polling never erases real, observed
        # fill progress.
        deadline = t0 + interval_sec - 2.0
        filled_set: set[str] = set()
        slice_fills: dict[str, int] = {}
        status_warnings: dict[str, int] = {}  # consecutive failure counts

        while time.monotonic() < deadline and len(filled_set) < len(in_flight):
            time.sleep(3.0)
            for sym, (oid, qty, amid, lpx) in in_flight.items():
                if sym in filled_set:
                    continue
                od = None
                try:
                    od = _check_order_status(client, account_hash, oid)
                except Exception as exc:
                    status_warnings[sym] = status_warnings.get(sym, 0) + 1
                    if dashboard:
                        dashboard.add_log(f"[yellow]Status check failed[/yellow] {sym}: {exc}")
                if od is not None:
                    status_warnings[sym] = 0
                    fq = int(float(od.get("filledQuantity", 0) or 0))
                    prev = slice_fills.get(sym, 0)
                    if fq > prev and dashboard:
                        fp = _avg_fill_price(od)
                        dashboard.update_fill(sym, action, fq - prev, fp)
                        dashboard.refresh()
                    slice_fills[sym] = max(slice_fills.get(sym, 0), fq)

                    if _is_filled(od, qty):
                        filled_set.add(sym)
                    elif str(od.get("status", "")).upper() in TERMINAL_ORDER_STATUSES:
                        filled_set.add(sym)
                # Pace consecutive order_details calls within one sweep so a
                # leg with many in-flight symbols (~25-30 ETFs) doesn't burst
                # far above a safe request rate and risk throttling (429s),
                # which would itself cause MORE missed-status retries.
                # Independent adversarial review, 2026-06-30.
                time.sleep(0.15)

        # -- Cancel unfilled orders, then CONFIRM terminal state before --
        # -- treating any shares as available to roll forward. Without   --
        # -- this confirmation, an order that is still WORKING or        --
        # -- PENDING_CANCELLATION could keep filling in the background   --
        # -- while we also resubmit the same shares in the next slice.   --
        to_cancel = [sym for sym in in_flight if sym not in filled_set]
        for sym in to_cancel:
            try:
                client.cancel_order(account_hash, in_flight[sym][0])
            except Exception:
                pass
            time.sleep(0.05)

        for sym in to_cancel:
            oid = in_flight[sym][0]
            for _ in range(5):
                od = _check_order_status_retrying(client, account_hash, oid, attempts=2, backoff_sec=0.5)
                if od is None:
                    break
                status = str(od.get("status", "")).upper()
                fq = int(float(od.get("filledQuantity", 0) or 0))
                slice_fills[sym] = max(slice_fills.get(sym, 0), fq)
                if status in TERMINAL_ORDER_STATUSES:
                    break
                time.sleep(0.5)
            time.sleep(0.15)  # pace between symbols -- see note above

        # -- Final fill check + record results --
        #
        # Sakana Fugu Ultra audit (2026-06-30): carrying forward
        # "qty - last_known_fill" is only SAFE when the order's terminal
        # status is broker-CONFIRMED (FILLED/CANCELED/REJECTED/EXPIRED/
        # REPLACED). If the order is still WORKING/PENDING_CANCELLATION, or
        # its status simply could not be checked, it may keep filling in
        # the background — resubmitting the "remainder" in that case risks
        # an unintended duplicate fill. The previous version computed carry
        # the same way regardless of confirmation. Now: if terminal state
        # is not confirmed (and we haven't already independently observed
        # a full fill via in-slice polling), STOP trading this symbol for
        # the rest of the leg, carry forward NOTHING, and flag it loudly
        # for manual reconciliation instead of guessing.
        fully_filled = 0
        for sym, (oid, qty, amid, lpx) in in_flight.items():
            od = _check_order_status_retrying(client, account_hash, oid, attempts=4, backoff_sec=1.0)
            time.sleep(0.15)  # pace consecutive calls -- see note above
            prev = slice_fills.get(sym, 0)
            status = str(od.get("status", "")).upper() if od is not None else None
            confirmed_terminal = od is not None and status in TERMINAL_ORDER_STATUSES

            if not confirmed_terminal and prev >= qty:
                # Already independently confirmed full fill via earlier
                # in-slice polling (a valid status check during the loop
                # above showed filledQuantity >= qty) — trust that, even
                # though this final check came back unconfirmed.
                confirmed_terminal = True
                fq, fp, is_complete, actual = prev, None, True, prev
                notes = ""
            elif not confirmed_terminal:
                fq = max(prev, int(float(od.get("filledQuantity", 0) or 0)) if od is not None else prev)
                fp = None
                is_complete = False
                actual = fq
                manual_required.add(sym)
                reason = (
                    "status check failed after retries" if od is None
                    else f"non-terminal status '{status}' persisted after cancel + retries"
                )
                notes = (
                    f"MANUAL RECONCILIATION REQUIRED: {reason}. {fq}/{qty} confirmed "
                    f"filled; remainder NOT carried forward (order state unconfirmed — "
                    f"automated trading stopped for this symbol)."
                )
                msg = f"{sym} ({action}): {notes}"
                if dashboard:
                    dashboard.add_log(f"[bold red]MANUAL REQUIRED[/bold red] {msg}")
                    dashboard.refresh()
                else:
                    print(f"    MANUAL REQUIRED: {msg}")
                notify_user(config, f"T2 TWAP: MANUAL RECONCILIATION REQUIRED ({sym})", msg)
            else:
                fq = int(float(od.get("filledQuantity", 0) or 0))
                fq = max(fq, prev)  # never regress below a previously observed fill
                fp = _avg_fill_price(od) if fq > 0 else None
                is_complete = _is_filled(od, qty)
                actual = qty if is_complete else fq
                notes = "" if is_complete else f"Partial: {fq}/{qty}"

            if actual > prev and dashboard:
                dashboard.update_fill(sym, action, actual - prev, fp)

            # Confirmed-terminal symbols carry their true remainder forward;
            # manual_required symbols carry nothing (see above) — they are
            # excluded from all further slices and from cleanup.
            #
            # MUST be += , not = : if the carry-cap above deferred an
            # over-cap excess for this symbol BEFORE this order was even
            # submitted, that excess is still sitting in
            # sym_state[sym]["carry"] right now (nothing else touches it
            # between submission and here). Overwriting it with just
            # `qty - actual` silently discards that deferred excess —
            # shares neither resubmitted NOR swept in cleanup. This only
            # bites after a string of failed/partial slices large enough to
            # make the cap actually bind, which is why it survived the
            # earlier fixes undetected (independent adversarial review,
            # 2026-06-30).
            if confirmed_terminal:
                sym_state[sym]["carry"] += (qty - actual)
            else:
                sym_state[sym]["carry"] = 0
            if is_complete:
                fully_filled += 1

            all_results.append(SliceResult(
                symbol=sym, action=action, slice_num=si + 1,
                target_qty=qty, filled_qty=actual,
                limit_price=lpx, fill_price=fp,
                arrival_mid=amid, filled=is_complete,
                order_id=oid, timestamp=datetime.now().isoformat(),
                notes=notes,
            ))
            time.sleep(0.05)

        if dashboard:
            dashboard.add_log(f"Slice {si+1} done: {fully_filled}/{len(in_flight)} filled")
            dashboard.refresh()
        else:
            print(f"    Slice {si+1} done: {fully_filled}/{len(in_flight)} filled")

        _wait_remainder(t0, interval_sec, si, num_slices)

    # -- Market order cleanup: sweep any remaining unfilled shares --
    residual_syms = {sym: st["carry"] for sym, st in sym_state.items() if st["carry"] > 0}
    if residual_syms:
        if dashboard:
            dashboard.add_log(
                f"[bold yellow]CLEANUP: {len(residual_syms)} symbols with residual shares — "
                f"submitting market orders[/bold yellow]"
            )
            dashboard.refresh()
        else:
            print(f"\n    CLEANUP: {len(residual_syms)} symbols with residual — market orders")

        for sym, qty in residual_syms.items():
            mkt_quote = None  # must be initialized before the try block —
            arrival = 0.0     # a LiveQuoteError below must not leave these unbound
            try:
                mkt_quote = get_live_quotes(client, [sym]).get(sym)
                if mkt_quote:
                    arrival = mkt_quote.mid
            except LiveQuoteError:
                pass

            if dashboard and mkt_quote:
                dashboard.update_quote(sym, action, mkt_quote.bid, mkt_quote.ask)

            # -- Spread guard + fail-CLOSED quote check: a blind market      --
            # -- order on a wide-spread ETF (or with NO quote at all) can    --
            # -- pay a large, unbounded cost. Skip the sweep for this        --
            # -- symbol and surface it loudly for manual handling instead    --
            # -- of silently eating a bad fill.                              --
            # -- Sakana Fugu Ultra audit (2026-06-30): the previous version  --
            # -- only skipped when `mkt_quote is not None AND spread too     --
            # -- wide`. If the quote fetch FAILED (mkt_quote is None), that  --
            # -- condition was False, so it fell through and submitted a    --
            # -- market order with ZERO spread protection — the opposite of --
            # -- fail-safe. "No quote" must mean "no cleanup order," not     --
            # -- "skip the guard."                                          --
            skip_reason: str | None = None
            if mkt_quote is None:
                skip_reason = "no live quote available (quote fetch failed)"
            elif mkt_quote.spread_bps > config.max_cleanup_spread_bps:
                skip_reason = (
                    f"spread {mkt_quote.spread_bps:.0f}bps exceeds "
                    f"{config.max_cleanup_spread_bps:.0f}bps cleanup limit"
                )
            if skip_reason is not None:
                msg = (
                    f"{sym}: {skip_reason} — SKIPPING market order. "
                    f"{qty} shares remain unfilled."
                )
                if dashboard:
                    dashboard.add_log(f"[bold red]CLEANUP SKIPPED[/bold red] {msg}")
                    dashboard.refresh()
                else:
                    print(f"    CLEANUP SKIPPED: {msg}")
                notify_user(config, "T2 TWAP: cleanup skipped", msg)
                all_results.append(SliceResult(
                    symbol=sym, action=action, slice_num=num_slices + 1,
                    target_qty=qty, filled_qty=0, limit_price=0.0,
                    fill_price=None, arrival_mid=arrival, filled=False,
                    order_id=None, timestamp=datetime.now().isoformat(),
                    notes=f"Cleanup skipped: {skip_reason}. Manual fill required.",
                ))
                time.sleep(0.1)
                continue

            payload = _build_market_payload(sym, qty, action)
            try:
                resp = client.place_order(account_hash, payload)
                sc = getattr(resp, "status_code", None)
                loc = resp.headers.get("Location") if hasattr(resp, "headers") else None
                oid = _extract_order_id(loc)

                if sc == 201 and oid is None:
                    # Market orders fill almost instantly, so a 201 with no
                    # Location header is actually the LIKELY case here, not
                    # an edge case — try to recover the real order ID
                    # before treating this as a submit failure.
                    time.sleep(1.0)
                    oid = _find_recent_order_id(client, account_hash, sym, action, qty)
                    if oid and dashboard:
                        dashboard.add_log(
                            f"[yellow]RECOVERED[/yellow] {sym}: market order {oid} "
                            "found via account lookup (201 had no Location header)"
                        )
                        dashboard.refresh()

                if sc != 201 or not oid:
                    msg = f"{sym}: market order submit failed/unrecoverable: HTTP {sc}"
                    if sc == 201:
                        msg = (
                            f"{sym}: market order returned HTTP 201 with no Location header "
                            f"and no matching order found in account history. It may have "
                            f"filled with an UNKNOWN ID. MANUAL RECONCILIATION REQUIRED."
                        )
                        notify_user(config, "T2 TWAP: UNRECOVERABLE market order ID", msg)
                    if dashboard:
                        dashboard.add_log(f"[red]MKT SUBMIT FAILED[/red] {msg}")
                        dashboard.refresh()
                    all_results.append(SliceResult(
                        symbol=sym, action=action, slice_num=num_slices + 1,
                        target_qty=qty, filled_qty=0, limit_price=0.0,
                        fill_price=None, arrival_mid=arrival, filled=False,
                        order_id=oid, timestamp=datetime.now().isoformat(),
                        notes=msg,
                    ))
                    continue

                time.sleep(3.0)
                od = _check_order_status_retrying(client, account_hash, oid, attempts=4, backoff_sec=1.0)

                if od is None:
                    # Truth unknown — do NOT assume zero fill or full fill.
                    # Record explicitly as unknown and alert immediately;
                    # this is the final settlement step with no further
                    # automated retry, so a human must verify.
                    fq, fp, is_complete, actual = 0, None, False, 0
                    notes = (
                        "STATUS CHECK FAILED after retries on market order — "
                        "fill state unknown. MANUAL RECONCILIATION REQUIRED."
                    )
                    if dashboard:
                        dashboard.add_log(f"[bold red]STATUS UNKNOWN[/bold red] {sym}: market order status check failed")
                        dashboard.refresh()
                    notify_user(
                        config, "T2 TWAP: cleanup order status UNKNOWN",
                        f"{sym}: market order (oid {oid}) status check failed after retries. "
                        "Verify manually in Schwab — it may have filled.",
                    )
                else:
                    fq = int(float(od.get("filledQuantity", 0) or 0))
                    fp = _avg_fill_price(od) if fq > 0 else None
                    is_complete = _is_filled(od, qty)
                    actual = qty if is_complete else fq
                    notes = "Market order cleanup" + ("" if is_complete else f" (partial {fq}/{qty})")

                if actual > 0 and dashboard:
                    dashboard.update_fill(sym, action, actual, fp)
                    slip_info = ""
                    if fp and arrival > 0:
                        s = ((arrival - fp) / arrival * 1e4) if action == "SELL" else ((fp - arrival) / arrival * 1e4)
                        slip_info = f" slip={s:.1f}bps"
                    dashboard.add_log(
                        f"[green]MKT FILLED[/green] {sym}: {actual} @ ${fp:.4f}{slip_info}"
                        if fp else f"[green]MKT FILLED[/green] {sym}: {actual}"
                    )
                    dashboard.refresh()

                sym_state[sym]["carry"] = qty - actual
                all_results.append(SliceResult(
                    symbol=sym, action=action, slice_num=num_slices + 1,
                    target_qty=qty, filled_qty=actual,
                    limit_price=0.0, fill_price=fp,
                    arrival_mid=arrival, filled=is_complete,
                    order_id=oid, timestamp=datetime.now().isoformat(),
                    notes=notes,
                ))

            except Exception as exc:
                all_results.append(SliceResult(
                    symbol=sym, action=action, slice_num=num_slices + 1,
                    target_qty=qty, filled_qty=0, limit_price=0.0,
                    fill_price=None, arrival_mid=arrival, filled=False,
                    order_id=None, timestamp=datetime.now().isoformat(),
                    notes=f"Market order error: {exc}",
                ))
            time.sleep(0.1)

        if dashboard:
            still_open = sum(1 for st in sym_state.values() if st["carry"] > 0)
            dashboard.add_log(
                f"Cleanup done: {len(residual_syms) - still_open}/{len(residual_syms)} filled"
            )
            dashboard.refresh()

    # Set final order statuses in dashboard
    if dashboard:
        for sym in all_syms:
            sym_results = [r for r in all_results if r.symbol == sym]
            total_filled = sum(r.filled_qty for r in sym_results)
            status = "FILLED" if total_filled >= sym_state[sym]["total_qty"] else "PARTIAL"
            dashboard.update_order_status(sym, action, status)
        dashboard.refresh()

    return all_results


# ============================================================================
# SECTION 6: iMESSAGE NOTIFICATIONS
# ============================================================================

def send_imessage(recipient: str, text: str) -> None:
    """Fire-and-forget iMessage via osascript."""
    safe_text = text.replace("\\", "\\\\").replace('"', '\\"')
    script = (
        'tell application "Messages"\n'
        '  set targetService to first service whose service type = iMessage\n'
        f'  set targetBuddy to buddy "{recipient}" of targetService\n'
        f'  send "{safe_text}" to targetBuddy\n'
        'end tell'
    )
    subprocess.Popen(
        ["osascript", "-e", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def notify_user(config: RunConfig, subject: str, body: str) -> None:
    """Send iMessage notification if --notify is set."""
    if not config.notify:
        return
    recipient = os.environ.get("T2_IMESSAGE_TO") or DEFAULT_IMESSAGE_RECIPIENT
    text = f"{subject}\n\n{body}"
    try:
        send_imessage(recipient, text)
    except Exception as exc:
        print(f"  WARNING: iMessage notification failed: {exc}")


# ============================================================================
# SECTION 7: OUTPUT WRITING
# ============================================================================

def write_trade_plan_workbook(
    output_dir: Path,
    plan: pd.DataFrame,
    summary: pd.DataFrame,
    target_weights: pd.Series,
    date_str: str,
) -> Path:
    """Write audit workbook with trade plan, summary, and target weights."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"schwab_trade_plan_{date_str}.xlsx"

    with pd.ExcelWriter(str(path), engine="xlsxwriter") as writer:
        plan.to_excel(writer, sheet_name="Trade Plan", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)

        tw_df = target_weights.reset_index()
        tw_df.columns = ["ETF", "Weight"]
        tw_df.to_excel(writer, sheet_name="Target Weights", index=False)

        workbook = writer.book
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            ws.set_column(0, 20, 18)

    print(f"  Trade plan written: {path}")
    return path


def write_execution_log(
    output_dir: Path,
    results: list[SliceResult],
    date_str: str,
) -> Path:
    """Write TWAP execution log as an Excel workbook."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"schwab_execution_log_{date_str}.xlsx"

    rows = []
    for r in results:
        slippage_bps = float("nan")
        if r.fill_price and r.arrival_mid > 0:
            if r.action == "SELL":
                slippage_bps = (r.arrival_mid - r.fill_price) / r.arrival_mid * 1e4
            else:
                slippage_bps = (r.fill_price - r.arrival_mid) / r.arrival_mid * 1e4
        rows.append({
            "Symbol": r.symbol,
            "Action": r.action,
            "Slice": r.slice_num,
            "Target Qty": r.target_qty,
            "Filled Qty": r.filled_qty,
            "Limit Price": r.limit_price,
            "Fill Price": r.fill_price,
            "Arrival Mid": r.arrival_mid,
            "Slippage (bps)": slippage_bps,
            "Filled": r.filled,
            "Order ID": r.order_id,
            "Timestamp": r.timestamp,
            "Notes": r.notes,
        })

    df = pd.DataFrame(rows)
    with pd.ExcelWriter(str(path), engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Execution Log", index=False)
        ws = writer.sheets["Execution Log"]
        ws.set_column(0, 20, 18)

    print(f"  Execution log written: {path}")
    return path


def write_marker(output_dir: Path, date_str: str, data: dict[str, Any]) -> Path:
    """Overwrite the live submission marker with updated status.

    Only safe to call AFTER this run has already claimed the marker via
    `claim_live_marker()` below — from that point on, this process owns
    the file for the rest of the run (status updates like SELLS_DONE,
    COMPLETED, etc. are plain overwrites, not a re-claim).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"schwab_live_marker_{date_str}.json"
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


def claim_live_marker(
    output_dir: Path, date_str: str, data: dict[str, Any], allow_overwrite: bool,
) -> Path:
    """Atomically create the live marker file for the FIRST time this run.

    The caller already checked `marker_path.exists()` earlier (and decided
    whether to proceed at all, including the --force-rerun open-orders
    check). But that existence check and this write are not the same
    instant — two near-simultaneous launches could both observe "no marker
    yet" and both proceed to trade the same account. This function closes
    that race with O_EXCL: the OS guarantees only ONE caller can win the
    create-if-not-exists, atomically. (Independent adversarial review,
    2026-06-30.)

    allow_overwrite=True is for the --force-rerun path, where the marker
    is EXPECTED to already exist (already validated above) and should be
    intentionally replaced — that case has no race to protect against
    since this same process already confirmed via Schwab that no orders
    are open before reaching here.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"schwab_live_marker_{date_str}.json"
    flags = os.O_CREAT | os.O_WRONLY | (os.O_TRUNC if allow_overwrite else os.O_EXCL)
    try:
        fd = os.open(path, flags)
    except FileExistsError as exc:
        raise TradingError(
            f"Live marker {path} appeared between the duplicate-run check and "
            "claiming it — another process likely started trading this account "
            "at almost the same moment. Refusing to proceed. Verify only one "
            "process is trading this account today before retrying."
        ) from exc
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(data, indent=2, default=str))
    return path


# ============================================================================
# SECTION 7b: TCA COMPUTATION & iMESSAGE FORMATTER
# ============================================================================

def _build_target_lookup(plan: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Map (Symbol, Action) -> the true original share count from the plan.

    Must be used instead of summing SliceResult.target_qty across slices:
    when a slice goes unfilled, its quantity rolls forward and is ADDED to
    a later slice's target_qty. Summing target_qty across all slices for a
    symbol therefore double-counts the rolled-forward portion and overstates
    the true order size (e.g. a 20-share order that misses 10 in slice 1
    and finishes 20 in slice 2 would sum to a fake "30 share target").
    """
    lookup: dict[tuple[str, str], int] = {}
    for _, row in plan.iterrows():
        qty = int(abs(row["Shares to Trade"]))
        if qty > 0:
            lookup[(row["Symbol"], row["Action"])] = qty
    return lookup


def _compute_tca(
    all_results: list[SliceResult],
    target_lookup: dict[tuple[str, str], int] | None = None,
) -> dict[str, Any]:
    """Compute detailed Transaction Cost Analysis from execution results.

    Returns a dict with aggregate stats plus per-symbol breakdowns for
    the sell and buy legs separately.
    """
    target_lookup = target_lookup or {}
    symbols_seen: set[str] = set()
    sell_results = [r for r in all_results if r.action == "SELL"]
    buy_results = [r for r in all_results if r.action == "BUY"]

    def _leg_stats(results: list[SliceResult], leg_action: str) -> dict[str, Any]:
        """Compute stats for one leg (SELL or BUY)."""
        by_sym: dict[str, list[SliceResult]] = {}
        for r in results:
            by_sym.setdefault(r.symbol, []).append(r)

        sym_details: list[dict[str, Any]] = []
        leg_filled = 0
        leg_notional = 0.0
        leg_slip_cost = 0.0

        for sym in sorted(by_sym):
            slices = by_sym[sym]
            filled = sum(s.filled_qty for s in slices)
            # Prefer the true plan-derived target; fall back to summing
            # slice target_qty only if the plan lookup is unavailable
            # (this fallback can overstate target — see docstring above).
            target = target_lookup.get((sym, leg_action))
            if target is None:
                target = sum(s.target_qty for s in slices)
            notional = 0.0
            slip_cost = 0.0
            prices = []
            for s in slices:
                if s.fill_price and s.arrival_mid > 0 and s.filled_qty > 0:
                    n = s.filled_qty * s.fill_price
                    notional += n
                    prices.append(s.fill_price)
                    if s.action == "SELL":
                        slip_cost += (s.arrival_mid - s.fill_price) * s.filled_qty
                    else:
                        slip_cost += (s.fill_price - s.arrival_mid) * s.filled_qty

            vwap = notional / filled if filled > 0 else 0.0
            slip_bps = (slip_cost / notional * 1e4) if notional > 0 else 0.0
            arrival = slices[0].arrival_mid if slices else 0.0

            sym_details.append({
                "symbol": sym, "filled": filled, "target": target,
                "notional": notional, "vwap": vwap,
                "arrival_mid": arrival, "slip_bps": slip_bps,
                "slip_cost": slip_cost,
            })
            leg_filled += filled
            leg_notional += notional
            leg_slip_cost += slip_cost

        leg_slip_bps = (leg_slip_cost / leg_notional * 1e4) if leg_notional > 0 else 0.0
        return {
            "symbols": sym_details, "filled_shares": leg_filled,
            "notional": leg_notional, "slip_bps": leg_slip_bps,
            "slip_cost": leg_slip_cost, "order_count": len(by_sym),
        }

    sell_tca = _leg_stats(sell_results, "SELL")
    buy_tca = _leg_stats(buy_results, "BUY")

    total_notional = sell_tca["notional"] + buy_tca["notional"]
    total_slip_cost = sell_tca["slip_cost"] + buy_tca["slip_cost"]
    avg_slip = (total_slip_cost / total_notional * 1e4) if total_notional > 0 else 0.0

    for r in all_results:
        symbols_seen.add(r.symbol)

    return {
        "sell": sell_tca,
        "buy": buy_tca,
        "total_symbols": len(symbols_seen),
        "total_slices": len(all_results),
        "total_filled_shares": sell_tca["filled_shares"] + buy_tca["filled_shares"],
        "total_notional": total_notional,
        "total_slippage_cost": total_slip_cost,
        "avg_slippage_bps": avg_slip,
    }


def _build_tca_imessage(
    tca: dict[str, Any], config: RunConfig, plan_path: Path, exec_path: Path,
) -> str:
    """Format a detailed TCA summary for iMessage notification."""
    lines: list[str] = []

    lines.append(f"Account: {config.account_name}")
    lines.append(f"TWAP: {config.twap_slices} slices / {config.twap_window_minutes} min")
    lines.append("")

    # Overall
    lines.append("--- OVERALL ---")
    lines.append(f"Symbols traded: {tca['total_symbols']}")
    lines.append(f"Total slices: {tca['total_slices']}")
    lines.append(f"Total shares: {tca['total_filled_shares']:,.0f}")
    lines.append(f"Total notional: ${tca['total_notional']:,.2f}")
    lines.append(f"Avg slippage: {tca['avg_slippage_bps']:.1f} bps")
    lines.append(f"Slippage cost: ${tca['total_slippage_cost']:,.2f}")
    lines.append("")

    # Sell leg
    sell = tca["sell"]
    if sell["order_count"] > 0:
        lines.append(f"--- SELLS ({sell['order_count']} orders) ---")
        lines.append(f"Shares: {sell['filled_shares']:,.0f}  Notional: ${sell['notional']:,.0f}")
        lines.append(f"Leg slippage: {sell['slip_bps']:.1f} bps (${sell['slip_cost']:,.2f})")
        for s in sell["symbols"]:
            tag = "OK" if s["slip_bps"] <= 5 else ("!" if s["slip_bps"] <= 15 else "!!")
            lines.append(
                f"  {s['symbol']:6s} {s['filled']:>6,}/{s['target']:>6,}  "
                f"VWAP ${s['vwap']:.4f}  arrival ${s['arrival_mid']:.4f}  "
                f"slip {s['slip_bps']:+.1f}bps {tag}"
            )
        lines.append("")

    # Buy leg
    buy = tca["buy"]
    if buy["order_count"] > 0:
        lines.append(f"--- BUYS ({buy['order_count']} orders) ---")
        lines.append(f"Shares: {buy['filled_shares']:,.0f}  Notional: ${buy['notional']:,.0f}")
        lines.append(f"Leg slippage: {buy['slip_bps']:.1f} bps (${buy['slip_cost']:,.2f})")
        for s in buy["symbols"]:
            tag = "OK" if s["slip_bps"] <= 5 else ("!" if s["slip_bps"] <= 15 else "!!")
            lines.append(
                f"  {s['symbol']:6s} {s['filled']:>6,}/{s['target']:>6,}  "
                f"VWAP ${s['vwap']:.4f}  arrival ${s['arrival_mid']:.4f}  "
                f"slip {s['slip_bps']:+.1f}bps {tag}"
            )
        lines.append("")

    lines.append(f"Plan: {plan_path.name}")
    lines.append(f"Log: {exec_path.name}")

    return "\n".join(lines)


# ============================================================================
# SECTION 8: MAIN ORCHESTRATOR
# ============================================================================

def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="T2 Factor Timing — Schwab TWAP Trader"
    )
    parser.add_argument(
        "--account-name", default=DEFAULT_ACCOUNT_NAME,
        help=f"Schwab account to trade. Default: {DEFAULT_ACCOUNT_NAME}",
    )
    parser.add_argument("--live", action="store_true", help="Submit orders to Schwab.")
    parser.add_argument("--confirm-live", action="store_true", help="Second safety confirmation for live trading.")
    parser.add_argument("--twap-window", type=int, default=DEFAULT_TWAP_WINDOW_MINUTES, help="TWAP window in minutes.")
    parser.add_argument("--twap-slices", type=int, default=DEFAULT_TWAP_SLICES, help="Number of TWAP slices.")
    parser.add_argument("--min-trade", type=float, default=DEFAULT_MIN_TRADE_DOLLARS, help="Minimum trade size in dollars.")
    parser.add_argument("--cash-buffer", type=float, default=DEFAULT_CASH_BUFFER_PCT, help="Cash buffer as fraction of equity.")
    parser.add_argument("--no-liquidity-cap", action="store_true", help="Disable ADV liquidity cap.")
    parser.add_argument("--liq-maxpart", type=float, default=DEFAULT_LIQ_MAXPART, help="Max fraction of daily ADV per position.")
    parser.add_argument("--notify", action="store_true", help="Send iMessage notifications.")
    parser.add_argument(
        "--max-unfilled-sell-pct", type=float, default=DEFAULT_MAX_UNFILLED_SELL_PCT,
        help="Abort the BUY phase if more than this fraction of planned sell "
             "notional remains unfilled after the sell leg (safety check for "
             "available cash before committing to buys).",
    )
    parser.add_argument(
        "--max-cleanup-spread-bps", type=float, default=DEFAULT_MAX_CLEANUP_SPREAD_BPS,
        help="Skip the end-of-window market-order cleanup sweep for any "
             "symbol whose bid/ask spread exceeds this many bps (avoids "
             "paying a large spread cost on a blind market order).",
    )
    parser.add_argument(
        "--max-slice-carry-multiple", type=float, default=DEFAULT_MAX_SLICE_CARRY_MULTIPLE,
        help="Cap any single TWAP slice's attempted quantity at this multiple "
             "of its base per-slice size, even if prior failures have built "
             "up a large carry-forward balance. Prevents a string of failed "
             "slices from forcing one giant order at the end.",
    )
    parser.add_argument(
        "--max-target-weights-age-days", type=float, default=DEFAULT_MAX_TARGET_WEIGHTS_AGE_DAYS,
        help="Refuse to trade live if T2_FINAL_T60_VALUE.xlsx is older than this "
             "many days (catches a forgotten monthly regeneration). Dry runs "
             "only print a warning.",
    )
    parser.add_argument(
        "--force-rerun", action="store_true",
        help="Allow a live re-run even though a marker already exists for "
             "today. Only proceeds after confirming via Schwab account_orders "
             "that there are no open/working orders on the account — if any "
             "are found, the run still refuses and you must resolve them in "
             "the Schwab UI first.",
    )
    args = parser.parse_args()

    return RunConfig(
        account_name=args.account_name,
        live=args.live,
        confirm_live=args.confirm_live,
        twap_window_minutes=args.twap_window,
        twap_slices=args.twap_slices,
        min_trade_dollars=args.min_trade,
        cash_buffer_pct=args.cash_buffer,
        apply_liquidity_cap=not args.no_liquidity_cap,
        liq_maxpart=args.liq_maxpart,
        notify=args.notify,
        output_dir=OUTPUT_DIR,
        max_unfilled_sell_pct=args.max_unfilled_sell_pct,
        max_cleanup_spread_bps=args.max_cleanup_spread_bps,
        max_slice_carry_multiple=args.max_slice_carry_multiple,
        max_target_weights_age_days=args.max_target_weights_age_days,
        force_rerun=args.force_rerun,
    )


def main() -> None:
    config = parse_args()
    date_str = datetime.now().strftime("%Y%m%d")
    is_live = config.live and config.confirm_live

    print("=" * 70)
    print("T2 FACTOR TIMING — SCHWAB TWAP TRADER")
    print("=" * 70)
    print(f"  Date:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Account:        {config.account_name}")
    print(f"  Mode:           {'*** LIVE ***' if is_live else 'DRY RUN'}")
    print(f"  TWAP Window:    {config.twap_window_minutes} min / {config.twap_slices} slices")
    print(f"  Min Trade:      ${config.min_trade_dollars:,.0f}")
    print(f"  Cash Buffer:    {config.cash_buffer_pct:.1%}")
    print(f"  Liquidity Cap:  {'ON (maxpart={:.0%})'.format(config.liq_maxpart) if config.apply_liquidity_cap else 'OFF'}")
    print(f"  Max Unfilled Sell %: {config.max_unfilled_sell_pct:.1%} (abort buys above this)")
    print(f"  Max Cleanup Spread:  {config.max_cleanup_spread_bps:.0f} bps (skip cleanup above this)")
    print(f"  Notifications:  {'ON' if config.notify else 'OFF'}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Load target weights
    # ------------------------------------------------------------------
    print("STEP 1: Loading target weights from T2_FINAL_T60_VALUE.xlsx...")
    check_target_weights_staleness(config.max_target_weights_age_days, is_live)
    check_market_hours(config.twap_window_minutes, is_live)
    target_weights = load_target_weights()
    if target_weights.sum() < 0.01:
        raise TradingError("Target weights sum to ~0. Nothing to trade.")
    print()

    # ------------------------------------------------------------------
    # Step 2: Connect to Schwab and pull holdings
    # ------------------------------------------------------------------
    print("STEP 2: Connecting to Schwab API...")
    client = get_schwab_client()
    account_hash, account_number = find_account_hash(client, config.account_name)
    masked_number = f"...{account_number[-4:]}" if len(account_number) >= 4 else (account_number or "UNKNOWN")
    print(f"  Account hash:   {account_hash[:8]}...")
    print(f"  Account number: {masked_number}  <-- VERIFY this matches the intended "
          f"'{config.account_name}' account before approving live trades")

    details = get_account_details(client, account_hash)
    check_margin_account(details)
    holdings, investable_cash, total_equity = parse_holdings(details)
    print(f"  Total equity:     ${total_equity:>14,.2f}")
    print(f"  Investable cash:  ${investable_cash:>14,.2f}")
    if not holdings.empty:
        print(f"  Positions:        {len(holdings)}")
        for _, row in holdings.iterrows():
            print(f"    {row['Symbol']:6s}  {row['Long Quantity']:>8.0f} shares  ${row['Market Value']:>12,.2f}")
    else:
        print("  No existing positions.")
    print()

    # ------------------------------------------------------------------
    # Step 3: Apply liquidity cap
    # ------------------------------------------------------------------
    if config.apply_liquidity_cap:
        print("STEP 3: Applying liquidity (ADV) position cap...")
        target_weights = apply_liquidity_cap_to_weights(
            target_weights, aum=total_equity, maxpart=config.liq_maxpart,
        )
        print(f"  Post-cap weights: {len(target_weights)} ETFs, sum = {target_weights.sum():.6f}")
        print()

    # ------------------------------------------------------------------
    # Step 4: Get reference prices (live quotes)
    # ------------------------------------------------------------------
    all_etfs = sorted(set(target_weights.index) | (set(holdings["Symbol"]) - {SNAXX_SYMBOL}) if not holdings.empty else set(target_weights.index))
    print(f"STEP 4: Fetching live quotes for {len(all_etfs)} ETFs...")
    quotes = get_live_quotes(client, all_etfs)
    ref_prices = pd.Series({sym: q.mid for sym, q in quotes.items()})
    for sym in sorted(ref_prices.index):
        q = quotes[sym]
        print(f"    {sym:6s}  bid={q.bid:>8.2f}  ask={q.ask:>8.2f}  "
              f"mid={q.mid:>8.2f}  spread={q.spread_bps:>5.1f}bps")
    print()

    # ------------------------------------------------------------------
    # Step 5: Build trade plan
    # ------------------------------------------------------------------
    print("STEP 5: Building trade plan...")
    plan, summary = build_trade_plan(
        target_weights, holdings, investable_cash, total_equity, ref_prices, config,
    )

    sells = plan[plan["Action"] == "SELL"]
    buys = plan[plan["Action"] == "BUY"]
    holds = plan[plan["Action"] == "HOLD"]

    print(f"\n  SELL orders: {len(sells[sells['Shares to Trade'] != 0])}")
    for _, row in sells[sells["Shares to Trade"] != 0].iterrows():
        print(f"    {row['Symbol']:6s}  SELL {abs(row['Shares to Trade']):>8.0f} shares  "
              f"~${abs(row['Trade Dollars']):>10,.2f}")

    print(f"  BUY orders:  {len(buys[buys['Shares to Trade'] != 0])}")
    for _, row in buys[buys["Shares to Trade"] != 0].iterrows():
        print(f"    {row['Symbol']:6s}  BUY  {row['Shares to Trade']:>8.0f} shares  "
              f"~${row['Trade Dollars']:>10,.2f}")

    print(f"  HOLD:        {len(holds)}")
    print()

    # ------------------------------------------------------------------
    # Step 6: Write trade plan
    # ------------------------------------------------------------------
    plan_path = write_trade_plan_workbook(config.output_dir, plan, summary, target_weights, date_str)
    print()

    # ------------------------------------------------------------------
    # Step 7: Check for duplicate live marker
    # ------------------------------------------------------------------
    marker_path = config.output_dir / f"schwab_live_marker_{date_str}.json"
    if is_live and marker_path.exists():
        existing = json.loads(marker_path.read_text())
        # Refuse to re-run on ANY existing marker for the day, not just
        # "completed" statuses. A marker showing "STARTED" means a prior
        # live run may have submitted orders to Schwab before crashing —
        # rerunning blind could resubmit those same orders.
        if not config.force_rerun:
            raise TradingError(
                f"Live marker already exists for {date_str} with status "
                f"'{existing.get('status')}'. Refusing to re-run — a prior live "
                "run may have already submitted orders to Schwab. Pass "
                "--force-rerun to retry (it will first verify via Schwab that "
                "no orders are still open before proceeding), or manually "
                "verify Schwab and delete the marker file."
            )
        # --force-rerun: only proceed if Schwab itself confirms there is no
        # order still working. If anything is open, refuse — recomputing a
        # fresh plan from current holdings while an old order is still live
        # could double up on shares that order is also trying to trade.
        print(f"  --force-rerun: marker shows '{existing.get('status')}'. "
              "Checking Schwab for open/working orders before proceeding...")
        open_orders = _get_open_orders(client, account_hash)
        if open_orders:
            details_str = ", ".join(
                f"{o.get('orderId')}({str(o.get('status'))})" for o in open_orders
            )
            raise TradingError(
                f"--force-rerun blocked: {len(open_orders)} open/working order(s) "
                f"found on the account: {details_str}. Resolve them in the Schwab "
                "UI first (let them fill or cancel them), then retry."
            )
        print("  No open orders found. Proceeding — plan recomputed from current "
              "Schwab holdings/cash above already reflects whatever already executed.")

    # ------------------------------------------------------------------
    # Step 8: Execute trades (or dry-run summary)
    # ------------------------------------------------------------------
    if not is_live:
        print("=" * 70)
        print("DRY RUN COMPLETE — no orders submitted.")
        print(f"To execute live: python \"{__file__}\" --live --confirm-live --notify")
        print("=" * 70)

        notify_user(config, "T2 Schwab DRY RUN", (
            f"Account: {config.account_name}\n"
            f"Total equity: ${total_equity:,.2f}\n"
            f"Sells: {len(sells[sells['Shares to Trade'] != 0])}\n"
            f"Buys: {len(buys[buys['Shares to Trade'] != 0])}\n"
            f"Plan: {plan_path}"
        ))
        return

    # ---- LIVE EXECUTION ----
    print("=" * 70)
    print("*** LIVE TRADING — SUBMITTING ORDERS TO SCHWAB ***")
    print("=" * 70)
    print()

    all_results: list[SliceResult] = []
    marker_data: dict[str, Any] = {
        "status": "STARTED",
        "timestamp": datetime.now().isoformat(),
        "account_name": config.account_name,
        "date": date_str,
    }
    claim_live_marker(config.output_dir, date_str, marker_data, allow_overwrite=config.force_rerun)

    # Build the live dashboard
    dashboard = TwapDashboard(
        account_name=config.account_name,
        total_equity=total_equity,
        cash_balance=investable_cash,
        mode="LIVE",
        twap_window=config.twap_window_minutes,
        twap_slices=config.twap_slices,
    )

    # Register SELL orders now. BUY orders are registered later, AFTER the
    # sell leg completes and the buy plan has potentially been rescaled to
    # live cash — registering them upfront with stale quantities would make
    # the dashboard's fill-percentage tracking wrong if the buy plan changes.
    for _, row in plan[plan["Action"] == "SELL"].iterrows():
        qty = int(abs(row["Shares to Trade"]))
        if qty >= 1:
            dashboard.add_order(row["Symbol"], "SELL", qty, config.twap_slices)

    buys_aborted = False
    buys_aborted_reason = ""
    buy_plan_for_tca = plan  # overwritten below if buys are rescaled to live cash

    with dashboard.live():
        # Phase 1: SELLS
        dashboard.add_log("[bold]PHASE 1: Executing SELL orders via TWAP...[/bold]")
        dashboard.refresh()
        sell_results = execute_twap_leg(client, account_hash, plan, "SELL", config, dashboard=dashboard)
        all_results.extend(sell_results)

        sell_total_filled = sum(r.filled_qty for r in sell_results)
        marker_data["status"] = "SELLS_DONE"
        marker_data["sell_results_count"] = len(sell_results)
        marker_data["sell_total_filled"] = sell_total_filled
        write_marker(config.output_dir, date_str, marker_data)

        # -- Aggregate unfilled sells using the PLAN's true target shares  --
        # -- (not a sum of slice target_qty, which double-counts rolled-  --
        # -- forward quantity — see _build_target_lookup docstring).      --
        sell_target_lookup = _build_target_lookup(plan)
        sell_by_sym: dict[str, list[SliceResult]] = {}
        for r in sell_results:
            sell_by_sym.setdefault(r.symbol, []).append(r)

        unfilled_syms = []
        unfilled_notional = 0.0
        planned_sell_notional = float(plan.loc[plan["Action"] == "SELL", "Trade Dollars"].abs().sum())
        for sym, slices in sell_by_sym.items():
            total_target = sell_target_lookup.get((sym, "SELL"), sum(s.target_qty for s in slices))
            total_filled = sum(s.filled_qty for s in slices)
            if total_filled < total_target:
                unfilled_syms.append(f"{sym}({total_filled}/{total_target})")
                px = float(ref_prices.get(sym, 0.0))
                unfilled_notional += (total_target - total_filled) * px

        if unfilled_syms:
            detail = ", ".join(unfilled_syms)
            dashboard.add_log(f"[bold red]WARNING: Unfilled sells: {detail}[/bold red]")
            dashboard.refresh()
            notify_user(config, "T2 TWAP: Sell slices partially unfilled", detail)
        else:
            detail = ""

        # -- Hard safety check: if too much of the planned sell notional  --
        # -- never filled, refuse to commit to buys that may have been   --
        # -- sized assuming that cash would be available. This is an     --
        # -- explicit, loud stop — not a silent fallback.                --
        unfilled_pct = (unfilled_notional / planned_sell_notional) if planned_sell_notional > 0 else 0.0
        if unfilled_pct > config.max_unfilled_sell_pct:
            buys_aborted = True
            buys_aborted_reason = (
                f"{unfilled_pct:.1%} of planned sell notional unfilled "
                f"(${unfilled_notional:,.0f} of ${planned_sell_notional:,.0f}), "
                f"exceeding the {config.max_unfilled_sell_pct:.1%} threshold."
            )
            dashboard.add_log(f"[bold red]ABORTING BUY PHASE: {buys_aborted_reason}[/bold red]")
            dashboard.refresh()
            marker_data["status"] = "SELLS_DONE_BUYS_ABORTED"
            marker_data["unfilled_sell_pct"] = unfilled_pct
            marker_data["abort_reason"] = buys_aborted_reason
            write_marker(config.output_dir, date_str, marker_data)
            notify_user(config, "T2 TWAP: BUY PHASE ABORTED", (
                f"{buys_aborted_reason}\nBuys were NOT submitted.\n"
                f"Unfilled: {detail or 'n/a'}"
            ))
        else:
            dashboard.add_log("[bold]SELL phase complete. Waiting 5s for settlement...[/bold]")
            dashboard.refresh()
            time.sleep(5.0)

            # Re-fetch LIVE account cash before sizing buys. Never trust the
            # pre-sell plan's dollar amounts blindly — actual sell proceeds
            # can differ from the plan due to partial fills, slippage, or
            # fees, and overspending risks margin usage or rejected orders.
            #
            # Sakana Fugu Ultra audit (2026-06-30): the previous version
            # fell back SILENTLY to the stale pre-sell cash estimate if this
            # refetch failed. That violates "fail is fail, no silent
            # fallbacks" — falling back to stale (pre-sell, lower) cash here
            # would under-size buys for no disclosed reason, and there is no
            # way to know whether the stale number is even close to right.
            # We now retry a few times, and if it still fails, ABORT the buy
            # phase loudly (same pattern as the unfilled-sells abort above)
            # instead of guessing.
            live_cash: float | None = None
            cash_fetch_error: str = ""
            for attempt in range(3):
                try:
                    fresh_details = get_account_details(client, account_hash)
                    _, live_cash, _live_equity = parse_holdings(fresh_details)
                    break
                except Exception as exc:
                    cash_fetch_error = str(exc)
                    if attempt < 2:
                        time.sleep(2.0 * (attempt + 1))

            if live_cash is None:
                buys_aborted = True
                buys_aborted_reason = (
                    f"Could not re-fetch live account cash after sells (3 attempts): "
                    f"{cash_fetch_error}. Refusing to size buys from a stale pre-sell "
                    f"estimate — that could under- or over-spend with no visibility."
                )
                dashboard.add_log(f"[bold red]ABORTING BUY PHASE: {buys_aborted_reason}[/bold red]")
                dashboard.refresh()
                marker_data["status"] = "SELLS_DONE_BUYS_ABORTED"
                marker_data["abort_reason"] = buys_aborted_reason
                write_marker(config.output_dir, date_str, marker_data)
                notify_user(config, "T2 TWAP: BUY PHASE ABORTED (cash refetch failed)", (
                    f"{buys_aborted_reason}\nBuys were NOT submitted. "
                    f"Sells already executed — reconcile and rerun with --force-rerun."
                ))
                live_cash = 0.0  # only used for the printed summary below; buy_plan unused if aborted

            if not buys_aborted:
                buy_plan = scale_buy_plan_to_cash(plan, live_cash, config.cash_buffer_pct, ref_prices, total_equity)
                buy_plan_for_tca = buy_plan

                # Register BUY orders now, using the (possibly rescaled) plan.
                for _, row in buy_plan[buy_plan["Action"] == "BUY"].iterrows():
                    qty = int(abs(row["Shares to Trade"]))
                    if qty >= 1:
                        dashboard.add_order(row["Symbol"], "BUY", qty, config.twap_slices)

            # Phase 2: BUYS (skipped entirely if the cash refetch above aborted)
            if not buys_aborted:
                dashboard.add_log(
                    f"[bold]PHASE 2: Executing BUY orders via TWAP (live cash: ${live_cash:,.0f})...[/bold]"
                )
                dashboard.refresh()
                buy_results = execute_twap_leg(client, account_hash, buy_plan, "BUY", config, dashboard=dashboard)
                all_results.extend(buy_results)

                marker_data["status"] = "COMPLETED"
                marker_data["buy_results_count"] = len(buy_results)
                marker_data["buy_total_filled"] = sum(r.filled_qty for r in buy_results)
                marker_data["completed_at"] = datetime.now().isoformat()
                write_marker(config.output_dir, date_str, marker_data)

                dashboard.add_log("[bold green]EXECUTION COMPLETE[/bold green]")
                dashboard.refresh()

        time.sleep(2.0)

    # Write execution log (after exiting dashboard's screen mode)
    exec_path = write_execution_log(config.output_dir, all_results, date_str)

    # Compute detailed TCA broken down by leg and symbol. Sells use the
    # original plan; buys use the (possibly rescaled-to-cash) buy plan so
    # the "target" reflects what was actually submitted, not the pre-sell
    # estimate.
    combined_target_lookup = _build_target_lookup(plan[plan["Action"] == "SELL"])
    combined_target_lookup.update(
        _build_target_lookup(buy_plan_for_tca[buy_plan_for_tca["Action"] == "BUY"])
    )
    tca = _compute_tca(all_results, combined_target_lookup)

    print()
    print("=" * 70)
    print("BUY PHASE ABORTED — SELLS ONLY" if buys_aborted else "EXECUTION COMPLETE")
    print("=" * 70)
    if buys_aborted:
        print(f"  Reason:             {buys_aborted_reason}")
    print(f"  Total orders:       {tca['total_symbols']}")
    print(f"  Total slices:       {tca['total_slices']}")
    print(f"  Total shares:       {tca['total_filled_shares']:,.0f}")
    print(f"  Total notional:     ${tca['total_notional']:>14,.2f}")
    print(f"  Avg slippage:       {tca['avg_slippage_bps']:>6.1f} bps")
    print(f"  Slippage cost:      ${tca['total_slippage_cost']:>10,.2f}")
    print(f"  Trade plan:         {plan_path}")
    print(f"  Execution log:      {exec_path}")
    print(f"  Marker:             {marker_path}")

    # Build the detailed iMessage body with TCA
    imsg_body = _build_tca_imessage(tca, config, plan_path, exec_path)
    if buys_aborted:
        imsg_body = f"*** BUY PHASE ABORTED ***\n{buys_aborted_reason}\n\n{imsg_body}"
    notify_user(
        config,
        "T2 Schwab TWAP: BUYS ABORTED (sells only)" if buys_aborted else "T2 Schwab TWAP COMPLETE",
        imsg_body,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except TradingError as e:
        print(f"\nTRADING ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
