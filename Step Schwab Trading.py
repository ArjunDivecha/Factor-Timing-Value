#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: Step Schwab Trading.py
=============================================================================

DESCRIPTION:
    Reads target country weights from T2_FINAL_T60_VALUE.xlsx, maps countries to
    ETF tickers via AssetList.xlsx, pulls current holdings from a Schwab
    brokerage account (Equity Momentum by default), calculates the trades
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

VERSION: 1.0
LAST UPDATED: 2026-06-29
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
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

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


# ============================================================================
# SECTION 1: LOAD TARGET WEIGHTS
# ============================================================================

ETF_OVERRIDES = {
    "U.S.": "VTV",
    "US SmallCap": "VBR",
}


def load_country_etf_mapping() -> dict[str, str]:
    """Build a {Country: ETF_Ticker} mapping from AssetList.xlsx + T2 Master country order.

    The Equity Value portfolio uses value-oriented ETFs for U.S. exposure:
      U.S.       → VTV  (Vanguard Value, replaces SPY)
      US SmallCap → VBR  (Vanguard Small-Cap Value, replaces IWM)
    These overrides are applied after loading the base AssetList mapping.
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
    return schwabdev.Client(client_id, client_secret, tokens_db=tokens_db)


def find_account_hash(client: Any, account_name: str) -> str:
    """Look up the Schwab API hash value for a named account."""
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
    return matches[0]["hashValue"]


def get_account_details(client: Any, account_hash: str) -> dict[str, Any]:
    return client.account_details(account_hash, fields="positions").json()


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


def get_live_quotes(client: Any, symbols: list[str]) -> dict[str, LiveQuote]:
    """Fetch live NBBO quotes from Schwab. FAIL IS FAIL — no silent fallback."""
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
    for sym in symbols:
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

    all_etfs = set(target_weights.index) | set(current_qty.index) - {SNAXX_SYMBOL}
    non_universe = sorted(set(current_qty.index) - set(target_weights.index) - {SNAXX_SYMBOL})
    non_universe_value = float(current_mv.reindex(non_universe).fillna(0.0).sum()) if non_universe else 0.0

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
        {"Metric": "Non-Universe Holdings", "Value": ", ".join(non_universe) if non_universe else "None"},
        {"Metric": "Non-Universe Value", "Value": f"${non_universe_value:,.2f}"},
        {"Metric": "TWAP Window (min)", "Value": config.twap_window_minutes},
        {"Metric": "TWAP Slices", "Value": config.twap_slices},
    ])
    return plan, summary


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


def _is_filled(order: dict[str, Any], expected_qty: int) -> bool:
    status = str(order.get("status", "")).upper()
    if status == "FILLED":
        return True
    filled_qty = float(order.get("filledQuantity", 0) or 0)
    remaining = float(order.get("remainingQuantity", expected_qty) or expected_qty)
    return filled_qty >= expected_qty or remaining <= 0


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


def execute_twap_single_order(
    client: Any,
    account_hash: str,
    symbol: str,
    action: str,
    total_quantity: int,
    num_slices: int,
    window_minutes: int,
    dashboard: TwapDashboard | None = None,
) -> list[SliceResult]:
    """Execute a single order via TWAP: split into slices over a time window.

    Returns a list of SliceResult, one per slice.
    """
    if total_quantity <= 0:
        return []

    action = action.upper()
    interval_sec = (window_minutes * 60.0) / num_slices
    base_slice_qty = total_quantity // num_slices
    remainder = total_quantity % num_slices

    slices_qty = []
    for i in range(num_slices):
        qty = base_slice_qty + (1 if i < remainder else 0)
        slices_qty.append(qty)

    results: list[SliceResult] = []
    unfilled_carry = 0

    for i, slice_qty in enumerate(slices_qty):
        slice_qty += unfilled_carry
        unfilled_carry = 0

        if slice_qty <= 0:
            continue

        is_last = (i == num_slices - 1)

        try:
            quote = get_live_quotes(client, [symbol])[symbol]
            arrival_mid = quote.mid

            if dashboard:
                dashboard.update_quote(symbol, action, quote.bid, quote.ask)

            if is_last:
                limit_px = _marketable_limit_price(quote, action, buffer_bps=10.0)
            else:
                limit_px = _marketable_limit_price(quote, action, buffer_bps=5.0)

            if dashboard:
                dashboard.update_slice_start(symbol, action, i + 1, limit_px, arrival_mid)
                dashboard.add_log(
                    f"[{'red' if action == 'SELL' else 'green'}]{action}[/] "
                    f"[bold]{symbol}[/bold] slice {i+1}/{num_slices}: "
                    f"{slice_qty} shares @ limit ${limit_px:.2f}  "
                    f"(bid=${quote.bid:.2f} ask=${quote.ask:.2f} spread={quote.spread_bps:.0f}bps)"
                )
                dashboard.refresh()

            payload = _build_limit_payload(symbol, slice_qty, action, limit_px)
            resp = client.place_order(account_hash, payload)
            status_code = getattr(resp, "status_code", None)
            location = resp.headers.get("Location") if hasattr(resp, "headers") else None
            order_id = _extract_order_id(location)

            if status_code != 201 or order_id is None:
                if dashboard:
                    dashboard.add_log(f"[red]FAILED[/red] {symbol} slice {i+1}: HTTP {status_code}")
                    dashboard.refresh()
                results.append(SliceResult(
                    symbol=symbol, action=action, slice_num=i + 1,
                    target_qty=slice_qty, filled_qty=0,
                    limit_price=limit_px, fill_price=None,
                    arrival_mid=arrival_mid, filled=False,
                    order_id=order_id,
                    timestamp=datetime.now().isoformat(),
                    notes=f"Submit failed: HTTP {status_code}",
                ))
                unfilled_carry = slice_qty
                time.sleep(interval_sec)
                continue

            wait_deadline = time.monotonic() + interval_sec - 2.0
            filled = False
            fill_price = None
            filled_so_far = 0

            while time.monotonic() < wait_deadline:
                time.sleep(2.0)
                order_detail = _check_order_status(client, account_hash, order_id)
                if _is_filled(order_detail, slice_qty):
                    filled = True
                    fill_price = _avg_fill_price(order_detail)
                    filled_so_far = slice_qty
                    break
                fq = float(order_detail.get("filledQuantity", 0) or 0)
                if fq > filled_so_far:
                    filled_so_far = int(fq)
                    if dashboard:
                        dashboard.update_fill(symbol, action, int(fq) - (filled_so_far - int(fq)), _avg_fill_price(order_detail))
                        dashboard.refresh()
                status = str(order_detail.get("status", "")).upper()
                if status in TERMINAL_ORDER_STATUSES and status != "FILLED":
                    break

            if not filled:
                try:
                    client.cancel_order(account_hash, order_id)
                except Exception:
                    pass
                time.sleep(0.5)
                order_detail = _check_order_status(client, account_hash, order_id)
                filled_so_far = int(float(order_detail.get("filledQuantity", 0) or 0))
                if filled_so_far > 0:
                    fill_price = _avg_fill_price(order_detail)
                unfilled_this_slice = slice_qty - filled_so_far
                unfilled_carry = unfilled_this_slice

            actual_filled = filled_so_far if not filled else slice_qty
            if dashboard and actual_filled > 0:
                dashboard.update_fill(symbol, action, actual_filled, fill_price)
                status_txt = "FILLED" if filled else "PARTIAL"
                slip_info = ""
                if fill_price and arrival_mid > 0:
                    s = ((arrival_mid - fill_price) / arrival_mid * 1e4) if action == "SELL" else ((fill_price - arrival_mid) / arrival_mid * 1e4)
                    slip_info = f"  slip={s:.1f}bps"
                dashboard.add_log(
                    f"[{'green' if filled else 'yellow'}]{status_txt}[/] {symbol} slice {i+1}: "
                    f"{actual_filled}/{slice_qty} @ ${fill_price:.4f}{slip_info}" if fill_price else
                    f"[yellow]UNFILLED[/yellow] {symbol} slice {i+1}: {actual_filled}/{slice_qty}"
                )
                dashboard.refresh()

            results.append(SliceResult(
                symbol=symbol, action=action, slice_num=i + 1,
                target_qty=slice_qty,
                filled_qty=actual_filled,
                limit_price=limit_px,
                fill_price=fill_price,
                arrival_mid=arrival_mid,
                filled=filled,
                order_id=order_id,
                timestamp=datetime.now().isoformat(),
                notes="" if filled else f"Partial: {filled_so_far}/{slice_qty} filled, {unfilled_carry} rolled forward",
            ))

        except LiveQuoteError as e:
            if dashboard:
                dashboard.add_log(f"[red]QUOTE ERROR[/red] {symbol}: {e}")
                dashboard.refresh()
            results.append(SliceResult(
                symbol=symbol, action=action, slice_num=i + 1,
                target_qty=slice_qty, filled_qty=0,
                limit_price=0.0, fill_price=None,
                arrival_mid=0.0, filled=False, order_id=None,
                timestamp=datetime.now().isoformat(),
                notes=f"Quote error: {e}",
            ))
            unfilled_carry = slice_qty

        if i < num_slices - 1:
            remaining_wait = max(0, interval_sec - 2.0)
            if remaining_wait > 0:
                time.sleep(remaining_wait)

    if dashboard:
        final_status = "FILLED" if all(r.filled for r in results) else "PARTIAL"
        dashboard.update_order_status(symbol, action, final_status)
        dashboard.refresh()

    return results


def execute_twap_leg(
    client: Any,
    account_hash: str,
    plan: pd.DataFrame,
    action: str,
    config: RunConfig,
    dashboard: TwapDashboard | None = None,
) -> list[SliceResult]:
    """Execute all orders for one leg (SELL or BUY) via TWAP."""
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

    if dashboard:
        dashboard.set_leg(action.upper())
        dashboard.add_log(f"[bold]Starting {action.upper()} leg: {len(leg_plan)} orders[/bold]")
        dashboard.refresh()

    all_results: list[SliceResult] = []
    for _, row in leg_plan.iterrows():
        symbol = row["Symbol"]
        qty = int(abs(row["Shares to Trade"]))

        if dashboard:
            dashboard.update_order_status(symbol, action.upper(), "WORKING")
            dashboard.refresh()

        results = execute_twap_single_order(
            client, account_hash, symbol, action.upper(), qty,
            config.twap_slices, config.twap_window_minutes,
            dashboard=dashboard,
        )
        all_results.extend(results)
        time.sleep(0.5)

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
    """Write (or append to) a live submission marker to prevent duplicate runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"schwab_live_marker_{date_str}.json"
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


# ============================================================================
# SECTION 7b: TCA COMPUTATION & iMESSAGE FORMATTER
# ============================================================================

def _compute_tca(all_results: list[SliceResult]) -> dict[str, Any]:
    """Compute detailed Transaction Cost Analysis from execution results.

    Returns a dict with aggregate stats plus per-symbol breakdowns for
    the sell and buy legs separately.
    """
    symbols_seen: set[str] = set()
    sell_results = [r for r in all_results if r.action == "SELL"]
    buy_results = [r for r in all_results if r.action == "BUY"]

    def _leg_stats(results: list[SliceResult]) -> dict[str, Any]:
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

    sell_tca = _leg_stats(sell_results)
    buy_tca = _leg_stats(buy_results)

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
    print(f"  Notifications:  {'ON' if config.notify else 'OFF'}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Load target weights
    # ------------------------------------------------------------------
    print("STEP 1: Loading target weights from T2_FINAL_T60_VALUE.xlsx...")
    target_weights = load_target_weights()
    if target_weights.sum() < 0.01:
        raise TradingError("Target weights sum to ~0. Nothing to trade.")
    print()

    # ------------------------------------------------------------------
    # Step 2: Connect to Schwab and pull holdings
    # ------------------------------------------------------------------
    print("STEP 2: Connecting to Schwab API...")
    client = get_schwab_client()
    account_hash = find_account_hash(client, config.account_name)
    print(f"  Account hash: {account_hash[:8]}...")

    details = get_account_details(client, account_hash)
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
        if existing.get("status") in ("COMPLETED", "SELLS_DONE", "BUYS_DONE"):
            raise TradingError(
                f"Live marker already exists for {date_str} with status "
                f"'{existing.get('status')}'. Refusing to re-run. "
                "Delete the marker file to force a re-run."
            )

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
    write_marker(config.output_dir, date_str, marker_data)

    # Build the live dashboard
    dashboard = TwapDashboard(
        account_name=config.account_name,
        total_equity=total_equity,
        cash_balance=investable_cash,
        mode="LIVE",
        twap_window=config.twap_window_minutes,
        twap_slices=config.twap_slices,
    )

    # Register all orders in the dashboard
    for _, row in plan[plan["Action"] == "SELL"].iterrows():
        qty = int(abs(row["Shares to Trade"]))
        if qty >= 1:
            dashboard.add_order(row["Symbol"], "SELL", qty, config.twap_slices)
    for _, row in plan[plan["Action"] == "BUY"].iterrows():
        qty = int(abs(row["Shares to Trade"]))
        if qty >= 1:
            dashboard.add_order(row["Symbol"], "BUY", qty, config.twap_slices)

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

        unfilled_sells = [r for r in sell_results if not r.filled and r.target_qty > 0]
        if unfilled_sells:
            detail = ", ".join(f"{r.symbol}({r.filled_qty}/{r.target_qty})" for r in unfilled_sells)
            dashboard.add_log(f"[bold red]WARNING: Unfilled sells: {detail}[/bold red]")
            dashboard.refresh()
            notify_user(config, "T2 TWAP: Sell slices partially unfilled", detail)

        dashboard.add_log("[bold]SELL phase complete. Waiting 5s for settlement...[/bold]")
        dashboard.refresh()
        time.sleep(5.0)

        # Phase 2: BUYS
        dashboard.add_log("[bold]PHASE 2: Executing BUY orders via TWAP...[/bold]")
        dashboard.refresh()
        buy_results = execute_twap_leg(client, account_hash, plan, "BUY", config, dashboard=dashboard)
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

    # Compute detailed TCA broken down by leg and symbol
    tca = _compute_tca(all_results)

    print()
    print("=" * 70)
    print("EXECUTION COMPLETE")
    print("=" * 70)
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
    notify_user(config, "T2 Schwab TWAP COMPLETE", imsg_body)


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
