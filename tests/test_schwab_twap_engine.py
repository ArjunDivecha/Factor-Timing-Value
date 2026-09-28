#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: tests/test_schwab_twap_engine.py
=============================================================================

INPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value/Step Schwab Trading.py
        The real production trading engine under test. This file is loaded
        directly (not imported as a package) so the tests always exercise
        whatever is currently on disk, with no risk of testing a stale copy.
        Identical execution engine to the sister 'T2 Factor Timing Fuzzy'
        (Momentum) repo; this file is a straight copy of that repo's test
        suite, ported alongside the engine fixes (2026-06-30).

OUTPUT FILES:
    None. This is a pytest test suite; results print to the console / pytest
    report only. No files are read or written by the engine itself in these
    tests — a FAKE Schwab broker is used, so NO real network calls, NO real
    orders, and NO real money are ever involved.

VERSION: 1.0
LAST UPDATED: 2026-06-30
AUTHOR: Arjun Divecha

DESCRIPTION (for a 10th grader):
    "Step Schwab Trading.py" is a robot that buys and sells stocks for you
    through Schwab. Before trusting it with real money, we want to check
    that it behaves safely even when things go wrong — a price quote fails
    to load, an order gets stuck in a weird state, the network hiccups,
    etc. Real Schwab won't let us "break things on purpose" to test this,
    so this file builds a FAKE, fully scripted version of Schwab
    (`FakeSchwabClient`) that we can program to misbehave in exactly the
    ways a real broker occasionally does. We then run the REAL trading
    engine's `execute_twap_leg()` function against this fake broker and
    check that it always does the SAFE thing: never double-buys/double-
    sells shares, never silently loses track of a partial fill, and stops
    to ask for human help instead of guessing when it genuinely doesn't
    know what happened to an order.

    This harness was built in response to a strategic code review (Sakana
    Fugu Ultra, 2026-06-30) that recommended exactly this kind of test
    before the engine is trusted with live money.

BACKGROUND — bugs this suite specifically guards against (fixed 2026-06-30):
    1. `_is_filled()` used to treat `remainingQuantity == 0` as "fully
       filled" even when the order was CANCELED with only a partial fill —
       silently dropping the unfilled remainder with no warning.
    2. Market-order cleanup would submit a BLIND market order (no spread
       protection) if the cleanup quote fetch failed, instead of skipping.
    3. The post-sell live-cash refetch silently fell back to a stale
       pre-sell cash number if it failed, instead of aborting loudly.
    4. Cancelled orders were carried forward (resubmitted) even when their
       final status could not be confirmed as broker-terminal — risking a
       duplicate fill if the "cancelled" order was secretly still working.
    5. A carry-cap deferred-excess field was overwritten instead of
       accumulated, silently losing shares after a burst of failed slices.
    6. A submit exception after Schwab may have already accepted the order
       triggered a blind resubmission with no account-history reconciliation.
    7. SNAXX's market value was counted into investable equity even though
       SNAXX is never sold, silently under-investing the book whenever a
       SNAXX balance is held (confirmed dormant -- $0 in both live accounts
       at fix time -- but real if a balance is ever swept in).

DEPENDENCIES:
    - pytest
    - pandas

USAGE:
    cd "/Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Value"
    python3 -m pytest tests/test_schwab_twap_engine.py -v

NOTES:
    - All tests pass (no xfail). The submit-exception-after-acceptance gap
      found while building this harness (place_order() raising right after
      Schwab actually accepted the order) was fixed 2026-06-30 by reusing
      the same account-history recovery path as the 201-no-Location case.
=============================================================================
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Load the REAL production script as a module (it has spaces in its filename
# and is not a package, so it can't be `import`ed normally).
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve().parent.parent / "Step Schwab Trading.py"
_spec = importlib.util.spec_from_file_location("schwab_trading_engine", SCRIPT_PATH)
sst = importlib.util.module_from_spec(_spec)
sys.modules["schwab_trading_engine"] = sst
_spec.loader.exec_module(sst)


# ============================================================================
# FAKE CLOCK — lets the test suite run in milliseconds instead of minutes by
# replacing time.sleep (no-op) and time.monotonic (fake, fast-forwarding
# counter) for the duration of each test.
# ============================================================================

class FakeClock:
    def __init__(self, step: float = 1.0):
        self.t = 0.0
        self.step = step

    def monotonic(self) -> float:
        self.t += self.step
        return self.t

    def sleep(self, _seconds: float) -> None:
        return None


@pytest.fixture()
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    clock = FakeClock(step=1.0)
    monkeypatch.setattr(time, "sleep", clock.sleep)
    monkeypatch.setattr(time, "monotonic", clock.monotonic)
    return clock


# ============================================================================
# FAKE SCHWAB BROKER
# ============================================================================

class FakeResponse:
    """Mimics the `requests.Response`-like object schwabdev returns."""

    def __init__(self, status_code: int = 200, json_data: Any = None,
                 headers: dict[str, str] | None = None, text: str = ""):
        self.status_code = status_code
        self._json_data = {} if json_data is None else json_data
        self.headers = headers or {}
        self.text = text

    def json(self) -> Any:
        return self._json_data


TERMINAL = {"FILLED", "CANCELED", "REJECTED", "EXPIRED", "REPLACED"}


class OrderScript:
    """Describes the full scripted lifecycle of ONE order, from the moment
    it is placed to its final status. Each test builds one of these per
    order to force a specific real-world broker behavior.

    poll_sequence: list of (status, filled_qty) tuples (or raw dict
        overrides) consumed one-per-call by order_details() BEFORE
        cancel_order() has been called on this order. Once exhausted, the
        last entry repeats.
    raise_first_n_polls: order_details() raises a ConnectionError for this
        many calls (across poll + cancel-confirm + final-check phases)
        before honoring poll_sequence / post-cancel values.
    post_cancel_status / post_cancel_filled_qty: what order_details()
        reports AFTER cancel_order() has been called on this order. If
        post_cancel_filled_qty is None, it reuses the last polled fill.
    post_cancel_raises: if True, order_details() keeps raising forever
        after cancel — simulating "we can never find out what happened".
    """

    def __init__(
        self,
        poll_sequence: list[tuple[str, int] | dict] | None = None,
        raise_first_n_polls: int = 0,
        post_cancel_status: str = "CANCELED",
        post_cancel_filled_qty: int | None = None,
        post_cancel_raises: bool = False,
    ):
        self.poll_sequence = poll_sequence or [("WORKING", 0)]
        self.raise_first_n_polls = raise_first_n_polls
        self.post_cancel_status = post_cancel_status
        self.post_cancel_filled_qty = post_cancel_filled_qty
        self.post_cancel_raises = post_cancel_raises

        self.cancel_called = False
        self._poll_idx = 0
        self._call_count = 0
        self.last_known_fill = 0


class FakeSchwabClient:
    """A fully scripted stand-in for `schwabdev.Client`. Implements exactly
    the methods `Step Schwab Trading.py` calls: quotes, place_order,
    cancel_order, order_details, account_orders, preview_order.
    """

    def __init__(self) -> None:
        self._next_id = 10_000
        self.orders: dict[str, OrderScript] = {}
        self.order_meta: dict[str, tuple[str, str, int]] = {}  # oid -> (symbol, action, qty)
        self.discoverable: set[str] = set()
        self.order_entered_time: dict[str, datetime] = {}

        self.quote_book: dict[str, dict[str, float]] = {}
        self.quote_fail_after_n_calls: dict[str, int] = {}
        self._quote_call_count: dict[str, int] = {}

        self.queued_scripts: dict[tuple[str, str], list[OrderScript]] = {}
        self.default_scripts: dict[tuple[str, str], OrderScript] = {}
        self.submit_behavior: dict[tuple[str, str], str] = {}  # "normal" (default) | "no_location_recoverable" | "no_location_unrecoverable" | "submit_fail" | "submit_raises_lost" | "submit_raises_accepted"
        self.preview_reject: set[str] = set()

        self.place_order_calls: list[dict] = []
        self.cancel_order_calls: list[str] = []
        self.order_details_calls: list[str] = []

    # -- test setup helpers --------------------------------------------
    def set_quote(self, symbol: str, bid: float, ask: float) -> None:
        self.quote_book[symbol] = {"bidPrice": bid, "askPrice": ask, "lastPrice": (bid + ask) / 2}

    def queue_script(self, symbol: str, action: str, script: OrderScript) -> None:
        self.queued_scripts.setdefault((symbol, action.upper()), []).append(script)

    def set_default_script(self, symbol: str, action: str, script: OrderScript) -> None:
        self.default_scripts[(symbol, action.upper())] = script

    def set_submit_behavior(self, symbol: str, action: str, behavior: str) -> None:
        self.submit_behavior[(symbol, action.upper())] = behavior

    def _new_id(self) -> str:
        self._next_id += 1
        return str(self._next_id)

    def _order_json(self, oid: str, symbol: str, action: str, qty: int, status: str, fq: int) -> dict:
        is_terminal = status in TERMINAL
        # Realistic Schwab semantic: remainingQuantity reflects what is
        # still WORKING at the exchange, which is 0 for ANY terminal order
        # -- including a CANCELED order that only partially filled. This is
        # the exact ambiguity that caused the original _is_filled() bug.
        remaining = 0 if is_terminal else max(0, qty - fq)
        activity = []
        if fq > 0:
            activity = [{"executionLegs": [{"quantity": fq, "price": 10.0}]}]
        return {
            "orderId": oid, "status": status, "filledQuantity": fq,
            "remainingQuantity": remaining, "quantity": qty,
            "orderLegCollection": [{
                "instruction": action.upper(), "quantity": qty,
                "instrument": {"symbol": symbol, "assetType": "EQUITY"},
            }],
            "orderActivityCollection": activity,
        }

    # -- schwabdev-shaped API --------------------------------------------
    def quotes(self, symbols: list[str], fields: str = "quote") -> FakeResponse:
        payload = {}
        for sym in symbols:
            self._quote_call_count[sym] = self._quote_call_count.get(sym, 0) + 1
            fail_after = self.quote_fail_after_n_calls.get(sym)
            if fail_after is not None and self._quote_call_count[sym] > fail_after:
                continue  # simulate "no quote block returned" for this symbol
            q = self.quote_book.get(sym)
            if q is None:
                continue
            payload[sym] = {"quote": q}
        return FakeResponse(200, json_data=payload)

    def preview_order(self, account_hash: str, payload: dict) -> FakeResponse:
        sym = payload["orderLegCollection"][0]["instrument"]["symbol"]
        if sym in self.preview_reject:
            return FakeResponse(200, json_data={
                "orderStrategy": {"status": "REJECTED"},
                "orderValidationResult": {"rejects": [{"activityMessage": "Simulated insufficient funds"}]},
            })
        return FakeResponse(200, json_data={"orderStrategy": {"status": "ACCEPTED"}, "orderValidationResult": {}})

    def place_order(self, account_hash: str, payload: dict) -> FakeResponse:
        self.place_order_calls.append(payload)
        leg = payload["orderLegCollection"][0]
        sym = leg["instrument"]["symbol"]
        action = leg["instruction"]
        qty = int(leg["quantity"])
        behavior = self.submit_behavior.get((sym, action), "normal")

        if behavior == "submit_fail":
            return FakeResponse(400, json_data={"message": "simulated reject"})

        if behavior == "submit_raises_lost":
            raise ConnectionError("simulated network failure (order truly never reached Schwab)")

        if behavior == "submit_raises_accepted":
            # The broker secretly DID create the order even though the
            # client's HTTP call raised before it could read the response
            # (e.g. a connection reset after the server processed it).
            oid = self._new_id()
            self._register_order(oid, sym, action, qty)
            self.discoverable.add(oid)
            raise ConnectionError("simulated network failure (but Schwab actually accepted the order)")

        oid = self._new_id()
        self._register_order(oid, sym, action, qty)

        if behavior == "no_location_recoverable":
            self.discoverable.add(oid)
            return FakeResponse(201, json_data={}, headers={})
        if behavior == "no_location_unrecoverable":
            return FakeResponse(201, json_data={}, headers={})  # NOT discoverable
        return FakeResponse(201, json_data={}, headers={"Location": f"https://api.schwab.com/orders/{oid}"})

    def _register_order(self, oid: str, sym: str, action: str, qty: int) -> None:
        key = (sym, action.upper())
        queue = self.queued_scripts.get(key)
        if queue:
            script = queue.pop(0)
        elif key in self.default_scripts:
            script = self.default_scripts[key]
        else:
            script = OrderScript(poll_sequence=[("FILLED", qty)])
        self.orders[oid] = script
        self.order_meta[oid] = (sym, action.upper(), qty)
        self.order_entered_time[oid] = datetime.now(timezone.utc)

    def add_decoy_discoverable_order(
        self, symbol: str, action: str, qty: int, seconds_before_now: float = 300.0,
    ) -> str:
        """Adds an unrelated, already-discoverable order with the SAME
        symbol+action+qty but an OLD `enteredTime` -- simulating a stale
        order from an earlier slice/run that would make naive
        symbol+side+qty-only matching ambiguous. Used to test that
        `_find_recent_order_id`'s `since` cutoff correctly excludes it."""
        oid = self._new_id()
        self._register_order(oid, symbol, action, qty)
        self.order_entered_time[oid] = datetime.now(timezone.utc) - timedelta(seconds=seconds_before_now)
        self.discoverable.add(oid)
        return oid

    def cancel_order(self, account_hash: str, order_id: str) -> FakeResponse:
        self.cancel_order_calls.append(order_id)
        script = self.orders.get(order_id)
        if script is not None:
            script.cancel_called = True
        return FakeResponse(200)

    def order_details(self, account_hash: str, order_id: str) -> FakeResponse:
        self.order_details_calls.append(order_id)
        script = self.orders[order_id]
        sym, action, qty = self.order_meta[order_id]
        script._call_count += 1

        if script._call_count <= script.raise_first_n_polls:
            raise ConnectionError("simulated transient order_details failure")

        if script.cancel_called:
            if script.post_cancel_raises:
                raise ConnectionError("simulated permanent order_details failure after cancel")
            fq = script.post_cancel_filled_qty
            if fq is None:
                fq = script.last_known_fill
            return FakeResponse(200, json_data=self._order_json(order_id, sym, action, qty, script.post_cancel_status, fq))

        idx = min(script._poll_idx, len(script.poll_sequence) - 1)
        entry = script.poll_sequence[idx]
        if script._poll_idx < len(script.poll_sequence) - 1:
            script._poll_idx += 1
        if isinstance(entry, dict):
            return FakeResponse(200, json_data=entry)
        status, fq = entry
        script.last_known_fill = fq
        return FakeResponse(200, json_data=self._order_json(order_id, sym, action, qty, status, fq))

    def account_orders(self, account_hash: str, start, end, maxResults: int = 200) -> FakeResponse:
        orders = []
        for oid in self.discoverable:
            sym, action, qty = self.order_meta[oid]
            entered = self.order_entered_time.get(oid, datetime.now(timezone.utc))
            orders.append({
                "orderId": oid,
                "orderLegCollection": [{
                    "instruction": action, "quantity": qty,
                    "instrument": {"symbol": sym},
                }],
                "status": "WORKING",
                "enteredTime": entered.isoformat(),
            })
        return FakeResponse(200, json_data=orders)


# ============================================================================
# TEST HELPERS
# ============================================================================

def make_config(**overrides) -> Any:
    defaults = dict(
        account_name="Test Account", live=True, confirm_live=True,
        twap_window_minutes=1, twap_slices=1, min_trade_dollars=0.0,
        cash_buffer_pct=0.03, apply_liquidity_cap=False, liq_maxpart=0.20,
        notify=False, output_dir=Path("."), max_unfilled_sell_pct=0.05,
        max_cleanup_spread_bps=50.0, max_slice_carry_multiple=3.0,
        max_target_weights_age_days=35.0, force_rerun=False,
        exposure_mode="off",
    )
    defaults.update(overrides)
    return sst.RunConfig(**defaults)


def make_plan(symbol: str, action: str, qty: int) -> pd.DataFrame:
    return pd.DataFrame([{"Symbol": symbol, "Action": action.upper(), "Shares to Trade": float(qty)}])


def make_plan_multi(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"Symbol": sym, "Action": action.upper(), "Shares to Trade": float(qty)}
        for sym, action, qty in rows
    ])


def total_filled(results: list, symbol: str) -> int:
    return sum(r.filled_qty for r in results if r.symbol == symbol)


# ============================================================================
# 1. HAPPY PATH
# ============================================================================

def test_happy_path_full_fill_no_cleanup(fake_clock):
    """All slices fill immediately -> total fill equals target, no cleanup."""
    client = FakeSchwabClient()
    client.set_quote("EWZ", 10.00, 10.02)
    client.set_default_script("EWZ", "SELL", OrderScript(poll_sequence=[("FILLED", 100)]))

    plan = make_plan("EWZ", "SELL", 100)
    results = sst.execute_twap_leg(client, "ACCT", plan, "SELL", make_config(twap_slices=1))

    assert total_filled(results, "EWZ") == 100
    assert all(r.filled for r in results)
    assert len(client.place_order_calls) == 1, "no cleanup market order should have been needed"


# ============================================================================
# 2. PARTIAL FILL, CONFIRMED TERMINAL -> TRUE REMAINDER CARRIES TO NEXT SLICE
# ============================================================================

def test_partial_fill_then_confirmed_canceled_carries_true_remainder(fake_clock):
    """Slice 1 partially fills (40/150) and is CONFIRMED CANCELED. The TRUE
    60-share remainder (150 - 90 already done across 2 slices) must show up
    correctly, not be silently dropped (the original Fugu-audit bug) and not
    be double-counted."""
    client = FakeSchwabClient()
    client.set_quote("EPHE", 20.00, 20.02)
    # Slice 1 (qty=75): partially fills 40, confirmed CANCELED at 40.
    client.queue_script("EPHE", "SELL", OrderScript(
        poll_sequence=[("WORKING", 40)],
        post_cancel_status="CANCELED", post_cancel_filled_qty=40,
    ))
    # Slice 2 (qty=75 base + 35 carry = 110... but slice qty resolved by engine):
    # whatever quantity slice 2 actually requests, fill it completely.
    client.set_default_script("EPHE", "SELL", OrderScript(poll_sequence=[("FILLED", 10_000)]))

    plan = make_plan("EPHE", "SELL", 150)
    results = sst.execute_twap_leg(client, "ACCT", plan, "SELL", make_config(twap_slices=2))

    # Slice 1 result must report the TRUE partial fill, not "complete".
    slice1 = [r for r in results if r.slice_num == 1][0]
    assert slice1.filled_qty == 40
    assert slice1.filled is False

    # Total across both slices must equal the full original order — the
    # unfilled 35 from slice 1 (75-40) must have rolled into slice 2.
    assert total_filled(results, "EPHE") == 150


# ============================================================================
# 3. FILL COUNT NEVER REGRESSES ON FLAKY / OUT-OF-ORDER READS
# ============================================================================

def test_fill_count_never_regresses_on_flaky_reads(fake_clock):
    """A later read reporting a LOWER fill count than a previous read (Schwab
    eventual-consistency lag) must never cause the engine to report less
    filled than it already confirmed."""
    client = FakeSchwabClient()
    client.set_quote("EWY", 15.00, 15.02)
    client.set_default_script("EWY", "SELL", OrderScript(
        poll_sequence=[("WORKING", 30), ("WORKING", 25), ("FILLED", 100)],
    ))

    plan = make_plan("EWY", "SELL", 100)
    results = sst.execute_twap_leg(client, "ACCT", plan, "SELL", make_config(twap_slices=1))

    assert total_filled(results, "EWY") == 100
    assert all(r.filled_qty >= 30 for r in results if r.symbol == "EWY")


def test_transient_status_check_exception_does_not_abort_run(fake_clock):
    """A few transient order_details() failures must not crash the run or
    cause incorrect carry — the engine should keep polling and eventually
    succeed."""
    client = FakeSchwabClient()
    client.set_quote("INDA", 50.00, 50.04)
    client.set_default_script("INDA", "SELL", OrderScript(
        raise_first_n_polls=2,
        poll_sequence=[("FILLED", 60)],
    ))

    plan = make_plan("INDA", "SELL", 60)
    results = sst.execute_twap_leg(client, "ACCT", plan, "SELL", make_config(twap_slices=1, twap_window_minutes=2))

    assert total_filled(results, "INDA") == 60


# ============================================================================
# 4. CANCEL CONFIRMED BUT STAYS NON-TERMINAL -> MANUAL REQUIRED, NO CARRY
# ============================================================================

def test_status_stays_nonterminal_after_cancel_marks_manual_required(fake_clock):
    """If, even after cancel + retries, the order's status is STILL not a
    confirmed terminal state, the engine must NOT carry the 'unfilled'
    remainder forward (that risks a duplicate fill from the still-possibly-
    live order) -- it must stop trading the symbol instead."""
    client = FakeSchwabClient()
    client.set_quote("EZA", 60.00, 60.06)
    client.set_default_script("EZA", "SELL", OrderScript(
        poll_sequence=[("WORKING", 0)],
        post_cancel_status="WORKING",  # cancel "succeeded" at the API call level, but status never moved to CANCELED
        post_cancel_filled_qty=0,
    ))

    plan = make_plan("EZA", "SELL", 200)
    config = make_config(twap_slices=1)
    results = sst.execute_twap_leg(client, "ACCT", plan, "SELL", config)

    # No cleanup market order should have been attempted for a manual-
    # required symbol -- only the original limit order was ever placed.
    assert len(client.place_order_calls) == 1
    assert "MANUAL RECONCILIATION REQUIRED" in results[-1].notes
    assert total_filled(results, "EZA") == 0


def test_order_details_permanently_unreachable_marks_manual_required(fake_clock):
    """If order_details() can never be reached at all (total API outage for
    this order), the engine must treat truth as unknown and stop, not
    assume zero and resubmit."""
    client = FakeSchwabClient()
    client.set_quote("KSA", 35.00, 35.05)
    client.set_default_script("KSA", "SELL", OrderScript(
        poll_sequence=[("WORKING", 0)],
        post_cancel_raises=True,
    ))

    plan = make_plan("KSA", "SELL", 80)
    results = sst.execute_twap_leg(client, "ACCT", plan, "SELL", make_config(twap_slices=1))

    assert len(client.place_order_calls) == 1, "must not resubmit an order whose fate is unknown"
    assert "MANUAL RECONCILIATION REQUIRED" in results[-1].notes


# ============================================================================
# 5. CANCEL ITSELF THROWS -- SUBSEQUENT STATUS CHECK STILL DRIVES THE OUTCOME
# ============================================================================

def test_cancel_order_raises_but_status_still_resolves_correctly(fake_clock):
    """cancel_order() throwing (e.g. already-filled-can't-cancel error from
    the broker) must not crash the run, and the TRUE state must still come
    from order_details(), not be guessed."""
    client = FakeSchwabClient()
    client.set_quote("MCHI", 45.00, 45.05)
    client.set_default_script("MCHI", "SELL", OrderScript(poll_sequence=[("WORKING", 50), ("FILLED", 90)]))

    real_cancel = client.cancel_order
    def raising_cancel(account_hash, order_id):
        raise RuntimeError("simulated: order already filled, cannot cancel")
    client.cancel_order = raising_cancel  # type: ignore[assignment]

    plan = make_plan("MCHI", "SELL", 90)
    results = sst.execute_twap_leg(client, "ACCT", plan, "SELL", make_config(twap_slices=1))

    assert total_filled(results, "MCHI") == 90


# ============================================================================
# 6. SUBMIT RAISES *AFTER* THE BROKER SECRETLY ACCEPTED THE ORDER
#    (fixed 2026-06-30 -- see module docstring)
# ============================================================================

def test_submit_raises_after_broker_accepted_order(fake_clock):
    """Regression test (fixed 2026-06-30, independent adversarial review):
    if place_order() raises an exception (e.g. a connection reset) AFTER
    Schwab already accepted the order, the engine must recover the real
    order via account history -- not blindly treat it as a clean failure
    and resubmit the same shares (a real double-execution risk)."""
    client = FakeSchwabClient()
    client.set_quote("VNM", 18.00, 18.02)
    client.set_submit_behavior("VNM", "SELL", "submit_raises_accepted")
    client.set_default_script("VNM", "SELL", OrderScript(poll_sequence=[("FILLED", 70)]))

    plan = make_plan("VNM", "SELL", 70)
    results = sst.execute_twap_leg(client, "ACCT", plan, "SELL", make_config(twap_slices=1))

    notes = " ".join(r.notes for r in results)
    assert "Submit error" not in notes, (
        "engine should have recovered the accepted order instead of "
        "treating the submit as a clean failure"
    )
    assert total_filled(results, "VNM") == 70
    assert any(r.order_id is not None for r in results)


def test_submit_raises_and_order_truly_unrecoverable_flags_manual_no_carry(fake_clock):
    """If place_order() raises AND the order is genuinely not findable in
    account history (it really never reached Schwab, or isn't discoverable
    yet), the engine must NOT carry the quantity forward -- that risks a
    duplicate execution if the order actually IS sitting there. It must
    flag for manual reconciliation and carry zero, exactly like the
    201-no-Location-unrecoverable case."""
    client = FakeSchwabClient()
    client.set_quote("VNM", 18.00, 18.02)
    client.set_submit_behavior("VNM", "SELL", "submit_raises_lost")  # truly never reached Schwab

    plan = make_plan("VNM", "SELL", 70)
    results = sst.execute_twap_leg(client, "ACCT", plan, "SELL", make_config(twap_slices=1))

    notes = " ".join(r.notes for r in results)
    assert "MANUAL RECONCILIATION REQUIRED" in notes
    assert total_filled(results, "VNM") == 0
    # No carry forward, and no market-cleanup sweep either -- manual_required
    # symbols are excluded from cleanup too.
    assert len(results) == 1


def test_find_recent_order_id_disambiguates_by_entered_time(fake_clock):
    """Symbol+side+qty alone is often ambiguous for equal-size TWAP slices
    (the exact same symbol/side/qty can legitimately recur). An old,
    unrelated discoverable order with the same symbol/side/qty must NOT
    block recovery of the order actually being recovered -- the `since`
    cutoff (added 2026-06-30) must filter it out by entered time."""
    client = FakeSchwabClient()
    client.set_quote("VNM", 18.00, 18.02)
    # A stale, unrelated order from 5 minutes ago with the identical
    # symbol/side/qty -- without time-based disambiguation this would make
    # the match ambiguous (2 candidates) and recovery would fail.
    client.add_decoy_discoverable_order("VNM", "SELL", 70, seconds_before_now=300.0)
    client.set_submit_behavior("VNM", "SELL", "submit_raises_accepted")
    client.set_default_script("VNM", "SELL", OrderScript(poll_sequence=[("FILLED", 70)]))

    plan = make_plan("VNM", "SELL", 70)
    results = sst.execute_twap_leg(client, "ACCT", plan, "SELL", make_config(twap_slices=1))

    notes = " ".join(r.notes for r in results)
    assert "UNRECOVERABLE" not in notes, (
        "the stale decoy order should have been excluded by the entered-time "
        "cutoff, leaving exactly one unambiguous match"
    )
    assert total_filled(results, "VNM") == 70


# ============================================================================
# 7 & 8. PLACE_ORDER RETURNS 201 WITH NO LOCATION HEADER
# ============================================================================

def test_201_no_location_recoverable_via_account_history(fake_clock):
    """schwabdev's own docstring warns: a 201 with no Location header is the
    LIKELY case for an instantly-filled marketable limit order. The engine
    must recover the real order ID via account_orders, not assume failure."""
    client = FakeSchwabClient()
    client.set_quote("EWH", 21.00, 21.02)
    client.set_submit_behavior("EWH", "SELL", "no_location_recoverable")
    client.set_default_script("EWH", "SELL", OrderScript(poll_sequence=[("FILLED", 120)]))

    plan = make_plan("EWH", "SELL", 120)
    results = sst.execute_twap_leg(client, "ACCT", plan, "SELL", make_config(twap_slices=1))

    assert total_filled(results, "EWH") == 120
    assert any(r.order_id is not None for r in results)


def test_201_no_location_unrecoverable_flags_manual_no_carry(fake_clock):
    """If the order can't be found in account history either, the engine
    must NOT carry the quantity forward (that risks a duplicate execution
    of an order that may have already filled) -- it must flag for manual
    reconciliation instead."""
    client = FakeSchwabClient()
    client.set_quote("EWW", 75.00, 75.05)
    client.set_submit_behavior("EWW", "SELL", "no_location_unrecoverable")

    plan = make_plan("EWW", "SELL", 50)
    results = sst.execute_twap_leg(client, "ACCT", plan, "SELL", make_config(twap_slices=1))

    assert total_filled(results, "EWW") == 0
    notes = " ".join(r.notes for r in results)
    assert "MANUAL RECONCILIATION REQUIRED" in notes
    assert "UNRECOVERABLE" in notes
    # No cleanup attempt either -- the share count is genuinely unknown, so
    # carrying it forward into a market sweep would risk double-execution.
    assert len(client.place_order_calls) == 1


# ============================================================================
# 9. CRITICAL REGRESSION TEST -- the exact bug Sakana Fugu Ultra found
# ============================================================================

def test_canceled_with_partial_fill_not_misread_as_complete(fake_clock):
    """THE core regression test for the critical _is_filled() bug: a broker
    response showing status=CANCELED, filledQuantity=40, remainingQuantity=0
    (Schwab's real convention -- remainingQuantity=0 means "nothing left
    WORKING", not "everything filled") must be read as a 40-share PARTIAL
    fill, never as a 100-share complete fill."""
    client = FakeSchwabClient()
    client.set_quote("THD", 30.00, 30.03)
    client.set_default_script("THD", "SELL", OrderScript(
        poll_sequence=[("WORKING", 40)],
        post_cancel_status="CANCELED", post_cancel_filled_qty=40,
    ))

    plan = make_plan("THD", "SELL", 100)
    config = make_config(twap_slices=1, max_cleanup_spread_bps=10_000)  # allow cleanup through
    results = sst.execute_twap_leg(client, "ACCT", plan, "SELL", config)

    slice_result = [r for r in results if r.slice_num == 1][0]
    assert slice_result.filled_qty == 40, "must report the TRUE partial fill, not the full 100"
    assert slice_result.filled is False

    # The 60-share remainder must show up as a cleanup market order, not be
    # silently dropped.
    cleanup_results = [r for r in results if r.slice_num > 1]
    assert len(cleanup_results) == 1
    assert cleanup_results[0].target_qty == 60


# ============================================================================
# 10 & 11. MARKET-CLEANUP FAIL-CLOSED BEHAVIOR
# ============================================================================

def test_cleanup_skipped_when_quote_fetch_fails(fake_clock):
    """If the cleanup-phase quote fetch fails, the engine must SKIP the
    market order, not submit one blindly with zero spread protection."""
    client = FakeSchwabClient()
    client.set_quote("EPOL", 25.00, 25.02)
    client.set_default_script("EPOL", "SELL", OrderScript(
        poll_sequence=[("WORKING", 0)],
        post_cancel_status="CANCELED", post_cancel_filled_qty=0,
    ))
    # Quote succeeds for slice submission (call #1) but fails from call #2
    # onward (the cleanup phase's single-symbol quote fetch).
    client.quote_fail_after_n_calls["EPOL"] = 1

    plan = make_plan("EPOL", "SELL", 50)
    results = sst.execute_twap_leg(client, "ACCT", plan, "SELL", make_config(twap_slices=1))

    cleanup_results = [r for r in results if r.slice_num > 1]
    assert len(cleanup_results) == 1
    assert cleanup_results[0].filled_qty == 0
    assert "no live quote" in cleanup_results[0].notes.lower()
    # Critically: NO market order was ever placed for the cleanup attempt.
    assert len(client.place_order_calls) == 1  # only the original limit order


def test_cleanup_skipped_when_spread_too_wide(fake_clock):
    """A wide bid/ask spread on a thin ETF must skip the market-order
    cleanup sweep rather than pay an unbounded crossing cost."""
    client = FakeSchwabClient()
    client.set_quote("EPHE", 10.00, 10.60)  # ~580bps spread
    client.set_default_script("EPHE", "SELL", OrderScript(
        poll_sequence=[("WORKING", 0)],
        post_cancel_status="CANCELED", post_cancel_filled_qty=0,
    ))

    plan = make_plan("EPHE", "SELL", 50)
    config = make_config(twap_slices=1, max_cleanup_spread_bps=50.0)
    results = sst.execute_twap_leg(client, "ACCT", plan, "SELL", config)

    cleanup_results = [r for r in results if r.slice_num > 1]
    assert len(cleanup_results) == 1
    assert "spread" in cleanup_results[0].notes.lower()
    assert len(client.place_order_calls) == 1


# ============================================================================
# 12. MALFORMED / INCOMPLETE ORDER DETAILS DOES NOT CRASH THE ENGINE
# ============================================================================

def test_malformed_order_details_does_not_crash(fake_clock):
    """A broker response missing expected fields must be handled
    conservatively (treated as NOT confirmed-filled), never crash the
    engine and never be treated as a silent success."""
    client = FakeSchwabClient()
    client.set_quote("ECH", 28.00, 28.03)
    client.set_default_script("ECH", "SELL", OrderScript(
        poll_sequence=[{"status": None}],  # missing filledQuantity, remainingQuantity, everything
    ))

    plan = make_plan("ECH", "SELL", 40)
    results = sst.execute_twap_leg(client, "ACCT", plan, "SELL", make_config(twap_slices=1))

    assert total_filled(results, "ECH") == 0
    assert not any(r.filled for r in results)


# ============================================================================
# 13. PREVIEW-ORDER PREFLIGHT BLOCKS A CONFIRMED REJECTION
# ============================================================================

def test_preview_order_rejection_blocks_first_slice_submission(fake_clock):
    """A confirmed Schwab pre-trade rejection (e.g. insufficient funds) must
    block the real LIMIT order submission, not be discovered only after the
    fact. (Note: the rejected quantity still rolls into the end-of-leg
    market-cleanup sweep today -- preview_order is only checked before the
    TWAP limit slices, not before cleanup. That sweep would presumably also
    get rejected by Schwab for the same underlying reason, so it is not a
    safety risk, just a redundant attempt; not in scope for this fix.)"""
    client = FakeSchwabClient()
    client.set_quote("TUR", 38.00, 38.05)
    client.preview_reject.add("TUR")

    plan = make_plan("TUR", "BUY", 1000)
    results = sst.execute_twap_leg(client, "ACCT", plan, "BUY", make_config(twap_slices=1))

    limit_orders = [c for c in client.place_order_calls if c.get("orderType") == "LIMIT"]
    assert len(limit_orders) == 0, "real LIMIT order must never be submitted after a confirmed preflight reject"
    assert "preflight rejected" in " ".join(r.notes for r in results).lower()


# ============================================================================
# 14. CARRY-FORWARD IS CAPPED, NOT UNBOUNDED
# ============================================================================

def test_carry_cap_limits_single_slice_size(fake_clock):
    """A string of failed slices must not force the LAST slice to attempt
    the entire original order at once -- the per-slice size must stay
    capped at max_slice_carry_multiple x the base slice size."""
    client = FakeSchwabClient()
    client.set_quote("ASHR", 30.00, 30.03)
    # Every order placed for ASHR fails to fill at all and is confirmed
    # CANCELED -- so the carry keeps accumulating across all 3 slices.
    client.set_default_script("ASHR", "SELL", OrderScript(
        poll_sequence=[("WORKING", 0)],
        post_cancel_status="CANCELED", post_cancel_filled_qty=0,
    ))

    plan = make_plan("ASHR", "SELL", 300)  # base slice = 100 over 3 slices
    config = make_config(twap_slices=3, max_slice_carry_multiple=1.0, max_cleanup_spread_bps=10_000)
    sst.execute_twap_leg(client, "ACCT", plan, "SELL", config)

    limit_order_qtys = [
        int(call["orderLegCollection"][0]["quantity"])
        for call in client.place_order_calls if call.get("orderType") == "LIMIT"
    ]
    assert limit_order_qtys == [100, 100, 100], (
        f"every TWAP slice should be capped at the 100-share base size "
        f"(max_slice_carry_multiple=1.0), got {limit_order_qtys}"
    )
    # Every one of the 3 capped 100-share attempts filled ZERO shares, so
    # the full original 300 shares are still outstanding and must ALL
    # reach cleanup -- not just the size of the last capped slice. (This
    # assertion used to read ==100, which was unknowingly encoding the
    # carry-cap orphaned-shares bug fixed 2026-06-30: with that bug, the
    # cap's own deferred excess got overwritten away each slice, and 200
    # of the 300 shares would have vanished with no carry and no cleanup.)
    market_orders = [c for c in client.place_order_calls if c.get("orderType") == "MARKET"]
    assert len(market_orders) == 1
    assert int(market_orders[0]["orderLegCollection"][0]["quantity"]) == 300


# ============================================================================
# 15. CRITICAL REGRESSION TEST -- carry-cap orphaned-shares bug
#    (found by an independent adversarial review, 2026-06-30, after the
#    first 14 tests above were already passing -- proof this kind of
#    interaction bug needs exactly this kind of end-to-end harness, not
#    just unit tests of individual helpers)
# ============================================================================

def test_carry_cap_deferred_excess_is_not_lost(fake_clock):
    """When the per-slice carry cap actually binds (a string of partial
    fills pushes the requested quantity above max_slice_carry_multiple x
    the base slice size), the EXCESS deferred by the cap must still be
    accounted for later -- not silently overwritten and lost when that
    slice's own (capped) order later turns out to be partially filled too.

    This drives 3 consecutive slices, each of which partially (or fully)
    fails to fill, with the cap deliberately set tight enough to bind on
    slices 2 and 3. The TRUE original 300 shares must be fully accounted
    for across (slice fills + final cleanup target) -- none may vanish.
    """
    client = FakeSchwabClient()
    client.set_quote("VWO", 40.00, 40.04)
    client.queue_script("VWO", "SELL", OrderScript(
        poll_sequence=[("WORKING", 60)], post_cancel_status="CANCELED", post_cancel_filled_qty=60,
    ))
    client.queue_script("VWO", "SELL", OrderScript(
        poll_sequence=[("WORKING", 70)], post_cancel_status="CANCELED", post_cancel_filled_qty=70,
    ))
    client.queue_script("VWO", "SELL", OrderScript(
        poll_sequence=[("WORKING", 0)], post_cancel_status="CANCELED", post_cancel_filled_qty=0,
    ))

    plan = make_plan("VWO", "SELL", 300)  # base slice = 100 over 3 slices
    config = make_config(twap_slices=3, max_slice_carry_multiple=1.0, max_cleanup_spread_bps=10_000)
    results = sst.execute_twap_leg(client, "ACCT", plan, "SELL", config)

    slice_filled_total = sum(r.filled_qty for r in results if r.slice_num <= 3)
    cleanup_results = [r for r in results if r.slice_num > 3]

    assert slice_filled_total == 130, f"expected 60+70+0=130 filled across 3 slices, got {slice_filled_total}"
    assert len(cleanup_results) == 1, "the true 170-share remainder must reach cleanup as ONE sweep"
    assert cleanup_results[0].target_qty == 170, (
        f"expected the full undiscovered remainder (300 - 130 = 170) to reach "
        f"cleanup, got {cleanup_results[0].target_qty} -- shares were silently "
        f"lost to the carry-cap overwrite bug"
    )


# ============================================================================
# 16. MARKET-HOURS / TIME-TO-CLOSE GATE
#    (new safety gate added 2026-06-30, independent adversarial review --
#    previously NOTHING stopped a live run from starting too close to the
#    close, or outside market hours entirely)
# ============================================================================

from zoneinfo import ZoneInfo  # noqa: E402

_ET = ZoneInfo("America/New_York")


def test_market_hours_blocks_live_run_on_weekend():
    saturday_noon = datetime(2026, 7, 4, 12, 0, tzinfo=_ET)  # a Saturday
    with pytest.raises(sst.TradingError, match="weekend"):
        sst.check_market_hours(15, is_live=True, now=saturday_noon)


def test_market_hours_blocks_live_run_before_open():
    early = datetime(2026, 6, 30, 8, 0, tzinfo=_ET)  # Tuesday, 8am ET
    with pytest.raises(sst.TradingError, match="not opened"):
        sst.check_market_hours(15, is_live=True, now=early)


def test_market_hours_blocks_live_run_after_close():
    late = datetime(2026, 6, 30, 17, 0, tzinfo=_ET)
    with pytest.raises(sst.TradingError, match="closed"):
        sst.check_market_hours(15, is_live=True, now=late)


def test_market_hours_blocks_live_run_too_close_to_close():
    # 15-min window x2 + 10-min cleanup buffer = 40 min required. 3:45pm ET
    # leaves only 15 minutes -- must block.
    almost_close = datetime(2026, 6, 30, 15, 45, tzinfo=_ET)
    with pytest.raises(sst.TradingError, match="minutes remain"):
        sst.check_market_hours(15, is_live=True, now=almost_close)


def test_market_hours_allows_live_run_mid_session():
    midday = datetime(2026, 6, 30, 11, 0, tzinfo=_ET)
    status = sst.check_market_hours(15, is_live=True, now=midday)
    assert "market open" in status


def test_one_bad_symbol_quote_does_not_stall_other_symbols(fake_clock):
    """Regression test (fixed 2026-06-30, independent adversarial review):
    get_live_quotes() used to be all-or-nothing for a batch -- ONE halted
    or bad-data symbol made the whole call raise, which carried EVERY
    symbol forward untouched for the slice. A good symbol in the same leg
    must still trade normally even while a bad symbol fails every slice."""
    client = FakeSchwabClient()
    client.set_quote("EWZ", 30.00, 30.02)  # good symbol
    # BADX gets NO quote set at all (simulates a halted/bad-data ETF) --
    # quote_book has no entry for it, so get_live_quotes treats it as "no
    # quote block returned" for BADX specifically, every slice.
    client.set_default_script("EWZ", "SELL", OrderScript(poll_sequence=[("FILLED", 100)]))

    plan = make_plan_multi([("EWZ", "SELL", 100), ("BADX", "SELL", 50)])
    results = sst.execute_twap_leg(client, "ACCT", plan, "SELL", make_config(twap_slices=1))

    assert total_filled(results, "EWZ") == 100, "EWZ should fill normally despite BADX having no quote"
    badx_notes = " ".join(r.notes for r in results if r.symbol == "BADX")
    assert "No quote" in badx_notes or "no quote" in badx_notes.lower()


def test_market_hours_never_raises_in_dry_run():
    # Dry run never submits real orders -- the gate must warn, not block.
    saturday_noon = datetime(2026, 7, 4, 12, 0, tzinfo=_ET)
    status = sst.check_market_hours(15, is_live=False, now=saturday_noon)
    assert "weekend" in status


# ============================================================================
# 17. ATOMIC LIVE-MARKER CLAIM
#    (new safety fix, 2026-06-30, independent adversarial review --
#    previously the duplicate-run check (.exists()) and the first marker
#    write were two separate steps with a race window between them)
# ============================================================================

def test_claim_live_marker_succeeds_when_no_marker_exists(tmp_path):
    path = sst.claim_live_marker(tmp_path, "20260630", {"status": "STARTED"}, allow_overwrite=False)
    assert path.exists()
    assert json.loads(path.read_text())["status"] == "STARTED"


def test_claim_live_marker_raises_if_marker_already_exists(tmp_path):
    # Simulates the race: another process's marker landed between this
    # process's .exists() check and its own claim attempt.
    marker_path = tmp_path / "schwab_live_marker_20260630.json"
    marker_path.write_text('{"status": "STARTED"}')
    with pytest.raises(sst.TradingError, match="another process"):
        sst.claim_live_marker(tmp_path, "20260630", {"status": "STARTED"}, allow_overwrite=False)


def test_claim_live_marker_allows_overwrite_for_force_rerun(tmp_path):
    marker_path = tmp_path / "schwab_live_marker_20260630.json"
    marker_path.write_text('{"status": "SELLS_DONE"}')
    path = sst.claim_live_marker(tmp_path, "20260630", {"status": "STARTED"}, allow_overwrite=True)
    assert json.loads(path.read_text())["status"] == "STARTED"


# ============================================================================
# 18. SCALE_BUY_PLAN_TO_CASH USES THE SAME (EQUITY-BASED) BUFFER BASIS AS
#     build_trade_plan -- NOT available_cash * pct
#    (fixed 2026-06-30, independent adversarial review)
# ============================================================================

def test_scale_buy_plan_uses_equity_based_buffer_not_cash_based():
    """If the rescale path used available_cash * cash_buffer_pct (the old
    behavior), a far-smaller-than-equity cash balance would reserve a tiny
    buffer and let buys consume almost all of it. With the fix, the buffer
    is total_equity * cash_buffer_pct, matching build_trade_plan exactly --
    a materially larger, more conservative reservation when cash is thin
    relative to total equity."""
    plan = pd.DataFrame([
        {"Symbol": "EWZ", "Action": "BUY", "Shares to Trade": 1000.0,
         "Trade Dollars": 30_000.0, "Reference Price": 30.0},
    ])
    reference_prices = pd.Series({"EWZ": 30.0})

    # $40,000 available cash, but total equity is $6,700,000 -- 3% of
    # equity ($201,000) swamps the available cash entirely.
    scaled = sst.scale_buy_plan_to_cash(
        plan, available_cash=40_000.0, cash_buffer_pct=0.03,
        reference_prices=reference_prices, total_equity=6_700_000.0,
    )
    # spendable = max(0, 40_000 - 201_000) = 0 -> buys scaled to zero, not
    # to ~$38,800 (which is what 40_000 * (1-0.03) would have allowed).
    assert scaled.loc[0, "Shares to Trade"] == 0


def test_scale_buy_plan_unaffected_when_cash_is_plentiful():
    """The fix must not change behavior in the normal case where cash
    comfortably covers the planned buys plus either buffer basis."""
    plan = pd.DataFrame([
        {"Symbol": "EWZ", "Action": "BUY", "Shares to Trade": 1000.0,
         "Trade Dollars": 30_000.0, "Reference Price": 30.0},
    ])
    reference_prices = pd.Series({"EWZ": 30.0})
    scaled = sst.scale_buy_plan_to_cash(
        plan, available_cash=4_000_000.0, cash_buffer_pct=0.03,
        reference_prices=reference_prices, total_equity=6_700_000.0,
    )
    assert scaled.loc[0, "Shares to Trade"] == 1000


# ============================================================================
# 19. SNAXX VALUE EXCLUDED FROM ALLOCATABLE (B3, fixed 2026-06-30, raised by
#     an independent GLM-style review and confirmed dormant -- $0 SNAXX in
#     both live accounts at fix time, but real if a balance is ever swept in)
# ============================================================================

def test_snaxx_value_excluded_from_allocatable_and_never_traded():
    """SNAXX is a cash-equivalent sweep holding outside the ETF rotation
    universe and is never sold by build_trade_plan -- but before this fix,
    its market value was still counted into `allocatable` (derived from
    Schwab's total_equity), so BUY targets were sized against money that
    could never actually be raised. With the fix, allocatable excludes
    SNAXX's value entirely, and SNAXX itself never appears as a plan row
    (no phantom BUY/SELL of a non-strategy symbol)."""
    holdings = pd.DataFrame([
        {"Symbol": "SNAXX", "Market Value": 1_000_000.0, "Long Quantity": 1_000_000.0},
        {"Symbol": "EWZ", "Market Value": 100_000.0, "Long Quantity": 2_000.0},
    ])
    target_weights = pd.Series({"EWZ": 1.0})
    reference_prices = pd.Series({"EWZ": 50.0})
    config = make_config(cash_buffer_pct=0.0, min_trade_dollars=0.0)

    # total_equity = $2,000,000 (the $1,000,000 SNAXX sweep + $1,000,000 of
    # other account value not reflected in `holdings` here, e.g. cash) --
    # mirrors a real account where total_equity (Schwab's liquidationValue)
    # is independent of what's summed from the positions list.
    plan, summary = sst.build_trade_plan(
        target_weights=target_weights, holdings=holdings,
        investable_cash=0.0, total_equity=2_000_000.0,
        reference_prices=reference_prices, config=config,
    )

    # SNAXX must never appear as a tradeable row -- it's not in the
    # universe and must not be bought, sold, or even listed as HOLD.
    assert "SNAXX" not in set(plan["Symbol"])

    # allocatable = total_equity - cash_buffer - snaxx_value
    #             = 2,000,000 - 0 - 1,000,000 = 1,000,000
    # EWZ target weight 1.0 -> target dollars = $1,000,000 -> target qty
    # 20,000 shares @ $50. WITHOUT the fix, allocatable would have been the
    # full $2,000,000 (target qty 40,000) -- double the correct size.
    ewz_row = plan.loc[plan["Symbol"] == "EWZ"].iloc[0]
    assert ewz_row["Target Shares"] == 20_000
    assert ewz_row["Target Dollars"] == pytest.approx(1_000_000.0)

    snaxx_summary_row = summary.loc[summary["Metric"].str.contains("SNAXX", case=False)]
    assert len(snaxx_summary_row) == 1
    assert "1,000,000.00" in snaxx_summary_row.iloc[0]["Value"]


def test_zero_snaxx_balance_is_a_no_op_for_allocatable():
    """Regression guard for the common (current real-account) case: if
    SNAXX is not held at all, the fix must be a complete no-op -- allocatable
    is exactly total_equity minus the cash buffer, same as before the fix."""
    holdings = pd.DataFrame([
        {"Symbol": "EWZ", "Market Value": 100_000.0, "Long Quantity": 2_000.0},
    ])
    target_weights = pd.Series({"EWZ": 1.0})
    reference_prices = pd.Series({"EWZ": 50.0})
    config = make_config(cash_buffer_pct=0.03, min_trade_dollars=0.0)

    plan, summary = sst.build_trade_plan(
        target_weights=target_weights, holdings=holdings,
        investable_cash=0.0, total_equity=1_000_000.0,
        reference_prices=reference_prices, config=config,
    )
    # allocatable = 1,000,000 * (1 - 0.03) - 0 = 970,000 -> 19,400 shares @ $50
    ewz_row = plan.loc[plan["Symbol"] == "EWZ"].iloc[0]
    assert ewz_row["Target Shares"] == 19_400

    snaxx_summary_row = summary.loc[summary["Metric"].str.contains("SNAXX", case=False)]
    assert snaxx_summary_row.iloc[0]["Value"] == "$0.00"


# ============================================================================
# 20. EXPOSURE DIAL (added 2026-09-27; spec: Experiments Deep Dive/Regime
#     Breadth Overlay/TRADER_EXPOSURE_SPEC.md). These drive the REAL main()
#     end to end against the fake broker: target weights, the Schwab client,
#     the market-hours gate and the Rich dashboard are stubbed; everything
#     else (dial validation, sizing, liquidity-cap wiring, buy funding,
#     leverage guard, marker, audit files) is production code.
# ============================================================================

import contextlib  # noqa: E402
import math  # noqa: E402

EXPOSURE_ACCOUNT_NUMBER = "12345" + sst.ACCOUNT_NAME_TO_LAST3[sst.DEFAULT_ACCOUNT_NAME]
LV = 1_000_000.0
_REAL_WRITE_TRADE_PLAN = sst.write_trade_plan_workbook


class NullDashboard:
    """Stand-in for TwapDashboard: every method is a no-op."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def live(self):
        return contextlib.nullcontext()

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class FakeAccountClient(FakeSchwabClient):
    """FakeSchwabClient plus the account endpoints main() calls. Each
    account_details() call returns the next scripted snapshot; the last
    one repeats (so [initial, post_sell] models the post-sell refetch)."""

    def __init__(self, details_sequence: list[dict]) -> None:
        super().__init__()
        self.details_sequence = list(details_sequence)
        self.account_details_calls = 0

    def linked_accounts(self) -> FakeResponse:
        return FakeResponse(200, json_data=[{"accountNumber": EXPOSURE_ACCOUNT_NUMBER, "hashValue": "HASH"}])

    def account_details(self, account_hash: str, fields: str = "positions") -> FakeResponse:
        idx = min(self.account_details_calls, len(self.details_sequence) - 1)
        self.account_details_calls += 1
        return FakeResponse(200, json_data=self.details_sequence[idx])


def account_snapshot(
    positions: dict[str, tuple[float, float]] | None = None,
    cash: float = LV, liquidation_value: float = LV,
    buying_power: float | None = 3 * LV, acct_type: str = "MARGIN",
) -> dict:
    """positions: {symbol: (shares, market_value)}."""
    balances = {"cashBalance": cash, "liquidationValue": liquidation_value}
    if buying_power is not None:
        balances["buyingPower"] = buying_power
    return {"securitiesAccount": {
        "type": acct_type,
        "positions": [
            {"instrument": {"symbol": s}, "longQuantity": q, "marketValue": mv}
            for s, (q, mv) in (positions or {}).items()
        ],
        "currentBalances": balances,
    }}


def dial_rows(**overrides) -> list[tuple[str, Any]]:
    now = datetime.now()
    rows = {
        "target_exposure": 1.0, "breadth": 0.5, "n_above": 17, "n_valid": 34,
        "asof_date": (now - timedelta(days=1)).date().isoformat(),
        "dial_low": 0.0, "dial_high": 2.0, "rule": "test",
        "computed_at": now.isoformat(timespec="seconds"),
        "strategy": "test", "source_file": "test.xlsx",
        "source_last_date": (now - timedelta(days=1)).date().isoformat(),
        "base_weight_sum": 1.0,
    }
    rows.update(overrides)
    return [(k, v) for k, v in rows.items() if v is not _DROP]


_DROP = object()


def write_target_workbook(path: Path, dial: list[tuple[str, Any]] | None) -> Path:
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame({"Country": ["Brazil"], "Country Alpha": [0.0], "Country Weight": [1.0]}).to_excel(
            xw, sheet_name="Latest_Country_Alpha_Weights", index=False)
        if dial is not None:
            pd.DataFrame(dial, columns=["Key", "Value"]).to_excel(xw, sheet_name="Exposure_Dial", index=False)
    return path


class MainHarness:
    """Runs sst.main() against a FakeAccountClient and records what it did."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *,
                 weights: dict[str, float], details_sequence: list[dict],
                 dial: list[tuple[str, Any]] | None, quotes: dict[str, tuple[float, float]],
                 approved: bool = True, run_name: str = "run") -> None:
        self.run_dir = tmp_path / run_name
        self.run_dir.mkdir()
        self.out_dir = self.run_dir / "outputs"
        self.client = FakeAccountClient(details_sequence)
        for sym, (bid, ask) in quotes.items():
            self.client.set_quote(sym, bid, ask)
        self.cap_calls: list[dict] = []
        self.plans: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        self.client_requested = 0
        self._weights = pd.Series(weights, dtype=float)
        self._mp = monkeypatch

        target_path = write_target_workbook(self.run_dir / "targets.xlsx", dial)
        monkeypatch.setattr(sst, "T2_FINAL_PATH", target_path)
        monkeypatch.setattr(sst, "OUTPUT_DIR", self.out_dir)
        monkeypatch.setattr(sst, "EXPOSURE_DIAL_LIVE_APPROVED", approved)
        monkeypatch.setattr(sst, "TwapDashboard", NullDashboard)
        monkeypatch.setattr(sst, "load_target_weights", lambda: self._weights.copy())
        monkeypatch.setattr(sst, "check_market_hours", lambda *a, **k: "patched open")

        def fake_get_client():
            self.client_requested += 1
            return self.client
        monkeypatch.setattr(sst, "get_schwab_client", fake_get_client)

        def spy_cap(weights, aum, maxpart):
            self.cap_calls.append({"aum": aum, "maxpart": maxpart})
            return weights
        monkeypatch.setattr(sst, "apply_liquidity_cap_to_weights", spy_cap)

        real_writer = _REAL_WRITE_TRADE_PLAN
        def spy_writer(output_dir, plan, summary, *args, **kwargs):
            self.plans.append((plan.copy(), summary.copy()))
            return real_writer(output_dir, plan, summary, *args, **kwargs)
        monkeypatch.setattr(sst, "write_trade_plan_workbook", spy_writer)

    def run(self, *argv: str) -> None:
        self._mp.setattr(sys, "argv", ["Step Schwab Trading.py", "--twap-slices", "1", "--twap-window", "1", *argv])
        sst.main()

    @property
    def plan(self) -> pd.DataFrame:
        return self.plans[-1][0]

    def marker(self) -> dict | None:
        files = list(self.out_dir.glob("schwab_live_marker_*.json")) if self.out_dir.exists() else []
        return json.loads(files[0].read_text()) if files else None

    def orders(self, action: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for call in self.client.place_order_calls:
            leg = call["orderLegCollection"][0]
            if leg["instruction"] == action:
                sym = leg["instrument"]["symbol"]
                out[sym] = out.get(sym, 0) + int(leg["quantity"])
        return out


TWO_ETF_WEIGHTS = {"EWZ": 0.5, "EWJ": 0.5}
TWO_ETF_QUOTES = {"EWZ": (10.00, 10.02), "EWJ": (20.00, 20.02)}


def test_exposure_mode_defaults_to_dial(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["Step Schwab Trading.py"])
    assert sst.parse_args().exposure_mode == "dial"


# (a) off and dial@1.0 are identical ----------------------------------------

def test_exposure_one_reproduces_off_plan_and_orders_exactly(monkeypatch, tmp_path, fake_clock):
    """--exposure-mode off and dial with target_exposure=1.0 must produce the
    identical trade plan AND the identical live order stream."""
    snap = account_snapshot({"EWZ": (30_000, 300_000.0), "EWH": (1_000, 20_000.0)}, cash=680_000.0)
    quotes = {**TWO_ETF_QUOTES, "EWH": (20.00, 20.02)}
    runs = {}
    for mode in ("off", "dial"):
        h = MainHarness(monkeypatch, tmp_path, weights=TWO_ETF_WEIGHTS, details_sequence=[snap],
                        dial=dial_rows(target_exposure=1.0), quotes=quotes, run_name=mode)
        h.run("--live", "--confirm-live", "--exposure-mode", mode)
        runs[mode] = h

    pd.testing.assert_frame_equal(runs["off"].plan, runs["dial"].plan)
    pd.testing.assert_frame_equal(runs["off"].plans[-1][1], runs["dial"].plans[-1][1])
    assert runs["off"].client.place_order_calls == runs["dial"].client.place_order_calls
    assert runs["off"].client.place_order_calls, "sanity: the scenario must actually trade"
    assert runs["off"].marker()["exposure_mode"] == "off"
    assert runs["dial"].marker()["exposure_mode"] == "dial"
    assert runs["dial"].marker()["target_exposure"] == 1.0


def test_off_mode_never_reads_the_dial_sheet(monkeypatch, tmp_path, fake_clock):
    """off must work even when the Exposure_Dial sheet does not exist."""
    h = MainHarness(monkeypatch, tmp_path, weights=TWO_ETF_WEIGHTS,
                    details_sequence=[account_snapshot()], dial=None, quotes=TWO_ETF_QUOTES)
    h.run("--exposure-mode", "off")
    assert h.plan["Target Dollars"].sum() == pytest.approx(0.97 * LV)


def test_build_trade_plan_exposure_one_is_bit_identical_to_default():
    holdings = pd.DataFrame([{"Symbol": "EWZ", "Market Value": 123_457.0, "Long Quantity": 12_333.0}])
    weights = pd.Series({"EWZ": 0.37, "EWJ": 0.63})
    prices = pd.Series({"EWZ": 10.01, "EWJ": 20.01})
    cfg = make_config()
    base = sst.build_trade_plan(weights, holdings, 1.0, 987_654.321, prices, cfg)
    one = sst.build_trade_plan(weights, holdings, 1.0, 987_654.321, prices, cfg, exposure=1.0)
    pd.testing.assert_frame_equal(base[0], one[0])
    pd.testing.assert_frame_equal(base[1], one[1])


# (b) exposure 1.7 with ample buying power -----------------------------------

def test_exposure_1_7_ample_buying_power_buys_full_size(monkeypatch, tmp_path, fake_clock):
    h = MainHarness(monkeypatch, tmp_path, weights=TWO_ETF_WEIGHTS,
                    details_sequence=[account_snapshot(buying_power=3 * LV)],
                    dial=dial_rows(target_exposure=1.7), quotes=TWO_ETF_QUOTES)
    h.run("--live", "--confirm-live")

    assert h.plan["Target Dollars"].sum() == pytest.approx(1.7 * 0.97 * LV)
    buy_rows = h.plan[h.plan["Action"] == "BUY"]
    planned = dict(zip(buy_rows["Symbol"], buy_rows["Shares to Trade"].astype(int)))
    assert planned == {"EWZ": math.floor(0.5 * 1.7 * 0.97 * LV / 10.01),
                       "EWJ": math.floor(0.5 * 1.7 * 0.97 * LV / 20.01)}
    assert h.orders("BUY") == planned, "buys must be executed at full 1.7x size"
    assert h.orders("SELL") == {}
    m = h.marker()
    assert m["status"] == "COMPLETED"
    assert m["target_exposure"] == 1.7 and m["breadth"] == 0.5
    assert m["asof_date"] and m["computed_at"]


# (c) exposure 1.7 with limited buying power ---------------------------------

def test_exposure_1_7_limited_buying_power_scales_to_bp_minus_reserve(monkeypatch, tmp_path, fake_clock):
    bp = 1_000_000.0
    h = MainHarness(monkeypatch, tmp_path, weights=TWO_ETF_WEIGHTS,
                    details_sequence=[account_snapshot(buying_power=bp)],
                    dial=dial_rows(target_exposure=1.7), quotes=TWO_ETF_QUOTES)
    h.run("--live", "--confirm-live")

    spendable = bp - sst.BUYING_POWER_RESERVE_PCT * LV   # 970,000
    bought = h.orders("BUY")
    bought_dollars = bought["EWZ"] * 10.01 + bought["EWJ"] * 20.01
    assert bought_dollars <= spendable
    assert bought_dollars >= spendable - 2 * 20.01, "scaled buys should use (almost) all spendable BP"
    planned_dollars = float(h.plan.loc[h.plan["Action"] == "BUY", "Trade Dollars"].sum())
    assert planned_dollars > 1.6 * LV, "sanity: the plan itself was sized to 1.7x before scaling"
    m = h.marker()
    assert m["status"] == "COMPLETED"
    assert "scaled to buying power" in m["buy_scale_note"]
    assert m["buying_power_spendable"] == pytest.approx(spendable)
    log_x = pd.read_excel(next(h.out_dir.glob("schwab_execution_log_*.xlsx")), sheet_name="Run Info")
    assert "scaled to buying power" in dict(zip(log_x["Key"], log_x["Value"]))["buy_scale_note"]


def test_scale_buy_plan_to_buying_power_unit():
    plan = pd.DataFrame([
        {"Symbol": "EWZ", "Action": "BUY", "Shares to Trade": 1000.0, "Trade Dollars": 10_000.0, "Reference Price": 10.0},
        {"Symbol": "EWJ", "Action": "SELL", "Shares to Trade": -50.0, "Trade Dollars": -1_000.0, "Reference Price": 20.0},
    ])
    prices = pd.Series({"EWZ": 10.0, "EWJ": 20.0})
    # spendable = 8,000 - 3% x 100,000 = 5,000 -> half of the planned $10,000
    scaled = sst.scale_buy_plan_to_buying_power(plan, 8_000.0, 100_000.0, prices)
    assert scaled.loc[0, "Shares to Trade"] == 500
    assert scaled.loc[1, "Shares to Trade"] == -50
    ample = sst.scale_buy_plan_to_buying_power(plan, 50_000.0, 100_000.0, prices)
    assert ample.loc[0, "Shares to Trade"] == 1000


# (d) exposure 0 --------------------------------------------------------------

def test_exposure_zero_sells_whole_country_book_and_buys_nothing(monkeypatch, tmp_path, fake_clock):
    snap = account_snapshot(
        {"EWZ": (20_000, 200_000.0), "EWJ": (10_000, 200_000.0), "SNAXX": (50_000, 50_000.0)},
        cash=550_000.0,
    )
    h = MainHarness(monkeypatch, tmp_path, weights=TWO_ETF_WEIGHTS, details_sequence=[snap],
                    dial=dial_rows(target_exposure=0.0), quotes=TWO_ETF_QUOTES)
    h.run("--live", "--confirm-live")

    assert h.orders("SELL") == {"EWZ": 20_000, "EWJ": 10_000}
    assert h.orders("BUY") == {}
    assert "SNAXX" not in set(h.plan["Symbol"])
    assert (h.plan["Target Dollars"] == 0).all()
    assert h.cap_calls == [], "liquidity cap must be skipped at exposure 0"
    assert h.marker()["status"] == "COMPLETED"
    assert h.marker()["target_exposure"] == 0.0


# (e) bad dial -> error before any order ----------------------------------------

@pytest.mark.parametrize("dial,match", [
    (None, "not found"),
    (dial_rows(target_exposure="abc"), "not a number"),
    (dial_rows(target_exposure=True), "not a number"),
    (dial_rows(target_exposure=float("nan")), "missing or blank"),
    (dial_rows(target_exposure=2.5), "outside"),
    (dial_rows(target_exposure=-0.1), "outside"),
    (dial_rows(target_exposure=_DROP), "missing"),
    (dial_rows(computed_at=_DROP), "missing"),
    (dial_rows(asof_date=(datetime.now() - timedelta(days=45)).date().isoformat()), "days old"),
    (dial_rows(asof_date=(datetime.now() + timedelta(days=2)).date().isoformat()), "later than today"),
    (dial_rows(computed_at=(datetime.now() - timedelta(days=36)).isoformat(timespec="seconds")), "days old"),
    (dial_rows(computed_at="not-a-date"), "ISO"),
])
def test_bad_exposure_dial_errors_before_any_order(monkeypatch, tmp_path, fake_clock, dial, match):
    h = MainHarness(monkeypatch, tmp_path, weights=TWO_ETF_WEIGHTS,
                    details_sequence=[account_snapshot()], dial=dial, quotes=TWO_ETF_QUOTES)
    with pytest.raises(sst.TradingError, match=match):
        h.run("--live", "--confirm-live")
    assert h.client.place_order_calls == []
    assert h.marker() is None
    with pytest.raises(sst.TradingError, match=match):
        h.run()  # dry run: still a hard error, no fallback to 1.0


def test_dial_asof_age_limit_is_inclusive_at_40_days(tmp_path):
    now = datetime(2026, 9, 27, 12, 0)
    ok = write_target_workbook(tmp_path / "ok.xlsx", dial_rows(
        asof_date="2026-08-18", computed_at="2026-09-27T09:00:00"))
    assert sst.load_exposure_dial(35.0, path=ok, now=now)["asof_date"] == "2026-08-18"
    bad = write_target_workbook(tmp_path / "bad.xlsx", dial_rows(
        asof_date="2026-08-17", computed_at="2026-09-27T09:00:00"))
    with pytest.raises(sst.TradingError, match="41 days old"):
        sst.load_exposure_dial(35.0, path=bad, now=now)


# (f) live + dial + not approved -------------------------------------------------

def test_live_dial_blocked_until_approved_before_any_order_or_marker(monkeypatch, tmp_path, fake_clock):
    h = MainHarness(monkeypatch, tmp_path, weights=TWO_ETF_WEIGHTS,
                    details_sequence=[account_snapshot()], dial=dial_rows(target_exposure=1.5),
                    quotes=TWO_ETF_QUOTES, approved=False)
    with pytest.raises(sst.TradingError, match="--exposure-mode off") as exc:
        h.run("--live", "--confirm-live")
    assert "EXPOSURE_DIAL_LIVE_APPROVED" in str(exc.value)
    assert h.client_requested == 0, "Schwab must not even be contacted"
    assert h.client.place_order_calls == []
    assert h.marker() is None

    # The same unapproved state still allows a dial dry run and an off live run.
    h.run()
    h.run("--live", "--confirm-live", "--exposure-mode", "off")
    assert h.marker()["exposure_mode"] == "off"


# (g) exposure > 1 on a non-MARGIN account ---------------------------------------

@pytest.mark.parametrize("acct_type", ["CASH", ""])
def test_exposure_above_one_requires_margin_account(monkeypatch, tmp_path, fake_clock, acct_type):
    h = MainHarness(monkeypatch, tmp_path, weights=TWO_ETF_WEIGHTS,
                    details_sequence=[account_snapshot(acct_type=acct_type)],
                    dial=dial_rows(target_exposure=1.2), quotes=TWO_ETF_QUOTES)
    with pytest.raises(sst.TradingError, match="MARGIN"):
        h.run("--live", "--confirm-live")
    assert h.client.place_order_calls == []
    assert h.marker() is None


def test_non_margin_account_only_warns_at_exposure_one(capsys):
    assert sst.check_margin_account(account_snapshot(acct_type="CASH"), 1.0) == "CASH"
    assert "WARNING" in capsys.readouterr().out


@pytest.mark.parametrize("bp", [None, float("nan")], ids=["missing", "nan"])
def test_exposure_above_one_without_buying_power_field_errors(monkeypatch, tmp_path, fake_clock, bp):
    h = MainHarness(monkeypatch, tmp_path, weights=TWO_ETF_WEIGHTS,
                    details_sequence=[account_snapshot(buying_power=bp)],
                    dial=dial_rows(target_exposure=1.5), quotes=TWO_ETF_QUOTES)
    with pytest.raises(sst.TradingError, match="buyingPower.*missing"):
        h.run("--live", "--confirm-live")
    assert h.client.place_order_calls == []


@pytest.mark.parametrize("post_sell_bp", [None, float("nan")], ids=["missing", "nan"])
def test_buying_power_missing_after_sells_aborts_buys_not_cash_fallback(monkeypatch, tmp_path, fake_clock, post_sell_bp):
    initial = account_snapshot({"EWH": (5_000, 100_000.0)}, cash=900_000.0, buying_power=3 * LV)
    post_sell = account_snapshot(cash=LV, buying_power=post_sell_bp)
    h = MainHarness(monkeypatch, tmp_path, weights=TWO_ETF_WEIGHTS, details_sequence=[initial, post_sell],
                    dial=dial_rows(target_exposure=1.5), quotes={**TWO_ETF_QUOTES, "EWH": (20.00, 20.02)})
    h.run("--live", "--confirm-live")
    assert h.orders("SELL") == {"EWH": 5_000}
    assert h.orders("BUY") == {}
    assert h.marker()["status"] == "SELLS_DONE_BUYS_ABORTED"
    assert "missing from the re-fetched account details" in h.marker()["abort_reason"]


# (h) leverage guard ----------------------------------------------------------------

def test_leverage_guard_trip_submits_no_buys(monkeypatch, tmp_path, fake_clock):
    """The post-sell refetch still shows the sold EWH position (stale/phantom
    holdings). Country book + planned buys would then be ~1.47x of account
    value at exposure 1.0 -> the guard must abort the buy phase."""
    initial = account_snapshot({"EWH": (25_000, 500_000.0)}, cash=500_000.0)
    post_sell = account_snapshot({"EWH": (25_000, 500_000.0)}, cash=LV)
    h = MainHarness(monkeypatch, tmp_path, weights=TWO_ETF_WEIGHTS, details_sequence=[initial, post_sell],
                    dial=dial_rows(target_exposure=1.0), quotes={**TWO_ETF_QUOTES, "EWH": (20.00, 20.02)})
    h.run("--live", "--confirm-live")

    assert h.orders("SELL") == {"EWH": 25_000}
    assert h.orders("BUY") == {}, "no buys may be submitted after the leverage guard trips"
    m = h.marker()
    assert m["status"] == "SELLS_DONE_BUYS_ABORTED"
    assert "LEVERAGE GUARD" in m["abort_reason"] and "MANUAL_REQUIRED" in m["abort_reason"]


def test_leverage_guard_not_applied_in_off_mode(monkeypatch, tmp_path, fake_clock):
    """off keeps the pre-dial flow: the same phantom-holdings scenario buys."""
    initial = account_snapshot({"EWH": (25_000, 500_000.0)}, cash=500_000.0)
    post_sell = account_snapshot({"EWH": (25_000, 500_000.0)}, cash=LV)
    h = MainHarness(monkeypatch, tmp_path, weights=TWO_ETF_WEIGHTS, details_sequence=[initial, post_sell],
                    dial=None, quotes={**TWO_ETF_QUOTES, "EWH": (20.00, 20.02)})
    h.run("--live", "--confirm-live", "--exposure-mode", "off")
    assert set(h.orders("BUY")) == {"EWZ", "EWJ"}
    assert h.marker()["status"] == "COMPLETED"


def test_check_leverage_guard_unit():
    buys = pd.DataFrame([{"Symbol": "EWZ", "Action": "BUY", "Shares to Trade": 100.0, "Trade Dollars": 1_000_000.0}])
    empty = pd.DataFrame(columns=["Symbol", "Market Value", "Long Quantity"])
    held = pd.DataFrame([{"Symbol": "EWJ", "Market Value": 700_000.0, "Long Quantity": 1.0}])
    snaxx = pd.DataFrame([{"Symbol": "SNAXX", "Market Value": 1_500_000.0, "Long Quantity": 1.0}])
    # 1.0M + 0.7M = 1.7M vs 1.7 x 1M x 1.02 = 1.734M -> passes
    assert sst.check_leverage_guard(buys, held, LV, 1.7) is None
    # same book at exposure 1.5 -> country limit 1.53M breached
    assert "exposure 1.50" in sst.check_leverage_guard(buys, held, LV, 1.5)
    # SNAXX is excluded from the country limit but counts toward MAX_EXPOSURE
    reason = sst.check_leverage_guard(buys, snaxx, LV, 1.0)
    assert reason is not None and "MAX_EXPOSURE" in reason and "country book" not in reason
    assert sst.check_leverage_guard(buys, empty, LV, 1.0) is None


# (i) liquidity cap AUM ---------------------------------------------------------------

@pytest.mark.parametrize("mode,exposure,expected_aum", [
    ("dial", 1.7, 1.7 * LV),
    ("dial", 0.4, 0.4 * LV),
    ("off", 1.7, LV),  # off ignores the dial entirely
])
def test_liquidity_cap_receives_aum_times_exposure(monkeypatch, tmp_path, fake_clock, mode, exposure, expected_aum):
    h = MainHarness(monkeypatch, tmp_path, weights=TWO_ETF_WEIGHTS,
                    details_sequence=[account_snapshot()], dial=dial_rows(target_exposure=exposure),
                    quotes=TWO_ETF_QUOTES)
    h.run("--exposure-mode", mode)  # dry run, cap ON (default)
    assert len(h.cap_calls) == 1
    assert h.cap_calls[0]["aum"] == pytest.approx(expected_aum)


def test_audit_trail_in_plan_and_execution_log(monkeypatch, tmp_path, fake_clock):
    h = MainHarness(monkeypatch, tmp_path, weights=TWO_ETF_WEIGHTS,
                    details_sequence=[account_snapshot()], dial=dial_rows(target_exposure=1.3, breadth=0.65),
                    quotes=TWO_ETF_QUOTES)
    h.run("--live", "--confirm-live")
    plan_x = pd.read_excel(next(h.out_dir.glob("schwab_trade_plan_*.xlsx")), sheet_name="Exposure")
    log_x = pd.read_excel(next(h.out_dir.glob("schwab_execution_log_*.xlsx")), sheet_name="Run Info")
    audit_keys = {"exposure_mode", "target_exposure", "breadth", "asof_date", "computed_at"}
    for frame in (plan_x, log_x):
        info = dict(zip(frame["Key"], frame["Value"]))
        assert audit_keys <= set(info)
        assert info["exposure_mode"] == "dial"
        assert float(info["target_exposure"]) == 1.3
        assert float(info["breadth"]) == 0.65
    assert set(dict(zip(plan_x["Key"], plan_x["Value"]))) == audit_keys
    log_info = dict(zip(log_x["Key"], log_x["Value"]))
    assert float(log_info["buying_power"]) == 3 * LV  # exposure > 1 records its funding


# Verifier follow-ups (2026-09-27) --------------------------------------------------

@pytest.mark.parametrize("post_sell_bp", [30_000.0, -10_000.0, 30_005.0], ids=["equal-reserve", "negative", "dust"])
def test_buying_power_exhausted_after_sells_aborts_manual_required(monkeypatch, tmp_path, fake_clock, post_sell_bp):
    """buyingPower <= the 3% reserve (or so little that every BUY floors to 0
    shares) must abort the buy leg loudly, never report COMPLETED with no buys."""
    initial = account_snapshot({"EWH": (5_000, 100_000.0)}, cash=900_000.0, buying_power=3 * LV)
    post_sell = account_snapshot(cash=LV, buying_power=post_sell_bp)
    h = MainHarness(monkeypatch, tmp_path, weights=TWO_ETF_WEIGHTS, details_sequence=[initial, post_sell],
                    dial=dial_rows(target_exposure=1.5), quotes={**TWO_ETF_QUOTES, "EWH": (20.00, 20.02)})
    h.run("--live", "--confirm-live")
    assert h.orders("SELL") == {"EWH": 5_000}
    assert h.orders("BUY") == {}
    m = h.marker()
    assert m["status"] == "SELLS_DONE_BUYS_ABORTED"
    assert "BUYING POWER EXHAUSTED" in m["abort_reason"] and "MANUAL_REQUIRED" in m["abort_reason"]
    assert m["buying_power"] == post_sell_bp


def test_leverage_guard_slack_is_absolute_five_pct_of_account():
    empty = pd.DataFrame(columns=["Symbol", "Market Value", "Long Quantity"])
    def buys(dollars):
        return pd.DataFrame([{"Symbol": "EWZ", "Action": "BUY", "Shares to Trade": 1.0, "Trade Dollars": dollars}])
    assert sst.LEVERAGE_GUARD_SLACK_PCT == 0.05
    for exposure in (0.5, 1.0, 1.7):
        limit = (exposure + 0.05) * LV
        assert sst.check_leverage_guard(buys(limit - 1.0), empty, LV, exposure) is None
        assert "country book" in sst.check_leverage_guard(buys(limit + 1.0), empty, LV, exposure)
    # The MAX_EXPOSURE ceiling has no slack: 1.97x + 0.04x SNAXX = 2.01x trips it
    # even though the country book (1.97x) is inside 1.95 + 0.05.
    snaxx = pd.DataFrame([{"Symbol": "SNAXX", "Market Value": 0.04 * LV, "Long Quantity": 1.0}])
    reason = sst.check_leverage_guard(buys(1.97 * LV), snaxx, LV, 1.95)
    assert reason is not None and "MAX_EXPOSURE" in reason and "country book" not in reason


def _half_exposure_unfilled_sell_scenario(monkeypatch, tmp_path, unfilled_shares, run_name, residue_mv=None):
    """Fully invested in EWH (50,000 sh @ $20), dial 0.5 rotates into EWZ/EWJ.
    The EWH sell leaves `unfilled_shares` unfilled (wide spread blocks the
    market-order cleanup); the post-sell snapshot shows that residue."""
    left = unfilled_shares
    residue = left * 20.0 if residue_mv is None else residue_mv
    initial = account_snapshot({"EWH": (50_000, 1_000_000.0)}, cash=0.0)
    post_sell = account_snapshot({"EWH": (left, residue)}, cash=LV - residue)
    h = MainHarness(monkeypatch, tmp_path, weights=TWO_ETF_WEIGHTS, details_sequence=[initial, post_sell],
                    dial=dial_rows(target_exposure=0.5),
                    quotes={**TWO_ETF_QUOTES, "EWH": (19.90, 20.10)}, run_name=run_name)
    h.client.queue_script("EWH", "SELL", OrderScript(
        poll_sequence=[("WORKING", 50_000 - left)],
        post_cancel_status="CANCELED", post_cancel_filled_qty=50_000 - left,
    ))
    h.run("--live", "--confirm-live")
    return h


def test_half_exposure_with_4pct_unfilled_sells_still_buys(monkeypatch, tmp_path, fake_clock):
    """4% of the sell notional unfilled (under the 5% sell-abort threshold):
    residue $40k + buys ~$485k = ~0.525x LV. The old 2%-of-exposure slack
    (limit 0.51x) tripped here; the absolute 5% slack (0.55x) must not."""
    h = _half_exposure_unfilled_sell_scenario(monkeypatch, tmp_path, 2_000, "four_pct")
    assert not [c for c in h.client.place_order_calls if c.get("orderType") == "MARKET"], (
        "cleanup must be skipped (wide spread), leaving the 2,000-share residue")
    assert h.marker()["sell_total_filled"] == 48_000
    assert set(h.orders("BUY")) == {"EWZ", "EWJ"}
    assert h.marker()["status"] == "COMPLETED"


def test_half_exposure_well_over_limit_trips_guard(monkeypatch, tmp_path, fake_clock):
    """Same flow, but the post-sell snapshot shows a residue worth 0.2x LV
    (e.g. stale/phantom position): 0.2x + ~0.485x buys > 0.55x -> abort."""
    h = _half_exposure_unfilled_sell_scenario(monkeypatch, tmp_path, 2_000, "over", residue_mv=200_000.0)
    assert h.orders("BUY") == {}
    m = h.marker()
    assert m["status"] == "SELLS_DONE_BUYS_ABORTED"
    assert "LEVERAGE GUARD" in m["abort_reason"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
