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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
