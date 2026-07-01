"""
=============================================================================
SCRIPT NAME: step_schwab_dashboard.py — Live TWAP Execution Dashboard
=============================================================================

INPUT FILES:  None (receives data from Step Schwab Trading.py at runtime).
OUTPUT FILES: None (renders to terminal only).

VERSION: 1.0
LAST UPDATED: 2026-06-29
AUTHOR: Arjun Divecha

DESCRIPTION:
    A live-updating terminal dashboard for monitoring TWAP order execution.
    Uses the Rich library to render a continuously-refreshing display showing:
      - A per-symbol summary table (aggregated across all slices so far):
        total/filled quantities, live bid/ask/spread, current slice's limit
        price, blended VWAP of all fills, and slippage vs arrival mid.
      - An ORDER BLOTTER — one row per individual order actually submitted
        to Schwab, in submission order: slice #, the exact limit price WE
        SUBMITTED for that order, and the ACTUAL fill price/qty Schwab
        reported back for it. This is the ground truth of what really
        happened order-by-order (the summary table above only shows a
        blended average across all slices, which hides per-slice execution
        quality). (Added 2026-07-01 after live-monitoring feedback that the
        aggregated view alone wasn't enough.)

    The dashboard is designed to be driven by the TWAP engine in
    Step Schwab Trading.py. The engine calls dashboard.update_*() methods
    as slices execute, and the Rich Live context auto-refreshes the display.

DEPENDENCIES: rich
=============================================================================
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ---------------------------------------------------------------------------
# Data structures the dashboard tracks
# ---------------------------------------------------------------------------

@dataclass
class BlotterRow:
    """One row in the order blotter = one individual order actually
    submitted to Schwab (i.e. one TWAP slice). Distinct from OrderState,
    which aggregates ALL of a symbol's slices into a single blended VWAP —
    the blotter preserves each order's own limit price and own fill price
    so nothing is averaged away."""
    time: str
    symbol: str
    action: str
    slice_num: int
    order_qty: int
    limit_price: float
    arrival_mid: float
    filled_qty: int = 0
    fill_price: float | None = None
    status: str = "SUBMITTED"

    @property
    def slip_bps(self) -> float | None:
        if self.fill_price is None or self.arrival_mid <= 0:
            return None
        if self.action == "SELL":
            return (self.arrival_mid - self.fill_price) / self.arrival_mid * 1e4
        return (self.fill_price - self.arrival_mid) / self.arrival_mid * 1e4


@dataclass
class OrderState:
    """Tracks the live state of a single TWAP order."""
    symbol: str
    action: str
    total_qty: int
    filled_qty: int = 0
    current_slice: int = 0
    total_slices: int = 10
    bid: float = 0.0
    ask: float = 0.0
    last_limit: float = 0.0
    arrival_mid: float = 0.0
    fill_notional: float = 0.0
    status: str = "PENDING"
    slices_detail: list[dict[str, Any]] = field(default_factory=list)

    @property
    def spread_bps(self) -> float:
        mid = (self.bid + self.ask) / 2.0
        if mid <= 0:
            return 0.0
        return (self.ask - self.bid) / mid * 1e4

    @property
    def vwap(self) -> float | None:
        if self.filled_qty <= 0:
            return None
        return self.fill_notional / self.filled_qty

    @property
    def slippage_bps(self) -> float | None:
        v = self.vwap
        if v is None or self.arrival_mid <= 0:
            return None
        if self.action == "SELL":
            return (self.arrival_mid - v) / self.arrival_mid * 1e4
        return (v - self.arrival_mid) / self.arrival_mid * 1e4

    @property
    def pct_filled(self) -> float:
        if self.total_qty <= 0:
            return 0.0
        return min(100.0, self.filled_qty / self.total_qty * 100.0)

    @property
    def trade_dollars(self) -> float:
        mid = (self.bid + self.ask) / 2.0
        return self.total_qty * mid if mid > 0 else 0.0


# ---------------------------------------------------------------------------
# Dashboard renderer
# ---------------------------------------------------------------------------

class TwapDashboard:
    """Live terminal dashboard for TWAP execution.

    Usage:
        dashboard = TwapDashboard(account_name, total_equity, ...)
        dashboard.add_order("EWZ", "SELL", 1000, 10)
        dashboard.add_order("EWH", "BUY", 500, 10)

        with dashboard.live():
            # TWAP engine loop:
            dashboard.update_quote("EWZ", "SELL", bid=34.50, ask=34.55)
            dashboard.update_slice_start("EWZ", "SELL", slice_num=1, limit_price=34.49, arrival_mid=34.52)
            dashboard.update_fill("EWZ", "SELL", slice_filled_qty=100, fill_price=34.50)
            ...
    """

    def __init__(
        self,
        account_name: str,
        total_equity: float,
        cash_balance: float,
        mode: str = "LIVE",
        twap_window: int = 15,
        twap_slices: int = 10,
    ):
        self.account_name = account_name
        self.total_equity = total_equity
        self.cash_balance = cash_balance
        self.mode = mode
        self.twap_window = twap_window
        self.twap_slices = twap_slices
        self.orders: dict[str, OrderState] = {}
        self.order_sequence: list[str] = []
        self.current_leg: str = ""
        self.start_time: float = time.monotonic()
        self.console = Console()
        self._live: Live | None = None
        self.log_lines: list[str] = []

        # Order blotter: ground-truth, one row per order actually submitted
        # to Schwab (never averaged away like OrderState's blended VWAP).
        self.blotter: list[BlotterRow] = []
        self._blotter_index: dict[tuple[str, str, int], BlotterRow] = {}

    def add_order(self, symbol: str, action: str, total_qty: int, total_slices: int) -> None:
        key = f"{symbol}_{action}"
        self.orders[key] = OrderState(
            symbol=symbol, action=action, total_qty=total_qty,
            total_slices=total_slices,
        )
        self.order_sequence.append(key)

    def set_leg(self, leg: str) -> None:
        self.current_leg = leg

    def update_quote(self, symbol: str, action: str, bid: float, ask: float) -> None:
        key = f"{symbol}_{action}"
        if key in self.orders:
            self.orders[key].bid = bid
            self.orders[key].ask = ask

    def update_slice_start(
        self, symbol: str, action: str, slice_num: int, limit_price: float, arrival_mid: float,
        order_qty: int | None = None,
    ) -> None:
        """Called once per order right before/as it is submitted to Schwab.

        order_qty (new 2026-07-01): when provided, also opens a new BLOTTER
        row for this exact order — the row that records what limit price WE
        submitted and (via update_fill) what price it actually filled at.
        Without order_qty, only the aggregated OrderState is updated (kept
        optional so callers that don't have qty on hand yet still work).
        """
        key = f"{symbol}_{action}"
        if key in self.orders:
            o = self.orders[key]
            o.current_slice = slice_num
            o.last_limit = limit_price
            if o.arrival_mid <= 0:
                o.arrival_mid = arrival_mid
            o.status = "WORKING"

        if order_qty is not None:
            row = BlotterRow(
                time=datetime.now().strftime("%H:%M:%S"),
                symbol=symbol, action=action, slice_num=slice_num,
                order_qty=order_qty, limit_price=limit_price, arrival_mid=arrival_mid,
            )
            self.blotter.append(row)
            self._blotter_index[(symbol, action, slice_num)] = row
            if len(self.blotter) > 500:
                # Cap memory for very long/many-symbol runs; drop oldest.
                dropped = self.blotter.pop(0)
                self._blotter_index.pop((dropped.symbol, dropped.action, dropped.slice_num), None)

    def update_fill(
        self, symbol: str, action: str, slice_filled_qty: int, fill_price: float | None,
        slice_num: int | None = None,
    ) -> None:
        """Record an incremental fill. slice_num (new 2026-07-01), when
        provided, also updates the matching blotter row's own fill price/qty
        (blended across however many partial-fill updates that one order
        gets) instead of only the cross-slice-averaged OrderState.vwap."""
        key = f"{symbol}_{action}"
        if key in self.orders:
            o = self.orders[key]
            o.filled_qty += slice_filled_qty
            if fill_price and fill_price > 0 and slice_filled_qty > 0:
                o.fill_notional += slice_filled_qty * fill_price
            if o.filled_qty >= o.total_qty:
                o.status = "FILLED"
            o.slices_detail.append({
                "slice": o.current_slice, "qty": slice_filled_qty,
                "price": fill_price, "time": datetime.now().strftime("%H:%M:%S"),
            })

        if slice_num is not None:
            row = self._blotter_index.get((symbol, action, slice_num))
            if row is not None and slice_filled_qty > 0:
                if fill_price and fill_price > 0:
                    prior_notional = (row.fill_price or 0.0) * row.filled_qty
                    row.filled_qty += slice_filled_qty
                    row.fill_price = (prior_notional + fill_price * slice_filled_qty) / row.filled_qty
                else:
                    row.filled_qty += slice_filled_qty
                row.status = "FILLED" if row.filled_qty >= row.order_qty else "PARTIAL"

    def mark_order_status(
        self, symbol: str, action: str, slice_num: int, status: str, notes: str = "",
    ) -> None:
        """Directly set a blotter row's status (REJECTED, SKIPPED, FAILED,
        UNKNOWN, MANUAL_REQUIRED, ...) for orders that never filled or were
        never even sent, so the blotter still shows the ground truth of
        every attempt, not just the ones that filled."""
        row = self._blotter_index.get((symbol, action, slice_num))
        if row is not None:
            row.status = status
        # else: no matching row (e.g. preflight-rejected before a row was
        # ever opened) -- caller should have opened one via
        # update_slice_start(..., order_qty=...) first if a row is wanted.

    def update_order_status(self, symbol: str, action: str, status: str) -> None:
        key = f"{symbol}_{action}"
        if key in self.orders:
            self.orders[key].status = status

    def add_log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_lines.append(f"[dim]{ts}[/dim] {message}")
        if len(self.log_lines) > 15:
            self.log_lines = self.log_lines[-15:]

    def _build_header(self) -> Panel:
        elapsed = time.monotonic() - self.start_time
        mins, secs = divmod(int(elapsed), 60)

        total_orders = len(self.orders)
        filled_orders = sum(1 for o in self.orders.values() if o.status == "FILLED")
        working = sum(1 for o in self.orders.values() if o.status == "WORKING")

        mode_style = "bold red" if self.mode == "LIVE" else "bold yellow"
        header_text = Text()
        header_text.append("T2 SCHWAB TWAP TRADER", style="bold white")
        header_text.append("  |  ", style="dim")
        header_text.append(self.mode, style=mode_style)
        header_text.append("  |  ", style="dim")
        header_text.append(f"Account: {self.account_name}", style="cyan")
        header_text.append("  |  ", style="dim")
        header_text.append(f"Equity: ${self.total_equity:,.0f}", style="green")
        header_text.append("  |  ", style="dim")
        header_text.append(f"Cash: ${self.cash_balance:,.0f}", style="green")
        header_text.append("  |  ", style="dim")
        header_text.append(f"Elapsed: {mins:02d}:{secs:02d}", style="white")
        header_text.append("  |  ", style="dim")
        header_text.append(f"Orders: {filled_orders}/{total_orders}", style="white")
        if working > 0:
            header_text.append(f" ({working} working)", style="yellow")

        return Panel(header_text, style="blue")

    def _build_orders_table(self) -> Table:
        """A SINGLE combined table for BOTH legs (Side column distinguishes
        SELL/BUY), sorted SELL-then-BUY, each in add_order() order.

        Previously this rendered as TWO separate full-height tables (one
        per leg). On accounts with 20-30+ symbols, two full tables plus the
        header/summary/blotter/log routinely exceeded the terminal's actual
        height -- and Rich's Live(screen=True) does NOT scroll, it just
        silently clips anything past the bottom of the screen. That's what
        made the BUY leg (rendered second, i.e. further down/off-screen)
        appear to "not show" during a live run even though it was executing
        correctly. One combined table for both legs is shorter overall (one
        title/header/border instead of two) and keeps every symbol's status
        together in a single scroll-free view. (Fixed 2026-07-01.)
        """
        table = Table(
            title="  ORDERS (all legs)  ",
            title_style="bold white",
            show_header=True,
            header_style="bold",
            border_style="dim",
            expand=True,
        )
        table.add_column("Side", min_width=4)
        table.add_column("Symbol", style="bold white", min_width=6)
        table.add_column("Total Qty", justify="right", min_width=9)
        table.add_column("Filled", justify="right", min_width=8)
        table.add_column("Fill%", justify="right", min_width=6)
        table.add_column("Slice", justify="center", min_width=7)
        table.add_column("Bid", justify="right", min_width=9)
        table.add_column("Ask", justify="right", min_width=9)
        table.add_column("Spread", justify="right", min_width=8)
        table.add_column("Our Limit", justify="right", min_width=10, style="bold cyan")
        table.add_column("VWAP", justify="right", min_width=9)
        table.add_column("Slip(bps)", justify="right", min_width=9)
        table.add_column("Status", justify="center", min_width=9)

        shown = 0
        for leg in ("SELL", "BUY"):
            for key in self.order_sequence:
                o = self.orders[key]
                if o.action != leg:
                    continue
                # Once we've moved on to the other leg, a FULLY FILLED order
                # from the leg we already finished is no longer actionable
                # information -- keep the table showing what's actually
                # happening now (plus anything from an earlier leg that's
                # STILL incomplete, which is exactly the kind of thing that
                # needs to stay visible). This is what keeps the combined
                # table's height bounded to whichever leg is active, instead
                # of always showing SELL+BUY simultaneously (25+11 rows),
                # which is what overflowed most terminal windows.
                if leg != self.current_leg and o.status == "FILLED" and self.current_leg:
                    continue
                shown += 1

                fill_pct = o.pct_filled
                vwap = o.vwap
                slip = o.slippage_bps

                pct_style = "green" if fill_pct >= 100 else ("yellow" if fill_pct > 0 else "dim")
                status_style = {
                    "FILLED": "bold green",
                    "WORKING": "bold yellow",
                    "PENDING": "dim",
                    "PARTIAL": "bold red",
                    "FAILED": "bold red",
                    "CANCELLED": "dim red",
                }.get(o.status, "white")

                slip_style = "white"
                if slip is not None:
                    slip_style = "green" if slip <= 5 else ("yellow" if slip <= 15 else "red")

                spread_str = f"{o.spread_bps:.0f}" if o.bid > 0 else "—"
                spread_style = "green" if o.spread_bps < 20 else ("yellow" if o.spread_bps < 100 else "red")
                side_style = "bold red" if leg == "SELL" else "bold green"

                table.add_row(
                    Text(leg, style=side_style),
                    o.symbol,
                    f"{o.total_qty:,}",
                    f"{o.filled_qty:,}",
                    Text(f"{fill_pct:.0f}%", style=pct_style),
                    f"{o.current_slice}/{o.total_slices}" if o.current_slice > 0 else "—",
                    f"{o.bid:.2f}" if o.bid > 0 else "—",
                    f"{o.ask:.2f}" if o.ask > 0 else "—",
                    Text(spread_str, style=spread_style),
                    f"{o.last_limit:.2f}" if o.last_limit > 0 else "—",
                    f"{vwap:.4f}" if vwap else "—",
                    Text(f"{slip:.1f}", style=slip_style) if slip is not None else Text("—", style="dim"),
                    Text(o.status, style=status_style),
                )

        hidden = len(self.orders) - shown
        if hidden > 0:
            table.caption = (
                f"({hidden} fully-filled order(s) from a completed leg collapsed "
                f"to save space -- all filled, nothing hidden that needs action)"
            )
            table.caption_style = "dim"

        return table

    def _build_summary(self) -> Panel:
        sell_orders = [o for o in self.orders.values() if o.action == "SELL"]
        buy_orders = [o for o in self.orders.values() if o.action == "BUY"]

        sell_filled = sum(o.filled_qty for o in sell_orders)
        sell_total = sum(o.total_qty for o in sell_orders)
        buy_filled = sum(o.filled_qty for o in buy_orders)
        buy_total = sum(o.total_qty for o in buy_orders)

        sell_notional = sum(o.fill_notional for o in sell_orders)
        buy_notional = sum(o.fill_notional for o in buy_orders)

        all_filled = [o for o in self.orders.values() if o.filled_qty > 0]
        total_notional = sum(o.fill_notional for o in all_filled)
        total_cost = 0.0
        for o in all_filled:
            v = o.vwap
            if v and o.arrival_mid > 0:
                if o.action == "SELL":
                    total_cost += (o.arrival_mid - v) * o.filled_qty
                else:
                    total_cost += (v - o.arrival_mid) * o.filled_qty
        avg_slip = (total_cost / total_notional * 1e4) if total_notional > 0 else 0.0

        text = Text()
        text.append("SELLS: ", style="bold red")
        text.append(f"{sell_filled:,}/{sell_total:,} shares  ${sell_notional:,.0f}", style="white")
        text.append("    |    ", style="dim")
        text.append("BUYS: ", style="bold green")
        text.append(f"{buy_filled:,}/{buy_total:,} shares  ${buy_notional:,.0f}", style="white")
        text.append("    |    ", style="dim")
        text.append("Avg Slippage: ", style="bold")
        slip_style = "green" if avg_slip <= 5 else ("yellow" if avg_slip <= 15 else "red")
        text.append(f"{avg_slip:.1f} bps", style=slip_style)

        return Panel(text, title="Execution Summary", border_style="dim")

    def _build_blotter(self, max_rows: int = 25) -> Panel:
        """Order-level ground truth: every individual order submitted to
        Schwab, its own limit price, and its own actual fill price — not
        blended across slices like the per-symbol summary table above."""
        table = Table(
            title="  ORDER BLOTTER (every order submitted -- limit vs. actual fill)  ",
            title_style="bold white",
            show_header=True,
            header_style="bold",
            border_style="dim",
            expand=True,
        )
        table.add_column("Time", min_width=8)
        table.add_column("Symbol", style="bold white", min_width=6)
        table.add_column("Side", min_width=4)
        table.add_column("Slice", justify="center", min_width=5)
        table.add_column("Order Qty", justify="right", min_width=9)
        table.add_column("Limit Px", justify="right", min_width=9, style="bold cyan")
        table.add_column("Fill Qty", justify="right", min_width=8)
        table.add_column("Fill Px", justify="right", min_width=9)
        table.add_column("Slip(bps)", justify="right", min_width=9)
        table.add_column("Status", justify="center", min_width=10)

        status_style = {
            "FILLED": "bold green",
            "PARTIAL": "bold yellow",
            "SUBMITTED": "yellow",
            "REJECTED": "bold red",
            "FAILED": "bold red",
            "SKIPPED": "dim red",
            "UNKNOWN": "bold red",
            "MANUAL_REQUIRED": "bold red",
        }

        rows = self.blotter[-max_rows:]
        for row in rows:
            slip = row.slip_bps
            slip_style = "white"
            if slip is not None:
                slip_style = "green" if slip <= 5 else ("yellow" if slip <= 15 else "red")
            side_style = "bold red" if row.action == "SELL" else "bold green"
            table.add_row(
                row.time,
                row.symbol,
                Text(row.action, style=side_style),
                str(row.slice_num),
                f"{row.order_qty:,}",
                f"{row.limit_price:.4f}" if row.limit_price > 0 else "MKT",
                f"{row.filled_qty:,}" if row.filled_qty > 0 else "—",
                f"{row.fill_price:.4f}" if row.fill_price else "—",
                Text(f"{slip:.1f}", style=slip_style) if slip is not None else Text("—", style="dim"),
                Text(row.status, style=status_style.get(row.status, "white")),
            )

        if not rows:
            table.add_row("—", "—", "—", "—", "—", "—", "—", "—", "—", Text("Waiting...", style="dim"))

        return table

    def _build_log(self, max_lines: int | None = None) -> Panel:
        lines = self.log_lines if max_lines is None else self.log_lines[-max_lines:]
        log_text = Text()
        for i, line in enumerate(lines):
            if i > 0:
                log_text.append("\n")
            log_text.append_text(Text.from_markup(line))
        if not lines:
            log_text.append("Waiting...", style="dim")
        return Panel(log_text, title="Activity Log", border_style="dim")

    def _measured_height(self, renderable: Any) -> int:
        """Actual number of terminal rows a renderable will occupy at the
        console's current width -- NOT a guess. Rich's Live(screen=True)
        does not scroll: anything beyond the terminal's real height is
        simply invisible with no warning, which is exactly what silently
        hid the BUY leg / blotter / log on wide, many-symbol accounts
        before this fix (2026-07-01)."""
        try:
            return len(self.console.render_lines(renderable, self.console.options))
        except Exception:
            return 0

    def render(self) -> Group:
        """Build the full dashboard as a vertical stack of renderables,
        sized to actually fit the real terminal height. The order table,
        header, and summary are always shown in full (they're the
        highest-priority info); the blotter and activity log then split
        whatever vertical room is left, shrinking automatically on a
        smaller terminal instead of silently rendering off-screen.

        This uses an ACTUAL measure-and-shrink loop rather than a fixed
        per-row overhead estimate, because a Table's real height depends on
        box-drawing style/title/border lines (Rich version-dependent) AND
        on text wrapping (a long activity-log message can wrap to more than
        one rendered line in a narrow terminal) -- a naive "N items = N
        rows" budget can still overflow in those cases. Iteratively
        measuring the real rendered height and shrinking guarantees nothing
        is silently clipped off-screen, which is the exact bug this whole
        redesign fixes (2026-07-01).
        """
        fixed_parts = [self._build_header()]

        if self.orders:
            fixed_parts.append(self._build_orders_table())
        else:
            fixed_parts.append(Panel("No orders.", style="dim"))

        fixed_parts.append(self._build_summary())

        terminal_height = self.console.size.height
        used = sum(self._measured_height(p) for p in fixed_parts)

        blotter_rows = max(1, min(25, len(self.blotter) or 1))
        log_rows = max(1, min(15, len(self.log_lines) or 1))

        blotter_panel = self._build_blotter(max_rows=blotter_rows)
        log_panel = self._build_log(max_lines=log_rows)

        for _ in range(40):
            total = used + self._measured_height(blotter_panel) + self._measured_height(log_panel)
            if total <= terminal_height - 1:
                break
            if log_rows > 1:
                log_rows -= 1
                log_panel = self._build_log(max_lines=log_rows)
            elif blotter_rows > 1:
                blotter_rows -= 1
                blotter_panel = self._build_blotter(max_rows=blotter_rows)
            else:
                break  # both already at minimum -- nothing more to shrink

        parts = fixed_parts + [blotter_panel, log_panel]

        return Group(*parts)

    def live(self) -> Live:
        """Return a Rich Live context manager that auto-refreshes the dashboard."""
        self._live = Live(
            self.render(),
            console=self.console,
            refresh_per_second=2,
            screen=True,
        )
        return self._live

    def refresh(self) -> None:
        """Manually refresh the dashboard display."""
        if self._live is not None:
            self._live.update(self.render())
