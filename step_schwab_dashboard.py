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
    Uses the Rich library to render a continuously-refreshing table showing:
      - Each order's symbol, action, total/filled quantities
      - Live bid/ask/spread from Schwab NBBO
      - Our limit price for the current slice
      - VWAP of fills so far and slippage vs arrival mid
      - Per-slice progress and overall completion

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
            dashboard.update_quote("EWZ", bid=34.50, ask=34.55)
            dashboard.update_slice_start("EWZ", slice_num=1, limit_price=34.49)
            dashboard.update_fill("EWZ", filled_qty=100, fill_price=34.50)
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
    ) -> None:
        key = f"{symbol}_{action}"
        if key in self.orders:
            o = self.orders[key]
            o.current_slice = slice_num
            o.last_limit = limit_price
            if o.arrival_mid <= 0:
                o.arrival_mid = arrival_mid
            o.status = "WORKING"

    def update_fill(
        self, symbol: str, action: str, slice_filled_qty: int, fill_price: float | None,
    ) -> None:
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

    def update_order_status(self, symbol: str, action: str, status: str) -> None:
        key = f"{symbol}_{action}"
        if key in self.orders:
            self.orders[key].status = status

    def add_log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_lines.append(f"[dim]{ts}[/dim] {message}")
        if len(self.log_lines) > 8:
            self.log_lines = self.log_lines[-8:]

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

    def _build_orders_table(self, leg: str) -> Table:
        title_style = "bold red" if leg == "SELL" else "bold green"
        table = Table(
            title=f"  {leg} LEG  ",
            title_style=title_style,
            show_header=True,
            header_style="bold",
            border_style="dim",
            expand=True,
        )
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

        for key in self.order_sequence:
            o = self.orders[key]
            if o.action != leg:
                continue

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

            table.add_row(
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

    def _build_log(self) -> Panel:
        log_text = Text()
        for i, line in enumerate(self.log_lines):
            if i > 0:
                log_text.append("\n")
            log_text.append_text(Text.from_markup(line))
        if not self.log_lines:
            log_text.append("Waiting...", style="dim")
        return Panel(log_text, title="Activity Log", border_style="dim")

    def render(self) -> Group:
        """Build the full dashboard as a vertical stack of renderables."""
        parts = [self._build_header()]

        has_sells = any(o.action == "SELL" for o in self.orders.values())
        has_buys = any(o.action == "BUY" for o in self.orders.values())

        if has_sells:
            parts.append(self._build_orders_table("SELL"))
        if has_buys:
            parts.append(self._build_orders_table("BUY"))
        if not has_sells and not has_buys:
            parts.append(Panel("No orders.", style="dim"))

        parts.append(self._build_summary())
        parts.append(self._build_log())

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
