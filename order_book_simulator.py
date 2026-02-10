"""
Limit Order Book Simulator
============================
Simulates a realistic limit order book with:
- Bid/ask price-time priority matching engine
- Market orders, limit orders, and cancel orders
- Basic market-making strategy with inventory management
- Order flow analysis and visualisation

This demonstrates understanding of market microstructure — critical
knowledge for roles at firms like Jane Street, HRT, and Citadel.

Author: ismailalt2
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import heapq
import time
import matplotlib.pyplot as plt


# =============================================================================
# Data Structures
# =============================================================================

class Side(Enum):
    BID = "BID"
    ASK = "ASK"


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    CANCEL = "CANCEL"


@dataclass
class Order:
    order_id: int
    side: Side
    price: float
    quantity: int
    order_type: OrderType = OrderType.LIMIT
    timestamp: float = 0.0
    filled: int = 0

    @property
    def remaining(self) -> int:
        return self.quantity - self.filled

    @property
    def is_filled(self) -> bool:
        return self.remaining <= 0


@dataclass
class Trade:
    price: float
    quantity: int
    aggressor_side: Side
    timestamp: float
    maker_id: int
    taker_id: int


@dataclass
class BookLevel:
    """Represents all orders at a single price level."""
    price: float
    orders: list = field(default_factory=list)

    @property
    def total_quantity(self) -> int:
        return sum(o.remaining for o in self.orders if not o.is_filled)

    def add_order(self, order: Order):
        self.orders.append(order)

    def remove_filled(self):
        self.orders = [o for o in self.orders if not o.is_filled]


# =============================================================================
# Matching Engine
# =============================================================================

class OrderBook:
    """
    Price-time priority matching engine.

    Bids stored as max-heap (highest price first).
    Asks stored as min-heap (lowest price first).
    Within same price level: FIFO ordering.
    """

    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        self.bids: dict[float, BookLevel] = {}     # price -> BookLevel
        self.asks: dict[float, BookLevel] = {}
        self.bid_prices: list = []                   # max-heap (negated)
        self.ask_prices: list = []                   # min-heap
        self.orders: dict[int, Order] = {}           # order_id -> Order
        self.trades: list[Trade] = []
        self._next_id = 1
        self._clock = 0.0

    @property
    def best_bid(self) -> Optional[float]:
        while self.bid_prices:
            price = -self.bid_prices[0]
            if price in self.bids and self.bids[price].total_quantity > 0:
                return price
            heapq.heappop(self.bid_prices)
            self.bids.pop(price, None)
        return None

    @property
    def best_ask(self) -> Optional[float]:
        while self.ask_prices:
            price = self.ask_prices[0]
            if price in self.asks and self.asks[price].total_quantity > 0:
                return price
            heapq.heappop(self.ask_prices)
            self.asks.pop(price, None)
        return None

    @property
    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return (bb + ba) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return ba - bb
        return None

    def submit_order(self, side: Side, price: float, quantity: int,
                     order_type: OrderType = OrderType.LIMIT) -> int:
        """Submit an order and return its ID."""
        self._clock += 1e-6
        order = Order(
            order_id=self._next_id, side=side, price=round(price / self.tick_size) * self.tick_size,
            quantity=quantity, order_type=order_type, timestamp=self._clock
        )
        self._next_id += 1
        self.orders[order.order_id] = order

        if order_type == OrderType.MARKET:
            self._match_market(order)
        else:
            self._match_limit(order)
            if not order.is_filled:
                self._add_to_book(order)

        return order.order_id

    def cancel_order(self, order_id: int) -> bool:
        """Cancel an existing order. Returns True if successfully cancelled."""
        if order_id not in self.orders:
            return False
        order = self.orders[order_id]
        order.filled = order.quantity  # mark as fully filled (effectively cancelled)
        return True

    def _match_market(self, order: Order):
        """Match a market order against resting orders."""
        if order.side == Side.BID:
            self._match_against(order, self.asks, self.ask_prices, is_bid=True)
        else:
            self._match_against(order, self.bids, self.bid_prices, is_bid=False)

    def _match_limit(self, order: Order):
        """Match a limit order (aggressive portion) against resting orders."""
        if order.side == Side.BID:
            # Match against asks at or below order price
            while not order.is_filled and self.best_ask is not None and order.price >= self.best_ask:
                self._execute_at_level(order, self.asks, self.best_ask)
        else:
            # Match against bids at or above order price
            while not order.is_filled and self.best_bid is not None and order.price <= self.best_bid:
                self._execute_at_level(order, self.bids, self.best_bid)

    def _match_against(self, order: Order, book: dict, prices: list, is_bid: bool):
        """Match aggressively until filled or book is empty."""
        while not order.is_filled:
            if is_bid:
                best = self.best_ask
            else:
                best = self.best_bid
            if best is None:
                break
            self._execute_at_level(order, book, best)

    def _execute_at_level(self, aggressor: Order, book: dict, price: float):
        """Execute against resting orders at a specific price level."""
        if price not in book:
            return
        level = book[price]
        for resting in level.orders:
            if aggressor.is_filled:
                break
            if resting.is_filled:
                continue
            fill_qty = min(aggressor.remaining, resting.remaining)
            aggressor.filled += fill_qty
            resting.filled += fill_qty

            trade = Trade(
                price=price, quantity=fill_qty,
                aggressor_side=aggressor.side, timestamp=self._clock,
                maker_id=resting.order_id, taker_id=aggressor.order_id
            )
            self.trades.append(trade)

        level.remove_filled()

    def _add_to_book(self, order: Order):
        """Add a resting limit order to the book."""
        if order.side == Side.BID:
            if order.price not in self.bids:
                self.bids[order.price] = BookLevel(order.price)
                heapq.heappush(self.bid_prices, -order.price)
            self.bids[order.price].add_order(order)
        else:
            if order.price not in self.asks:
                self.asks[order.price] = BookLevel(order.price)
                heapq.heappush(self.ask_prices, order.price)
            self.asks[order.price].add_order(order)

    def get_depth(self, levels: int = 10) -> dict:
        """Get order book depth (top N levels on each side)."""
        bid_depth, ask_depth = [], []

        seen_bids = set()
        for neg_p in sorted(self.bid_prices):
            p = -neg_p
            if p in seen_bids:
                continue
            seen_bids.add(p)
            if p in self.bids and self.bids[p].total_quantity > 0:
                bid_depth.append((p, self.bids[p].total_quantity))
                if len(bid_depth) >= levels:
                    break

        seen_asks = set()
        for p in sorted(self.ask_prices):
            if p in seen_asks:
                continue
            seen_asks.add(p)
            if p in self.asks and self.asks[p].total_quantity > 0:
                ask_depth.append((p, self.asks[p].total_quantity))
                if len(ask_depth) >= levels:
                    break

        return {"bids": bid_depth, "asks": ask_depth}

    def __str__(self) -> str:
        depth = self.get_depth(5)
        lines = ["\n========== ORDER BOOK =========="]
        lines.append(f"{'ASKS':>30}")
        for price, qty in reversed(depth["asks"]):
            bar = "█" * min(qty // 10, 30)
            lines.append(f"  {price:>10.2f}  |  {qty:>6}  {bar}")
        lines.append(f"  {'--- SPREAD: ':>10}{self.spread:.2f} ---" if self.spread else "  --- EMPTY ---")
        for price, qty in depth["bids"]:
            bar = "█" * min(qty // 10, 30)
            lines.append(f"  {price:>10.2f}  |  {qty:>6}  {bar}")
        lines.append(f"{'BIDS':>30}")
        lines.append("=" * 32)
        return "\n".join(lines)


# =============================================================================
# Market Making Strategy
# =============================================================================

class MarketMaker:
    """
    Simple market-making strategy with inventory management.

    Places symmetric quotes around the mid-price, adjusting for:
    - Inventory risk (skew quotes away from accumulated position)
    - Volatility (widen spread in volatile conditions)
    """

    def __init__(self, base_spread: float = 0.10, quote_size: int = 100,
                 max_inventory: int = 500, skew_factor: float = 0.02):
        self.base_spread = base_spread
        self.quote_size = quote_size
        self.max_inventory = max_inventory
        self.skew_factor = skew_factor
        self.inventory = 0
        self.pnl = 0.0
        self.active_bids: list[int] = []
        self.active_asks: list[int] = []
        self.pnl_history = []
        self.inventory_history = []

    def update(self, book: OrderBook, volatility: float = 0.01):
        """Cancel old quotes and place new ones."""
        # Cancel existing orders
        for oid in self.active_bids + self.active_asks:
            book.cancel_order(oid)
        self.active_bids.clear()
        self.active_asks.clear()

        mid = book.mid_price
        if mid is None:
            return

        # Inventory skew: shift quotes to reduce position
        inventory_ratio = self.inventory / self.max_inventory if self.max_inventory > 0 else 0
        skew = self.skew_factor * inventory_ratio * mid

        # Volatility-adjusted spread
        half_spread = (self.base_spread + volatility * mid) / 2

        bid_price = mid - half_spread - skew
        ask_price = mid + half_spread - skew

        # Place quotes (reduce size near inventory limits)
        bid_size = self.quote_size if self.inventory < self.max_inventory else self.quote_size // 4
        ask_size = self.quote_size if self.inventory > -self.max_inventory else self.quote_size // 4

        bid_id = book.submit_order(Side.BID, bid_price, bid_size, OrderType.LIMIT)
        ask_id = book.submit_order(Side.ASK, ask_price, ask_size, OrderType.LIMIT)
        self.active_bids.append(bid_id)
        self.active_asks.append(ask_id)

    def process_fills(self, book: OrderBook):
        """Check for fills and update inventory/PnL."""
        for oid in self.active_bids:
            if oid in book.orders:
                order = book.orders[oid]
                if order.filled > 0:
                    self.inventory += order.filled
                    self.pnl -= order.filled * order.price

        for oid in self.active_asks:
            if oid in book.orders:
                order = book.orders[oid]
                if order.filled > 0:
                    self.inventory -= order.filled
                    self.pnl += order.filled * order.price

        mid = book.mid_price or 0
        mark_pnl = self.pnl + self.inventory * mid
        self.pnl_history.append(mark_pnl)
        self.inventory_history.append(self.inventory)


# =============================================================================
# Simulation
# =============================================================================

def simulate_market(n_steps: int = 5000, seed: int = 42):
    """
    Run a full market simulation with random order flow and a market maker.

    Order flow model:
    - 60% limit orders (uniform around mid ± 2%)
    - 30% market orders (random side)
    - 10% cancellations
    """
    rng = np.random.default_rng(seed)
    book = OrderBook(tick_size=0.01)
    mm = MarketMaker(base_spread=0.10, quote_size=100, max_inventory=500)

    # Seed the book with initial orders
    mid_init = 100.0
    for i in range(20):
        book.submit_order(Side.BID, mid_init - 0.05 * (i + 1), rng.integers(50, 200), OrderType.LIMIT)
        book.submit_order(Side.ASK, mid_init + 0.05 * (i + 1), rng.integers(50, 200), OrderType.LIMIT)

    mid_prices = []
    spreads = []
    volumes = []
    trade_count_before = 0

    for step in range(n_steps):
        # Market maker updates
        mm.update(book, volatility=0.005)

        # Random order flow
        event = rng.random()
        mid = book.mid_price or mid_init

        if event < 0.60:
            # Limit order
            side = Side.BID if rng.random() < 0.5 else Side.ASK
            offset = rng.exponential(0.3)
            if side == Side.BID:
                price = mid - offset
            else:
                price = mid + offset
            qty = rng.integers(10, 150)
            book.submit_order(side, price, qty, OrderType.LIMIT)

        elif event < 0.90:
            # Market order
            side = Side.BID if rng.random() < 0.5 else Side.ASK
            qty = rng.integers(10, 100)
            book.submit_order(side, 0 if side == Side.ASK else 1e6, qty, OrderType.MARKET)

        else:
            # Cancel a random order
            active_ids = [oid for oid, o in book.orders.items()
                         if not o.is_filled and o.remaining > 0]
            if active_ids:
                book.cancel_order(rng.choice(active_ids))

        # Track market maker fills
        mm.process_fills(book)

        # Record metrics
        mid_prices.append(book.mid_price or mid_prices[-1] if mid_prices else mid_init)
        spreads.append(book.spread or 0)
        new_trades = len(book.trades) - trade_count_before
        volumes.append(new_trades)
        trade_count_before = len(book.trades)

    return book, mm, mid_prices, spreads, volumes


# =============================================================================
# Visualisation
# =============================================================================

def plot_simulation(book, mm, mid_prices, spreads, volumes):
    """Plot simulation results."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=False)

    # 1. Mid price evolution
    axes[0].plot(mid_prices, color="#4fc3f7", linewidth=0.6)
    axes[0].set_title("Mid Price", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Price ($)")
    axes[0].grid(alpha=0.3)

    # 2. Bid-ask spread
    axes[1].plot(spreads, color="#ffa726", linewidth=0.5, alpha=0.6)
    # Rolling average
    window = 50
    if len(spreads) > window:
        rolling_spread = np.convolve(spreads, np.ones(window)/window, mode="valid")
        axes[1].plot(range(window-1, len(spreads)), rolling_spread, color="#ef5350", linewidth=1.5,
                     label=f"Rolling {window}")
    axes[1].set_title("Bid-Ask Spread", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Spread ($)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 3. Market maker PnL and inventory
    ax3a = axes[2]
    ax3b = ax3a.twinx()
    ax3a.plot(mm.pnl_history, color="#66bb6a", linewidth=1, label="Mark-to-Market PnL")
    ax3b.plot(mm.inventory_history, color="#ab47bc", linewidth=0.6, alpha=0.6, label="Inventory")
    ax3a.set_title("Market Maker Performance", fontsize=12, fontweight="bold")
    ax3a.set_ylabel("PnL ($)", color="#66bb6a")
    ax3b.set_ylabel("Inventory (shares)", color="#ab47bc")
    ax3a.legend(loc="upper left")
    ax3b.legend(loc="upper right")
    ax3a.grid(alpha=0.3)

    # 4. Order book snapshot (final state)
    depth = book.get_depth(15)
    bid_prices_d = [p for p, q in depth["bids"]]
    bid_qtys = [q for p, q in depth["bids"]]
    ask_prices_d = [p for p, q in depth["asks"]]
    ask_qtys = [q for p, q in depth["asks"]]

    axes[3].barh(range(len(bid_prices_d)), bid_qtys, color="#66bb6a", alpha=0.7, label="Bids")
    axes[3].barh(range(len(bid_prices_d), len(bid_prices_d) + len(ask_prices_d)),
                 ask_qtys, color="#ef5350", alpha=0.7, label="Asks")

    all_prices = bid_prices_d + ask_prices_d
    axes[3].set_yticks(range(len(all_prices)))
    axes[3].set_yticklabels([f"${p:.2f}" for p in all_prices], fontsize=8)
    axes[3].set_title("Order Book Depth (Final Snapshot)", fontsize=12, fontweight="bold")
    axes[3].set_xlabel("Quantity")
    axes[3].legend()
    axes[3].grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("order_book_simulation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: order_book_simulation.png")


def plot_trade_analysis(book):
    """Analyse trade flow."""
    if not book.trades:
        return

    prices = [t.price for t in book.trades]
    sizes = [t.quantity for t in book.trades]
    sides = [1 if t.aggressor_side == Side.BID else -1 for t in book.trades]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Trade price distribution
    axes[0].hist(prices, bins=50, color="#4fc3f7", alpha=0.7, edgecolor="none")
    axes[0].set_title("Trade Price Distribution")
    axes[0].set_xlabel("Price")
    axes[0].set_ylabel("Count")

    # Order flow imbalance (rolling)
    cum_imbalance = np.cumsum([s * q for s, q in zip(sides, sizes)])
    axes[1].plot(cum_imbalance, color="#ab47bc", linewidth=0.8)
    axes[1].set_title("Cumulative Order Flow Imbalance")
    axes[1].set_xlabel("Trade #")
    axes[1].set_ylabel("Net Buy Volume")
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)

    # Trade size distribution
    axes[2].hist(sizes, bins=30, color="#ffa726", alpha=0.7, edgecolor="none")
    axes[2].set_title("Trade Size Distribution")
    axes[2].set_xlabel("Quantity")
    axes[2].set_ylabel("Count")

    for ax in axes:
        ax.grid(alpha=0.3)

    plt.suptitle("Trade Flow Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("trade_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: trade_analysis.png")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("LIMIT ORDER BOOK SIMULATOR")
    print("=" * 60)

    print("\nRunning simulation (5000 steps)...")
    t0 = time.perf_counter()
    book, mm, mid_prices, spreads, volumes = simulate_market(n_steps=5000)
    elapsed = time.perf_counter() - t0
    print(f"Simulation completed in {elapsed:.2f}s")

    # Print final book state
    print(book)

    # Statistics
    print(f"\nMarket Statistics:")
    print(f"  Total trades:        {len(book.trades)}")
    print(f"  Avg spread:          ${np.mean(spreads):.4f}")
    print(f"  Price range:         ${min(mid_prices):.2f} – ${max(mid_prices):.2f}")
    print(f"  Final mid:           ${mid_prices[-1]:.2f}")
    print(f"\nMarket Maker:")
    print(f"  Final PnL:           ${mm.pnl_history[-1]:.2f}")
    print(f"  Final inventory:     {mm.inventory_history[-1]} shares")
    print(f"  Max inventory:       {max(abs(i) for i in mm.inventory_history)} shares")

    # Plots
    plot_simulation(book, mm, mid_prices, spreads, volumes)
    plot_trade_analysis(book)
    print("\nDone!")


if __name__ == "__main__":
    main()
