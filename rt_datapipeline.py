"""
Real-Time Market Data Pipeline
================================
Streams live market data via WebSocket, computes rolling statistics
(VWAP, volatility, momentum signals), and displays a live terminal
dashboard or generates a summary report.

Supports:
- Binance WebSocket for crypto (no API key needed)
- Simulated data fallback for offline demonstration

Dependencies: pip install websockets numpy matplotlib

Author: ismailalt2
"""

import numpy as np
import json
import time
import asyncio
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
import threading

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    print("websockets not installed — using simulated data stream.")


# =============================================================================
# 1. Rolling Statistics Engine
# =============================================================================

class RollingStats:
    """
    Efficient online computation of rolling statistics using circular buffers.

    Computes: VWAP, EMA, Bollinger Bands, RSI, rolling volatility,
    momentum signals, and microstructure metrics.
    """

    def __init__(self, window: int = 200, fast_ema: int = 12, slow_ema: int = 26):
        self.window = window
        self.prices = deque(maxlen=window)
        self.volumes = deque(maxlen=window)
        self.timestamps = deque(maxlen=window)
        self.returns = deque(maxlen=window)

        # EMA state
        self.fast_ema_val = None
        self.slow_ema_val = None
        self.fast_alpha = 2 / (fast_ema + 1)
        self.slow_alpha = 2 / (slow_ema + 1)

        # RSI state
        self.rsi_window = 14
        self.gains = deque(maxlen=self.rsi_window)
        self.losses = deque(maxlen=self.rsi_window)

        # Trade tracking
        self.trade_count = 0
        self.total_volume = 0.0
        self.total_turnover = 0.0  # price * volume

        # Full history for plotting
        self.price_history = []
        self.vwap_history = []
        self.fast_ema_history = []
        self.slow_ema_history = []
        self.vol_history = []
        self.rsi_history = []
        self.signal_history = []
        self.time_history = []

    def update(self, price: float, volume: float, timestamp: float):
        """Process a new tick."""
        # Return
        if len(self.prices) > 0:
            ret = (price - self.prices[-1]) / self.prices[-1]
            self.returns.append(ret)
            # RSI
            if ret > 0:
                self.gains.append(ret)
                self.losses.append(0)
            else:
                self.gains.append(0)
                self.losses.append(abs(ret))

        self.prices.append(price)
        self.volumes.append(volume)
        self.timestamps.append(timestamp)
        self.trade_count += 1
        self.total_volume += volume
        self.total_turnover += price * volume

        # EMA update
        if self.fast_ema_val is None:
            self.fast_ema_val = price
            self.slow_ema_val = price
        else:
            self.fast_ema_val = self.fast_alpha * price + (1 - self.fast_alpha) * self.fast_ema_val
            self.slow_ema_val = self.slow_alpha * price + (1 - self.slow_alpha) * self.slow_ema_val

        # Store history
        self.price_history.append(price)
        self.vwap_history.append(self.vwap)
        self.fast_ema_history.append(self.fast_ema_val)
        self.slow_ema_history.append(self.slow_ema_val)
        self.vol_history.append(self.rolling_volatility)
        self.rsi_history.append(self.rsi)
        self.signal_history.append(self.momentum_signal)
        self.time_history.append(timestamp)

    @property
    def vwap(self) -> float:
        """Volume-weighted average price."""
        if not self.volumes or sum(self.volumes) == 0:
            return self.prices[-1] if self.prices else 0
        return sum(p * v for p, v in zip(self.prices, self.volumes)) / sum(self.volumes)

    @property
    def rolling_volatility(self) -> float:
        """Annualised rolling volatility."""
        if len(self.returns) < 2:
            return 0.0
        return float(np.std(self.returns)) * np.sqrt(252 * 24 * 60)  # per-minute data

    @property
    def bollinger_bands(self) -> tuple:
        """Upper, middle, lower Bollinger Bands (2σ)."""
        if len(self.prices) < 20:
            p = self.prices[-1] if self.prices else 0
            return p, p, p
        arr = np.array(list(self.prices)[-20:])
        mid = np.mean(arr)
        std = np.std(arr)
        return mid + 2 * std, mid, mid - 2 * std

    @property
    def rsi(self) -> float:
        """Relative Strength Index."""
        if len(self.gains) < self.rsi_window:
            return 50.0
        avg_gain = np.mean(self.gains)
        avg_loss = np.mean(self.losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)

    @property
    def momentum_signal(self) -> int:
        """
        Combined momentum signal: -1 (bearish), 0 (neutral), +1 (bullish).

        Based on EMA crossover + RSI confirmation.
        """
        if self.fast_ema_val is None or self.slow_ema_val is None:
            return 0

        ema_signal = 1 if self.fast_ema_val > self.slow_ema_val else -1
        rsi_val = self.rsi

        if ema_signal == 1 and rsi_val > 50:
            return 1
        elif ema_signal == -1 and rsi_val < 50:
            return -1
        return 0

    def summary(self) -> dict:
        """Return current statistics summary."""
        bb_upper, bb_mid, bb_lower = self.bollinger_bands
        return {
            "price": self.prices[-1] if self.prices else 0,
            "vwap": self.vwap,
            "fast_ema": self.fast_ema_val or 0,
            "slow_ema": self.slow_ema_val or 0,
            "volatility": self.rolling_volatility,
            "rsi": self.rsi,
            "bb_upper": bb_upper,
            "bb_mid": bb_mid,
            "bb_lower": bb_lower,
            "signal": self.momentum_signal,
            "trade_count": self.trade_count,
            "total_volume": self.total_volume,
        }


# =============================================================================
# 2. Data Sources
# =============================================================================

async def binance_stream(symbol: str, stats: RollingStats, duration: int = 120):
    """Stream real-time trade data from Binance WebSocket."""
    url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade"
    print(f"Connecting to Binance WebSocket for {symbol.upper()}...")

    async with websockets.connect(url) as ws:
        start = time.time()
        while time.time() - start < duration:
            msg = await ws.recv()
            data = json.loads(msg)
            price = float(data["p"])
            volume = float(data["q"])
            timestamp = data["T"] / 1000

            stats.update(price, volume, timestamp)

            if stats.trade_count % 100 == 0:
                s = stats.summary()
                signal_str = {1: "🟢 BULL", -1: "🔴 BEAR", 0: "⚪ NEUTRAL"}[s["signal"]]
                print(f"  [{stats.trade_count:>6}] ${s['price']:.2f} | "
                      f"VWAP: ${s['vwap']:.2f} | Vol: {s['volatility']:.1%} | "
                      f"RSI: {s['rsi']:.1f} | {signal_str}")


def simulated_stream(stats: RollingStats, n_ticks: int = 5000, seed: int = 42):
    """Generate simulated market data with realistic microstructure."""
    rng = np.random.default_rng(seed)
    price = 100.0
    t = time.time()

    # Regime switching: trending vs mean-reverting
    regime = "trending"
    regime_counter = 0

    for i in range(n_ticks):
        # Regime switch
        regime_counter += 1
        if regime_counter > rng.integers(200, 500):
            regime = "mean_revert" if regime == "trending" else "trending"
            regime_counter = 0

        # Price dynamics
        if regime == "trending":
            drift = 0.0001 * (1 if rng.random() > 0.45 else -1)
            vol = 0.002
        else:
            drift = -0.0005 * (price - 100) / 100
            vol = 0.003

        # Clustered volatility (GARCH-like)
        if rng.random() < 0.02:
            vol *= 3  # vol spike

        ret = drift + vol * rng.standard_normal()
        price *= (1 + ret)
        price = max(price, 1.0)  # floor

        # Volume: higher at extremes and random bursts
        base_vol = rng.exponential(50)
        volume = base_vol * (1 + abs(ret) * 100)

        stats.update(price, volume, t + i * 0.5)

        if (i + 1) % 500 == 0:
            s = stats.summary()
            signal_str = {1: "BULL", -1: "BEAR", 0: "NEUTRAL"}[s["signal"]]
            print(f"  [{i+1:>6}] ${s['price']:.2f} | VWAP: ${s['vwap']:.2f} | "
                  f"Vol: {s['volatility']:.1%} | RSI: {s['rsi']:.1f} | {signal_str}")


# =============================================================================
# 3. Visualisation / Dashboard
# =============================================================================

def plot_dashboard(stats: RollingStats, symbol: str = ""):
    """Generate static dashboard from collected data."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1, 1]})

    n = len(stats.price_history)
    x = range(n)

    # 1. Price with overlays
    ax = axes[0]
    ax.plot(x, stats.price_history, color="#4fc3f7", linewidth=0.8, label="Price", alpha=0.9)
    ax.plot(x, stats.vwap_history, color="#ffa726", linewidth=1.2, label="VWAP", linestyle="--")
    ax.plot(x, stats.fast_ema_history, color="#66bb6a", linewidth=1, label="Fast EMA (12)")
    ax.plot(x, stats.slow_ema_history, color="#ef5350", linewidth=1, label="Slow EMA (26)")

    # Signal markers
    for i in range(1, n):
        if stats.signal_history[i] != stats.signal_history[i-1]:
            if stats.signal_history[i] == 1:
                ax.axvline(i, color="green", alpha=0.1, linewidth=3)
            elif stats.signal_history[i] == -1:
                ax.axvline(i, color="red", alpha=0.1, linewidth=3)

    ax.set_title(f"Real-Time Market Data Pipeline {f'({symbol})' if symbol else ''}",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # 2. Rolling Volatility
    ax = axes[1]
    vol_pct = [v * 100 for v in stats.vol_history]
    ax.fill_between(x, vol_pct, alpha=0.3, color="#ab47bc")
    ax.plot(x, vol_pct, color="#ab47bc", linewidth=0.8)
    ax.set_ylabel("Volatility (%)")
    ax.set_title("Rolling Annualised Volatility", fontsize=10)
    ax.grid(alpha=0.3)

    # 3. RSI
    ax = axes[2]
    ax.plot(x, stats.rsi_history, color="#4fc3f7", linewidth=0.8)
    ax.axhline(70, color="#ef5350", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axhline(30, color="#66bb6a", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.fill_between(x, stats.rsi_history, 70,
                     where=[r > 70 for r in stats.rsi_history], alpha=0.2, color="#ef5350")
    ax.fill_between(x, stats.rsi_history, 30,
                     where=[r < 30 for r in stats.rsi_history], alpha=0.2, color="#66bb6a")
    ax.set_ylabel("RSI")
    ax.set_ylim(0, 100)
    ax.set_title("Relative Strength Index (14)", fontsize=10)
    ax.grid(alpha=0.3)

    # 4. Momentum Signal
    ax = axes[3]
    colors = ["#66bb6a" if s == 1 else "#ef5350" if s == -1 else "#666666"
              for s in stats.signal_history]
    ax.bar(x, stats.signal_history, color=colors, width=1.0)
    ax.set_ylabel("Signal")
    ax.set_xlabel("Tick")
    ax.set_title("Momentum Signal (+1 Bull / -1 Bear)", fontsize=10)
    ax.set_yticks([-1, 0, 1])
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("market_pipeline_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: market_pipeline_dashboard.png")


def print_final_report(stats: RollingStats):
    """Print summary report."""
    s = stats.summary()
    print(f"\n{'=' * 55}")
    print(f"{'PIPELINE SUMMARY':^55}")
    print(f"{'=' * 55}")
    print(f"  Total ticks processed:  {s['trade_count']:,}")
    print(f"  Total volume:           {s['total_volume']:,.1f}")
    print(f"  Final price:            ${s['price']:.2f}")
    print(f"  VWAP:                   ${s['vwap']:.2f}")
    print(f"  Fast EMA:               ${s['fast_ema']:.2f}")
    print(f"  Slow EMA:               ${s['slow_ema']:.2f}")
    print(f"  Rolling Volatility:     {s['volatility']:.2%}")
    print(f"  RSI:                    {s['rsi']:.1f}")
    print(f"  Bollinger Upper:        ${s['bb_upper']:.2f}")
    print(f"  Bollinger Lower:        ${s['bb_lower']:.2f}")

    signal_map = {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}
    print(f"  Momentum Signal:        {signal_map[s['signal']]}")

    # Signal distribution
    bull = sum(1 for s in stats.signal_history if s == 1)
    bear = sum(1 for s in stats.signal_history if s == -1)
    neut = sum(1 for s in stats.signal_history if s == 0)
    total = len(stats.signal_history)
    print(f"\n  Signal Distribution:")
    print(f"    Bullish: {bull/total:.1%} | Bearish: {bear/total:.1%} | Neutral: {neut/total:.1%}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("REAL-TIME MARKET DATA PIPELINE")
    print("=" * 60)

    stats = RollingStats(window=200, fast_ema=12, slow_ema=26)

    if HAS_WEBSOCKETS:
        print("\nStreaming live data from Binance (BTCUSDT, 2 minutes)...")
        try:
            asyncio.run(binance_stream("btcusdt", stats, duration=120))
            symbol = "BTC/USDT"
        except Exception as e:
            print(f"WebSocket failed: {e}. Falling back to simulation.")
            stats = RollingStats(window=200, fast_ema=12, slow_ema=26)
            simulated_stream(stats, n_ticks=5000)
            symbol = "Simulated"
    else:
        print("\nRunning simulated market data stream (5000 ticks)...")
        simulated_stream(stats, n_ticks=5000)
        symbol = "Simulated"

    print_final_report(stats)
    plot_dashboard(stats, symbol)
    print("\nDone!")


if __name__ == "__main__":
    main()
