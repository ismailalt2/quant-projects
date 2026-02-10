"""
Statistical Arbitrage – Pairs Trading Backtest
===============================================
Identifies cointegrated stock pairs, implements a z-score mean-reversion
strategy, and backtests with full P&L tracking and risk metrics.

Methodology:
1. Engle-Granger cointegration test to identify valid pairs
2. Kalman filter for dynamic hedge ratio estimation
3. Z-score based entry/exit signals
4. Backtest with transaction costs, Sharpe ratio, max drawdown

Dependencies: pip install yfinance numpy scipy statsmodels matplotlib pandas

Author: ismailalt2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    from statsmodels.tsa.stattools import coint, adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("statsmodels not installed — using simplified cointegration test.")


# =============================================================================
# 1. Data Loading
# =============================================================================

def fetch_pair_data(ticker1: str, ticker2: str, period: str = "3y") -> pd.DataFrame:
    """Fetch adjusted close prices for a pair of stocks."""
    data = yf.download([ticker1, ticker2], period=period, auto_adjust=True)["Close"]
    data.columns = [ticker1, ticker2]
    return data.dropna()


def generate_synthetic_pair(n: int = 750, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic cointegrated pair for demonstration."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2025-01-31", periods=n)

    # Common stochastic trend
    trend = np.cumsum(rng.normal(0.0003, 0.015, n))

    # Stock A follows trend
    noise_a = np.cumsum(rng.normal(0, 0.005, n))
    price_a = 100 * np.exp(trend + noise_a)

    # Stock B is cointegrated with A (mean-reverting spread)
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.92 * spread[i-1] + rng.normal(0, 0.8)  # OU process

    price_b = 0.6 * price_a + spread + 20  # linear relationship + noise

    return pd.DataFrame({"STOCK_A": price_a, "STOCK_B": price_b}, index=dates)


# =============================================================================
# 2. Cointegration Testing
# =============================================================================

def test_cointegration(prices: pd.DataFrame) -> dict:
    """Run Engle-Granger cointegration test and ADF on the spread."""
    y = prices.iloc[:, 0].values
    x = prices.iloc[:, 1].values
    names = prices.columns.tolist()

    # OLS hedge ratio
    slope, intercept, r_value, p_ols, std_err = stats.linregress(x, y)
    spread = y - slope * x - intercept

    # ADF test on spread
    if HAS_STATSMODELS:
        coint_stat, coint_pval, _ = coint(y, x)
        adf_stat, adf_pval, _, _, adf_crit, _ = adfuller(spread)
    else:
        # Simplified: check if spread is mean-reverting via autocorrelation
        autocorr = np.corrcoef(spread[:-1], spread[1:])[0, 1]
        coint_pval = 0.01 if autocorr < 0.95 else 0.10
        coint_stat = -3.5 if autocorr < 0.95 else -1.5
        adf_pval = coint_pval
        adf_stat = coint_stat
        adf_crit = {"1%": -3.43, "5%": -2.86, "10%": -2.57}

    result = {
        "pair": f"{names[0]} / {names[1]}",
        "hedge_ratio": slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "coint_stat": coint_stat,
        "coint_pval": coint_pval,
        "adf_stat": adf_stat if not HAS_STATSMODELS else adf_stat,
        "adf_pval": adf_pval if not HAS_STATSMODELS else adf_pval,
        "spread_mean": np.mean(spread),
        "spread_std": np.std(spread),
        "half_life": compute_half_life(spread),
        "is_cointegrated": coint_pval < 0.05,
    }
    return result


def compute_half_life(spread: np.ndarray) -> float:
    """Estimate mean-reversion half-life via AR(1) regression."""
    lag = spread[:-1]
    diff = np.diff(spread)
    slope, _, _, _, _ = stats.linregress(lag, diff)
    if slope >= 0:
        return np.inf
    return -np.log(2) / slope


# =============================================================================
# 3. Kalman Filter for Dynamic Hedge Ratio
# =============================================================================

def kalman_hedge_ratio(y: np.ndarray, x: np.ndarray,
                       delta: float = 1e-4) -> np.ndarray:
    """
    Estimate time-varying hedge ratio using a Kalman filter.

    State: beta (hedge ratio), intercept
    Observation: y_t = beta_t * x_t + intercept_t + noise
    """
    n = len(y)
    # State: [beta, intercept]
    theta = np.zeros((n, 2))
    P = np.eye(2) * 1.0          # state covariance
    R = np.eye(2) * delta         # process noise
    Ve = 1e-3                     # measurement noise (will be updated)

    theta[0] = [1.0, 0.0]

    for t in range(1, n):
        # Predict
        P = P + R

        # Observation model
        F = np.array([x[t], 1.0])
        y_hat = F @ theta[t-1]
        e = y[t] - y_hat

        # Update measurement noise
        S = F @ P @ F + Ve
        K = P @ F / S
        theta[t] = theta[t-1] + K * e
        P = P - np.outer(K, K) * S

        # Adaptive measurement noise
        Ve = max(0.99 * Ve + 0.01 * e**2, 1e-6)

    return theta[:, 0]  # return hedge ratios


# =============================================================================
# 4. Backtest Engine
# =============================================================================

def backtest_pairs(prices: pd.DataFrame, lookback: int = 60,
                   entry_z: float = 2.0, exit_z: float = 0.5,
                   stop_z: float = 4.0, transaction_cost: float = 0.001,
                   use_kalman: bool = True) -> pd.DataFrame:
    """
    Backtest a pairs trading strategy.

    Signals:
    - Enter long spread when z < -entry_z  (buy A, sell B)
    - Enter short spread when z > +entry_z (sell A, buy B)
    - Exit when |z| < exit_z
    - Stop loss when |z| > stop_z

    Returns DataFrame with full P&L history.
    """
    y = prices.iloc[:, 0].values
    x = prices.iloc[:, 1].values
    n = len(y)
    names = prices.columns.tolist()

    # Compute hedge ratios
    if use_kalman:
        hedge = kalman_hedge_ratio(y, x)
    else:
        # Rolling OLS
        hedge = np.full(n, np.nan)
        for i in range(lookback, n):
            s, inc, _, _, _ = stats.linregress(x[i-lookback:i], y[i-lookback:i])
            hedge[i] = s

    # Compute spread and z-score
    spread = np.full(n, np.nan)
    z_score = np.full(n, np.nan)

    for i in range(lookback, n):
        spread[i] = y[i] - hedge[i] * x[i]

    for i in range(2 * lookback, n):
        window = spread[i-lookback:i]
        valid = window[~np.isnan(window)]
        if len(valid) > 10:
            z_score[i] = (spread[i] - np.mean(valid)) / max(np.std(valid), 1e-8)

    # Generate signals
    position = np.zeros(n)    # +1 long spread, -1 short spread, 0 flat
    pnl = np.zeros(n)
    trades = []

    for i in range(2 * lookback + 1, n):
        if np.isnan(z_score[i]):
            position[i] = position[i-1]
            continue

        prev_pos = position[i-1]

        # Entry
        if prev_pos == 0:
            if z_score[i] < -entry_z:
                position[i] = 1    # long spread: buy A, sell B
            elif z_score[i] > entry_z:
                position[i] = -1   # short spread: sell A, buy B
            else:
                position[i] = 0
        # Exit or stop
        elif prev_pos != 0:
            if abs(z_score[i]) < exit_z:
                position[i] = 0    # mean-reversion achieved
            elif abs(z_score[i]) > stop_z:
                position[i] = 0    # stop loss
            else:
                position[i] = prev_pos

        # P&L calculation (spread return)
        if prev_pos != 0:
            ret_y = (y[i] - y[i-1]) / y[i-1]
            ret_x = (x[i] - x[i-1]) / x[i-1]
            spread_ret = prev_pos * (ret_y - hedge[i] * ret_x * (x[i-1] / y[i-1]))
            pnl[i] = spread_ret

        # Transaction costs on position changes
        if position[i] != prev_pos:
            pnl[i] -= transaction_cost
            trades.append({"date_idx": i, "action": "enter" if position[i] != 0 else "exit",
                           "position": position[i], "z_score": z_score[i]})

    # Build results DataFrame
    results = pd.DataFrame({
        "date": prices.index[:n],
        names[0]: y, names[1]: x,
        "hedge_ratio": hedge,
        "spread": spread,
        "z_score": z_score,
        "position": position,
        "pnl": pnl,
        "cumulative_pnl": np.cumsum(pnl),
    })

    return results, trades


# =============================================================================
# 5. Performance Metrics
# =============================================================================

def compute_metrics(results: pd.DataFrame) -> dict:
    """Compute strategy performance metrics."""
    pnl = results["pnl"].values
    cum_pnl = results["cumulative_pnl"].values
    active = pnl[pnl != 0]

    # Annualised Sharpe (252 trading days)
    if len(active) > 0 and np.std(active) > 0:
        sharpe = np.mean(active) / np.std(active) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - peak
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0

    # Win rate
    winning = np.sum(active > 0)
    total = len(active)
    win_rate = winning / total if total > 0 else 0.0

    # Profit factor
    gross_profit = np.sum(active[active > 0])
    gross_loss = abs(np.sum(active[active < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    positions = results["position"].values
    n_trades = np.sum(np.diff(positions) != 0)

    return {
        "total_return": cum_pnl[-1],
        "annualised_sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": int(n_trades),
        "avg_daily_pnl": np.mean(active) if len(active) > 0 else 0,
        "pnl_std": np.std(active) if len(active) > 0 else 0,
    }


# =============================================================================
# 6. Visualisation
# =============================================================================

def plot_backtest(results: pd.DataFrame, metrics: dict, coint_info: dict):
    """Create comprehensive backtest visualisation."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.5, 1.5, 1.5]})

    dates = results["date"]
    names = [c for c in results.columns if c not in
             ["date", "hedge_ratio", "spread", "z_score", "position", "pnl", "cumulative_pnl"]]

    # 1. Price chart with normalised prices
    ax = axes[0]
    p1 = results[names[0]] / results[names[0]].iloc[0] * 100
    p2 = results[names[1]] / results[names[1]].iloc[0] * 100
    ax.plot(dates, p1, label=names[0], color="#4fc3f7", linewidth=1.2)
    ax.plot(dates, p2, label=names[1], color="#ef5350", linewidth=1.2)
    ax.set_ylabel("Normalised Price")
    ax.set_title(f"Pairs Trading: {coint_info['pair']}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # 2. Z-Score with entry/exit bands
    ax = axes[1]
    ax.plot(dates, results["z_score"], color="#ab47bc", linewidth=0.8, label="Z-Score")
    ax.axhline(2, color="#66bb6a", linestyle="--", alpha=0.7, label="Entry ±2")
    ax.axhline(-2, color="#66bb6a", linestyle="--", alpha=0.7)
    ax.axhline(0.5, color="#ffa726", linestyle=":", alpha=0.7, label="Exit ±0.5")
    ax.axhline(-0.5, color="#ffa726", linestyle=":", alpha=0.7)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)

    # Shade positions
    pos = results["position"].values
    for i in range(1, len(pos)):
        if pos[i] == 1:
            ax.axvspan(dates.iloc[i-1], dates.iloc[i], alpha=0.05, color="green")
        elif pos[i] == -1:
            ax.axvspan(dates.iloc[i-1], dates.iloc[i], alpha=0.05, color="red")

    ax.set_ylabel("Z-Score")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Cumulative P&L
    ax = axes[2]
    cum_pnl = results["cumulative_pnl"] * 100
    ax.fill_between(dates, cum_pnl, where=cum_pnl >= 0, color="#66bb6a", alpha=0.3)
    ax.fill_between(dates, cum_pnl, where=cum_pnl < 0, color="#ef5350", alpha=0.3)
    ax.plot(dates, cum_pnl, color="white", linewidth=1)
    ax.set_ylabel("Cumulative P&L (%)")
    ax.grid(alpha=0.3)

    # 4. Dynamic hedge ratio
    ax = axes[3]
    ax.plot(dates, results["hedge_ratio"], color="#4fc3f7", linewidth=0.8)
    ax.set_ylabel("Hedge Ratio (β)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.3)

    # Metrics annotation
    metrics_text = (
        f"Sharpe: {metrics['annualised_sharpe']:.2f}  |  "
        f"Return: {metrics['total_return']*100:.1f}%  |  "
        f"Max DD: {metrics['max_drawdown']*100:.1f}%  |  "
        f"Win Rate: {metrics['win_rate']:.1%}  |  "
        f"Trades: {metrics['num_trades']}"
    )
    fig.text(0.5, 0.01, metrics_text, ha="center", fontsize=11,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#333333", edgecolor="#666666"),
             color="white")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("pairs_trading_backtest.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: pairs_trading_backtest.png")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("STATISTICAL ARBITRAGE – PAIRS TRADING BACKTEST")
    print("=" * 60)

    # Load data
    if HAS_YFINANCE:
        print("\nFetching data for KO/PEP (Coca-Cola / PepsiCo)...")
        try:
            prices = fetch_pair_data("KO", "PEP", period="3y")
        except Exception as e:
            print(f"Fetch failed: {e}. Using synthetic data.")
            prices = generate_synthetic_pair()
    else:
        print("Using synthetic cointegrated pair...")
        prices = generate_synthetic_pair()

    print(f"Data: {len(prices)} trading days, {prices.index[0].date()} to {prices.index[-1].date()}")

    # Cointegration test
    coint_info = test_cointegration(prices)
    print(f"\nCointegration Test: {coint_info['pair']}")
    print(f"  p-value:      {coint_info['coint_pval']:.4f} {'✓' if coint_info['is_cointegrated'] else '✗'}")
    print(f"  Hedge ratio:  {coint_info['hedge_ratio']:.4f}")
    print(f"  R²:           {coint_info['r_squared']:.4f}")
    print(f"  Half-life:    {coint_info['half_life']:.1f} days")

    if not coint_info["is_cointegrated"]:
        print("\n⚠ Pair is not cointegrated at 5% level. Proceeding anyway for demonstration.")

    # Backtest
    print("\nRunning backtest...")
    results, trades = backtest_pairs(prices, lookback=60, entry_z=2.0,
                                      exit_z=0.5, stop_z=4.0,
                                      transaction_cost=0.001, use_kalman=True)

    # Metrics
    metrics = compute_metrics(results)
    print(f"\n{'PERFORMANCE METRICS':=^50}")
    print(f"  Total Return:        {metrics['total_return']*100:+.2f}%")
    print(f"  Annualised Sharpe:   {metrics['annualised_sharpe']:.2f}")
    print(f"  Max Drawdown:        {metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate:            {metrics['win_rate']:.1%}")
    print(f"  Profit Factor:       {metrics['profit_factor']:.2f}")
    print(f"  Number of Trades:    {metrics['num_trades']}")

    # Plot
    plot_backtest(results, metrics, coint_info)
    print("\nDone!")


if __name__ == "__main__":
    main()
