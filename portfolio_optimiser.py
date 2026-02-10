"""
Portfolio Optimiser
====================
Implements portfolio construction methods:
1. Markowitz Mean-Variance Optimisation (MVO)
2. Risk Parity (Equal Risk Contribution)
3. Equal Weight benchmark
4. Maximum Sharpe Ratio (Tangency Portfolio)
5. Minimum Variance Portfolio

Includes efficient frontier plotting, transaction cost modelling,
and out-of-sample backtesting.

Dependencies: pip install numpy scipy matplotlib pandas yfinance

Author: ismailalt2
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


# =============================================================================
# 1. Data Loading
# =============================================================================

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ", "XOM", "GLD", "TLT", "VNQ"]

def fetch_returns(tickers: list = UNIVERSE, period: str = "5y") -> pd.DataFrame:
    """Fetch daily returns from Yahoo Finance."""
    prices = yf.download(tickers, period=period, auto_adjust=True)["Close"]
    returns = prices.pct_change().dropna()
    return returns


def generate_synthetic_returns(n_assets: int = 10, n_days: int = 1260, seed: int = 42):
    """Generate realistic synthetic returns with correlation structure."""
    rng = np.random.default_rng(seed)
    names = [f"Asset_{i+1}" for i in range(n_assets)]

    # Generate correlation matrix via random factor model
    n_factors = 3
    B = rng.normal(0, 0.3, (n_assets, n_factors))
    cov_factors = np.eye(n_factors)
    specific_var = np.diag(rng.uniform(0.01, 0.05, n_assets)**2)
    cov = B @ cov_factors @ B.T + specific_var

    # Ensure positive definite
    eigvals = np.linalg.eigvalsh(cov)
    if np.min(eigvals) < 0:
        cov += (-np.min(eigvals) + 1e-6) * np.eye(n_assets)

    # Expected returns with some dispersion
    mu_annual = rng.uniform(0.04, 0.15, n_assets)
    mu_daily = mu_annual / 252

    L = np.linalg.cholesky(cov)
    raw = rng.normal(size=(n_days, n_assets))
    returns = raw @ L.T + mu_daily

    dates = pd.bdate_range(end="2025-01-31", periods=n_days)
    return pd.DataFrame(returns, index=dates, columns=names)


# =============================================================================
# 2. Portfolio Optimisation
# =============================================================================

def portfolio_stats(weights, mu, cov, rf=0.045/252):
    """Compute portfolio return, volatility, and Sharpe ratio (annualised)."""
    ret = np.dot(weights, mu) * 252
    vol = np.sqrt(weights @ cov @ weights) * np.sqrt(252)
    sharpe = (ret - rf * 252) / vol if vol > 0 else 0
    return ret, vol, sharpe


def max_sharpe_portfolio(mu, cov, rf=0.045/252) -> np.ndarray:
    """Find the tangency (maximum Sharpe) portfolio."""
    n = len(mu)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n

    def neg_sharpe(w):
        r, v, s = portfolio_stats(w, mu, cov, rf)
        return -s

    result = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP",
                      bounds=bounds, constraints=constraints)
    return result.x


def min_variance_portfolio(cov) -> np.ndarray:
    """Find the global minimum variance portfolio."""
    n = cov.shape[0]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n

    def variance(w):
        return w @ cov @ w

    result = minimize(variance, np.ones(n) / n, method="SLSQP",
                      bounds=bounds, constraints=constraints)
    return result.x


def target_return_portfolio(mu, cov, target_ret) -> np.ndarray:
    """Find minimum variance portfolio for a given target return."""
    n = len(mu)
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: np.dot(w, mu) * 252 - target_ret},
    ]
    bounds = [(0, 1)] * n

    def variance(w):
        return w @ cov @ w

    result = minimize(variance, np.ones(n) / n, method="SLSQP",
                      bounds=bounds, constraints=constraints)
    if result.success:
        return result.x
    return None


def risk_parity_portfolio(cov) -> np.ndarray:
    """
    Risk parity: equalise risk contribution from each asset.

    Risk contribution of asset i = w_i * (Σw)_i / σ_p²
    """
    n = cov.shape[0]

    def risk_budget_objective(w):
        port_var = w @ cov @ w
        marginal_risk = cov @ w
        risk_contrib = w * marginal_risk / port_var
        target_rc = 1.0 / n
        return np.sum((risk_contrib - target_rc) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 1)] * n

    result = minimize(risk_budget_objective, np.ones(n) / n, method="SLSQP",
                      bounds=bounds, constraints=constraints)
    return result.x


# =============================================================================
# 3. Efficient Frontier
# =============================================================================

def compute_efficient_frontier(mu, cov, n_points: int = 100):
    """Compute the efficient frontier by varying target returns."""
    # Find return range
    min_var_w = min_variance_portfolio(cov)
    min_ret = np.dot(min_var_w, mu) * 252
    max_ret = np.max(mu) * 252

    target_returns = np.linspace(min_ret, max_ret * 0.95, n_points)
    frontier_vol = []
    frontier_ret = []
    frontier_weights = []

    for target in target_returns:
        w = target_return_portfolio(mu, cov, target)
        if w is not None:
            r, v, _ = portfolio_stats(w, mu, cov)
            frontier_ret.append(r)
            frontier_vol.append(v)
            frontier_weights.append(w)

    return np.array(frontier_ret), np.array(frontier_vol), frontier_weights


# =============================================================================
# 4. Backtest with Rebalancing
# =============================================================================

def backtest_strategy(returns: pd.DataFrame, strategy_fn, lookback: int = 252,
                      rebal_freq: int = 21, transaction_cost: float = 0.001):
    """
    Walk-forward backtest with periodic rebalancing.

    Parameters
    ----------
    returns : daily returns DataFrame
    strategy_fn : function(mu, cov) -> weights
    lookback : estimation window (days)
    rebal_freq : rebalancing frequency (days)
    transaction_cost : one-way cost per trade
    """
    n_assets = returns.shape[1]
    weights = np.ones(n_assets) / n_assets
    portfolio_returns = []
    weight_history = []
    dates = []

    for t in range(lookback, len(returns)):
        # Daily return
        daily_ret = returns.iloc[t].values
        port_ret = np.dot(weights, daily_ret)

        # Rebalance
        if (t - lookback) % rebal_freq == 0:
            hist = returns.iloc[t-lookback:t]
            mu = hist.mean().values
            cov = hist.cov().values

            new_weights = strategy_fn(mu, cov)
            # Transaction costs
            turnover = np.sum(np.abs(new_weights - weights))
            port_ret -= turnover * transaction_cost
            weights = new_weights

        # Update weights for drift
        weights = weights * (1 + daily_ret)
        weights = weights / np.sum(weights)

        portfolio_returns.append(port_ret)
        weight_history.append(weights.copy())
        dates.append(returns.index[t])

    return pd.Series(portfolio_returns, index=dates), np.array(weight_history)


# =============================================================================
# 5. Visualisation
# =============================================================================

def plot_efficient_frontier(mu, cov, returns, asset_names):
    """Plot efficient frontier with key portfolios marked."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Efficient frontier
    f_ret, f_vol, _ = compute_efficient_frontier(mu, cov)
    ax.plot(f_vol * 100, f_ret * 100, color="#4fc3f7", linewidth=2.5, label="Efficient Frontier")

    # Individual assets
    asset_ret = mu * 252 * 100
    asset_vol = np.sqrt(np.diag(cov)) * np.sqrt(252) * 100
    ax.scatter(asset_vol, asset_ret, color="#ffa726", s=80, zorder=5, edgecolors="white", linewidth=1)
    for i, name in enumerate(asset_names):
        ax.annotate(name, (asset_vol[i], asset_ret[i]), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")

    # Key portfolios
    portfolios = {
        "Max Sharpe": max_sharpe_portfolio(mu, cov),
        "Min Variance": min_variance_portfolio(cov),
        "Risk Parity": risk_parity_portfolio(cov),
        "Equal Weight": np.ones(len(mu)) / len(mu),
    }
    colors = {"Max Sharpe": "#ef5350", "Min Variance": "#66bb6a",
              "Risk Parity": "#ab47bc", "Equal Weight": "#ffa726"}
    markers = {"Max Sharpe": "★", "Min Variance": "◆", "Risk Parity": "●", "Equal Weight": "■"}

    for name, w in portfolios.items():
        r, v, s = portfolio_stats(w, mu, cov)
        ax.scatter(v * 100, r * 100, color=colors[name], s=200, zorder=10,
                   edgecolors="white", linewidth=2, label=f"{name} (SR={s:.2f})")

    ax.set_xlabel("Annualised Volatility (%)", fontsize=12)
    ax.set_ylabel("Annualised Return (%)", fontsize=12)
    ax.set_title("Efficient Frontier & Portfolio Strategies", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("efficient_frontier.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: efficient_frontier.png")


def plot_backtest_results(results: dict, returns: pd.DataFrame):
    """Plot cumulative returns and drawdowns for all strategies."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

    colors = {"Max Sharpe": "#ef5350", "Min Variance": "#66bb6a",
              "Risk Parity": "#ab47bc", "Equal Weight": "#ffa726"}

    # 1. Cumulative returns
    for name, (rets, _) in results.items():
        cum = (1 + rets).cumprod()
        axes[0].plot(cum.index, cum, label=name, color=colors.get(name, "gray"), linewidth=1.5)

    axes[0].set_title("Cumulative Returns (Walk-Forward Backtest)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. Drawdowns
    for name, (rets, _) in results.items():
        cum = (1 + rets).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak * 100
        axes[1].plot(dd.index, dd, label=name, color=colors.get(name, "gray"), linewidth=0.8)

    axes[1].set_title("Drawdowns", fontsize=12)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # 3. Rolling Sharpe (60-day)
    for name, (rets, _) in results.items():
        rolling_mean = rets.rolling(60).mean() * 252
        rolling_std = rets.rolling(60).std() * np.sqrt(252)
        rolling_sharpe = rolling_mean / rolling_std
        axes[2].plot(rolling_sharpe.index, rolling_sharpe, label=name,
                     color=colors.get(name, "gray"), linewidth=0.8)

    axes[2].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_title("Rolling 60-Day Sharpe Ratio", fontsize=12)
    axes[2].set_ylabel("Sharpe Ratio")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("portfolio_backtest.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: portfolio_backtest.png")


def print_results_table(results: dict):
    """Print performance comparison table."""
    print(f"\n{'Strategy':<18} {'Return':>8} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8} {'Calmar':>8}")
    print("-" * 60)

    for name, (rets, _) in results.items():
        ann_ret = rets.mean() * 252
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = (1 + rets).cumprod()
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        print(f"{name:<18} {ann_ret:>7.1%} {ann_vol:>7.1%} {sharpe:>8.2f} {max_dd:>7.1%} {calmar:>8.2f}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("PORTFOLIO OPTIMISER")
    print("=" * 60)

    # Load data
    if HAS_YFINANCE:
        print(f"\nFetching data for {len(UNIVERSE)} assets...")
        try:
            returns = fetch_returns(UNIVERSE, period="5y")
            asset_names = UNIVERSE
        except Exception as e:
            print(f"Fetch failed: {e}. Using synthetic data.")
            returns = generate_synthetic_returns()
            asset_names = returns.columns.tolist()
    else:
        print("Using synthetic return data...")
        returns = generate_synthetic_returns()
        asset_names = returns.columns.tolist()

    print(f"Data: {len(returns)} days, {returns.shape[1]} assets")
    print(f"Period: {returns.index[0].date()} to {returns.index[-1].date()}")

    # Full-sample estimates
    mu = returns.mean().values
    cov = returns.cov().values

    # Key portfolios
    print(f"\n{'PORTFOLIO ALLOCATIONS':=^60}")
    strategies = {
        "Max Sharpe": max_sharpe_portfolio(mu, cov),
        "Min Variance": min_variance_portfolio(cov),
        "Risk Parity": risk_parity_portfolio(cov),
        "Equal Weight": np.ones(len(mu)) / len(mu),
    }

    for name, w in strategies.items():
        r, v, s = portfolio_stats(w, mu, cov)
        print(f"\n{name} (Return={r:.1%}, Vol={v:.1%}, Sharpe={s:.2f}):")
        for i, asset in enumerate(asset_names):
            if w[i] > 0.01:
                print(f"  {asset:<8} {w[i]:>6.1%}")

    # Efficient frontier plot
    plot_efficient_frontier(mu, cov, returns, asset_names)

    # Walk-forward backtest
    print(f"\n{'WALK-FORWARD BACKTEST':=^60}")
    strategy_fns = {
        "Max Sharpe": lambda mu, cov: max_sharpe_portfolio(mu, cov),
        "Min Variance": lambda mu, cov: min_variance_portfolio(cov),
        "Risk Parity": lambda mu, cov: risk_parity_portfolio(cov),
        "Equal Weight": lambda mu, cov: np.ones(len(mu)) / len(mu),
    }

    results = {}
    for name, fn in strategy_fns.items():
        print(f"  Backtesting {name}...")
        rets, weights = backtest_strategy(returns, fn, lookback=252,
                                           rebal_freq=21, transaction_cost=0.001)
        results[name] = (rets, weights)

    print_results_table(results)
    plot_backtest_results(results, returns)
    print("\nDone!")


if __name__ == "__main__":
    main()
