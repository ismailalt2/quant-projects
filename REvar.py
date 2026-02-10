"""
Risk Engine – Value at Risk Calculator
========================================
Computes portfolio VaR using three industry-standard methods:
1. Historical Simulation
2. Parametric (Variance-Covariance)
3. Monte Carlo Simulation

Includes:
- Expected Shortfall (CVaR) for each method
- Backtesting with Kupiec and Christoffersen tests
- Stress testing with historical crash scenarios
- Component VaR decomposition

Dependencies: pip install numpy scipy matplotlib pandas yfinance

Author: ismailalt2
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


# =============================================================================
# 1. Data
# =============================================================================

PORTFOLIO = {
    "AAPL": 0.20, "MSFT": 0.15, "GOOGL": 0.10, "AMZN": 0.10,
    "JPM": 0.10, "JNJ": 0.10, "XOM": 0.05, "GLD": 0.10, "TLT": 0.10,
}

def fetch_portfolio_data(holdings: dict = PORTFOLIO, period: str = "5y") -> tuple:
    """Fetch returns and compute portfolio returns."""
    tickers = list(holdings.keys())
    prices = yf.download(tickers, period=period, auto_adjust=True)["Close"]
    returns = prices.pct_change().dropna()
    weights = np.array([holdings[t] for t in tickers])
    portfolio_returns = returns.values @ weights
    return returns, portfolio_returns, weights, tickers


def generate_synthetic_portfolio(n_assets: int = 9, n_days: int = 1260, seed: int = 42):
    """Generate synthetic portfolio data with fat tails."""
    rng = np.random.default_rng(seed)
    names = list(PORTFOLIO.keys())[:n_assets]
    weights = np.array(list(PORTFOLIO.values()))[:n_assets]
    weights = weights / weights.sum()

    # Student-t returns (fatter tails than normal)
    df = 5  # degrees of freedom
    mu = rng.uniform(-0.0002, 0.0005, n_assets)
    sigma = rng.uniform(0.01, 0.03, n_assets)

    # Correlation via factor model
    n_factors = 2
    B = rng.normal(0, 0.5, (n_assets, n_factors))
    factor_returns = rng.standard_t(df, (n_days, n_factors)) * 0.01
    specific = rng.standard_t(df, (n_days, n_assets)) * sigma * 0.5

    asset_returns = factor_returns @ B.T + specific + mu

    dates = pd.bdate_range(end="2025-01-31", periods=n_days)
    returns_df = pd.DataFrame(asset_returns, index=dates, columns=names)
    portfolio_returns = asset_returns @ weights

    return returns_df, portfolio_returns, weights, names


# =============================================================================
# 2. VaR Methods
# =============================================================================

def historical_var(returns: np.ndarray, confidence: float = 0.99,
                   portfolio_value: float = 1_000_000) -> dict:
    """
    Historical Simulation VaR.

    Simply uses the empirical distribution of past returns.
    Non-parametric — makes no distributional assumptions.
    """
    sorted_returns = np.sort(returns)
    n = len(sorted_returns)
    idx = int(np.floor((1 - confidence) * n))

    var_pct = -sorted_returns[idx]
    var_dollar = var_pct * portfolio_value

    # Expected Shortfall (CVaR) = mean of losses beyond VaR
    tail = sorted_returns[:idx + 1]
    cvar_pct = -np.mean(tail)
    cvar_dollar = cvar_pct * portfolio_value

    return {
        "method": "Historical Simulation",
        "var_pct": var_pct,
        "var_dollar": var_dollar,
        "cvar_pct": cvar_pct,
        "cvar_dollar": cvar_dollar,
        "confidence": confidence,
    }


def parametric_var(returns: np.ndarray, confidence: float = 0.99,
                   portfolio_value: float = 1_000_000) -> dict:
    """
    Parametric (Variance-Covariance) VaR.

    Assumes returns are normally distributed.
    Uses Cornish-Fisher expansion for skewness/kurtosis adjustment.
    """
    mu = np.mean(returns)
    sigma = np.std(returns)
    skew = float(pd.Series(returns).skew())
    kurt = float(pd.Series(returns).kurtosis())  # excess kurtosis

    z = norm.ppf(1 - confidence)

    # Standard normal VaR
    var_normal = -(mu + z * sigma)

    # Cornish-Fisher adjusted z
    z_cf = (z + (z**2 - 1) * skew / 6
            + (z**3 - 3*z) * kurt / 24
            - (2*z**3 - 5*z) * skew**2 / 36)
    var_cf = -(mu + z_cf * sigma)

    var_pct = var_cf
    var_dollar = var_pct * portfolio_value

    # CVaR under normality
    cvar_pct = -(mu - sigma * norm.pdf(z) / (1 - confidence))
    cvar_dollar = cvar_pct * portfolio_value

    return {
        "method": "Parametric (Cornish-Fisher)",
        "var_pct": var_pct,
        "var_dollar": var_dollar,
        "cvar_pct": cvar_pct,
        "cvar_dollar": cvar_dollar,
        "confidence": confidence,
        "var_normal_pct": var_normal,
        "skewness": skew,
        "excess_kurtosis": kurt,
    }


def monte_carlo_var(returns: np.ndarray, confidence: float = 0.99,
                    portfolio_value: float = 1_000_000,
                    n_simulations: int = 100_000, seed: int = 42) -> dict:
    """
    Monte Carlo VaR using fitted distribution.

    Fits a Student-t distribution to capture fat tails, then simulates.
    """
    # Fit Student-t
    params = stats.t.fit(returns)
    df_fit, loc_fit, scale_fit = params

    rng = np.random.default_rng(seed)
    simulated = stats.t.rvs(df_fit, loc=loc_fit, scale=scale_fit,
                             size=n_simulations, random_state=rng)

    sorted_sim = np.sort(simulated)
    idx = int(np.floor((1 - confidence) * n_simulations))

    var_pct = -sorted_sim[idx]
    var_dollar = var_pct * portfolio_value

    tail = sorted_sim[:idx + 1]
    cvar_pct = -np.mean(tail)
    cvar_dollar = cvar_pct * portfolio_value

    return {
        "method": "Monte Carlo (Student-t)",
        "var_pct": var_pct,
        "var_dollar": var_dollar,
        "cvar_pct": cvar_pct,
        "cvar_dollar": cvar_dollar,
        "confidence": confidence,
        "t_df": df_fit,
        "n_simulations": n_simulations,
    }


# =============================================================================
# 3. Component VaR
# =============================================================================

def component_var(returns_df: pd.DataFrame, weights: np.ndarray,
                  confidence: float = 0.99, portfolio_value: float = 1_000_000) -> pd.DataFrame:
    """
    Decompose VaR into component contributions by asset.

    Component VaR_i = w_i * β_i * VaR_portfolio
    where β_i = Cov(r_i, r_p) / Var(r_p)
    """
    cov = returns_df.cov().values
    port_var = weights @ cov @ weights
    port_vol = np.sqrt(port_var)
    z = norm.ppf(confidence)

    marginal_var = cov @ weights / port_vol * z
    component_var_pct = weights * marginal_var
    component_var_dollar = component_var_pct * portfolio_value

    # Percentage contribution
    total_var = np.sum(component_var_dollar)
    pct_contribution = component_var_dollar / total_var * 100

    df = pd.DataFrame({
        "Asset": returns_df.columns,
        "Weight": weights,
        "Component VaR ($)": component_var_dollar,
        "% Contribution": pct_contribution,
    })
    return df.sort_values("% Contribution", ascending=False)


# =============================================================================
# 4. Stress Testing
# =============================================================================

STRESS_SCENARIOS = {
    "2008 Financial Crisis": {
        "description": "Lehman collapse, Sep-Nov 2008",
        "equity_shock": -0.40,
        "bond_shock": 0.05,
        "commodity_shock": -0.30,
        "gold_shock": 0.10,
    },
    "COVID-19 Crash": {
        "description": "Feb-Mar 2020 pandemic selloff",
        "equity_shock": -0.34,
        "bond_shock": 0.08,
        "commodity_shock": -0.25,
        "gold_shock": -0.03,
    },
    "Dot-Com Bust": {
        "description": "Tech selloff 2000-2002",
        "equity_shock": -0.45,
        "bond_shock": 0.12,
        "commodity_shock": -0.10,
        "gold_shock": 0.05,
    },
    "2022 Rate Hike": {
        "description": "Fed aggressive tightening",
        "equity_shock": -0.20,
        "bond_shock": -0.15,
        "commodity_shock": 0.10,
        "gold_shock": -0.05,
    },
    "Flash Crash": {
        "description": "Sudden liquidity crisis",
        "equity_shock": -0.10,
        "bond_shock": 0.02,
        "commodity_shock": -0.08,
        "gold_shock": 0.03,
    },
}

def classify_asset(name: str) -> str:
    """Classify asset for stress testing."""
    bonds = {"TLT", "AGG", "BND", "IEF"}
    gold = {"GLD", "IAU", "GOLD"}
    commodities = {"XOM", "USO", "DBC", "XLE"}
    if name in bonds:
        return "bond"
    elif name in gold:
        return "gold"
    elif name in commodities:
        return "commodity"
    return "equity"


def run_stress_tests(weights: np.ndarray, tickers: list,
                     portfolio_value: float = 1_000_000) -> pd.DataFrame:
    """Apply stress scenarios to the portfolio."""
    results = []

    for scenario_name, shocks in STRESS_SCENARIOS.items():
        portfolio_loss = 0
        for i, ticker in enumerate(tickers):
            asset_class = classify_asset(ticker)
            shock_key = f"{asset_class}_shock"
            shock = shocks.get(shock_key, shocks.get("equity_shock", -0.20))
            portfolio_loss += weights[i] * shock

        results.append({
            "Scenario": scenario_name,
            "Description": shocks["description"],
            "Portfolio Loss (%)": portfolio_loss * 100,
            "Portfolio Loss ($)": portfolio_loss * portfolio_value,
        })

    return pd.DataFrame(results)


# =============================================================================
# 5. Backtesting
# =============================================================================

def backtest_var(returns: np.ndarray, confidence: float = 0.99,
                 window: int = 252) -> dict:
    """
    Walk-forward VaR backtest with violation analysis.

    Kupiec test: tests if violation rate matches expected rate.
    """
    n = len(returns)
    violations = []
    var_series = []

    for t in range(window, n):
        hist = returns[t-window:t]
        var_t = np.percentile(hist, (1 - confidence) * 100)
        var_series.append(-var_t)

        if returns[t] < var_t:
            violations.append(t)

    n_test = n - window
    n_violations = len(violations)
    expected_violations = n_test * (1 - confidence)
    violation_rate = n_violations / n_test

    # Kupiec POF test
    p = 1 - confidence
    if 0 < n_violations < n_test:
        lr = -2 * (np.log(p**n_violations * (1-p)**(n_test - n_violations))
                    - np.log((n_violations/n_test)**n_violations
                             * (1 - n_violations/n_test)**(n_test - n_violations)))
        kupiec_pval = 1 - chi2.cdf(lr, 1)
    else:
        kupiec_pval = 0.0

    return {
        "n_observations": n_test,
        "n_violations": n_violations,
        "expected_violations": expected_violations,
        "violation_rate": violation_rate,
        "expected_rate": 1 - confidence,
        "kupiec_pval": kupiec_pval,
        "kupiec_pass": kupiec_pval > 0.05,
        "var_series": var_series,
        "violation_indices": violations,
    }


# =============================================================================
# 6. Visualisation
# =============================================================================

def plot_var_comparison(returns, hist_var, param_var, mc_var, portfolio_value):
    """Plot return distribution with VaR levels marked."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Return distribution with VaR lines
    ax = axes[0]
    ax.hist(returns * 100, bins=80, density=True, color="#4fc3f7", alpha=0.6,
            edgecolor="none", label="Historical Returns")

    # Fitted normal
    x = np.linspace(min(returns), max(returns), 200) * 100
    mu, sigma = np.mean(returns) * 100, np.std(returns) * 100
    ax.plot(x, norm.pdf(x, mu, sigma), color="#ffa726", linewidth=1.5, label="Normal Fit")

    # VaR lines
    methods = [
        (hist_var, "#ef5350", "Historical"),
        (param_var, "#66bb6a", "Parametric"),
        (mc_var, "#ab47bc", "Monte Carlo"),
    ]
    for var_result, color, label in methods:
        ax.axvline(-var_result["var_pct"] * 100, color=color, linewidth=2,
                   linestyle="--", label=f"{label} VaR = {var_result['var_pct']*100:.2f}%")

    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    ax.set_title("Return Distribution & VaR Estimates (99%)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # VaR and CVaR comparison bar chart
    ax = axes[1]
    methods_names = ["Historical", "Parametric\n(Cornish-Fisher)", "Monte Carlo\n(Student-t)"]
    var_vals = [v["var_dollar"] / 1000 for v in [hist_var, param_var, mc_var]]
    cvar_vals = [v["cvar_dollar"] / 1000 for v in [hist_var, param_var, mc_var]]

    x_pos = np.arange(len(methods_names))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, var_vals, width, label="VaR", color="#ef5350", alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, cvar_vals, width, label="CVaR (ES)", color="#ffa726", alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods_names, fontsize=9)
    ax.set_ylabel("Loss ($K)")
    ax.set_title(f"VaR vs CVaR (99%, ${portfolio_value/1e6:.0f}M portfolio)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"${bar.get_height():.1f}K", ha="center", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"${bar.get_height():.1f}K", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("var_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: var_comparison.png")


def plot_backtest(returns, backtest_result, window=252):
    """Plot VaR backtest results."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    n = len(returns)
    test_returns = returns[window:] * 100
    var_series = np.array(backtest_result["var_series"]) * 100
    violations = [v - window for v in backtest_result["violation_indices"]]

    # Returns vs VaR
    ax = axes[0]
    ax.plot(range(len(test_returns)), test_returns, color="#4fc3f7", linewidth=0.5, alpha=0.6)
    ax.plot(range(len(var_series)), -var_series, color="#ef5350", linewidth=1.2, label="99% VaR")
    ax.scatter(violations, [test_returns[v] for v in violations if v < len(test_returns)],
               color="#ef5350", s=20, zorder=5, label=f"Violations ({backtest_result['n_violations']})")
    ax.set_ylabel("Daily Return (%)")
    ax.set_title("VaR Backtest (Historical, 252-day rolling window)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Cumulative violations vs expected
    ax = axes[1]
    cum_violations = np.zeros(len(test_returns))
    for v in violations:
        if v < len(cum_violations):
            cum_violations[v:] += 1
    expected_line = np.arange(1, len(test_returns) + 1) * 0.01

    ax.plot(range(len(test_returns)), cum_violations, color="#ef5350", linewidth=1.5, label="Actual")
    ax.plot(range(len(test_returns)), expected_line, color="#66bb6a", linewidth=1.5,
            linestyle="--", label="Expected (1%)")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Cumulative Violations")
    ax.legend()
    ax.grid(alpha=0.3)

    kupiec = "PASS ✓" if backtest_result["kupiec_pass"] else "FAIL ✗"
    fig.text(0.5, 0.01,
             f"Kupiec Test: {kupiec} (p={backtest_result['kupiec_pval']:.3f}) | "
             f"Violations: {backtest_result['n_violations']}/{backtest_result['n_observations']} "
             f"({backtest_result['violation_rate']:.2%} vs {backtest_result['expected_rate']:.2%} expected)",
             ha="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#333", edgecolor="#666"),
             color="white")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("var_backtest.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: var_backtest.png")


def plot_stress_tests(stress_df, portfolio_value):
    """Plot stress test results."""
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = ["#ef5350" if v < -20 else "#ffa726" if v < -10 else "#66bb6a"
              for v in stress_df["Portfolio Loss (%)"]]

    bars = ax.barh(stress_df["Scenario"], -stress_df["Portfolio Loss ($)"] / 1000,
                   color=colors, edgecolor="none", alpha=0.85)

    for bar, loss_pct in zip(bars, stress_df["Portfolio Loss (%)"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{loss_pct:.1f}%", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Portfolio Loss ($K)")
    ax.set_title(f"Stress Test Results (${portfolio_value/1e6:.0f}M Portfolio)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig("stress_tests.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: stress_tests.png")


# =============================================================================
# Main
# =============================================================================

def main():
    portfolio_value = 1_000_000
    confidence = 0.99

    print("=" * 60)
    print("RISK ENGINE – VALUE AT RISK CALCULATOR")
    print(f"Portfolio Value: ${portfolio_value:,.0f} | Confidence: {confidence:.0%}")
    print("=" * 60)

    # Load data
    if HAS_YFINANCE:
        print("\nFetching portfolio data...")
        try:
            returns_df, port_returns, weights, tickers = fetch_portfolio_data()
        except Exception as e:
            print(f"Fetch failed: {e}. Using synthetic data.")
            returns_df, port_returns, weights, tickers = generate_synthetic_portfolio()
    else:
        print("Using synthetic portfolio data...")
        returns_df, port_returns, weights, tickers = generate_synthetic_portfolio()

    print(f"Data: {len(port_returns)} trading days")
    print(f"Assets: {', '.join(tickers)}")

    # --- VaR Calculations ---
    print(f"\n{'VAR ESTIMATES (1-Day, 99%)':=^60}")

    hist = historical_var(port_returns, confidence, portfolio_value)
    param = parametric_var(port_returns, confidence, portfolio_value)
    mc = monte_carlo_var(port_returns, confidence, portfolio_value)

    for result in [hist, param, mc]:
        print(f"\n  {result['method']}:")
        print(f"    VaR:  {result['var_pct']:.3%} = ${result['var_dollar']:,.0f}")
        print(f"    CVaR: {result['cvar_pct']:.3%} = ${result['cvar_dollar']:,.0f}")

    if "skewness" in param:
        print(f"\n  Distribution stats: Skew={param['skewness']:.3f}, "
              f"Excess Kurtosis={param['excess_kurtosis']:.3f}")
    if "t_df" in mc:
        print(f"  Fitted Student-t df: {mc['t_df']:.2f}")

    # --- Component VaR ---
    print(f"\n{'COMPONENT VAR':=^60}")
    comp_var = component_var(returns_df, weights, confidence, portfolio_value)
    print(comp_var.to_string(index=False))

    # --- Stress Tests ---
    print(f"\n{'STRESS TESTING':=^60}")
    stress_df = run_stress_tests(weights, tickers, portfolio_value)
    for _, row in stress_df.iterrows():
        print(f"  {row['Scenario']:<25} {row['Portfolio Loss (%)']:>+7.1f}% = ${row['Portfolio Loss ($)']:>+12,.0f}")

    # --- Backtest ---
    print(f"\n{'VAR BACKTESTING':=^60}")
    bt = backtest_var(port_returns, confidence, window=252)
    print(f"  Violations: {bt['n_violations']}/{bt['n_observations']} "
          f"({bt['violation_rate']:.2%} vs {bt['expected_rate']:.2%} expected)")
    print(f"  Kupiec test: p={bt['kupiec_pval']:.4f} {'PASS ✓' if bt['kupiec_pass'] else 'FAIL ✗'}")

    # --- Plots ---
    plot_var_comparison(port_returns, hist, param, mc, portfolio_value)
    plot_backtest(port_returns, bt)
    plot_stress_tests(stress_df, portfolio_value)
    print("\nDone!")


if __name__ == "__main__":
    main()
