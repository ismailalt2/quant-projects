"""
Options Pricing Engine
======================
Implements three methods for pricing European options:
1. Black-Scholes analytical solution
2. Monte Carlo simulation
3. Binomial Tree (Cox-Ross-Rubinstein)

Compares accuracy, convergence, and computational cost across methods.

Author: ismailalt2
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time


# =============================================================================
# 1. Black-Scholes Analytical Solution
# =============================================================================

def black_scholes(S: float, K: float, T: float, r: float, sigma: float,
                  option_type: str = "call") -> dict:
    """
    Black-Scholes closed-form solution for European options.

    Parameters
    ----------
    S : float – Current stock price
    K : float – Strike price
    T : float – Time to expiration (years)
    r : float – Risk-free rate (annualised)
    sigma : float – Volatility (annualised)
    option_type : str – 'call' or 'put'

    Returns
    -------
    dict with price and Greeks (delta, gamma, vega, theta, rho)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100,       # per 1% move in vol
        "theta": theta / 365,     # per calendar day
        "rho": rho / 100,         # per 1% move in rate
    }


# =============================================================================
# 2. Monte Carlo Simulation
# =============================================================================

def monte_carlo(S: float, K: float, T: float, r: float, sigma: float,
                option_type: str = "call", n_simulations: int = 100_000,
                seed: int = 42) -> dict:
    """
    Monte Carlo pricing via geometric Brownian motion with antithetic variates.

    Returns dict with price, standard error, and 95% confidence interval.
    """
    rng = np.random.default_rng(seed)

    # Generate paths with antithetic variates for variance reduction
    Z = rng.standard_normal(n_simulations // 2)
    Z = np.concatenate([Z, -Z])  # antithetic

    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    discounted = np.exp(-r * T) * payoffs
    price = np.mean(discounted)
    std_error = np.std(discounted) / np.sqrt(n_simulations)

    return {
        "price": price,
        "std_error": std_error,
        "ci_95": (price - 1.96 * std_error, price + 1.96 * std_error),
    }


# =============================================================================
# 3. Binomial Tree (Cox-Ross-Rubinstein)
# =============================================================================

def binomial_tree(S: float, K: float, T: float, r: float, sigma: float,
                  option_type: str = "call", n_steps: int = 500) -> dict:
    """
    CRR binomial tree for European options.

    Returns dict with price and the tree parameters (u, d, p).
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))          # up factor
    d = 1 / u                                 # down factor
    p = (np.exp(r * dt) - d) / (u - d)       # risk-neutral probability
    discount = np.exp(-r * dt)

    # Terminal payoffs
    asset_prices = S * u ** np.arange(n_steps, -1, -1) * d ** np.arange(0, n_steps + 1)

    if option_type == "call":
        values = np.maximum(asset_prices - K, 0)
    else:
        values = np.maximum(K - asset_prices, 0)

    # Backward induction
    for step in range(n_steps - 1, -1, -1):
        values = discount * (p * values[:-1] + (1 - p) * values[1:])

    return {
        "price": values[0],
        "u": u,
        "d": d,
        "p": p,
    }


# =============================================================================
# Comparison & Visualisation
# =============================================================================

def convergence_analysis(S=100, K=100, T=1, r=0.05, sigma=0.2):
    """Plot convergence of Monte Carlo and Binomial Tree to Black-Scholes price."""
    bs = black_scholes(S, K, T, r, sigma)
    bs_price = bs["price"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Monte Carlo convergence ---
    sim_counts = [1000, 5000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
    mc_prices, mc_errors = [], []
    for n in sim_counts:
        result = monte_carlo(S, K, T, r, sigma, n_simulations=n)
        mc_prices.append(result["price"])
        mc_errors.append(result["std_error"])

    axes[0].errorbar(sim_counts, mc_prices, yerr=[1.96 * e for e in mc_errors],
                     fmt="o-", capsize=4, color="#4fc3f7", label="MC estimate ± 95% CI")
    axes[0].axhline(bs_price, color="#ef5350", linestyle="--", label=f"BS = {bs_price:.4f}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Number of Simulations")
    axes[0].set_ylabel("Option Price ($)")
    axes[0].set_title("Monte Carlo Convergence")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- Binomial Tree convergence ---
    step_counts = [10, 25, 50, 100, 200, 500, 1000]
    bt_prices = [binomial_tree(S, K, T, r, sigma, n_steps=n)["price"] for n in step_counts]

    axes[1].plot(step_counts, bt_prices, "o-", color="#66bb6a", label="Binomial Tree")
    axes[1].axhline(bs_price, color="#ef5350", linestyle="--", label=f"BS = {bs_price:.4f}")
    axes[1].set_xlabel("Number of Steps")
    axes[1].set_ylabel("Option Price ($)")
    axes[1].set_title("Binomial Tree Convergence")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Convergence to Black-Scholes Analytical Price", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("options_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: options_convergence.png")


def greeks_surface(S=100, K=100, T=1, r=0.05):
    """Plot Greeks as a function of spot price and volatility."""
    spots = np.linspace(60, 140, 80)
    vols = np.linspace(0.05, 0.60, 60)
    S_grid, V_grid = np.meshgrid(spots, vols)

    delta_grid = np.zeros_like(S_grid)
    gamma_grid = np.zeros_like(S_grid)

    for i in range(S_grid.shape[0]):
        for j in range(S_grid.shape[1]):
            g = black_scholes(S_grid[i, j], K, T, r, V_grid[i, j])
            delta_grid[i, j] = g["delta"]
            gamma_grid[i, j] = g["gamma"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), subplot_kw={"projection": "3d"})

    axes[0].plot_surface(S_grid, V_grid, delta_grid, cmap="coolwarm", alpha=0.85)
    axes[0].set_xlabel("Spot Price")
    axes[0].set_ylabel("Volatility")
    axes[0].set_zlabel("Delta")
    axes[0].set_title("Delta Surface")

    axes[1].plot_surface(S_grid, V_grid, gamma_grid, cmap="viridis", alpha=0.85)
    axes[1].set_xlabel("Spot Price")
    axes[1].set_ylabel("Volatility")
    axes[1].set_zlabel("Gamma")
    axes[1].set_title("Gamma Surface")

    plt.suptitle("Greeks Surfaces (European Call)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("greeks_surface.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: greeks_surface.png")


def main():
    # Parameters
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    print("=" * 65)
    print("OPTIONS PRICING ENGINE – European Call")
    print(f"S={S}, K={K}, T={T}y, r={r:.1%}, σ={sigma:.1%}")
    print("=" * 65)

    # Black-Scholes
    t0 = time.perf_counter()
    bs = black_scholes(S, K, T, r, sigma)
    bs_time = time.perf_counter() - t0
    print(f"\n{'Black-Scholes':.<30} ${bs['price']:.4f}  ({bs_time*1e6:.0f} μs)")
    print(f"  Delta={bs['delta']:.4f}  Gamma={bs['gamma']:.4f}  "
          f"Vega={bs['vega']:.4f}  Theta={bs['theta']:.4f}  Rho={bs['rho']:.4f}")

    # Monte Carlo
    t0 = time.perf_counter()
    mc = monte_carlo(S, K, T, r, sigma, n_simulations=1_000_000)
    mc_time = time.perf_counter() - t0
    print(f"\n{'Monte Carlo (1M sims)':.<30} ${mc['price']:.4f}  ({mc_time:.3f}s)")
    print(f"  Std Error: {mc['std_error']:.6f}")
    print(f"  95% CI: [{mc['ci_95'][0]:.4f}, {mc['ci_95'][1]:.4f}]")

    # Binomial Tree
    t0 = time.perf_counter()
    bt = binomial_tree(S, K, T, r, sigma, n_steps=1000)
    bt_time = time.perf_counter() - t0
    print(f"\n{'Binomial Tree (1000 steps)':.<30} ${bt['price']:.4f}  ({bt_time:.3f}s)")
    print(f"  u={bt['u']:.6f}  d={bt['d']:.6f}  p={bt['p']:.6f}")

    # Errors
    print(f"\n{'Method':<28} {'Price':>8} {'Error vs BS':>12}")
    print("-" * 50)
    print(f"{'Black-Scholes':<28} {bs['price']:>8.4f} {'—':>12}")
    print(f"{'Monte Carlo':<28} {mc['price']:>8.4f} {mc['price']-bs['price']:>+12.6f}")
    print(f"{'Binomial Tree':<28} {bt['price']:>8.4f} {bt['price']-bs['price']:>+12.6f}")

    # Plots
    convergence_analysis(S, K, T, r, sigma)
    greeks_surface(S, K, T, r)

    # Put-Call Parity check
    put_bs = black_scholes(S, K, T, r, sigma, "put")
    parity_lhs = bs["price"] - put_bs["price"]
    parity_rhs = S - K * np.exp(-r * T)
    print(f"\nPut-Call Parity Check: C - P = {parity_lhs:.6f}, S - Ke^(-rT) = {parity_rhs:.6f}")
    print(f"Parity error: {abs(parity_lhs - parity_rhs):.2e}")


if __name__ == "__main__":
    main()
