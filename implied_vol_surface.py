"""
Implied Volatility Surface
===========================
Fetches real options chain data, computes implied volatility using
Newton-Raphson root finding, and plots a 3D volatility surface with
smile/skew analysis.

Dependencies: pip install yfinance numpy scipy matplotlib

Author: ismailalt2
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import warnings

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("yfinance not installed — using synthetic data for demonstration.")


# =============================================================================
# Black-Scholes formula (needed for IV computation)
# =============================================================================

def bs_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes European option price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# =============================================================================
# Implied Volatility via Brent's method
# =============================================================================

def implied_vol(market_price, S, K, T, r, option_type="call"):
    """
    Compute implied volatility using Brent's method on the interval [0.001, 5.0].

    Returns NaN if no root is found (e.g., arbitrage violation).
    """
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    if market_price < intrinsic - 0.01:
        return np.nan
    if T <= 1e-8:
        return np.nan

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    try:
        return brentq(objective, 1e-4, 5.0, xtol=1e-8, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan


# =============================================================================
# Fetch real options data
# =============================================================================

def fetch_options_data(ticker: str = "SPY"):
    """Fetch options chain from Yahoo Finance and compute IVs."""
    stock = yf.Ticker(ticker)
    S = stock.history(period="1d")["Close"].iloc[-1]
    expirations = stock.options

    # Use up to 8 expiration dates
    expirations = expirations[:8]
    r = 0.045  # approximate risk-free rate

    records = []
    for exp_str in expirations:
        chain = stock.option_chain(exp_str)
        T = (datetime.strptime(exp_str, "%Y-%m-%d") - datetime.now()).days / 365.25
        if T <= 0.01:
            continue

        for _, row in chain.calls.iterrows():
            K = row["strike"]
            mid = (row["bid"] + row["ask"]) / 2 if row["bid"] > 0 and row["ask"] > 0 else row["lastPrice"]
            if mid <= 0 or row.get("volume", 0) is None:
                continue
            moneyness = K / S
            if 0.7 < moneyness < 1.3:  # filter extreme strikes
                iv = implied_vol(mid, S, K, T, r, "call")
                if not np.isnan(iv) and iv < 2.0:
                    records.append({
                        "expiration": exp_str, "T": T, "strike": K,
                        "moneyness": moneyness, "mid_price": mid, "iv": iv,
                    })

    return records, S, r


# =============================================================================
# Generate synthetic data (fallback if yfinance unavailable)
# =============================================================================

def generate_synthetic_data():
    """Generate realistic synthetic IV surface data with smile and term structure."""
    S = 450.0
    r = 0.045
    records = []

    expiries_days = [7, 14, 30, 60, 90, 120, 180, 365]
    moneyness_range = np.linspace(0.80, 1.20, 40)

    for days in expiries_days:
        T = days / 365.25
        for m in moneyness_range:
            K = S * m
            # Realistic IV with smile (quadratic in log-moneyness) and term structure
            log_m = np.log(m)
            base_vol = 0.18                                # ATM base
            smile = 0.15 * log_m**2                        # curvature
            skew = -0.08 * log_m                           # negative skew
            term_adj = -0.03 * np.sqrt(T)                  # vol term structure
            noise = np.random.normal(0, 0.003)
            iv_val = max(base_vol + smile + skew + term_adj + noise, 0.05)

            mid = bs_price(S, K, T, r, iv_val, "call")
            records.append({
                "expiration": f"{days}d", "T": T, "strike": K,
                "moneyness": m, "mid_price": mid, "iv": iv_val,
            })

    return records, S, r


# =============================================================================
# Plotting
# =============================================================================

def plot_vol_surface(records, S, title_suffix=""):
    """Create 3D implied volatility surface and 2D smile/skew plots."""
    moneyness = np.array([r["moneyness"] for r in records])
    T = np.array([r["T"] for r in records])
    iv = np.array([r["iv"] for r in records]) * 100  # to percentage

    # --- 3D Surface ---
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(moneyness, T * 365, iv, c=iv, cmap="plasma",
                         s=12, alpha=0.8, edgecolors="none")
    ax.set_xlabel("Moneyness (K/S)", fontsize=11)
    ax.set_ylabel("Days to Expiry", fontsize=11)
    ax.set_zlabel("Implied Vol (%)", fontsize=11)
    ax.set_title(f"Implied Volatility Surface {title_suffix}", fontsize=14, fontweight="bold")
    fig.colorbar(scatter, ax=ax, shrink=0.5, label="IV (%)")
    ax.view_init(elev=25, azim=-45)
    plt.tight_layout()
    plt.savefig("iv_surface_3d.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: iv_surface_3d.png")

    # --- 2D Smile by Expiry ---
    unique_expiries = sorted(set(r["expiration"] for r in records))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cmap = plt.cm.viridis(np.linspace(0, 1, len(unique_expiries)))
    for idx, exp in enumerate(unique_expiries):
        subset = [r for r in records if r["expiration"] == exp]
        m = [r["moneyness"] for r in subset]
        v = [r["iv"] * 100 for r in subset]
        sort_idx = np.argsort(m)
        m, v = np.array(m)[sort_idx], np.array(v)[sort_idx]
        axes[0].plot(m, v, color=cmap[idx], label=exp, linewidth=1.5)

    axes[0].axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="ATM")
    axes[0].set_xlabel("Moneyness (K/S)")
    axes[0].set_ylabel("Implied Vol (%)")
    axes[0].set_title("Volatility Smile by Expiry")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)

    # --- ATM Term Structure ---
    atm_data = [(r["T"], r["iv"] * 100) for r in records if 0.97 < r["moneyness"] < 1.03]
    if atm_data:
        atm_T, atm_iv = zip(*sorted(atm_data))
        axes[1].scatter(np.array(atm_T) * 365, atm_iv, alpha=0.4, s=15, color="#4fc3f7")
        # Rolling average
        from scipy.ndimage import uniform_filter1d
        if len(atm_iv) > 5:
            sorted_pairs = sorted(zip(atm_T, atm_iv))
            t_arr = np.array([p[0] for p in sorted_pairs]) * 365
            iv_arr = np.array([p[1] for p in sorted_pairs])
            axes[1].plot(t_arr, uniform_filter1d(iv_arr, size=min(5, len(iv_arr))),
                         color="#ef5350", linewidth=2, label="Smoothed")

    axes[1].set_xlabel("Days to Expiry")
    axes[1].set_ylabel("ATM Implied Vol (%)")
    axes[1].set_title("ATM Volatility Term Structure")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"Volatility Analysis {title_suffix}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("iv_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: iv_analysis.png")


def print_summary(records, S):
    """Print summary statistics of the IV surface."""
    ivs = [r["iv"] for r in records]
    atm_ivs = [r["iv"] for r in records if 0.98 < r["moneyness"] < 1.02]

    print(f"\n{'=' * 55}")
    print(f"IMPLIED VOLATILITY SURFACE SUMMARY")
    print(f"{'=' * 55}")
    print(f"Spot Price:           ${S:.2f}")
    print(f"Total data points:    {len(records)}")
    print(f"Expiries:             {len(set(r['expiration'] for r in records))}")
    print(f"IV range:             {min(ivs)*100:.1f}% – {max(ivs)*100:.1f}%")
    if atm_ivs:
        print(f"ATM IV (mean):        {np.mean(atm_ivs)*100:.1f}%")

    # Skew metric: IV(90% moneyness) - IV(110% moneyness)
    put_wing = [r["iv"] for r in records if 0.88 < r["moneyness"] < 0.92]
    call_wing = [r["iv"] for r in records if 1.08 < r["moneyness"] < 1.12]
    if put_wing and call_wing:
        skew = (np.mean(put_wing) - np.mean(call_wing)) * 100
        print(f"25-delta skew:        {skew:+.1f}pp")
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    if HAS_YFINANCE:
        print("Fetching SPY options data from Yahoo Finance...")
        try:
            records, S, r = fetch_options_data("SPY")
            title_suffix = "(SPY – Live Data)"
        except Exception as e:
            print(f"Failed to fetch data: {e}")
            print("Falling back to synthetic data...")
            records, S, r = generate_synthetic_data()
            title_suffix = "(Synthetic)"
    else:
        records, S, r = generate_synthetic_data()
        title_suffix = "(Synthetic)"

    if not records:
        print("No valid data points. Exiting.")
        return

    print_summary(records, S)
    plot_vol_surface(records, S, title_suffix)
    print("Done! Check iv_surface_3d.png and iv_analysis.png")


if __name__ == "__main__":
    main()
