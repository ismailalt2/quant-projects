# quant-projects

Small projects exploring quantitative finance concepts — pricing, risk, portfolio construction, market microstructure, and real-time data processing.

## Projects

| # | Project | Key Concepts | File |
|---|---------|-------------|------|
| 1 | **Options Pricing Engine** | Black-Scholes, Monte Carlo, Binomial Tree, Greeks | `options_pricing_engine.py` |
| 2 | **Implied Volatility Surface** | IV computation (Brent's method), smile/skew analysis, term structure | `implied_vol_surface.py` |
| 3 | **Pairs Trading Backtest** | Cointegration (Engle-Granger), Kalman filter hedge ratio, z-score signals | `pairs_trading_backtest.py` |
| 4 | **Order Book Simulator** | Price-time priority matching, market making, inventory management | `order_book_simulator.py` |
| 5 | **Portfolio Optimiser** | Markowitz MVO, risk parity, efficient frontier, walk-forward backtest | `portfolio_optimiser.py` |
| 6 | **Real-Time Data Pipeline** | WebSocket streaming, VWAP, EMA crossover, RSI, Bollinger Bands | `rt_datapipeline.py` |
| 7 | **Risk Engine (VaR)** | Historical/Parametric/Monte Carlo VaR, CVaR, stress testing, Kupiec backtest | `REvar.py` | 
# Some file names may be different
## Setup

```bash
pip install numpy scipy matplotlib pandas yfinance statsmodels websockets
```

Each script runs standalone:

```bash
python options_pricing_engine.py
python implied_vol_surface.py
python pairs_trading_backtest.py
python order_book_simulator.py
python portfolio_optimiser.py
python realtime_data_pipeline.py
python risk_engine_var.py
```

All projects include **synthetic data fallback** — they run without internet or API keys, generating realistic simulated data for demonstration.

## Project Details

### 1. Options Pricing Engine

Prices European options using three methods and compares convergence:

- **Black-Scholes** closed-form with full Greeks (Δ, Γ, ν, Θ, ρ)
- **Monte Carlo** with antithetic variates for variance reduction
- **Binomial Tree** (Cox-Ross-Rubinstein) with configurable steps

Outputs: convergence plots, 3D Greeks surfaces, put-call parity verification.

### 2. Implied Volatility Surface

Fetches live SPY options chain data, computes implied volatility via root-finding (Brent's method on the BS equation), and visualises:

- 3D volatility surface (moneyness × time × IV)
- Smile curves by expiry
- ATM volatility term structure
- Skew metrics (25-delta)

### 3. Statistical Arbitrage — Pairs Trading

Full pipeline from pair selection to backtesting:

- **Engle-Granger cointegration test** for pair validation
- **Kalman filter** for dynamic hedge ratio estimation
- **Z-score signals** with entry/exit/stop-loss thresholds
- Performance: Sharpe ratio, max drawdown, win rate, profit factor

### 4. Order Book Simulator

Implements a realistic limit order book from scratch:

- **Matching engine** with price-time priority (FIFO within price level)
- Bid/ask heaps for O(log n) best price access
- **Market making strategy** with inventory-aware quote skewing
- Order flow analysis: trade distribution, cumulative imbalance

### 5. Portfolio Optimiser

Compares portfolio construction approaches:

- **Maximum Sharpe** (tangency portfolio)
- **Minimum Variance**
- **Risk Parity** (equal risk contribution via optimisation)
- **Equal Weight** benchmark
- **Walk-forward backtest** with monthly rebalancing and transaction costs

### 6. Real-Time Data Pipeline

Streams and processes market data in real time:

- **Binance WebSocket** for live BTC/USDT trades (or simulated fallback)
- Rolling statistics engine: VWAP, dual EMA, Bollinger Bands, RSI
- **Momentum signal** combining EMA crossover with RSI confirmation
- Regime-switching simulation with clustered volatility

### 7. Risk Engine — Value at Risk

Industry-standard risk measurement:

- **Historical Simulation** VaR (non-parametric)
- **Parametric VaR** with Cornish-Fisher skewness/kurtosis adjustment
- **Monte Carlo VaR** with fitted Student-t distribution
- **Expected Shortfall** (CVaR) for all methods
- **Component VaR** decomposition by asset
- **Stress testing** with historical scenarios (2008, COVID, dot-com, 2022)
- **Kupiec backtest** for model validation

## Tech Stack

Python, NumPy, SciPy, pandas, matplotlib, yfinance, statsmodels, websockets

## License

MIT
