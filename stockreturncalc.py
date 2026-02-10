# stock_return_calculator.py
import yfinance as yf
import pandas as pd

# Download Apple stock data
symbol = "AAPL"
data = yf.download(symbol, start="2023-01-01", end="2024-01-01")

# Use 'Close' instead of 'Adj Close'
data['Daily Return'] = data['Close'].pct_change()

# Show last 5 rows
print(f"{symbol} Daily Returns (Last 5 Days):")
print(data[['Close', 'Daily Return']].tail())

# Save to CSV
data[['Close', 'Daily Return']].to_csv('aapl_returns.csv')   
