import pandas as pd
import numpy as np
import os

# Load raw stock data
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
csv_path = os.path.join(data_dir, 'raw_stock_data.csv')

print("Loading raw stock data...")
stock_data = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)

print("\n" + "="*50)
print("PHASE 3: DATA PROCESSING & RETURNS CALCULATION")
print("="*50)

# Extract only Close prices
close_prices = stock_data['Close']
print("\nClose prices extracted:")
print(close_prices.head())

# Calculate daily percentage returns
daily_returns = close_prices.pct_change()

# Remove first NaN row
daily_returns = daily_returns.dropna()

print("\nFirst 10 rows of daily returns:")
print(daily_returns.head(10))

print("\nSummary statistics of daily returns:")
print(daily_returns.describe())

# Calculate cumulative returns
cumulative_returns = (1 + daily_returns).cumprod()
cumulative_returns_pct = (cumulative_returns - 1) * 100

print("\nCumulative returns (%) - Last 5 values:")
print(cumulative_returns_pct.tail())

# Function to calculate annualized returns
def calculate_annualized_return(daily_returns):
    """
    Calculate annualized returns from daily returns.
    Formula: (1 + total_return) ^ (252 / number_of_trading_days) - 1
    """
    total_return = (1 + daily_returns).prod() - 1
    num_days = len(daily_returns)
    annualized_return = (1 + total_return) ** (252 / num_days) - 1
    return annualized_return

# Apply to each stock
annualized_returns = daily_returns.apply(calculate_annualized_return)

print("\nAnnualized returns for each stock:")
for ticker, ret in annualized_returns.items():
    print(f"{ticker}: {ret:.2%}")

print("\n" + "="*50)
print("PHASE 4: RISK ANALYSIS")
print("="*50)

# Calculate daily volatility (standard deviation)
daily_volatility = daily_returns.std()

print("\nDaily volatility (standard deviation) for each stock:")
for ticker, vol in daily_volatility.items():
    print(f"{ticker}: {vol:.4f}")

# Function to calculate annualized volatility
def calculate_annualized_volatility(daily_returns):
    """
    Calculate annualized volatility from daily returns.
    Formula: daily_std * sqrt(252)
    """
    daily_std = daily_returns.std()
    annualized_vol = daily_std * np.sqrt(252)
    return annualized_vol

# Apply to all stocks
annualized_volatility = daily_returns.apply(calculate_annualized_volatility)

print("\nAnnualized volatility for each stock:")
for ticker, vol in annualized_volatility.items():
    print(f"{ticker}: {vol:.2%}")

# Function to calculate Sharpe Ratio
def calculate_sharpe_ratio(annualized_return, annualized_volatility, risk_free_rate=0.02):
    """
    Calculate Sharpe Ratio.
    Formula: (annual_return - risk_free_rate) / annual_volatility
    """
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    return sharpe_ratio

# Calculate Sharpe Ratios for all stocks
risk_free_rate = 0.02
sharpe_ratios = {
    ticker: calculate_sharpe_ratio(annualized_returns[ticker], annualized_volatility[ticker], risk_free_rate)
    for ticker in annualized_returns.index
}

# Create results dataframe
results_df = pd.DataFrame({
    'Stock Ticker': annualized_returns.index,
    'Annualized Return': annualized_returns.values,
    'Annualized Volatility': annualized_volatility.values,
    'Sharpe Ratio': [sharpe_ratios[ticker] for ticker in annualized_returns.index]
})

# Sort by Sharpe Ratio in descending order
results_df = results_df.sort_values('Sharpe Ratio', ascending=False)

print("\n" + "="*50)
print("PORTFOLIO RISK METRICS (sorted by Sharpe Ratio)")
print("="*50)
print(results_df.to_string(index=False))

# Export results
outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
os.makedirs(outputs_dir, exist_ok=True)
results_path = os.path.join(outputs_dir, 'portfolio_metrics.csv')
results_df.to_csv(results_path, index=False)
print(f"\nResults exported to: {results_path}")

# Export daily returns
returns_path = os.path.join(outputs_dir, 'daily_returns.csv')
daily_returns.to_csv(returns_path)
print(f"Daily returns exported to: {returns_path}")

# Export cumulative returns
cum_returns_path = os.path.join(outputs_dir, 'cumulative_returns.csv')
cumulative_returns_pct.to_csv(cum_returns_path)
print(f"Cumulative returns exported to: {cum_returns_path}")