"""
Configuration file for Financial Portfolio Analysis project.
Contains all configurable parameters for the analysis.
"""

import os

# Stock tickers to analyze
# These are the major tech stocks included in the portfolio
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']

# Date range for historical data
# Start date for fetching stock data
START_DATE = '2023-01-01'

# End date for fetching stock data
END_DATE = '2025-11-22'

# Risk-free rate for Sharpe ratio calculation (2% annual)
# Typically based on US Treasury bond yields
RISK_FREE_RATE = 0.02

# Equal weight allocation for each stock (20% each for 5 stocks)
EQUAL_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]

# Visualization parameters
# DPI (dots per inch) for saving high-quality images
OUTPUT_DPI = 300

# Figure sizes for different chart types
# Large figures for detailed visualizations (width, height in inches)
FIGURE_FIGSIZE_LARGE = (14, 8)

# Medium figures for standard charts
FIGURE_FIGSIZE_MEDIUM = (10, 6)

# Small figures for compact displays
FIGURE_FIGSIZE_SMALL = (8, 5)

# Directory paths
# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory for storing raw and processed data files
DATA_PATH = os.path.join(BASE_DIR, 'data')

# Directory for saving generated visualizations
FIGURES_PATH = os.path.join(BASE_DIR, 'figures')

# Directory for Jupyter notebooks
NOTEBOOKS_PATH = os.path.join(BASE_DIR, 'notebooks')

# Directory for output reports and analysis results
OUTPUTS_PATH = os.path.join(BASE_DIR, 'outputs')

# Logging configuration
# Path to log file
LOG_FILE = os.path.join(OUTPUTS_PATH, 'portfolio_analysis.log')

# Logging format
LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s'

# Date format for logs
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Analysis parameters
# Number of trading days per year (standard for US markets)
TRADING_DAYS_PER_YEAR = 252

# Rolling window periods for volatility analysis
ROLLING_WINDOW_SHORT = 30  # 30-day rolling window
ROLLING_WINDOW_LONG = 90   # 90-day rolling window

# Candlestick chart parameters
CANDLESTICK_DAYS = 60  # Number of days to display in candlestick charts

# Visualization colors
COLORS_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Portfolio strategy names
STRATEGY_EQUAL_WEIGHT = 'Equal Weight'
STRATEGY_INVERSE_VOLATILITY = 'Inverse Volatility'

# Real-time data configuration
# Update interval for real-time data (in seconds)
REALTIME_UPDATE_INTERVAL = 15

# Cache TTL for real-time data (in seconds)
REALTIME_CACHE_TTL = 15

# WebSocket server configuration
WEBSOCKET_HOST = '127.0.0.1'
WEBSOCKET_PORT = 5000

# Real-time data intervals
INTRADAY_INTERVAL = '5m'  # 5-minute candles
INTRADAY_PERIOD = '1d'    # 1 day of intraday data

