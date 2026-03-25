"""
Data Fetcher Module for Financial Portfolio Analysis
Fetches historical and real-time stock data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_stock_data(tickers, start_date, end_date, data_path):
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        data_path: Path to save the data
        
    Returns:
        DataFrame with historical stock data
    """
    logger.info(f"Fetching historical data for {', '.join(tickers)}")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    try:
        # Download historical data
        data = yf.download(tickers, start=start_date, end=end_date, group_by='column')
        
        if data.empty:
            logger.error("No data downloaded!")
            return None
        
        # Save to CSV
        os.makedirs(data_path, exist_ok=True)
        csv_path = os.path.join(data_path, 'raw_stock_data.csv')
        data.to_csv(csv_path)
        
        logger.info(f"Successfully downloaded {len(data)} days of data")
        logger.info(f"Data saved to: {csv_path}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        return None


if __name__ == '__main__':
    # Test the data fetcher
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
    data = fetch_stock_data(
        tickers,
        '2023-01-01',
        '2025-11-30',
        '../data'
    )
    
    if data is not None:
        print("\nData fetched successfully!")
        print(f"Shape: {data.shape}")
        print("\nFirst few rows:")
        print(data.head())