"""
Real-Time Data Fetcher for Financial Portfolio Analysis
Fetches live stock prices with caching to avoid API rate limits.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeDataFetcher:
    """Fetches real-time stock data with intelligent caching."""
    
    def __init__(self, cache_ttl: int = 15):
        """
        Initialize the real-time data fetcher.
        
        Args:
            cache_ttl: Time-to-live for cached data in seconds (default: 15)
        """
        self.cache_ttl = cache_ttl
        self.cache = {}
        self.cache_timestamps = {}
        
    def _is_cache_valid(self, ticker: str) -> bool:
        """Check if cached data for ticker is still valid."""
        if ticker not in self.cache_timestamps:
            return False
        
        elapsed = (datetime.now() - self.cache_timestamps[ticker]).total_seconds()
        return elapsed < self.cache_ttl
    
    def get_current_price(self, ticker: str) -> Optional[Dict]:
        """
        Get current price for a single ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            Dictionary containing price information or None on error
        """
        # Check cache first
        if self._is_cache_valid(ticker):
            logger.debug(f"Returning cached data for {ticker}")
            return self.cache[ticker]
        
        try:
            # Fetch real-time data
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get fast info for current price
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            previous_close = info.get('previousClose') or info.get('regularMarketPreviousClose')
            
            if current_price is None:
                # Fallback: use history
                hist = stock.history(period='1d', interval='1m')
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    if previous_close is None:
                        previous_close = hist['Close'].iloc[0]
            
            if current_price is None:
                logger.warning(f"Could not fetch price for {ticker}")
                return None
            
            # Calculate changes
            if previous_close:
                change = current_price - previous_close
                change_percent = (change / previous_close) * 100
            else:
                change = 0
                change_percent = 0
            
            price_data = {
                'ticker': ticker,
                'price': round(float(current_price), 2),
                'previous_close': round(float(previous_close), 2) if previous_close else None,
                'change': round(float(change), 2),
                'change_percent': round(float(change_percent), 2),
                'timestamp': datetime.now().isoformat(),
                'market_cap': info.get('marketCap'),
                'volume': info.get('volume'),
                'day_high': info.get('dayHigh'),
                'day_low': info.get('dayLow'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow')
            }
            
            # Update cache
            self.cache[ticker] = price_data
            self.cache_timestamps[ticker] = datetime.now()
            
            logger.info(f"Fetched {ticker}: ${price_data['price']} ({price_data['change_percent']:+.2f}%)")
            return price_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def get_multiple_prices(self, tickers: List[str]) -> Dict[str, Dict]:
        """
        Get current prices for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Dictionary mapping tickers to their price data
        """
        results = {}
        
        for ticker in tickers:
            price_data = self.get_current_price(ticker)
            if price_data:
                results[ticker] = price_data
            else:
                # Return placeholder data on error
                results[ticker] = {
                    'ticker': ticker,
                    'price': None,
                    'change': None,
                    'change_percent': None,
                    'timestamp': datetime.now().isoformat(),
                    'error': True
                }
        
        return results
    
    def get_intraday_data(self, ticker: str, interval: str = '1m', period: str = '1d') -> pd.DataFrame:
        """
        Get intraday data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            interval: Data interval ('1m', '5m', '15m', '30m', '60m', '90m', '1h')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y')
            
        Returns:
            DataFrame with intraday OHLCV data
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No intraday data available for {ticker}")
                return pd.DataFrame()
            
            logger.info(f"Fetched {len(data)} {interval} candles for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching intraday data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def get_live_metrics(self, tickers: List[str]) -> Dict:
        """
        Calculate live portfolio metrics.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Dictionary containing portfolio metrics
        """
        prices = self.get_multiple_prices(tickers)
        
        # Calculate aggregate metrics
        total_change = sum(p['change'] for p in prices.values() if p.get('change'))
        avg_change_percent = np.mean([p['change_percent'] for p in prices.values() if p.get('change_percent')])
        
        # Count gainers and losers
        gainers = sum(1 for p in prices.values() if p.get('change_percent', 0) > 0)
        losers = sum(1 for p in prices.values() if p.get('change_percent', 0) < 0)
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_stocks': len(tickers),
            'gainers': gainers,
            'losers': losers,
            'neutral': len(tickers) - gainers - losers,
            'avg_change_percent': round(float(avg_change_percent), 2) if avg_change_percent else 0,
            'total_change': round(float(total_change), 2) if total_change else 0,
            'prices': prices
        }
        
        return metrics
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache = {}
        self.cache_timestamps = {}
        logger.info("Cache cleared")


def fetch_stock_data(tickers: List[str], start_date: str, end_date: str, data_path: str) -> pd.DataFrame:
    """
    Fetch historical stock data (backward compatible with existing code).
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        data_path: Path to save the data
        
    Returns:
        DataFrame with historical stock data
    """
    import os
    
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


# Example usage
if __name__ == '__main__':
    # Initialize fetcher
    fetcher = RealtimeDataFetcher(cache_ttl=15)
    
    # Test with example tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
    
    print("=" * 70)
    print("REAL-TIME STOCK PRICE FETCHER - TEST")
    print("=" * 70)
    
    # Get current prices
    print("\nFetching current prices...")
    metrics = fetcher.get_live_metrics(tickers)
    
    print(f"\nPortfolio Summary:")
    print(f"  Gainers: {metrics['gainers']}")
    print(f"  Losers: {metrics['losers']}")
    print(f"  Average Change: {metrics['avg_change_percent']:+.2f}%")
    
    print(f"\nIndividual Stock Prices:")
    for ticker, data in metrics['prices'].items():
        if not data.get('error'):
            print(f"  {ticker}: ${data['price']} ({data['change_percent']:+.2f}%)")
        else:
            print(f"  {ticker}: Error fetching data")
    
    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)
