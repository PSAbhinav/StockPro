"""
NSE Stock Fetcher - Real-time data from NSE India
Provides real-time quotes, historical data, and market information for Indian stocks.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import time
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NSEFetcher:
    """Fetch real-time stock data from NSE India."""
    
    BASE_URL = "https://www.nseindia.com"
    QUOTE_API = f"{BASE_URL}/api/quote-equity"
    HISTORICAL_API = f"{BASE_URL}/api/historical/cm/equity"
    MARKET_DATA_API = f"{BASE_URL}/api/market-data-pre-open"
    
    def __init__(self):
        """Initialize NSE fetcher with session and headers."""
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.nseindia.com/',
            'X-Requested-With': 'XMLHttpRequest'
        }
        self.cache = {}
        self.cache_ttl = 30  # 30 seconds cache
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize session by visiting NSE homepage to get cookies."""
        try:
            self.session.get(self.BASE_URL, headers=self.headers, timeout=5)
            logger.info("NSE session initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize NSE session: {e}")
    
    def _clean_symbol(self, ticker: str) -> str:
        """Convert ticker to NSE symbol format."""
        # Remove .NS or .BO suffix
        symbol = ticker.replace('.NS', '').replace('.BO', '').upper()
        return symbol
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self.cache:
            return False
        data, timestamp = self.cache[key]
        return (datetime.now() - timestamp).total_seconds() < self.cache_ttl
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached data if valid."""
        if self._is_cache_valid(key):
            return self.cache[key][0]
        return None
    
    def _set_cache(self, key: str, data: Dict):
        """Cache data with timestamp."""
        self.cache[key] = (data, datetime.now())
    
    def is_market_open(self) -> bool:
        """Check if Indian market is currently open (9:15 AM - 3:30 PM IST)."""
        now = datetime.now()
        
        # Check if weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Market hours: 9:15 AM to 3:30 PM IST
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def get_quote(self, ticker: str) -> Optional[Dict]:
        """
        Get real-time quote for a stock from NSE.
        
        Args:
            ticker: Stock ticker (e.g., 'RELIANCE.NS' or 'RELIANCE')
        
        Returns:
            Dictionary with stock data or None on failure
        """
        symbol = self._clean_symbol(ticker)
        cache_key = f"quote_{symbol}"
        
        # Check cache
        cached = self._get_cached(cache_key)
        if cached:
            logger.info(f"Returning cached quote for {symbol}")
            return cached
        
        try:
            url = f"{self.QUOTE_API}?symbol={symbol}"
            response = self.session.get(url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"NSE API returned status {response.status_code} for {symbol}")
                return None
            
            data = response.json()
            
            # Extract relevant data
            if 'priceInfo' not in data or 'metadata' not in data:
                logger.error(f"Invalid response structure for {symbol}")
                return None
            
            price_info = data['priceInfo']
            metadata = data['metadata']
            
            result = {
                'ticker': ticker,
                'name': metadata.get('companyName', symbol),
                'current_price': float(price_info.get('lastPrice', 0)),
                'change': float(price_info.get('change', 0)),
                'change_percent': float(price_info.get('pChange', 0)),
                'open': float(price_info.get('open', 0)),
                'high': float(price_info.get('intraDayHighLow', {}).get('max', 0)),
                'low': float(price_info.get('intraDayHighLow', {}).get('min', 0)),
                'previous_close': float(price_info.get('previousClose', 0)),
                'volume': int(data.get('preOpenMarket', {}).get('totalTradedVolume', 0)),
                'day_high': float(price_info.get('intraDayHighLow', {}).get('max', 0)),
                'day_low': float(price_info.get('intraDayHighLow', {}).get('min', 0)),
                'fifty_two_week_high': float(price_info.get('weekHighLow', {}).get('max', 0)),
                'fifty_two_week_low': float(price_info.get('weekHighLow', {}).get('min', 0)),
                'market_cap': metadata.get('marketCap', 0),
                'pe_ratio': metadata.get('pdSectorPe', 0),
                'sector': metadata.get('industry', 'Unknown'),
                'isin': metadata.get('isin', ''),
                'last_update_time': datetime.now().isoformat(),
                'source': 'NSE'
            }
            
            # Cache the result
            self._set_cache(cache_key, result)
            logger.info(f"Successfully fetched quote for {symbol} from NSE")
            
            return result
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching quote for {symbol}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {symbol}: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Data parsing error for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching quote for {symbol}: {e}")
            return None
    
    def get_historical_data(self, ticker: str, period: str = '1mo', interval: str = '1d') -> Optional[Dict]:
        """
        Get historical OHLCV data for a stock.
        Note: NSE provides limited historical data. This is a simplified implementation.
        For comprehensive historical data, yfinance fallback is recommended.
        
        Args:
            ticker: Stock ticker
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y)
            interval: Data interval (1d, 1wk, 1mo)
        
        Returns:
            Dictionary with historical data or None on failure
        """
        symbol = self._clean_symbol(ticker)
        
        # NSE historical API has limitations - recommend using yfinance for historical data
        # This is a placeholder for future enhancement
        logger.info(f"Historical data request for {symbol} - consider using yfinance fallback")
        return None
    
    def get_market_depth(self, ticker: str) -> Optional[Dict]:
        """
        Get market depth (bid/ask) for a stock.
        
        Args:
            ticker: Stock ticker
        
        Returns:
            Dictionary with market depth data or None
        """
        symbol = self._clean_symbol(ticker)
        
        try:
            quote_data = self.get_quote(ticker)
            if not quote_data:
                return None
            
            # Market depth is included in quote response
            # This is a simplified version - full implementation would parse bid/ask levels
            return {
                'symbol': symbol,
                'total_buy_quantity': 0,  # Not available in basic quote
                'total_sell_quantity': 0,
                'bid_price': quote_data.get('current_price', 0),
                'ask_price': quote_data.get('current_price', 0),
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching market depth for {symbol}: {e}")
            return None
    
    def search_stocks(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for stocks by name or symbol.
        
        Args:
            query: Search query
            limit: Maximum results to return
        
        Returns:
            List of matching stocks
        """
        # NSE search API - simplified implementation
        # In production, this would use NSE's search endpoint
        query = query.upper()
        
        # Common NSE stocks for demo purposes
        common_stocks = [
            {'symbol': 'RELIANCE', 'name': 'Reliance Industries Ltd.', 'exchange': 'NSE'},
            {'symbol': 'TCS', 'name': 'Tata Consultancy Services Ltd.', 'exchange': 'NSE'},
            {'symbol': 'INFY', 'name': 'Infosys Ltd.', 'exchange': 'NSE'},
            {'symbol': 'HDFCBANK', 'name': 'HDFC Bank Ltd.', 'exchange': 'NSE'},
            {'symbol': 'ICICIBANK', 'name': 'ICICI Bank Ltd.', 'exchange': 'NSE'},
            {'symbol': 'SBIN', 'name': 'State Bank of India', 'exchange': 'NSE'},
            {'symbol': 'BHARTIARTL', 'name': 'Bharti Airtel Ltd.', 'exchange': 'NSE'},
            {'symbol': 'HINDUNILVR', 'name': 'Hindustan Unilever Ltd.', 'exchange': 'NSE'},
            {'symbol': 'ITC', 'name': 'ITC Ltd.', 'exchange': 'NSE'},
            {'symbol': 'KOTAKBANK', 'name': 'Kotak Mahindra Bank Ltd.', 'exchange': 'NSE'},
        ]
        
        results = [
            {**stock, 'ticker': f"{stock['symbol']}.NS"}
            for stock in common_stocks
            if query in stock['symbol'] or query in stock['name'].upper()
        ]
        
        return results[:limit]
    
    def get_indices(self) -> Optional[Dict]:
        """
        Get current values of major indices.
        
        Returns:
            Dictionary with index values or None
        """
        try:
            # Simplified - would fetch from NSE indices API
            return {
                'NIFTY 50': {'value': 0, 'change': 0, 'change_percent': 0},
                'SENSEX': {'value': 0, 'change': 0, 'change_percent': 0},
                'NIFTY BANK': {'value': 0, 'change': 0, 'change_percent': 0},
            }
        except Exception as e:
            logger.error(f"Error fetching indices: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        logger.info("NSE cache cleared")


# Global instance
nse_fetcher = NSEFetcher()


if __name__ == "__main__":
    # Test the fetcher
    print("Testing NSE Fetcher...")
    
    # Test quote
    print("\n1. Testing get_quote for RELIANCE:")
    quote = nse_fetcher.get_quote("RELIANCE.NS")
    if quote:
        print(f"   ✓ Price: ₹{quote['current_price']}")
        print(f"   ✓ Change: {quote['change_percent']}%")
        print(f"   ✓ Source: {quote['source']}")
    else:
        print("   ✗ Failed to fetch quote")
    
    # Test market status
    print("\n2. Testing market status:")
    is_open = nse_fetcher.is_market_open()
    print(f"   Market is: {'OPEN ✓' if is_open else 'CLOSED ✗'}")
    
    # Test search
    print("\n3. Testing search for 'RELIANCE':")
    results = nse_fetcher.search_stocks("RELIANCE")
    print(f"   Found {len(results)} results")
    
    print("\nNSE Fetcher test complete!")
