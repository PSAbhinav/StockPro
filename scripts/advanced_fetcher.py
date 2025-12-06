"""
Advanced Data Fetcher for Stock Market Platform
ENHANCED VERSION - Finds ANY company including all subsidiaries
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedStockFetcher:
    """Advanced stock data fetcher with comprehensive search."""
    
    # Comprehensive stock database for search
    INDIAN_STOCKS = {
        # Large Cap
        'RELIANCE.NS': 'Reliance Industries Ltd',
        'TCS.NS': 'Tata Consultancy Services',
        'HDFCBANK.NS': 'HDFC Bank Ltd',
        'INFY.NS': 'Infosys Ltd',
        'ICICIBANK.NS': 'ICICI Bank Ltd',
        'HINDUNILVR.NS': 'Hindustan Unilever',
        'SBIN.NS': 'State Bank of India',
        'BHARTIARTL.NS': 'Bharti Airtel',
        'ITC.NS': 'ITC Ltd',
        'KOTAKBANK.NS': 'Kotak Mahindra Bank',
        'LT.NS': 'Larsen & Toubro',
        'AXISBANK.NS': 'Axis Bank',
        'ASIANPAINT.NS': 'Asian Paints',
        'MARUTI.NS': 'Maruti Suzuki',
        'TATAMOTORS.NS': 'Tata Motors Ltd',
        'TATASTEEL.NS': 'Tata Steel',
        'WIPRO.NS': 'Wipro Ltd',
        'ULTRACEMCO.NS': 'UltraTech Cement',
        'SUNPHARMA.NS': 'Sun Pharmaceutical',
        'M&M.NS': 'Mahindra & Mahindra',
        'NTPC.NS': 'NTPC Ltd',
        'ONGC.NS': 'Oil & Natural Gas Corp',
        'POWERGRID.NS': 'Power Grid Corp',
        'BAJFINANCE.NS': 'Bajaj Finance',
        'BAJAJFINSV.NS': 'Bajaj Finserv',
        'HCLTECH.NS': 'HCL Technologies',
        'TECHM.NS': 'Tech Mahindra',
        
        # Mid Cap  
        'ADANIPORTS.NS': 'Adani Ports',
        'ADANIENT.NS': 'Adani Enterprises',
        'GODREJCP.NS': 'Godrej Consumer',
        'DIVISLAB.NS': 'Divi\'s Laboratories',
        'DRREDDY.NS': 'Dr Reddy\'s Labs',
        'EICHERMOT.NS': 'Eicher Motors',
        'GRASIM.NS': 'Grasim Industries',
        'JSWSTEEL.NS': 'JSW Steel',
        'VEDL.NS': 'Vedanta Ltd',
        
        # Additional Stocks
        'TATAPOWER.NS': 'Tata Power Company',
        'TATAELXSI.NS': 'Tata Elxsi',
        'TATACHEM.NS': 'Tata Chemicals',
        'TATACOMM.NS': 'Tata Communications',
        'TITAN.NS': 'Titan Company',
        'HDFCLIFE.NS': 'HDFC Life Insurance',
        'HDFCAMC.NS': 'HDFC Asset Management',
        'ICICIGI.NS': 'ICICI Lombard General Insurance',
        'ICICIPRULI.NS': 'ICICI Prudential Life',
        'SBILIFE.NS': 'SBI Life Insurance',
        'INDUSINDBK.NS': 'IndusInd Bank',
        'BANDHANBNK.NS': 'Bandhan Bank',
        'FEDERALBNK.NS': 'Federal Bank',
        'IDFCFIRSTB.NS': 'IDFC First Bank',
        'YESBANK.NS': 'Yes Bank',
        'PNB.NS': 'Punjab National Bank',
        'BANKBARODA.NS': 'Bank of Baroda',
        'CANBK.NS': 'Canara Bank',
        'HINDALCO.NS': 'Hindalco Industries',
        'COALINDIA.NS': 'Coal India',
        'IOC.NS': 'Indian Oil Corporation',
        'BPCL.NS': 'Bharat Petroleum',
        'GAIL.NS': 'GAIL India',
        'HEROMOTOCO.NS': 'Hero MotoCorp',
        'BAJAJ-AUTO.NS': 'Bajaj Auto',
        'TVSMOTOR.NS': 'TVS Motor Company',
        'APOLLOHOSP.NS': 'Apollo Hospitals',
        'CIPLA.NS': 'Cipla',
        'LUPIN.NS': 'Lupin',
        'BIOCON.NS': 'Biocon',
        'ZOMATO.NS': 'Zomato',
        'PAYTM.NS': 'One 97 Communications (Paytm)',
        'NYKAA.NS': 'FSN E-Commerce (Nykaa)',
        'DMART.NS': 'Avenue Supermarts (DMart)',
    }
    
    US_STOCKS = {
        # Tech Giants
        'AAPL': 'Apple Inc',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc',
        'AMZN': 'Amazon.com Inc',
        'META': 'Meta Platforms Inc',
        'TSLA': 'Tesla Inc',
        'NVDA': 'NVIDIA Corporation',
        'NFLX': 'Netflix Inc',
        
        # Other Large Cap
        'JPM': 'JPMorgan Chase',
        'V': 'Visa Inc',
        'JNJ': 'Johnson & Johnson',
        'WMT': 'Walmart Inc',
        'PG': 'Procter & Gamble',
        'MA': 'Mastercard Inc',
        'UNH': 'UnitedHealth Group',
        'HD': 'Home Depot',
        'DIS': 'Walt Disney',
        'BAC': 'Bank of America',
        'INTC': 'Intel Corporation',
        'AMD': 'Advanced Micro Devices',
        
        # Additional US Stocks
        'CRM': 'Salesforce Inc',
        'ORCL': 'Oracle Corporation',
        'ADBE': 'Adobe Inc',
        'PYPL': 'PayPal Holdings',
        'UBER': 'Uber Technologies',
        'ABNB': 'Airbnb Inc',
        'COIN': 'Coinbase Global',
        'SQ': 'Block Inc (Square)',
        'SHOP': 'Shopify Inc',
        'ZM': 'Zoom Video Communications',
        'SNOW': 'Snowflake Inc',
        'PLTR': 'Palantir Technologies',
        'RIVN': 'Rivian Automotive',
        'LCID': 'Lucid Group',
        'F': 'Ford Motor Company',
        'GM': 'General Motors',
        'XOM': 'Exxon Mobil',
        'CVX': 'Chevron Corporation',
        'KO': 'Coca-Cola Company',
        'PEP': 'PepsiCo Inc',
        'MCD': 'McDonalds Corporation',
        'SBUX': 'Starbucks Corporation',
        'NKE': 'Nike Inc',
    }
    
    # Company aliases for better search
    COMPANY_ALIASES = {
        'tata': ['TCS.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TATAPOWER.NS', 'TATAELXSI.NS', 'TATACHEM.NS', 'TATACOMM.NS', 'TITAN.NS'],
        'hdfc': ['HDFCBANK.NS', 'HDFCLIFE.NS', 'HDFCAMC.NS'],
        'icici': ['ICICIBANK.NS', 'ICICIGI.NS', 'ICICIPRULI.NS'],
        'sbi': ['SBIN.NS', 'SBILIFE.NS'],
        'reliance': ['RELIANCE.NS'],
        'infosys': ['INFY.NS'],
        'wipro': ['WIPRO.NS'],
        'adani': ['ADANIPORTS.NS', 'ADANIENT.NS'],
        'bajaj': ['BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BAJAJ-AUTO.NS'],
        'mahindra': ['M&M.NS'],
        'apple': ['AAPL'],
        'amazon': ['AMZN'],
        'google': ['GOOGL'],
        'microsoft': ['MSFT'],
        'meta': ['META'],
        'facebook': ['META'],
        'tesla': ['TSLA'],
        'nvidia': ['NVDA'],
        'netflix': ['NFLX'],
        'bank': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'INDUSINDBK.NS', 'FEDERALBNK.NS', 'YESBANK.NS', 'PNB.NS', 'BANKBARODA.NS', 'CANBK.NS', 'JPM', 'BAC'],
        'auto': ['TATAMOTORS.NS', 'MARUTI.NS', 'M&M.NS', 'HEROMOTOCO.NS', 'BAJAJ-AUTO.NS', 'TVSMOTOR.NS', 'EICHERMOT.NS', 'TSLA', 'F', 'GM', 'RIVN', 'LCID'],
        'pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'LUPIN.NS', 'BIOCON.NS', 'DIVISLAB.NS', 'JNJ'],
        'it': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'TATAELXSI.NS'],
        'tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE'],
        'oil': ['ONGC.NS', 'IOC.NS', 'BPCL.NS', 'GAIL.NS', 'RELIANCE.NS', 'XOM', 'CVX'],
        'steel': ['TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'VEDL.NS'],
        'power': ['TATAPOWER.NS', 'NTPC.NS', 'POWERGRID.NS'],
        'fmcg': ['HINDUNILVR.NS', 'ITC.NS', 'GODREJCP.NS', 'PG', 'KO', 'PEP'],
        'motors': ['TATAMOTORS.NS', 'MARUTI.NS', 'HEROMOTOCO.NS', 'BAJAJ-AUTO.NS', 'TVSMOTOR.NS'],
        'cement': ['ULTRACEMCO.NS', 'GRASIM.NS'],
        'insurance': ['HDFCLIFE.NS', 'SBILIFE.NS', 'ICICIGI.NS', 'ICICIPRULI.NS'],
        'food': ['ZOMATO.NS', 'MCD', 'SBUX', 'KO', 'PEP'],
        'ecommerce': ['AMZN', 'NYKAA.NS', 'DMART.NS'],
    }
    
    def __init__(self, cache_ttl: int = 60):
        """Initialize with caching."""
        self.cache_ttl = cache_ttl
        self.cache = {}
        self.ALL_STOCKS = {**self.INDIAN_STOCKS, **self.US_STOCKS}
    
    def search_stock(self, query: str, limit: int = 15) -> List[Dict]:
        """
        Enhanced search with fuzzy matching and company aliases.
        Finds ANY company including Tata Motors, HDFC, banks, etc.
        """
        query_upper = query.upper().strip()
        query_lower = query.lower().strip()
        results = []
        seen_tickers = set()
        
        # 1. Check company aliases first (e.g., "tata" -> all Tata stocks)
        alias_tickers = []
        for alias, tickers in self.COMPANY_ALIASES.items():
            if alias in query_lower or query_lower in alias:
                alias_tickers.extend(tickers)
        
        # Add stocks from aliases
        for ticker in alias_tickers:
            if ticker not in seen_tickers and ticker in self.ALL_STOCKS:
                name = self.ALL_STOCKS[ticker]
                results.append({
                    'ticker': ticker,
                    'name': name,
                    'exchange': 'NSE' if '.NS' in ticker else 'NYSE/NASDAQ',
                    'type': 'Equity',
                    'currency': 'INR' if '.NS' in ticker or '.BO' in ticker else 'USD'
                })
                seen_tickers.add(ticker)
                if len(results) >= limit:
                    return results
        
        # 2. Try exact ticker match
        for ticker in [query_upper, f'{query_upper}.NS', f'{query_upper}.BO']:
            if ticker not in seen_tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    if info and info.get('symbol'):
                        results.append(self._format_result(ticker, info))
                        seen_tickers.add(ticker)
                        if len(results) >= limit:
                            return results
                except:
                    pass
        
        # 3. Search in our comprehensive database with fuzzy matching
        for ticker, name in self.ALL_STOCKS.items():
            if ticker in seen_tickers:
                continue
            
            # Match by ticker or company name (fuzzy)
            ticker_match = query_upper in ticker
            name_lower = name.lower()
            
            # Fuzzy name matching - check if any query word appears in name
            query_words = query_lower.split()
            name_match = any(word in name_lower for word in query_words) or query_lower in name_lower
            
            if ticker_match or name_match:
                results.append({
                    'ticker': ticker,
                    'name': name,
                    'exchange': 'NSE' if '.NS' in ticker else 'NYSE/NASDAQ',
                    'type': 'Equity',
                    'currency': 'INR' if '.NS' in ticker or '.BO' in ticker else 'USD'
                })
                seen_tickers.add(ticker)
                if len(results) >= limit:
                    return results
        
        # 4. Try common ticker variations for Indian stocks
        if '.' not in query_upper:
            for suffix in ['.NS', '.BO']:
                ticker = query_upper + suffix
                if ticker not in seen_tickers:
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        if info and info.get('symbol'):
                            results.append(self._format_result(ticker, info))
                            seen_tickers.add(ticker)
                            if len(results) >= limit:
                                return results
                    except:
                        pass
        
        return results
    
    def _format_result(self, ticker: str, info: Dict, default_name: str = None) -> Dict:
        """Format search result."""
        return {
            'ticker': ticker,
            'name': info.get('longName') or info.get('shortName') or default_name or ticker,
            'exchange': info.get('exchange', 'NSE' if '.NS' in ticker else 'Unknown'),
            'type': info.get('quoteType', 'Equity'),
            'currency': info.get('currency', 'INR' if '.NS' in ticker or '.BO' in ticker else 'USD')
        }
    
    def get_comprehensive_data(self, ticker: str) -> Optional[Dict]:
        """Get comprehensive market data for a stock."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or not info.get('symbol'):
                return None
            
            # Get current price data
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            previous_close = info.get('previousClose') or info.get('regularMarketPreviousClose')
            
            # Calculate changes
            if current_price and previous_close:
                change = current_price - previous_close
                change_percent = (change / previous_close) * 100
            else:
                change = 0
                change_percent = 0
            
            # Compile comprehensive data
            data = {
                # Basic Price Data
                'ticker': ticker,
                'name': info.get('longName') or info.get('shortName'),
                'current_price': float(current_price) if current_price else None,
                'previous_close': float(previous_close) if previous_close else None,
                'change': float(change),
                'change_percent': float(change_percent),
                'open': info.get('open'),
                'day_high': info.get('dayHigh'),
                'day_low': info.get('dayLow'),
                'volume': info.get('volume'),
                'avg_volume': info.get('averageVolume'),
                
                # 52 Week Data
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                
                # Market Data
                'market_cap': info.get('marketCap'),
                'shares_outstanding': info.get('sharesOutstanding'),
                
                # Valuation Ratios
                'pe_ratio': info.get('trailingPE') or info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                
                # Financial Metrics
                'eps': info.get('trailingEps'),
                'revenue': info.get('totalRevenue'),
                'profit_margin': info.get('profitMargins'),
                'return_on_equity': info.get('returnOnEquity'),
                
                # Dividends
                'dividend_rate': info.get('dividendRate'),
                'dividend_yield': info.get('dividendYield'),
                
                # Trading Data
                'beta': info.get('beta'),
                'bid': info.get('bid'),
                'ask': info.get('ask'),
                
                # Company Info
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'country': info.get('country'),
                'exchange': info.get('exchange'),
                'currency': info.get('currency'),
                'business_summary': info.get('longBusinessSummary'),
                
                # Analyst Data
                'target_mean_price': info.get('targetMeanPrice'),
                'recommendation': info.get('recommendationKey'),
                
                # Timestamp
                'timestamp': datetime.now().isoformat(),
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def get_historical_data(self, ticker: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
        """Get historical OHLCV data."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def get_company_news(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get latest news - FIXED to return 10 articles."""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                logger.info(f"No news found for {ticker}")
                return []
            
            formatted_news = []
            for article in news[:limit]:  # Get up to limit articles
                formatted_news.append({
                    'title': article.get('title', 'No title'),
                    'publisher': article.get('publisher', 'Unknown'),
                    'link': article.get('link', ''),
                    'published': datetime.fromtimestamp(article.get('providerPublishTime', 0)).isoformat() if article.get('providerPublishTime') else datetime.now().isoformat(),
                    'type': article.get('type', 'news')
                })
            
            logger.info(f"Found {len(formatted_news)} news articles for {ticker}")
            return formatted_news
            
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {str(e)}")
            return []
    
    def get_quick_stats(self, ticker: str) -> Optional[Dict]:
        """Get quick stats for dashboard display."""
        data = self.get_comprehensive_data(ticker)
        
        if not data:
            return None
        
        return {
            'ticker': data['ticker'],
            'name': data['name'],
            'price': data['current_price'],
            'change': data['change'],
            'change_percent': data['change_percent'],
            'volume': data['volume'],
            'market_cap': data['market_cap'],
            'pe_ratio': data['pe_ratio'],
            'day_high': data['day_high'],
            'day_low': data['day_low']
        }


if __name__ == '__main__':
    print("=" * 70)
    print("ENHANCED STOCK FETCHER - TEST")
    print("=" * 70)
    
    fetcher = AdvancedStockFetcher()
    
    # Test Tata Motors search
    print("\n1. Testing search for 'Tata Motors'...")
    results = fetcher.search_stock('Tata Motors', limit=5)
    for stock in results:
        print(f"   • {stock['ticker']}: {stock['name']} ({stock['exchange']})")
    
    # Test HDFC search
    print("\n2. Testing search for 'HDFC'...")
    results = fetcher.search_stock('HDFC', limit=5)
    for stock in results:
        print(f"   • {stock['ticker']}: {stock['name']} ({stock['exchange']})")
    
    # Test news
    print("\n3. Testing news for TATAMOTORS.NS...")
    news = fetcher.get_company_news('TATAMOTORS.NS', limit=10)
    print(f"   Found {len(news)} articles")
    for i, article in enumerate(news[:3], 1):
        print(f"   {i}. {article['title'][:60]}...")
    
    print("\n" + "=" * 70)
