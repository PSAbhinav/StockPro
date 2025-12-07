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
        logger.info("AdvancedStockFetcher initialized (yfinance)")
    
    def _is_indian_stock(self, ticker: str) -> bool:
        """Check if ticker is an Indian stock."""
        return ticker.endswith('.NS') or ticker.endswith('.BO')
    

    
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
        """Get comprehensive market data for a stock using yfinance."""
        try:
            logger.info(f"Fetching {ticker} from yfinance...")
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
                
                # Data source
                'source': 'yfinance'
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
        """Get latest news - uses multiple sources with proper fallback."""
        formatted_news = []
        
        # Try yfinance first (handles new API format)
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if news and len(news) > 0:
                for article in news[:limit]:
                    # Handle both old and new yfinance formats
                    title = article.get('title') or article.get('content', {}).get('title', 'No title')
                    link = article.get('link') or article.get('content', {}).get('clickThroughUrl', {}).get('url', '')
                    publisher = article.get('publisher') or article.get('content', {}).get('provider', {}).get('displayName', 'Unknown')
                    pub_time = article.get('providerPublishTime') or article.get('content', {}).get('pubDate', 0)
                    
                    if isinstance(pub_time, int) and pub_time > 0:
                        pub_date = datetime.fromtimestamp(pub_time).isoformat()
                    else:
                        pub_date = datetime.now().isoformat()
                    
                    if title and title != 'No title':
                        formatted_news.append({
                            'title': title,
                            'publisher': publisher,
                            'link': link,
                            'published': pub_date,
                            'type': 'news'
                        })
                logger.info(f"Found {len(formatted_news)} news articles from yfinance for {ticker}")
        except Exception as e:
            logger.error(f"yfinance news error for {ticker}: {str(e)}")
        
        # Fallback to Google News RSS if yfinance returns empty or insufficient
        if len(formatted_news) < 3:
            try:
                import urllib.request
                import urllib.parse
                import re
                
                # Get company name for search
                company_name = self.ALL_STOCKS.get(ticker, ticker.replace('.NS', '').replace('.BO', ''))
                if isinstance(company_name, str):
                    search_term = company_name.split()[0]  # Use first word of company name
                else:
                    search_term = ticker.replace('.NS', '').replace('.BO', '')
                
                search_query = urllib.parse.quote(f"{search_term} stock")
                rss_url = f"https://news.google.com/rss/search?q={search_query}&hl=en-IN&gl=IN&ceid=IN:en"
                
                req = urllib.request.Request(rss_url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                with urllib.request.urlopen(req, timeout=10) as response:
                    rss_content = response.read().decode('utf-8')
                
                # Parse RSS manually for better control
                title_pattern = r'<title><!\[CDATA\[(.*?)\]\]></title>|<title>(.*?)</title>'
                link_pattern = r'<link>(.*?)</link>'
                pubdate_pattern = r'<pubDate>(.*?)</pubDate>'
                source_pattern = r'<source[^>]*>(.*?)</source>'
                
                # Find all items
                items = re.findall(r'<item>(.*?)</item>', rss_content, re.DOTALL)
                
                for item in items[:limit - len(formatted_news)]:
                    title_match = re.search(title_pattern, item)
                    link_match = re.search(link_pattern, item)
                    pubdate_match = re.search(pubdate_pattern, item)
                    source_match = re.search(source_pattern, item)
                    
                    title = ''
                    if title_match:
                        title = title_match.group(1) or title_match.group(2) or ''
                        title = title.strip()
                    
                    link = link_match.group(1) if link_match else ''
                    pub_date = pubdate_match.group(1) if pubdate_match else datetime.now().isoformat()
                    source = source_match.group(1) if source_match else 'Google News'
                    
                    if title and len(title) > 5:
                        formatted_news.append({
                            'title': title,
                            'publisher': source,
                            'link': link,
                            'published': pub_date,
                            'type': 'news'
                        })
                
                logger.info(f"Added {len(items)} news from Google News RSS for {ticker}")
            except Exception as e:
                logger.error(f"Google News RSS error for {ticker}: {str(e)}")
        
        # Last resort: Generate placeholder news with market data
        if len(formatted_news) < 2:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                company_name = info.get('longName', ticker)
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                prev_close = info.get('previousClose', current_price)
                change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
                
                # Generate contextual news based on market data
                if change_pct > 2:
                    formatted_news.append({
                        'title': f'{company_name} surges {change_pct:.1f}% in today\'s trading session',
                        'publisher': 'StockPro Analysis',
                        'link': f'https://finance.yahoo.com/quote/{ticker}',
                        'published': datetime.now().isoformat(),
                        'type': 'analysis'
                    })
                elif change_pct < -2:
                    formatted_news.append({
                        'title': f'{company_name} falls {abs(change_pct):.1f}% amid market volatility',
                        'publisher': 'StockPro Analysis',
                        'link': f'https://finance.yahoo.com/quote/{ticker}',
                        'published': datetime.now().isoformat(),
                        'type': 'analysis'
                    })
                else:
                    formatted_news.append({
                        'title': f'{company_name} trades steadily at current levels',
                        'publisher': 'StockPro Analysis',
                        'link': f'https://finance.yahoo.com/quote/{ticker}',
                        'published': datetime.now().isoformat(),
                        'type': 'analysis'
                    })
            except Exception as e:
                logger.error(f"Fallback news generation error: {str(e)}")
        
        return formatted_news[:limit]
    
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
