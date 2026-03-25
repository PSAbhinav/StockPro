from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import math
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
from datetime import datetime

load_dotenv()

# Lazy load predictor to avoid import crashes on Vercel
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        from ai_predictor import StockPredictor
        _predictor = StockPredictor()
    return _predictor
import requests

# Fix for yfinance on certain hosting providers
# Set a default user-agent for all requests
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

@app.errorhandler(Exception)
def handle_exception(e):
    """Global error handler to return JSON instead of HTML on error."""
    import traceback
    err_msg = str(e)
    err_type = type(e).__name__
    app.logger.error(f"Unhandled Exception: {err_msg}\n{traceback.format_exc()}")
    return jsonify({
        "status": "error",
        "message": err_msg,
        "type": err_type,
        "trace": traceback.format_exc() if os.environ.get("DEBUG") else None
    }), 500

# Firebase setup (optional)
try:
    import firebase_admin
    from firebase_admin import credentials, db as firebase_db
    cred_path = os.path.join(os.path.dirname(__file__), 'service-account.json')
    if os.path.exists(cred_path) and not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            "databaseURL": os.environ.get("FIREBASE_DB_URL", "https://stockpro-4d381-default-rtdb.firebaseio.com/")
        })
except Exception as e:
    app.logger.warning(f"Firebase not initialized: {e}")

def _flatten_cols(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

def safe_float(val, default=0.0):
    """Convert a value to float, returning default if NaN/Inf."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default

# ── Popular ticker database for name-based search ──
TICKER_DB = {
    # Indian Stocks (NSE)
    "RELIANCE": {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "exchange": "NSE"},
    "TCS": {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "exchange": "NSE"},
    "HDFCBANK": {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Ltd.", "exchange": "NSE"},
    "INFY": {"symbol": "INFY.NS", "name": "Infosys Ltd.", "exchange": "NSE"},
    "ICICIBANK": {"symbol": "ICICIBANK.NS", "name": "ICICI Bank Ltd.", "exchange": "NSE"},
    "SBIN": {"symbol": "SBIN.NS", "name": "State Bank of India", "exchange": "NSE"},
    "HINDUNILVR": {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever", "exchange": "NSE"},
    "ITC": {"symbol": "ITC.NS", "name": "ITC Ltd.", "exchange": "NSE"},
    "BHARTIARTL": {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel", "exchange": "NSE"},
    "KOTAKBANK": {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "exchange": "NSE"},
    "LT": {"symbol": "LT.NS", "name": "Larsen & Toubro", "exchange": "NSE"},
    "WIPRO": {"symbol": "WIPRO.NS", "name": "Wipro Ltd.", "exchange": "NSE"},
    "HCLTECH": {"symbol": "HCLTECH.NS", "name": "HCL Technologies", "exchange": "NSE"},
    "MARUTI": {"symbol": "MARUTI.NS", "name": "Maruti Suzuki India", "exchange": "NSE"},
    "TATAMOTORS": {"symbol": "TATAMOTORS.NS", "name": "Tata Motors", "exchange": "NSE"},
    "TATASTEEL": {"symbol": "TATASTEEL.NS", "name": "Tata Steel", "exchange": "NSE"},
    "ADANIENT": {"symbol": "ADANIENT.NS", "name": "Adani Enterprises", "exchange": "NSE"},
    "BAJFINANCE": {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance", "exchange": "NSE"},
    "SUNPHARMA": {"symbol": "SUNPHARMA.NS", "name": "Sun Pharmaceutical", "exchange": "NSE"},
    "TITAN": {"symbol": "TITAN.NS", "name": "Titan Company", "exchange": "NSE"},
    "AXISBANK": {"symbol": "AXISBANK.NS", "name": "Axis Bank", "exchange": "NSE"},
    "ASIANPAINT": {"symbol": "ASIANPAINT.NS", "name": "Asian Paints", "exchange": "NSE"},
    "ULTRACEMCO": {"symbol": "ULTRACEMCO.NS", "name": "UltraTech Cement", "exchange": "NSE"},
    "POWERGRID": {"symbol": "POWERGRID.NS", "name": "Power Grid Corp", "exchange": "NSE"},
    "NTPC": {"symbol": "NTPC.NS", "name": "NTPC Ltd.", "exchange": "NSE"},
    "ONGC": {"symbol": "ONGC.NS", "name": "Oil & Natural Gas Corp", "exchange": "NSE"},
    "COALINDIA": {"symbol": "COALINDIA.NS", "name": "Coal India", "exchange": "NSE"},
    # Global Stocks (US)
    "GOOGLE": {"symbol": "GOOGL", "name": "Alphabet Inc. (Google)", "exchange": "NASDAQ"},
    "GOOGL": {"symbol": "GOOGL", "name": "Alphabet Inc. (Google)", "exchange": "NASDAQ"},
    "APPLE": {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
    "AAPL": {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
    "MICROSOFT": {"symbol": "MSFT", "name": "Microsoft Corp.", "exchange": "NASDAQ"},
    "MSFT": {"symbol": "MSFT", "name": "Microsoft Corp.", "exchange": "NASDAQ"},
    "AMAZON": {"symbol": "AMZN", "name": "Amazon.com Inc.", "exchange": "NASDAQ"},
    "AMZN": {"symbol": "AMZN", "name": "Amazon.com Inc.", "exchange": "NASDAQ"},
    "META": {"symbol": "META", "name": "Meta Platforms Inc.", "exchange": "NASDAQ"},
    "FACEBOOK": {"symbol": "META", "name": "Meta Platforms Inc.", "exchange": "NASDAQ"},
    "TESLA": {"symbol": "TSLA", "name": "Tesla Inc.", "exchange": "NASDAQ"},
    "TSLA": {"symbol": "TSLA", "name": "Tesla Inc.", "exchange": "NASDAQ"},
    "NVIDIA": {"symbol": "NVDA", "name": "NVIDIA Corp.", "exchange": "NASDAQ"},
    "NVDA": {"symbol": "NVDA", "name": "NVIDIA Corp.", "exchange": "NASDAQ"},
    "NETFLIX": {"symbol": "NFLX", "name": "Netflix Inc.", "exchange": "NASDAQ"},
    "NFLX": {"symbol": "NFLX", "name": "Netflix Inc.", "exchange": "NASDAQ"},
    "AMD": {"symbol": "AMD", "name": "Advanced Micro Devices", "exchange": "NASDAQ"},
    "INTEL": {"symbol": "INTC", "name": "Intel Corp.", "exchange": "NASDAQ"},
    "INTC": {"symbol": "INTC", "name": "Intel Corp.", "exchange": "NASDAQ"},
    "DISNEY": {"symbol": "DIS", "name": "Walt Disney Co.", "exchange": "NYSE"},
    "DIS": {"symbol": "DIS", "name": "Walt Disney Co.", "exchange": "NYSE"},
    "JPMORGAN": {"symbol": "JPM", "name": "JPMorgan Chase", "exchange": "NYSE"},
    "JPM": {"symbol": "JPM", "name": "JPMorgan Chase", "exchange": "NYSE"},
    "VISA": {"symbol": "V", "name": "Visa Inc.", "exchange": "NYSE"},
    "MASTERCARD": {"symbol": "MA", "name": "Mastercard Inc.", "exchange": "NYSE"},
}

def resolve_ticker(raw_ticker):
    """Resolve a user input to a valid yfinance ticker symbol."""
    key = raw_ticker.upper().strip()
    if key in TICKER_DB:
        return TICKER_DB[key]["symbol"], TICKER_DB[key]["name"], TICKER_DB[key]["exchange"]
    # Try direct yfinance lookup
    if '.' in key:
        return key, key, "Unknown"
    # Try as NSE ticker first
    try:
        t = yf.Ticker(key + ".NS")
        info = t.info
        if info and info.get("regularMarketPrice"):
            name = info.get("longName") or info.get("shortName") or key
            return key + ".NS", name, "NSE"
    except:
        pass
    # Try as US ticker
    try:
        t = yf.Ticker(key)
        info = t.info
        if info and info.get("regularMarketPrice"):
            name = info.get("longName") or info.get("shortName") or key
            return key, name, info.get("exchange", "US")
    except:
        pass
    return None, None, None

def get_currency(symbol):
    if '.NS' in symbol or '.BO' in symbol:
        return 'INR'
    return 'USD'


@app.route('/api/python/search-ticker')
def search_ticker():
    """Fuzzy search for tickers by name or symbol."""
    q = request.args.get('q', '').upper().strip()
    if not q or len(q) < 1:
        return jsonify({"results": []})
    
    results = []
    for key, val in TICKER_DB.items():
        if q in key or q in val["name"].upper():
            entry = {"ticker": key, "symbol": val["symbol"], "name": val["name"], "exchange": val["exchange"]}
            if entry not in results:
                results.append(entry)
    
    # Deduplicate by symbol
    seen = set()
    unique = []
    for r in results:
        if r["symbol"] not in seen:
            seen.add(r["symbol"])
            unique.append(r)

    return jsonify({"results": unique[:10]})


@app.route('/api/python/history')
def history():
    """OHLCV historical data for charts."""
    raw_ticker = request.args.get('ticker', 'RELIANCE')
    period = request.args.get('period', '6mo')
    interval = request.args.get('interval', '1d')
    
    symbol, name, exchange = resolve_ticker(raw_ticker)
    if not symbol:
        return jsonify({"status": "error", "message": f"Could not find ticker: {raw_ticker}"}), 404
    
    # Validate period/interval combinations
    valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max']
    valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo']
    
    if period not in valid_periods:
        period = '6mo'
    if interval not in valid_intervals:
        interval = '1d'
    
    # yfinance restrictions: intraday data limited to last 60 days
    if interval in ['1m', '2m', '5m', '15m', '30m'] and period not in ['1d', '5d']:
        period = '5d'
    if interval in ['60m', '90m', '1h'] and period not in ['1d', '5d', '1mo']:
        period = '1mo'
    
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False, session=session)
        data = _flatten_cols(data)
        if data.empty:
            return jsonify({"status": "error", "message": "No data available"}), 404
        
        records = []
        for idx, row in data.iterrows():
            ts = idx
            if hasattr(ts, 'timestamp'):
                time_val = int(ts.timestamp())
                # Add IST offset (5.5 hours) for Indian stocks to show local market time
                if symbol.endswith('.NS') or symbol.endswith('.BO'):
                    time_val += 19800
            else:
                time_val = str(ts)
            
            close_val = safe_float(row['Close'])
            if close_val == 0:
                continue  # skip rows with no data
            record = {
                "time": time_val,
                "open": round(safe_float(row['Open']), 2),
                "high": round(safe_float(row['High']), 2),
                "low": round(safe_float(row['Low']), 2),
                "close": round(close_val, 2),
            }
            if 'Volume' in row:
                record["volume"] = int(safe_float(row['Volume']))
            records.append(record)
        
        return jsonify({
            "status": "success",
            "ticker": raw_ticker.upper(),
            "symbol": symbol,
            "name": name,
            "exchange": exchange,
            "currency": get_currency(symbol),
            "data": records,
            "period": period,
            "interval": interval,
        })
    except Exception as e:
        app.logger.error(f"History error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/python/predict')
def predict():
    raw_ticker = request.args.get('ticker', 'RELIANCE')
    symbol, name, exchange = resolve_ticker(raw_ticker)
    if not symbol:
        return jsonify({"status": "error", "message": f"Could not find ticker: {raw_ticker}. Try the full symbol."}), 404
    
    predictor = get_predictor()
    analysis = predictor.get_comprehensive_analysis(symbol)
    if analysis:
        analysis['ticker'] = raw_ticker.upper().replace('.NS', '').replace('.BO', '')
        analysis['display_name'] = name
        analysis['exchange'] = exchange
        analysis['currency'] = get_currency(symbol)
        analysis['symbol'] = symbol
        return jsonify({"status": "success", "analysis": analysis})
    
    return jsonify({"status": "error", "message": f"Could not analyze {raw_ticker}. Check ticker symbol."}), 500


@app.route('/api/python/news')
def news():
    """Fetch news for a stock ticker."""
    raw_ticker = request.args.get('ticker', 'RELIANCE')
    symbol, name, exchange = resolve_ticker(raw_ticker)
    if not symbol:
        return jsonify({"status": "error", "message": f"Unknown ticker: {raw_ticker}"}), 404
    
    try:
        ticker_obj = yf.Ticker(symbol)
        raw_news = ticker_obj.news if hasattr(ticker_obj, 'news') else []
        
        articles = []
        items = raw_news if isinstance(raw_news, list) else []
        for item in items[:15]:
            # Handle different yfinance news formats
            if isinstance(item, dict):
                article = {
                    "title": item.get("title") or item.get("content", {}).get("title", ""),
                    "publisher": item.get("publisher") or item.get("content", {}).get("provider", {}).get("displayName", ""),
                    "link": item.get("link") or item.get("content", {}).get("canonicalUrl", {}).get("url", ""),
                    "published": item.get("providerPublishTime") or item.get("content", {}).get("pubDate", ""),
                    "thumbnail": "",
                }
                # Try to extract thumbnail
                if "thumbnail" in item and item["thumbnail"]:
                    resolutions = item["thumbnail"].get("resolutions", [])
                    if resolutions:
                        article["thumbnail"] = resolutions[-1].get("url", "")
                elif "content" in item:
                    thumb = item.get("content", {}).get("thumbnail", {})
                    if thumb and "resolutions" in thumb:
                        resolutions = thumb["resolutions"]
                        if resolutions:
                            article["thumbnail"] = resolutions[-1].get("url", "")
                
                if article["title"]:
                    articles.append(article)
        
        return jsonify({
            "status": "success",
            "ticker": raw_ticker.upper(),
            "name": name,
            "articles": articles
        })
    except Exception as e:
        app.logger.error(f"News error: {e}")
        return jsonify({"status": "success", "ticker": raw_ticker.upper(), "articles": []})


@app.route('/api/python/watchlist')
def watchlist():
    """Live prices for watchlist stocks."""
    indian = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'SBIN.NS',
              'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'WIPRO.NS', 'TATAMOTORS.NS']
    global_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA']
    
    names = {
        'RELIANCE.NS': 'Reliance Industries', 'TCS.NS': 'Tata Consultancy Services',
        'HDFCBANK.NS': 'HDFC Bank Ltd.', 'INFY.NS': 'Infosys Ltd.',
        'ICICIBANK.NS': 'ICICI Bank Ltd.', 'SBIN.NS': 'State Bank of India',
        'BHARTIARTL.NS': 'Bharti Airtel', 'ITC.NS': 'ITC Ltd.',
        'KOTAKBANK.NS': 'Kotak Mahindra Bank', 'LT.NS': 'Larsen & Toubro',
        'WIPRO.NS': 'Wipro Ltd.', 'TATAMOTORS.NS': 'Tata Motors',
        'AAPL': 'Apple Inc.', 'GOOGL': 'Alphabet (Google)',
        'MSFT': 'Microsoft Corp.', 'AMZN': 'Amazon.com Inc.',
        'TSLA': 'Tesla Inc.', 'NVDA': 'NVIDIA Corp.',
    }
    
    def fetch_group(tickers, currency):
        results = []
        try:
            data = yf.download(tickers, period='5d', interval='1d', progress=False, group_by='ticker', session=session)
            for t in tickers:
                try:
                    if len(tickers) > 1:
                        stock_data = data[t] if t in data.columns.get_level_values(0) else None
                    else:
                        stock_data = data
                    if stock_data is None or stock_data.empty:
                        continue
                    last = stock_data.tail(1)
                    prev = stock_data.tail(2).head(1)
                    cp = safe_float(last['Close'].values[0])
                    if cp == 0:
                        continue
                    pp = safe_float(prev['Close'].values[0]) if len(prev) > 0 else cp
                    if pp == 0:
                        pp = cp
                    ch = safe_float(((cp - pp) / pp) * 100)
                    results.append({
                        "ticker": t.replace('.NS', '').replace('.BO', ''),
                        "name": names.get(t, t),
                        "price": round(cp, 2),
                        "change": round(ch, 2),
                        "currency": currency,
                        "exchange": "NSE" if '.NS' in t else "US",
                    })
                except:
                    continue
        except:
            pass
        return results
    
    indian_data = fetch_group(indian, "INR")
    global_data = fetch_group(global_stocks, "USD")
    
    return jsonify({
        "status": "success",
        "indian": indian_data,
        "global": global_data,
    })


if __name__ == "__main__":
    app.run(port=8000, host='0.0.0.0', debug=True)
