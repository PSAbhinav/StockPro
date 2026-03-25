from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import math
from dotenv import load_dotenv
from datetime import datetime
import traceback

load_dotenv()

# --- Lazy Loaders & Config ---
_session = None
def get_session():
    global _session
    if _session is None:
        import requests
        _session = requests.Session()
        _session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    return _session

def get_yf():
    import yfinance as yf
    return yf

def get_pd():
    import pandas as pd
    return pd

# --- Minimal Technical Analysis (Replaces ML) ---
class TechnicalAnalyzer:
    def _flatten_columns(self, df):
        import pandas as pd
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df

    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def get_analysis(self, ticker):
        yf = get_yf()
        try:
            data = yf.download(ticker, period='6mo', interval='1d', progress=False, session=get_session())
            data = self._flatten_columns(data)
            if data.empty: return None
            
            curr = data.tail(1)
            price = float(curr['Close'].values[0])
            rsi = float(self._calculate_rsi(data['Close']).tail(1).values[0])
            
            # Simple Logic
            recommendation = "BUY" if rsi < 40 else "SELL" if rsi > 60 else "HOLD"
            sentiment = 70 if rsi < 30 else 30 if rsi > 70 else 50
            
            info = yf.Ticker(ticker, session=get_session()).info
            return {
                "ticker": ticker,
                "current_price": price,
                "predicted_price": round(price * (1.02 if recommendation == "BUY" else 0.98), 2),
                "expected_change": 2.0 if recommendation == "BUY" else -2.0,
                "recommendation": recommendation,
                "sentiment_score": sentiment,
                "signals": {"short_term": recommendation, "medium_term": "HOLD", "long_term": "BUY"},
                "technical_indicators": {"rsi": round(rsi, 2)},
                "key_stats": {"name": info.get("longName") or ticker, "market_cap": info.get("marketCap")},
                "timestamp": datetime.now().isoformat()
            }
        except: return None

# --- Flask App ---
app = Flask(__name__)
CORS(app)

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")
    return jsonify({"status": "error", "message": "Backend engine crash. Check Vercel logs."}), 500

def _flatten_cols(df):
    if hasattr(df.columns, 'levels'):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

@app.route('/api/python/health')
def health():
    return jsonify({"status": "ok", "message": "Python API is live (Minimal Edition)"})

@app.route('/api/python/watchlist')
def get_watchlist():
    yf = get_yf()
    indian = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
    glob = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    
    def fetch_group(tickers):
        data = yf.download(tickers, period='5d', interval='1d', progress=False, group_by='ticker', session=get_session())
        res = []
        for t in tickers:
            try:
                hist = data[t] if len(tickers) > 1 else data
                hist = _flatten_cols(hist).dropna()
                if hist.empty: continue
                curr = hist['Close'].values[-1]
                prev = hist['Close'].values[-2]
                res.append({"symbol": t.replace(".NS", ""), "price": round(curr, 2), "change": round(curr-prev, 2), "change_percent": round((curr-prev)/prev*100, 2)})
            except: continue
        return res
    return jsonify({"status": "success", "indian": fetch_group(indian), "global": fetch_group(glob)})

@app.route('/api/python/stock-data')
def get_stock_data():
    ticker = request.args.get('ticker', 'AAPL')
    period = request.args.get('period', '1mo')
    interval = request.args.get('interval', '1d')
    if not ticker.endswith(('.NS', '.BO')) and len(ticker) < 6:
        ticker = ticker # Assume US or already complex
    yf = get_yf()
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, session=get_session())
        df = _flatten_cols(df).dropna()
        chart_data = [{"time": int(i.timestamp()), "open": float(r['Open']), "high": float(r['High']), "low": float(r['Low']), "close": float(r['Close']), "volume": float(r['Volume'])} for i, r in df.iterrows()]
        return jsonify({"status": "success", "symbol": ticker, "data": chart_data})
    except: return jsonify({"status": "error", "message": "Fetch error"}), 500

@app.route('/api/python/predict')
def predict():
    ticker = request.args.get('ticker', 'AAPL')
    analysis = TechnicalAnalyzer().get_analysis(ticker)
    if analysis: return jsonify({"status": "success", "data": analysis})
    return jsonify({"status": "error", "message": "Analysis failed"}), 500

@app.route('/api/python/news')
def get_news():
    ticker = request.args.get('ticker', 'AAPL')
    yf = get_yf()
    try:
        news = [{"title": i['title'], "publisher": i['publisher'], "link": i['link']} for i in yf.Ticker(ticker, session=get_session()).news[:5]]
        return jsonify({"status": "success", "news": news})
    except: return jsonify({"status": "success", "news": []})

@app.route('/api/python/search-ticker')
def search():
    query = request.args.get('query', '').upper()
    if not query: return jsonify([])
    return jsonify([{"symbol": query, "name": query, "full_symbol": query}])

if __name__ == '__main__':
    app.run(port=8000)
