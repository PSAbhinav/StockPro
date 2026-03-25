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

# --- Inlined AI Predictor ---
class StockPredictor:
    def __init__(self):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.used_features = ['Close', 'RSI', 'MACD', 'SMA_20', 'EMA_10', 'SMA_50']

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

    def _calculate_macd(self, series):
        ema12 = series.ewm(span=12).mean()
        ema26 = series.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd, signal

    def _calculate_bollinger(self, series, period=20):
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, sma, lower

    def _get_support_resistance(self, df, window=20):
        recent = df.tail(window)
        return float(recent['Low'].min()), float(recent['High'].max())

    def _get_sentiment_score(self, rsi, macd_val, price, sma20):
        score = 50
        if rsi < 30: score += 20
        elif rsi > 70: score -= 20
        if macd_val > 0: score += 15
        else: score -= 15
        if price > sma20: score += 10
        else: score -= 10
        return max(0, min(100, score))

    def prepare_data(self, df):
        df = self._flatten_columns(df.copy())
        df['RSI'] = self._calculate_rsi(df['Close'])
        macd, signal = self._calculate_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_10'] = df['Close'].ewm(span=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df = df.dropna()
        features = self.used_features
        available = [f for f in features if f in df.columns]
        X = df[available].values
        y = df['Close'].shift(-1).dropna().values
        X = X[:len(y)]
        return X, y, available

    def train(self, ticker):
        yf = get_yf()
        try:
            data = yf.download(ticker, period='5y', interval='1d', progress=False, session=get_session())
            data = self._flatten_columns(data)
            if data.empty or len(data) < 50: return False
            X, y, features = self.prepare_data(data)
            if len(X) == 0: return False
            self.model.fit(X, y)
            self.used_features = features
            return True
        except Exception as e:
            return False

    def get_comprehensive_analysis(self, ticker):
        yf = get_yf()
        if not self.train(ticker): return None
        try:
            data = yf.download(ticker, period='6mo', interval='1d', progress=False, session=get_session())
            data = self._flatten_columns(data)
            if data.empty: return None
            df = data.copy()
            df['RSI'] = self._calculate_rsi(df['Close'])
            macd, macd_sig = self._calculate_macd(df['Close'])
            df['MACD'] = macd
            df['MACD_Signal'] = macd_sig
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['EMA_10'] = df['Close'].ewm(span=10).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            bb_u, bb_m, bb_l = self._calculate_bollinger(df['Close'])
            df['BB_Upper'] = bb_u
            df['BB_Lower'] = bb_l
            curr_data = df.dropna().tail(1)
            if curr_data.empty: return None
            
            features = curr_data[self.used_features].values
            prediction = self.model.predict(features)[0]
            curr_price = float(curr_data['Close'].values[0])
            change_pct = ((prediction - curr_price) / curr_price) * 100
            
            rsi_val = float(curr_data['RSI'].values[0])
            macd_val = float(curr_data['MACD'].values[0])
            sma20 = float(curr_data['SMA_20'].values[0])
            
            support, resistance = self._get_support_resistance(df)
            sentiment = self._get_sentiment_score(rsi_val, macd_val, curr_price, sma20)
            
            info = yf.Ticker(ticker, session=get_session()).info
            return {
                "ticker": ticker,
                "current_price": curr_price,
                "predicted_price": float(prediction),
                "expected_change": float(change_pct),
                "recommendation": "BUY" if change_pct > 1.0 else "SELL" if change_pct < -1.0 else "HOLD",
                "sentiment_score": sentiment,
                "signals": {
                    "short_term": "BUY" if rsi_val < 40 else "SELL" if rsi_val > 60 else "HOLD",
                    "medium_term": "BUY" if curr_price > sma20 else "SELL",
                    "long_term": "HOLD"
                },
                "technical_indicators": {
                    "rsi": round(rsi_val, 2),
                    "macd": round(macd_val, 2),
                    "support": round(support, 2),
                    "resistance": round(resistance, 2)
                },
                "key_stats": {
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "name": info.get("longName") or ticker
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return None

# --- Flask App ---
app = Flask(__name__)
CORS(app)

_predictor = None
def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = StockPredictor()
    return _predictor

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")
    return jsonify({"status": "error", "message": str(e), "type": type(e).__name__}), 500

def _flatten_cols(df):
    if isinstance(df.columns, (list, tuple)) or hasattr(df.columns, 'levels'):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

def resolve_ticker(symbol):
    if not symbol: return None, None, None
    s = symbol.upper()
    if s.endswith(('.NS', '.BO')): return s, s, "NSE/BSE"
    yf = get_yf()
    try:
        t = yf.Ticker(s, session=get_session())
        if t.info and 'symbol' in t.info: return t.info['symbol'], t.info.get('shortName', s), "US"
    except: pass
    try:
        t_ns = yf.Ticker(s + ".NS", session=get_session())
        if t_ns.info and 'symbol' in t_ns.info: return s + ".NS", t_ns.info.get('shortName', s), "NSE"
    except: pass
    return s, s, "Unknown"

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
                last = hist.tail(2)
                if len(last) < 2: continue
                prev, curr = last['Close'].values[-2], last['Close'].values[-1]
                change = curr - prev
                pct = (change / prev) * 100
                res.append({"symbol": t.replace(".NS", ""), "price": round(curr, 2), "change": round(change, 2), "change_percent": round(pct, 2)})
            except: continue
        return res

    return jsonify({"status": "success", "indian": fetch_group(indian), "global": fetch_group(glob)})

@app.route('/api/python/stock-data')
def get_stock_data():
    ticker = request.args.get('ticker', 'AAPL')
    period = request.args.get('period', '1mo')
    interval = request.args.get('interval', '1d')
    symbol, _, _ = resolve_ticker(ticker)
    yf = get_yf()
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, session=get_session())
        df = _flatten_cols(df).dropna()
        chart_data = []
        for idx, row in df.iterrows():
            chart_data.append({
                "time": int(idx.timestamp()),
                "open": float(row['Open']), "high": float(row['High']),
                "low": float(row['Low']), "close": float(row['Close']),
                "volume": float(row['Volume'])
            })
        return jsonify({"status": "success", "symbol": symbol, "data": chart_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/python/predict')
def predict():
    ticker = request.args.get('ticker', 'AAPL')
    symbol, _, _ = resolve_ticker(ticker)
    analysis = get_predictor().get_comprehensive_analysis(symbol)
    if analysis: return jsonify({"status": "success", "data": analysis})
    return jsonify({"status": "error", "message": "Analysis failed"}), 500

@app.route('/api/python/news')
def get_news():
    ticker = request.args.get('ticker', 'AAPL')
    symbol, _, _ = resolve_ticker(ticker)
    yf = get_yf()
    try:
        t = yf.Ticker(symbol, session=get_session())
        news = []
        for item in t.news[:5]:
            news.append({"title": item['title'], "publisher": item['publisher'], "link": item['link'], "type": item['type']})
        return jsonify({"status": "success", "news": news})
    except:
        return jsonify({"status": "success", "news": []})

@app.route('/api/python/search-ticker')
def search():
    query = request.args.get('query', '').upper()
    if not query: return jsonify([])
    symbol, name, _ = resolve_ticker(query)
    return jsonify([{"symbol": symbol.replace(".NS", ""), "name": name, "full_symbol": symbol}])

if __name__ == '__main__':
    app.run(port=8000)
