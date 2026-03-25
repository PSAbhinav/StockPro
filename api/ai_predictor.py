import numpy as np
import pandas as pd
from datetime import datetime
import logging
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


class StockPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def _flatten_columns(self, df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df

    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
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

    def _calculate_atr(self, df, period=14):
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _get_support_resistance(self, df, window=20):
        recent = df.tail(window)
        return float(recent['Low'].min()), float(recent['High'].max())

    def _get_sentiment_score(self, rsi, macd_val, price, sma20):
        score = 50
        if rsi < 30: score += 20
        elif rsi > 70: score -= 20
        elif rsi < 50: score += 5
        else: score -= 5

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
        bb_upper, bb_mid, bb_lower = self._calculate_bollinger(df['Close'])
        df['BB_Upper'] = bb_upper
        df['BB_Lower'] = bb_lower
        if 'Volume' in df.columns:
            df['Vol_SMA'] = df['Volume'].rolling(window=20).mean()

        df = df.dropna()
        features = ['Close', 'RSI', 'MACD', 'SMA_20', 'EMA_10', 'SMA_50']
        available = [f for f in features if f in df.columns]
        X = df[available].values
        y = df['Close'].shift(-1).dropna().values
        X = X[:len(y)]
        return X, y, available

    def train(self, ticker):
        try:
            data = yf.download(ticker, period='5y', interval='1d', progress=False)
            data = self._flatten_columns(data)
            if data.empty or len(data) < 50:
                return False
            X, y, features = self.prepare_data(data)
            if len(X) == 0:
                return False
            self.model.fit(X, y)
            self.used_features = features
            return True
        except Exception as e:
            logger.error(f"Training error for {ticker}: {e}")
            return False

    def get_comprehensive_analysis(self, ticker):
        if not self.train(ticker):
            return None

        data = yf.download(ticker, period='6mo', interval='1d', progress=False)
        data = self._flatten_columns(data)
        if data.empty:
            return None

        data['RSI'] = self._calculate_rsi(data['Close'])
        macd, macd_signal = self._calculate_macd(data['Close'])
        data['MACD'] = macd
        data['MACD_Signal'] = macd_signal
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['EMA_10'] = data['Close'].ewm(span=10).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        bb_upper, bb_mid, bb_lower = self._calculate_bollinger(data['Close'])
        data['BB_Upper'] = bb_upper
        data['BB_Lower'] = bb_lower

        current_data = data.dropna().tail(1)
        if current_data.empty:
            return None

        current_features = current_data[self.used_features].values
        prediction = self.model.predict(current_features)[0]
        current_price = float(current_data['Close'].values[0])
        change_pct = ((prediction - current_price) / current_price) * 100

        rsi_val = float(current_data['RSI'].values[0])
        macd_val = float(current_data['MACD'].values[0])
        macd_sig = float(current_data['MACD_Signal'].values[0])
        sma20 = float(current_data['SMA_20'].values[0])
        sma50 = float(current_data['SMA_50'].values[0])
        bb_u = float(current_data['BB_Upper'].values[0])
        bb_l = float(current_data['BB_Lower'].values[0])

        support, resistance = self._get_support_resistance(data)
        sentiment = self._get_sentiment_score(rsi_val, macd_val, current_price, sma20)

        # Multi-timeframe signals
        short_signal = "BUY" if rsi_val < 40 and macd_val > macd_sig else "SELL" if rsi_val > 60 and macd_val < macd_sig else "HOLD"
        medium_signal = "BUY" if current_price > sma20 and macd_val > 0 else "SELL" if current_price < sma20 and macd_val < 0 else "HOLD"
        long_signal = "BUY" if current_price > sma50 and sma20 > sma50 else "SELL" if current_price < sma50 and sma20 < sma50 else "HOLD"

        # Get stock info for key stats
        try:
            info = yf.Ticker(ticker).info
            key_stats = {
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "week52_high": info.get("fiftyTwoWeekHigh"),
                "week52_low": info.get("fiftyTwoWeekLow"),
                "avg_volume": info.get("averageVolume"),
                "dividend_yield": info.get("dividendYield"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "name": info.get("longName") or info.get("shortName") or ticker,
            }
        except:
            key_stats = {}

        return {
            "ticker": ticker,
            "current_price": current_price,
            "predicted_price": float(prediction),
            "expected_change": float(change_pct),
            "recommendation": "BUY" if change_pct > 1.0 else "SELL" if change_pct < -1.0 else "HOLD",
            "sentiment_score": sentiment,
            "signals": {
                "short_term": short_signal,
                "medium_term": medium_signal,
                "long_term": long_signal,
            },
            "model_info": {
                "type": "Random Forest Adaptive (Ensemble)",
                "features": self.used_features,
                "training_period": "5 Years",
                "n_estimators": 100,
            },
            "technical_indicators": {
                "rsi": round(rsi_val, 2),
                "macd": round(macd_val, 2),
                "macd_signal": round(macd_sig, 2),
                "sma_20": round(sma20, 2),
                "sma_50": round(sma50, 2),
                "bollinger_upper": round(bb_u, 2),
                "bollinger_lower": round(bb_l, 2),
                "support": round(support, 2),
                "resistance": round(resistance, 2),
            },
            "key_stats": key_stats,
            "timestamp": datetime.now().isoformat()
        }
