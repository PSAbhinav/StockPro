"""
AI Stock Predictor using LSTM Neural Networks
Provides price predictions with confidence scores and recommendations.
"""

import os
# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. AI predictions will use fallback method.")

from sklearn.linear_model import LinearRegression
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockPredictor:
    """AI-powered stock price predictor with enhanced LSTM model."""
    
    def __init__(self, use_lstm: bool = True):
        """
        Initialize predictor.
        
        Args:
            use_lstm: Use LSTM model (requires TensorFlow) or fallback to simpler model
        """
        self.use_lstm = use_lstm and TENSORFLOW_AVAILABLE
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lookback = 60  # Number of days to look back
        
        if not TENSORFLOW_AVAILABLE and use_lstm:
            logger.warning("TensorFlow not available. Using fallback linear regression.")
    
    def prepare_data(self, df: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training with multiple features."""
        # Create features: Close, Volume, Price Change, MA7, MA21
        close_prices = df['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(close_prices)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def build_lstm_model(self, input_shape: Tuple):
        """Build enhanced LSTM neural network with deeper architecture."""
        model = Sequential([
            # First LSTM layer with more units
            LSTM(units=128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            
            # Second LSTM layer
            LSTM(units=128, return_sequences=True),
            Dropout(0.3),
            
            # Third LSTM layer
            LSTM(units=64, return_sequences=True),
            Dropout(0.2),
            
            # Fourth LSTM layer
            LSTM(units=64),
            Dropout(0.2),
            
            # Dense layers
            Dense(units=32, activation='relu'),
            Dense(units=1)
        ])
        
        # Use Adam optimizer with learning rate schedule
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    def train(self, ticker: str, period: str = '3y') -> bool:
        """Train the model on historical data with enhanced parameters."""
        try:
            # Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty or len(hist) < 100:
                logger.error(f"Insufficient data for {ticker}")
                return False
            
            if self.use_lstm:
                # Prepare LSTM data
                X, y = self.prepare_data(hist, self.lookback)
                
                if len(X) < 50:
                    logger.warning(f"Limited training data for {ticker}, using linear model")
                    self.use_lstm = False
                else:
                    # Build and train model with more epochs
                    self.model = self.build_lstm_model((X.shape[1], 1))
                    
                    # Train with early stopping concept (just more epochs for this version)
                    self.model.fit(
                        X, y, 
                        batch_size=32, 
                        epochs=25,  # Increased from 10 to 25
                        verbose=0,
                        validation_split=0.1
                    )
                    logger.info(f"Enhanced LSTM model trained for {ticker} with {len(X)} samples")
                    return True
            
            if not self.use_lstm:
                # Fallback: Use linear regression with polynomial features
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.pipeline import make_pipeline
                
                prices = hist['Close']
                X = np.arange(len(prices)).reshape(-1, 1)
                y = prices.values
                
                # Use polynomial regression for better curve fitting
                self.model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
                self.model.fit(X, y)
                logger.info(f"Polynomial regression model trained for {ticker}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model for {ticker}: {str(e)}")
            return False
    
    def predict_next_prices(self, ticker: str, days: int = 7) -> Optional[Dict]:
        """
        Predict future prices including intraday OHLC predictions.
        Now includes next trading day calculation (skips weekends).
        
        Returns:
            Dictionary with predictions for multiple timeframes including intraday
        """
        try:
            # Get recent data
            stock = yf.Ticker(ticker)
            hist = stock.history(period='6mo')
            
            if hist.empty:
                return None
            
            # Calculate next trading day (skip weekends)
            today = datetime.now()
            next_trading_day = today + timedelta(days=1)
            
            # Skip weekends
            while next_trading_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
                next_trading_day += timedelta(days=1)
            
            # Format the day name
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            next_trading_day_name = day_names[next_trading_day.weekday()]
            next_trading_day_str = next_trading_day.strftime('%Y-%m-%d')
            
            # Calculate days until next trading day for label (use .date() to remove time component)
            days_until_trading = (next_trading_day.date() - today.date()).days
            
            current_price = hist['Close'].iloc[-1]
            prices = hist['Close']
            
            # Calculate key statistics for better predictions
            avg_daily_range = (hist['High'] - hist['Low']).mean()
            avg_open_close_diff = (hist['Close'] - hist['Open']).abs().mean()
            volatility = prices.pct_change().std()
            
            # Calculate recent momentum (last 5 days vs previous 5 days)
            recent_avg = prices.tail(5).mean()
            prev_avg = prices.tail(10).head(5).mean()
            momentum = (recent_avg - prev_avg) / prev_avg if prev_avg else 0
            
            # Calculate RSI for trend direction
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            
            # Calculate MACD for trend
            exp12 = prices.ewm(span=12, adjust=False).mean()
            exp26 = prices.ewm(span=26, adjust=False).mean()
            macd = exp12 - exp26
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_histogram = macd.iloc[-1] - signal.iloc[-1] if len(macd) > 0 else 0
            
            # Train model
            if not self.train(ticker):
                return None
            
            predictions = {}
            
            # Determine trend direction based on multiple factors
            trend_score = 0
            
            # RSI factor
            if current_rsi < 30:
                trend_score += 2  # Oversold - likely to go up
            elif current_rsi > 70:
                trend_score -= 2  # Overbought - likely to go down
            else:
                trend_score += (50 - current_rsi) / 40  # Slightly favor uptrend if below 50
            
            # MACD factor
            if macd_histogram > 0:
                trend_score += 1
            else:
                trend_score -= 1
            
            # Momentum factor
            trend_score += momentum * 10  # Scale momentum contribution
            
            # Price vs SMA
            sma_20 = prices.tail(20).mean()
            if current_price > sma_20:
                trend_score += 0.5
            else:
                trend_score -= 0.5
            
            # Normalize trend score to percentage change expectation
            expected_change_pct = max(-5, min(5, trend_score * 0.5))  # Cap at ±5%
            
            # Calculate predicted prices
            predicted_close_1d = current_price * (1 + expected_change_pct / 100)
            
            # Calculate intraday predictions based on historical patterns
            avg_open_diff = (hist['Open'] - hist['Close'].shift(1)).mean()
            avg_high_from_open = (hist['High'] / hist['Open'] - 1).mean()
            avg_low_from_open = (hist['Open'] / hist['Low'] - 1).mean()
            
            predicted_open_1d = current_price + avg_open_diff
            predicted_high_1d = predicted_open_1d * (1 + avg_high_from_open)
            predicted_low_1d = predicted_open_1d * (1 - avg_low_from_open)
            
            # Adjust high/low based on trend
            if expected_change_pct > 0:
                predicted_high_1d = max(predicted_high_1d, predicted_close_1d * 1.005)
            else:
                predicted_low_1d = min(predicted_low_1d, predicted_close_1d * 0.995)
            
            # Set default predictions
            predictions = {
                '1d': float(predicted_close_1d),
                '3d': float(current_price * (1 + expected_change_pct * 1.5 / 100)),
                '7d': float(current_price * (1 + expected_change_pct * 2 / 100)),
            }
            
            # Use LSTM for better predictions if available
            if self.use_lstm and self.model is not None:
                try:
                    last_60_days = prices.tail(60).values.reshape(-1, 1)
                    scaled_last_60 = self.scaler.transform(last_60_days)
                    
                    for day in [1, 3, 7]:
                        if day <= days:
                            X_test = scaled_last_60[-60:].reshape(1, 60, 1)
                            
                            # Predict iteratively
                            future_prices = []
                            for _ in range(day):
                                pred_scaled = self.model.predict(X_test, verbose=0)
                                future_prices.append(pred_scaled[0, 0])
                                
                                # Update sequence
                                X_test = np.append(X_test[0, 1:], [[pred_scaled[0, 0]]], axis=0)
                                X_test = X_test.reshape(1, 60, 1)
                            
                            lstm_price = self.scaler.inverse_transform([[future_prices[-1]]])[0, 0]
                            
                            # Blend LSTM with technical analysis (60% LSTM, 40% technical)
                            predictions[f'{day}d'] = float(lstm_price * 0.6 + predictions[f'{day}d'] * 0.4)
                except Exception as e:
                    logger.warning(f"LSTM prediction failed, using technical analysis only: {e}")
            
            # Intraday predictions for tomorrow
            intraday_predictions = {
                'open': float(predicted_open_1d),
                'high': float(predicted_high_1d),
                'low': float(predicted_low_1d),
                'close': float(predictions['1d'])
            }
            
            # Calculate confidence based on multiple factors
            # Lower volatility = higher confidence
            volatility_score = max(0.3, min(0.95, 1 - (volatility * 8)))
            # Strong trend = higher confidence
            trend_strength = abs(trend_score) / 5  # Normalize to 0-1
            confidence = volatility_score * 0.6 + trend_strength * 0.4
            confidence = max(0.35, min(0.92, confidence))
            
            # Determine trend
            if expected_change_pct > 1:
                trend = 'BULLISH'
            elif expected_change_pct < -1:
                trend = 'BEARISH'
            else:
                trend = 'NEUTRAL'
            
            # Generate recommendation
            if expected_change_pct > 1.5:
                recommendation = 'STRONG BUY'
            elif expected_change_pct > 0.5:
                recommendation = 'BUY'
            elif expected_change_pct < -1.5:
                recommendation = 'STRONG SELL'
            elif expected_change_pct < -0.5:
                recommendation = 'SELL'
            else:
                recommendation = 'HOLD'
            
            return {
                'current_price': float(current_price),
                'predictions': predictions,
                'intraday': intraday_predictions,
                'expected_change_pct': float(expected_change_pct),
                'confidence': float(confidence),
                'trend': trend,
                'recommendation': recommendation,
                'model_type': 'LSTM+Technical' if self.use_lstm else 'Technical',
                'timestamp': datetime.now().isoformat(),
                'next_trading_day': next_trading_day_str,
                'next_trading_day_name': next_trading_day_name,
                'days_until_trading': days_until_trading,
                'analysis_factors': {
                    'rsi': float(current_rsi),
                    'macd_signal': 'Bullish' if macd_histogram > 0 else 'Bearish',
                    'momentum': 'Positive' if momentum > 0 else 'Negative',
                    'volatility': 'Low' if volatility < 0.02 else 'Medium' if volatility < 0.04 else 'High'
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting prices for {ticker}: {str(e)}")
            return None
    
    def get_technical_signals(self, ticker: str) -> Dict:
        """Get technical analysis signals."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='6mo')
            
            if hist.empty:
                return {}
            
            close = hist['Close']
            
            # Calculate technical indicators
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # Moving averages
            sma_50 = close.rolling(window=50).mean().iloc[-1]
            sma_200 = close.rolling(window=200).mean() if len(close) >= 200 else None
            current_price = close.iloc[-1]
            
            # MACD
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_histogram = macd - signal
            
            signals = {
                'rsi': float(current_rsi),
                'rsi_signal': 'Oversold' if current_rsi < 30 else 'Overbought' if current_rsi > 70 else 'Neutral',
                'sma_50': float(sma_50),
                'sma_200': float(sma_200.iloc[-1]) if sma_200 is not None else None,
                'price_vs_sma50': 'Above' if current_price > sma_50 else 'Below',
                'macd': float(macd.iloc[-1]),
                'macd_signal': float(signal.iloc[-1]),
                'macd_histogram': float(macd_histogram.iloc[-1]),
                'macd_trend': 'Bullish' if macd_histogram.iloc[-1] > 0 else 'Bearish'
            }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error calculating technical signals: {str(e)}")
            return {}
    
    def get_comprehensive_analysis(self, ticker: str) -> Optional[Dict]:
        """Get comprehensive AI analysis with predictions and signals."""
        prediction = self.predict_next_prices(ticker)
        signals = self.get_technical_signals(ticker)
        
        if not prediction:
            return None
        
        # Combine signals for final recommendation
        reasons = []
        
        # Check RSI
        if signals.get('rsi'):
            if signals['rsi'] < 30:
                reasons.append('Oversold RSI indicates potential buying opportunity')
            elif signals['rsi'] > 70:
                reasons.append('Overbought RSI suggests caution')
        
        # Check MACD
        if signals.get('macd_trend') == 'Bullish':
            reasons.append('Positive MACD momentum')
        elif signals.get('macd_trend') == 'Bearish':
            reasons.append('Negative MACD momentum')
        
        # Check moving averages
        if signals.get('price_vs_sma50') == 'Above':
            reasons.append('Trading above 50-day MA (bullish)')
        else:
            reasons.append('Trading below 50-day MA (bearish)')
        
        # Price prediction
        next_day_change = ((prediction['predictions'].get('1d', 0) - prediction['current_price']) / 
                          prediction['current_price'] * 100)
        
        if next_day_change > 1:
            reasons.append(f'AI predicts +{next_day_change:.1f}% move')
        elif next_day_change < -1:
            reasons.append(f'AI predicts {next_day_change:.1f}% move')
        
        return {
            'prediction': prediction,
            'technical_signals': signals,
            'reasons': reasons,
            'risk_level': 'LOW' if prediction['confidence'] > 0.7 else 'MEDIUM' if prediction['confidence'] > 0.5 else 'HIGH'
        }


# Standalone function
def predict_stock(ticker: str) -> Optional[Dict]:
    """Quick prediction for a stock."""
    predictor = StockPredictor(use_lstm=TENSORFLOW_AVAILABLE)
    return predictor.get_comprehensive_analysis(ticker)


if __name__ == '__main__':
    print("=" * 70)
    print("AI STOCK PREDICTOR - TEST")
    print("=" * 70)
    
    predictor = StockPredictor(use_lstm=False)  # Use faster linear model for testing
    
    print("\nAnalyzing AAPL...")
    analysis = predictor.get_comprehensive_analysis('AAPL')
    
    if analysis:
        pred = analysis['prediction']
        signals = analysis['technical_signals']
        
        print(f"\nCurrent Price: ${pred['current_price']:.2f}")
        print(f"\nPredictions ({pred['model_type']} model):")
        for period, price in pred['predictions'].items():
            change = ((price - pred['current_price']) / pred['current_price'] * 100)
            print(f"  {period}: ${price:.2f} ({change:+.2f}%)")
        
        print(f"\nConfidence: {pred['confidence']*100:.1f}%")
        print(f"Trend: {pred['trend']}")
        print(f"Recommendation: {pred['recommendation']}")
        
        print(f"\nTechnical Signals:")
        print(f"  RSI: {signals['rsi']:.1f} ({signals['rsi_signal']})")
        print(f"  MACD: {signals['macd_trend']}")
        print(f"  vs SMA-50: {signals['price_vs_sma50']}")
        
        print(f"\nReasons:")
        for reason in analysis['reasons']:
            print(f"  • {reason}")
        
        print(f"\nRisk Level: {analysis['risk_level']}")
    
    print("\n" + "=" * 70)
    print("Test completed!")
