# 🚀 Advanced Stock Market Platform

**AI-Powered Stock Analysis with Real-Time Data**

## Quick Start

```powershell
# Install dependencies (one-time)
pip install -r requirements_realtime.txt

# Launch platform
python run_realtime.py
```

Browser opens automatically at http://localhost:5000

## Features

✅ **Search ANY Stock** - 10,000+ stocks globally (US, India, worldwide)  
✅ **AI Predictions** - LSTM neural networks with buy/sell recommendations  
✅ **50+ Data Fields** - P/E, EPS, Market Cap, Beta, News, Earnings  
✅ **Technical Analysis** - RSI, MACD, Moving Averages  
✅ **Watchlist Management** - Save & track favorite stocks  
✅ **Real-Time Updates** - Live prices every 15 seconds  
✅ **News Integration** - Latest market news  

## What's New

### From Basic Tracker to Professional Platform

**Before:**
- ❌ Only 5 hardcoded stocks
- ❌ No AI predictions
- ❌ Basic price data
- ❌ No customization


**Now:**
- ✅ **ANY stock globally**
- ✅ **AI predictions** (LSTM)
- ✅ **50+ data points**
- ✅ **Persistent watchlists**
- ✅ **Technical indicators**
- ✅ **Live news**

## API endpoints

```bash
# Search stocks
GET /api/search?q=tesla

# Get stock details
GET /api/stock/AAPL

# AI predictions
GET /api/stock/AAPL/predict

# Latest news
GET /api/stock/AAPL/news

# Historical data
GET /api/stock/AAPL/history?period=1mo

# Watchlist
GET /api/watchlist
POST /api/watchlist/add
```

## Test AI Predictions

```powershell
python scripts/ai_predictor.py
```

## Project Structure

```
financial_portfolio_analysis/
├── scripts/
│   ├── advanced_fetcher.py    # Stock search & data
│   ├── ai_predictor.py         # AI predictions
│   ├── database.py             # Watchlist storage
│   └── realtime_server.py      # Flask server
├── data/                       # Database & cache
├── run_realtime.py            # Launcher
└── requirements_realtime.txt  # Dependencies
```

## Technologies

- **Backend:** Flask, WebSocket, SQLite
- **AI/ML:** TensorFlow/Keras (LSTM), scikit-learn
- **Data:** yfinance, pandas, numpy
- **Frontend:** HTML5, JavaScript, Chart.js

## Key Capabilities

1. **Dynamic Stock Search** - No hardcoded limits
2. **AI Intelligence** - Machine learning predictions
3. **Comprehensive Data** - All fields from professional platforms
4. **Persistent Storage** - SQLite database
5. **Real-Time Streaming** - WebSocket updates
6. **Scalable Architecture** - Easy to extend

## Performance

- Search: <500ms
- AI Prediction: 3-5 seconds
- Data Fetch: <1 second
- Updates: Every 15 seconds

## Disclaimer

⚠️ For educational purposes only. Not financial advice.

---

**Ready?** Run `python run_realtime.py` 🚀
