# 🚀 Real-Time Portfolio Dashboard - Installation & Launch Guide

## Quick Start (3 Steps)

### Step 1: Install Dependencies
```powershell
pip install -r requirements_realtime.txt
```

This installs:
- Flask (web server)
- Flask-SocketIO (WebSocket support)
- Flask-CORS (cross-origin requests)
- yfinance (stock data)
- pandas, numpy (data processing)

### Step 2: Launch the Dashboard
```powershell
python run_realtime.py
```

The script will:
- ✅ Check all dependencies
- ✅ Start the Flask server on http://localhost:5000
- ✅ Automatically open your browser
- ✅ Begin streaming live stock prices

### Step 3: Watch Live Data!
Your browser shows:
- 🔴 Live price ticker scrolling across the top
- 📊 Real-time metrics (gainers, losers, avg change)
- 📈 Auto-updating charts (every 15 seconds)
- 📋 Detailed stock table with current prices
- ⚡ Connection status indicator

---

## Features You'll See

### Live Price Ticker
Continuously scrolling across the top showing:
- Stock symbol (AAPL, MSFT, etc.)
- Current price
- Percentage change (color-coded)

### Connection Status
- **Green dot + "Connected"** = Live data streaming
- **Red dot + "Disconnected"** = Fallback polling mode

### Metrics Cards
- Total stocks monitored: 5
- Gainers: Count of stocks up today
- Losers: Count of stocks down today
- Average change: Portfolio average

### Stock Table
Detailed view with:
- Current price
- Dollar change
- Percentage change
- Previous close
- Day high/low

### Live Charts
Two auto-updating charts:
1. **Price Distribution**: Current prices by stock
2. **Daily Change %**: Performance comparison

---

## Manual Refresh
Click the "🔄 Refresh Now" button anytime for immediate update

---

## Alternative: Manual Server Start

If you prefer to start components separately:

```powershell
# Terminal 1: Start the server
cd scripts
python realtime_server.py

# Terminal 2 or Browser: Open the dashboard
# Open http://localhost:5000 in your browser
```

---

## API Usage

Access the REST API directly:

```powershell
# Get all current prices
curl http://localhost:5000/api/prices

# Get specific stock price
curl http://localhost:5000/api/prices/AAPL

# Get portfolio metrics
curl http://localhost:5000/api/metrics

# Get server status
curl http://localhost:5000/api/status
```

---

## Troubleshooting

### Port Already in Use
If port 5000 is taken, edit `run_realtime.py`:
```python
port = 5001  # Change to any available port
```

### Dependencies Not Found
Make sure you're in the virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements_realtime.txt
```

### Data Not Updating
- Check internet connection
- Verify Yahoo Finance is accessible
- Check console for error messages

### WebSocket Connection Failed
- Check firewall settings
- Dashboard will automatically fall back to polling
- Still works, just uses REST API instead

---

## What's Happening Behind the Scenes

1. **Background Thread** fetches prices every 15 seconds
2. **Cache** stores data to avoid API rate limits
3. **WebSocket** broadcasts updates to all connected browsers
4. **Charts** smoothly animate with new data
5. **Fallback** switches to polling if WebSocket fails

---

## Next Steps

- 📖 Read `REALTIME_QUICKSTART.md` for detailed documentation
- 📊 Check `walkthrough.md` for technical details
- 🎨 Customize update intervals in `scripts/config.py`
- 💻 Build on the API for your own applications

---

## Support

For issues or questions:
- Check the console output for errors
- Review `outputs/portfolio_analysis.log`
- Ensure all dependencies are installed
- Verify internet connectivity

---

**Ready to go?**
```powershell
python run_realtime.py
```

Enjoy your real-time portfolio dashboard! 🎉
