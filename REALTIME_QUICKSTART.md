# Real-Time Financial Portfolio Analysis - Quick Start Guide

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements_realtime.txt
```

### 2. Run the Real-Time Dashboard
```powershell
python run_realtime.py
```

The script will:
- Start the Flask server on http://localhost:5000
- Automatically open the dashboard in your default browser
- Begin streaming live stock prices every 15 seconds

## 📊 Features

### Real-Time Updates
- **Live Price Ticker**: Scrolling ticker with current prices
- **Auto-Refresh**: Data updates every 15 seconds automatically
- **WebSocket Streaming**: Instant updates via WebSocket connection
- **Fallback Support**: Automatically switches to polling if WebSocket fails

### Interactive Dashboard
- **Connection Status**: See if you're connected to live data
- **Live Metrics**: Gainers, losers, and average change
- **Stock Table**: Detailed view of all stocks with prices and changes
- **Real-Time Charts**: Auto-updating visualizations

### API Endpoints
The server provides several REST API endpoints:

- `GET /api/status` - Server status and uptime
- `GET /api/prices` - Current prices for all stocks
- `GET /api/prices/<ticker>` - Price for specific stock
- `GET /api/metrics` - Portfolio metrics summary
- `GET /api/intraday/<ticker>` - Intraday candlestick data

## 🛠️ Configuration

Edit `scripts/config.py` to customize:

```python
# Update interval (seconds)
REALTIME_UPDATE_INTERVAL = 15

# Cache time-to-live (seconds)
REALTIME_CACHE_TTL = 15

# Server settings
WEBSOCKET_HOST = '127.0.0.1'
WEBSOCKET_PORT = 5000
```

## 📱 Usage Examples

### Access Dashboard
Open your browser to: http://localhost:5000

### Manual Refresh
Click the "🔄 Refresh Now" button for immediate update

### API Usage
```bash
# Get current prices
curl http://localhost:5000/api/prices

# Get specific stock
curl http://localhost:5000/api/prices/AAPL

# Get portfolio metrics
curl http://localhost:5000/api/metrics
```

## 🔧 Troubleshooting

### WebSocket Connection Issues
- Check firewall settings
- Ensure port 5000 is not in use
- Try using polling fallback (automatic)

### Data Not Updating
- Verify internet connection
- Check Yahoo Finance API status
- Review server logs for errors

### Port Already in Use
Edit `run_realtime.py` to change the port:
```python
port = 5001  # Change to available port
```

## 📝 Architecture

```
User Browser (dashboard_realtime.html)
         ↓
    WebSocket / REST API
         ↓
Flask Server (realtime_server.py)
         ↓
Data Fetcher (realtime_fetcher.py)
         ↓
Yahoo Finance API
```

## ⚠️ Important Notes

1. **Market Hours**: Real-time data is most useful during market hours
2. **API Limits**: Yahoo Finance has rate limits; caching helps avoid them
3. **Network**: Requires active internet connection
4. **Educational Use**: This tool is for educational purposes only

## 🎯 Next Steps

After running the real-time dashboard:
1. Monitor live price changes
2. Observe gainers and losers in real-time
3. Use the API endpoints in your own applications
4. Customize update intervals as needed

## 📚 Related Files

- `scripts/realtime_fetcher.py` - Data fetching logic
- `scripts/realtime_server.py` - Flask server implementation
- `dashboard_realtime.html` - Frontend interface
- `scripts/config.py` - Configuration settings

## 🔗 Additional Resources

- [Yahoo Finance Documentation](https://finance.yahoo.com)
- [Flask Documentation](https://flask.palletsprojects.com)
- [Socket.IO Documentation](https://socket.io)

---

**Project**: Financial Portfolio Analysis  
**Version**: 2.0 (Real-Time Edition)  
**Last Updated**: November 30, 2025
