"""
Enhanced Real-Time Server with Advanced Features
Supports stock search, AI predictions, watchlists, and comprehensive data.
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import time
import logging
from datetime import datetime
import os
import sys

# Add to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_fetcher import AdvancedStockFetcher
from database import WatchlistManager, DatabaseManager
from ai_predictor import StockPredictor
from portfolio_manager import PortfolioManager
from realtime_updater import RealtimeUpdater, setup_socketio_handlers
from portfolio_analytics import get_real_time_portfolio_value, get_sector_allocation, export_portfolio_to_csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            static_folder='../figures',
            template_folder='..')
app.config['SECRET_KEY'] = 'advanced-stock-platform-secret-key'
CORS(app)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize components
fetcher = AdvancedStockFetcher(cache_ttl=60)
watchlist_manager = WatchlistManager()
db = DatabaseManager()
predictor = StockPredictor(use_lstm=False)  # Start with faster model
portfolio = PortfolioManager()

# Initialize real-time updater
realtime_updater = RealtimeUpdater(socketio, fetcher)
setup_socketio_handlers(socketio, realtime_updater)

# Global state
server_start_time = datetime.now()
update_count = 0
connected_clients = 0
UPDATE_INTERVAL = 15



@app.route('/')
def index():
    """Serve the advanced dashboard."""
    return send_from_directory('..', 'dashboard_advanced.html')


@app.route('/api/search')
def api_search():
    """Search for stocks."""
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))
    
    if not query:
        return jsonify({'success': False, 'error': 'Query parameter required'}), 400
    
    try:
        results = fetcher.search_stock(query, limit=limit)
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        })
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stock/<ticker>')
def api_stock_details(ticker):
    """Get comprehensive stock details."""
    try:
        data = fetcher.get_comprehensive_data(ticker.upper())
        
        if not data:
            return jsonify({'success': False, 'error': 'Stock not found'}), 404
        
        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stock/<ticker>/predict')
def api_predict(ticker):
    """Get AI predictions for stock."""
    try:
        analysis = predictor.get_comprehensive_analysis(ticker.upper())
        
        if not analysis:
            return jsonify({'success': False, 'error': 'Prediction failed'}), 500
        
        # Save prediction to database
        pred = analysis['prediction']
        if 'predictions' in pred and '1d' in pred['predictions']:
            db.save_prediction(
                ticker.upper(),
                datetime.now().strftime('%Y-%m-%d'),
                pred['predictions']['1d'],
                pred['confidence']
            )
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
    except Exception as e:
        logger.error(f"Prediction error for {ticker}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stock/<ticker>/news')
def api_news(ticker):
    """Get news for stock."""
    try:
        limit = int(request.args.get('limit', 10))
        news = fetcher.get_company_news(ticker.upper(), limit=limit)
        
        return jsonify({
            'success': True,
            'news': news,
            'count': len(news)
        })
    except Exception as e:
        logger.error(f"News error for {ticker}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stock/<ticker>/history')
def api_history(ticker):
    """Get historical data."""
    try:
        period = request.args.get('period', '1mo')
        interval = request.args.get('interval', '1d')
        
        data = fetcher.get_historical_data(ticker.upper(), period=period, interval=interval)
        
        if data.empty:
            return jsonify({'success': False, 'error': 'No data available'}), 404
        
        # Convert to JSON-friendly format
        history = {
            'timestamps': data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'open': data['Open'].tolist(),
            'high': data['High'].tolist(),
            'low': data['Low'].tolist(),
            'close': data['Close'].tolist(),
            'volume': data['Volume'].tolist()
        }
        
        return jsonify({
            'success': True,
            'data': history
        })
    except Exception as e:
        logger.error(f"History error for {ticker}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/watchlist', methods=['GET'])
def api_get_watchlist():
    """Get user's watchlist."""
    try:
        name = request.args.get('name', 'Default')
        tickers = watchlist_manager.get_watchlist_tickers(name)
        
        # Get quick stats for all tickers
        stocks = []
        for ticker in tickers:
            stats = fetcher.get_quick_stats(ticker)
            if stats:
                stocks.append(stats)
        
        return jsonify({
            'success': True,
            'watchlist': name,
            'tickers': tickers,
            'stocks': stocks
        })
    except Exception as e:
        logger.error(f"Watchlist error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/watchlist/add', methods=['POST'])
def api_add_to_watchlist():
    """Add stock to watchlist."""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').upper()
        watchlist_name = data.get('watchlist', 'Default')
        
        if not ticker:
            return jsonify({'success': False, 'error': 'Ticker required'}), 400
        
        success = watchlist_manager.add_stock(ticker, watchlist_name)
        
        if success:
            return jsonify({'success': True, 'message': f'{ticker} added to {watchlist_name}'})
        else:
            return jsonify({'success': False, 'error': 'Failed to add stock'}), 500
    except Exception as e:
        logger.error(f"Add to watchlist error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/watchlist/remove', methods=['POST'])
def api_remove_from_watchlist():
    """Remove stock from watchlist."""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').upper()
        watchlist_name = data.get('watchlist', 'Default')
        
        if not ticker:
            return jsonify({'success': False, 'error': 'Ticker required'}), 400
        
        success = watchlist_manager.remove_stock(ticker, watchlist_name)
        
        if success:
            return jsonify({'success': True, 'message': f'{ticker} removed from {watchlist_name}'})
        else:
            return jsonify({'success': False, 'error': 'Failed to remove stock'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ===== PORTFOLIO ANALYTICS ENDPOINTS =====

@app.route('/api/portfolio/analytics/<int:user_id>')
def api_portfolio_analytics(user_id):
    """Get comprehensive portfolio analytics with real-time values."""
    try:
        portfolio_data = get_real_time_portfolio_value(portfolio, user_id, fetcher)
        
        if not portfolio_data:
            return jsonify({'success': False, 'error': 'Failed to load portfolio'}), 500
        
        sector_data = get_sector_allocation(portfolio, user_id, fetcher)
        
        return jsonify({
            'success': True,
            'portfolio': portfolio_data,
            'sector_allocation': sector_data
        })
    except Exception as e:
        logger.error(f"Portfolio analytics error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/portfolio/export/<int:user_id>')
def api_export_portfolio(user_id):
    """Export portfolio to CSV file."""
    try:
        from flask import send_file
        
        filepath = export_portfolio_to_csv(portfolio, user_id, fetcher)
        
        if not filepath:
            return jsonify({'success': False, 'error': 'Export failed'}), 500
        
        return send_file(filepath, as_attachment=True, 
                        download_name=f'portfolio_export_{user_id}.csv',
                        mimetype='text/csv')
    except Exception as e:
        logger.error(f"Portfolio export error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/api/preferences', methods=['GET', 'POST'])
def api_preferences():
    """Get or set user preferences."""
    try:
        if request.method == 'GET':
            theme = db.get_preference('theme', 'light')
            return jsonify({
                'success': True,
                'preferences': {
                    'theme': theme
                }
            })
        else:  # POST
            data = request.get_json()
            theme = data.get('theme')
            
            if theme:
                db.set_preference('theme', theme)
            
            return jsonify({'success': True, 'message': 'Preferences updated'})
    except Exception as e:
        logger.error(f"Preferences error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# Portfolio & Trading endpoints
@app.route('/api/auth/register', methods=['POST'])
def api_register():
    """Register new user."""
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not all([username, email, password]):
            return jsonify({'success': False, 'error': 'All fields required'}), 400
        
        user_id = portfolio.create_user(username, email, password)
        
        if user_id:
            return jsonify({'success': True, 'user_id': user_id})
        else:
            return jsonify({'success': False, 'error': 'User already exists'}), 400
    except Exception as e:
        logger.error(f"Register error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """Login user."""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not all([username, password]):
            return jsonify({'success': False, 'error': 'Username and password required'}), 400
        
        user = portfolio.login_user(username, password)
        
        if user:
            return jsonify({'success': True, 'user': user})
        else:
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/portfolio/<int:user_id>')
def api_get_portfolio(user_id):
    """Get user portfolio."""
    try:
        holdings = portfolio.get_portfolio(user_id)
        balance = portfolio.get_user_balance(user_id)
        
        # Get current prices
        current_prices = {}
        for holding in holdings:
            data = fetcher.get_quick_stats(holding['ticker'])
            if data:
                current_prices[holding['ticker']] = data['price']
        
        value = portfolio.calculate_portfolio_value(user_id, current_prices)
        
        return jsonify({
            'success': True,
            'holdings': holdings,
            'value': value
        })
    except Exception as e:
        logger.error(f"Portfolio error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/trade/buy', methods=['POST'])
def api_buy_stock():
    """Buy stock."""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        ticker = data.get('ticker', '').upper()
        quantity = float(data.get('quantity', 0))
        
        if not all([user_id, ticker, quantity]):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # Get current price
        stock_data = fetcher.get_quick_stats(ticker)
        if not stock_data or not stock_data.get('price'):
            return jsonify({'success': False, 'error': 'Cannot get stock price'}), 400
        
        price = stock_data['price']
        
        success = portfolio.buy_stock(user_id, ticker, quantity, price)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Bought {quantity} shares of {ticker} at ${price}',
                'total': quantity * price
            })
        else:
            return jsonify({'success': False, 'error': 'Purchase failed (insufficient balance?)'}), 400
    except Exception as e:
        logger.error(f"Buy error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/trade/sell', methods=['POST'])
def api_sell_stock():
    """Sell stock."""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        ticker = data.get('ticker', '').upper()
        quantity = float(data.get('quantity', 0))
        
        if not all([user_id, ticker, quantity]):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # Get current price
        stock_data = fetcher.get_quick_stats(ticker)
        if not stock_data or not stock_data.get('price'):
            return jsonify({'success': False, 'error': 'Cannot get stock price'}), 400
        
        price = stock_data['price']
        
        success = portfolio.sell_stock(user_id, ticker, quantity, price)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Sold {quantity} shares of {ticker} at ${price}',
                'total': quantity * price
            })
        else:
            return jsonify({'success': False, 'error': 'Sale failed (insufficient holdings?)'}), 400
    except Exception as e:
        logger.error(f"Sell error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/transactions/<int:user_id>')
def api_get_transactions(user_id):
    """Get user transaction history."""
    try:
        limit = int(request.args.get('limit', 100))
        transactions = portfolio.get_transactions(user_id, limit=limit)
        
        return jsonify({
            'success': True,
            'transactions': transactions,
            'count': len(transactions)
        })
    except Exception as e:
        logger.error(f"Transactions error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/categories')
def api_categories():
    """Get stock categories - FIXED with proper market caps."""
    try:
        # INDIAN STOCKS FIRST - as requested
        categories = {
            'top_gainers': [],
            'top_losers': [],
            # Indian stocks - prioritized
            'indian_large_cap': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS'],
            'indian_mid_cap': ['TATAMOTORS.NS', 'M&M.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'EICHERMOT.NS'],
            'indian_small_cap': ['ZOMATO.NS', 'PAYTM.NS', 'NYKAA.NS', 'POLICYBZR.NS'],
            # US stocks
            'us_large_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
            'us_mid_cap': ['AMD', 'NFLX', 'UBER', 'ADBE', 'CRM'],
            'us_small_cap': ['ROKU', 'PLTR', 'SNAP', 'PINS']
        }
        
        # Get data for ALL stocks to find gainers/losers
        all_stocks = (categories['indian_large_cap'] + categories['indian_mid_cap'] + 
                     categories['us_large_cap'][:5])  # Limited for performance
        stocks_data = []
        
        for ticker in all_stocks:
            data = fetcher.get_quick_stats(ticker)
            if data:
                stocks_data.append(data)
        
        # Sort by change to get gainers/losers
        stocks_data.sort(key=lambda x: x.get('change_percent', 0), reverse=True)
        categories['top_gainers'] = [s for s in stocks_data[:6] if s.get('change_percent', 0) > 0]
        categories['top_losers'] = [s for s in stocks_data[-6:] if s.get('change_percent', 0) < 0]
        categories['top_losers'].reverse()  # Show worst first
        
        return jsonify({
            'success': True,
            'categories': categories
        })
    except Exception as e:
        logger.error(f"Categories error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/status')
def api_status():
    """Get server status."""
    uptime = (datetime.now() - server_start_time).total_seconds()
    
    return jsonify({
        'status': 'running',
        'uptime_seconds': int(uptime),
        'update_count': update_count,
        'connected_clients': connected_clients,
        'features': {
            'search': True,
            'ai_predictions': True,
            'watchlists': True,
            'news': True,
            'technical_analysis': True,
            'trading': True,
            'authentication': True
        },
        'timestamp': datetime.now().isoformat()
    })


# WebSocket handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    global connected_clients
    connected_clients += 1
    logger.info(f"Client connected. Total: {connected_clients}")


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    global connected_clients
    connected_clients = max(0, connected_clients - 1)
    logger.info(f"Client disconnected. Total: {connected_clients}")


@socketio.on('subscribe')
def handle_subscribe(data):
    """Subscribe to stock updates."""
    ticker = data.get('ticker', '').upper()
    if ticker:
        logger.info(f"Client subscribed to {ticker}")
        # Send initial data
        stock_data = fetcher.get_quick_stats(ticker)
        if stock_data:
            emit('stock_update', {'ticker': ticker, 'data': stock_data})


def background_updater():
    """Background thread for watchlist updates."""
    global update_count
    
    logger.info("Background updater started")
    
    while True:
        try:
            if connected_clients > 0:
                # Get default watchlist
                tickers = watchlist_manager.get_watchlist_tickers('Default')
                
                # Fetch data for all tickers
                updates = {}
                for ticker in tickers:
                    stats = fetcher.get_quick_stats(ticker)
                    if stats:
                        updates[ticker] = stats
                
                # Broadcast updates
                socketio.emit('watchlist_update', updates, broadcast=True)
                update_count += 1
                logger.info(f"Broadcast update #{update_count} to {connected_clients} clients")
            
            time.sleep(UPDATE_INTERVAL)
        except Exception as e:
            logger.error(f"Background updater error: {str(e)}")
            time.sleep(UPDATE_INTERVAL)


def start_server(host='127.0.0.1', port=5000, debug=False):
    """Start the Flask server."""
    logger.info("=" * 70)
    logger.info("ADVANCED STOCK MARKET PLATFORM SERVER")
    logger.info("=" * 70)
    logger.info(f"Server starting on http://{host}:{port}")
    logger.info("Features: Stock Search, AI Predictions, Watchlists, News, Real-Time Updates")
    logger.info("=" * 70)
    
    # Start background updater
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()
    logger.info("Background updater started")
    
    # Start real-time price updater
    realtime_updater.start()
    logger.info("Real-time price updater started")
    
    # Start server
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    start_server(debug=False)
