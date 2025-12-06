"""
Real-Time Stock Price Updater
Provides WebSocket-based live price streaming during market hours.
"""

import threading
import time
from datetime import datetime
from typing import Dict, Set, Optional
import logging
from flask_socketio import SocketIO, emit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeUpdater:
    """Manages real-time price updates via WebSocket."""
    
    def __init__(self, socketio: SocketIO, fetcher):
        """
        Initialize real-time updater.
        
        Args:
            socketio: Flask-SocketIO instance
            fetcher: AdvancedStockFetcher instance
        """
        self.socketio = socketio
        self.fetcher = fetcher
        self.subscribed_tickers: Set[str] = set()
        self.update_thread: Optional[threading.Thread] = None
        self.running = False
        self.update_interval = 5  # seconds
        logger.info("RealtimeUpdater initialized")
    
    def subscribe(self, ticker: str):
        """
        Subscribe to real-time updates for a ticker.
        
        Args:
            ticker: Stock ticker to subscribe to
        """
        self.subscribed_tickers.add(ticker)
        logger.info(f"Subscribed to {ticker} (Total: {len(self.subscribed_tickers)})")
    
    def unsubscribe(self, ticker: str):
        """
        Unsubscribe from real-time updates for a ticker.
        
        Args:
            ticker: Stock ticker to unsubscribe from
        """
        self.subscribed_tickers.discard(ticker)
        logger.info(f"Unsubscribed from {ticker} (Total: {len(self.subscribed_tickers)})")
    
    def clear_subscriptions(self):
        """Clear all subscriptions."""
        count = len(self.subscribed_tickers)
        self.subscribed_tickers.clear()
        logger.info(f"Cleared {count} subscriptions")
    
    def is_market_hours(self) -> bool:
        """
        Check if it's currently market hours (9:15 AM - 3:30 PM IST for weekdays).
        
        Returns:
            True if market is open, False otherwise
        """
        now = datetime.now()
        
        # Check if weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Market hours: 9:15 AM to 3:30 PM IST
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def _update_loop(self):
        """Background thread that fetches and broadcasts price updates."""
        logger.info("Real-time update loop started")
        
        while self.running:
            try:
                # Only update during market hours or if we have subscriptions
                if not self.subscribed_tickers:
                    time.sleep(self.update_interval)
                    continue
                
                is_market_open = self.is_market_hours()
                
                # Fetch updates for all subscribed tickers
                updates = {}
                for ticker in list(self.subscribed_tickers):  # Copy to avoid modification during iteration
                    try:
                        data = self.fetcher.get_comprehensive_data(ticker)
                        if data:
                            # Extract only the necessary real-time data
                            updates[ticker] = {
                                'ticker': ticker,
                                'price': data.get('current_price'),
                                'change': data.get('change'),
                                'change_percent': data.get('change_percent'),
                                'volume': data.get('volume'),
                                'timestamp': datetime.now().isoformat(),
                                'source': data.get('source', 'unknown'),
                                'market_open': is_market_open
                            }
                    except Exception as e:
                        logger.error(f"Error fetching update for {ticker}: {e}")
                
                # Broadcast updates via WebSocket
                if updates:
                    self.socketio.emit('price_update', updates, namespace='/')
                    logger.debug(f"Broadcasted updates for {len(updates)} tickers")
                
                # Sleep before next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(self.update_interval)
        
        logger.info("Real-time update loop stopped")
    
    def start(self):
        """Start the real-time update service."""
        if self.running:
            logger.warning("Real-time updater already running")
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        logger.info("Real-time updater started")
    
    def stop(self):
        """Stop the real-time update service."""
        if not self.running:
            logger.warning("Real-time updater not running")
            return
        
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        logger.info("Real-time updater stopped")
    
    def get_status(self) -> Dict:
        """
        Get current status of the updater.
        
        Returns:
            Dictionary with status information
        """
        return {
            'running': self.running,
            'subscribed_tickers': list(self.subscribed_tickers),
            'subscription_count': len(self.subscribed_tickers),
            'market_open': self.is_market_hours(),
            'update_interval': self.update_interval
        }


# Socket.IO event handlers (to be registered in realtime_server.py)
def setup_socketio_handlers(socketio: SocketIO, updater: RealtimeUpdater):
    """
    Setup Socket.IO event handlers for real-time updates.
    
    Args:
        socketio: Flask-SocketIO instance
        updater: RealtimeUpdater instance
    """
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        logger.info(f"Client connected")
        emit('connection_status', {'status': 'connected', 'market_open': updater.is_market_hours()})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        logger.info("Client disconnected")
    
    @socketio.on('subscribe')
    def handle_subscribe(data):
        """
        Handle subscription request.
        
        Args:
            data: Dictionary with 'ticker' field
        """
        ticker = data.get('ticker')
        if ticker:
            updater.subscribe(ticker)
            emit('subscribed', {'ticker': ticker, 'status': 'success'})
            logger.info(f"Client subscribed to {ticker}")
        else:
            emit('error', {'message': 'No ticker provided'})
    
    @socketio.on('unsubscribe')
    def handle_unsubscribe(data):
        """
        Handle unsubscription request.
        
        Args:
            data: Dictionary with 'ticker' field
        """
        ticker = data.get('ticker')
        if ticker:
            updater.unsubscribe(ticker)
            emit('unsubscribed', {'ticker': ticker, 'status': 'success'})
            logger.info(f"Client unsubscribed from {ticker}")
        else:
            emit('error', {'message': 'No ticker provided'})
    
    @socketio.on('get_status')
    def handle_get_status():
        """Handle status request."""
        status = updater.get_status()
        emit('updater_status', status)
    
    logger.info("Socket.IO handlers registered")


if __name__ == "__main__":
    print("RealtimeUpdater module - Use in realtime_server.py")
