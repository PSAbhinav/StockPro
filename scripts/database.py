"""
Database Manager for Stock Market Platform
Handles watchlists, preferences, and caching.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database for watchlists and caching."""
    
    def __init__(self, db_path: str = None):
        """Initialize database manager."""
        if db_path is None:
            # Default to data directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, 'stock_platform.db')
        
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Watchlists table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watchlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                tickers TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Stock cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_cache (
                ticker TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # AI predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                prediction_date DATE NOT NULL,
                predicted_price REAL,
                confidence REAL,
                actual_price REAL,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # User preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    # Watchlist methods
    def create_watchlist(self, name: str, tickers: List[str]) -> bool:
        """Create a new watchlist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            tickers_json = json.dumps(tickers)
            cursor.execute(
                'INSERT INTO watchlists (name, tickers) VALUES (?, ?)',
                (name, tickers_json)
            )
            
            conn.commit()
            conn.close()
            logger.info(f"Created watchlist: {name}")
            return True
        except sqlite3.IntegrityError:
            logger.error(f"Watchlist {name} already exists")
            return False
        except Exception as e:
            logger.error(f"Error creating watchlist: {str(e)}")
            return False
    
    def get_watchlist(self, name: str) -> Optional[Dict]:
        """Get a specific watchlist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT name, tickers, created_at, updated_at FROM watchlists WHERE name = ?',
                (name,)
            )
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'name': row[0],
                    'tickers': json.loads(row[1]),
                    'created_at': row[2],
                    'updated_at': row[3]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting watchlist: {str(e)}")
            return None
    
    def get_all_watchlists(self) -> List[Dict]:
        """Get all watchlists."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT name, tickers, created_at, updated_at FROM watchlists')
            rows = cursor.fetchall()
            conn.close()
            
            return [{
                'name': row[0],
                'tickers': json.loads(row[1]),
                'created_at': row[2],
                'updated_at': row[3]
            } for row in rows]
        except Exception as e:
            logger.error(f"Error getting watchlists: {str(e)}")
            return []
    
    def update_watchlist(self, name: str, tickers: List[str]) -> bool:
        """Update watchlist tickers."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            tickers_json = json.dumps(tickers)
            cursor.execute(
                'UPDATE watchlists SET tickers = ?, updated_at = CURRENT_TIMESTAMP WHERE name = ?',
                (tickers_json, name)
            )
            
            conn.commit()
            conn.close()
            logger.info(f"Updated watchlist: {name}")
            return True
        except Exception as e:
            logger.error(f"Error updating watchlist: {str(e)}")
            return False
    
    def delete_watchlist(self, name: str) -> bool:
        """Delete a watchlist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM watchlists WHERE name = ?', (name,))
            
            conn.commit()
            conn.close()
            logger.info(f"Deleted watchlist: {name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting watchlist: {str(e)}")
            return False
    
    def add_to_watchlist(self, name: str, ticker: str) -> bool:
        """Add a ticker to watchlist."""
        watchlist = self.get_watchlist(name)
        if not watchlist:
            return False
        
        tickers = watchlist['tickers']
        if ticker not in tickers:
            tickers.append(ticker)
            return self.update_watchlist(name, tickers)
        return True
    
    def remove_from_watchlist(self, name: str, ticker: str) -> bool:
        """Remove a ticker from watchlist."""
        watchlist = self.get_watchlist(name)
        if not watchlist:
            return False
        
        tickers = watchlist['tickers']
        if ticker in tickers:
            tickers.remove(ticker)
            return self.update_watchlist(name, tickers)
        return True
    
    # Cache methods
    def cache_stock_data(self, ticker: str, data: Dict) -> bool:
        """Cache stock data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            data_json = json.dumps(data)
            cursor.execute(
                'INSERT OR REPLACE INTO stock_cache (ticker, data, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)',
                (ticker, data_json)
            )
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error caching stock data: {str(e)}")
            return False
    
    def get_cached_stock_data(self, ticker: str, max_age_minutes: int = 5) -> Optional[Dict]:
        """Get cached stock data if not expired."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT data, updated_at FROM stock_cache WHERE ticker = ?',
                (ticker,)
            )
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                updated_at = datetime.fromisoformat(row[1])
                age = datetime.now() - updated_at
                
                if age.total_seconds() / 60 < max_age_minutes:
                    return json.loads(row[0])
            
            return None
        except Exception as e:
            logger.error(f"Error getting cached data: {str(e)}")
            return None
    
    # Preferences methods
    def set_preference(self, key: str, value: str) -> bool:
        """Set a user preference."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'INSERT OR REPLACE INTO preferences (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)',
                (key, value)
            )
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error setting preference: {str(e)}")
            return False
    
    def get_preference(self, key: str, default: str = None) -> Optional[str]:
        """Get a user preference."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT value FROM preferences WHERE key = ?', (key,))
            row = cursor.fetchone()
            conn.close()
            
            return row[0] if row else default
        except Exception as e:
            logger.error(f"Error getting preference: {str(e)}")
            return default
    
    # Prediction methods
    def save_prediction(self, ticker: str, prediction_date: str, predicted_price: float,
                       confidence: float, model_version: str = '1.0') -> bool:
        """Save an AI prediction."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                '''INSERT INTO predictions 
                   (ticker, prediction_date, predicted_price, confidence, model_version)
                   VALUES (?, ?, ?, ?, ?)''',
                (ticker, prediction_date, predicted_price, confidence, model_version)
            )
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            return False
    
    def get_predictions(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get predictions for a ticker."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                '''SELECT prediction_date, predicted_price, confidence, actual_price, created_at
                   FROM predictions WHERE ticker = ? ORDER BY created_at DESC LIMIT ?''',
                (ticker, limit)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            return [{
                'prediction_date': row[0],
                'predicted_price': row[1],
                'confidence': row[2],
                'actual_price': row[3],
                'created_at': row[4]
            } for row in rows]
        except Exception as e:
            logger.error(f"Error getting predictions: {str(e)}")
            return []


class WatchlistManager:
    """High-level watchlist management."""
    
    def __init__(self):
        self.db = DatabaseManager()
        self._ensure_default_watchlist()
    
    def _ensure_default_watchlist(self):
        """Ensure a default watchlist exists."""
        if not self.db.get_watchlist('Default'):
            self.db.create_watchlist('Default', ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA'])
    
    def add_stock(self, ticker: str, watchlist_name: str = 'Default') -> bool:
        """Add a stock to watchlist."""
        return self.db.add_to_watchlist(watchlist_name, ticker.upper())
    
    def remove_stock(self, ticker: str, watchlist_name: str = 'Default') -> bool:
        """Remove a stock from watchlist."""
        return self.db.remove_from_watchlist(watchlist_name, ticker.upper())
    
    def get_watchlist_tickers(self, watchlist_name: str = 'Default') -> List[str]:
        """Get tickers in a watchlist."""
        watchlist = self.db.get_watchlist(watchlist_name)
        return watchlist['tickers'] if watchlist else []
    
    def get_all_watchlists(self) -> List[str]:
        """Get all watchlist names."""
        watchlists = self.db.get_all_watchlists()
        return [w['name'] for w in watchlists]


if __name__ == '__main__':
    print("=" * 70)
    print("DATABASE MANAGER - TEST")
    print("=" * 70)
    
    # Test watchlist management
    manager = WatchlistManager()
    
    print("\n1. Default watchlist:")
    tickers = manager.get_watchlist_tickers('Default')
    print(f"   {tickers}")
    
    print("\n2. Adding TSLA to watchlist...")
    manager.add_stock('TSLA')
    tickers = manager.get_watchlist_tickers('Default')
    print(f"   {tickers}")
    
    print("\n3. Creating custom watchlist...")
    manager.db.create_watchlist('Tech Giants', ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN'])
    
    print("\n4. All watchlists:")
    watchlists = manager.get_all_watchlists()
    for name in watchlists:
        tickers = manager.get_watchlist_tickers(name)
        print(f"   • {name}: {tickers}")
    
    print("\n5. Testing preferences...")
    manager.db.set_preference('theme', 'dark')
    theme = manager.db.get_preference('theme')
    print(f"   Theme preference: {theme}")
    
    print("\n" + "=" * 70)
    print("Test completed!")
