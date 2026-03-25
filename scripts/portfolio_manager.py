"""
Portfolio Manager - FIXED VERSION
Simple password storage for demo purposes
"""

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages user portfolios and trading."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, 'stock_platform.db')
        
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                balance REAL DEFAULT 100000.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS holdings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                quantity REAL NOT NULL,
                avg_price REAL NOT NULL,
                current_price REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, ticker)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                type TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                total REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, username: str, email: str, password: str) -> Optional[int]:
        """Create new user - stores password directly for demo."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                (username, email, password)
            )
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Created user: {username}")
            return user_id
        except sqlite3.IntegrityError:
            return None
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return None
    
    def login_user(self, username: str, password: str) -> Optional[Dict]:
        """Login user."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT id, username, email, balance FROM users WHERE username = ? AND password_hash = ?',
                (username, password)
            )
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'id': row[0],
                    'username': row[1],
                    'email': row[2],
                    'balance': row[3]
                }
            return None
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return None
    
    def get_user_balance(self, user_id: int) -> float:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT balance FROM users WHERE id = ?', (user_id,))
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else 0.0
        except:
            return 0.0
    
    def buy_stock(self, user_id: int, ticker: str, quantity: float, price: float) -> bool:
        try:
            total_cost = quantity * price
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT balance FROM users WHERE id = ?', (user_id,))
            balance = cursor.fetchone()[0]
            
            if balance < total_cost:
                conn.close()
                return False
            
            cursor.execute('UPDATE users SET balance = balance - ? WHERE id = ?', (total_cost, user_id))
            
            cursor.execute(
                'SELECT quantity, avg_price FROM holdings WHERE user_id = ? AND ticker = ?',
                (user_id, ticker)
            )
            existing = cursor.fetchone()
            
            if existing:
                old_qty, old_avg = existing
                new_qty = old_qty + quantity
                new_avg = ((old_qty * old_avg) + (quantity * price)) / new_qty
                cursor.execute(
                    'UPDATE holdings SET quantity = ?, avg_price = ?, current_price = ? WHERE user_id = ? AND ticker = ?',
                    (new_qty, new_avg, price, user_id, ticker)
                )
            else:
                cursor.execute(
                    'INSERT INTO holdings (user_id, ticker, quantity, avg_price, current_price) VALUES (?, ?, ?, ?, ?)',
                    (user_id, ticker, quantity, price, price)
                )
            
            cursor.execute(
                'INSERT INTO transactions (user_id, ticker, type, quantity, price, total) VALUES (?, ?, ?, ?, ?, ?)',
                (user_id, ticker, 'BUY', quantity, price, total_cost)
            )
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Buy error: {str(e)}")
            return False
    
    def sell_stock(self, user_id: int, ticker: str, quantity: float, price: float) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT quantity FROM holdings WHERE user_id = ? AND ticker = ?', (user_id, ticker))
            holding = cursor.fetchone()
            
            if not holding or holding[0] < quantity:
                conn.close()
                return False
            
            total_proceeds = quantity * price
            cursor.execute('UPDATE users SET balance = balance + ? WHERE id = ?', (total_proceeds, user_id))
            
            new_qty = holding[0] - quantity
            if new_qty > 0:
                cursor.execute(
                    'UPDATE holdings SET quantity = ?, current_price = ? WHERE user_id = ? AND ticker = ?',
                    (new_qty, price, user_id, ticker)
                )
            else:
                cursor.execute('DELETE FROM holdings WHERE user_id = ? AND ticker = ?', (user_id, ticker))
            
            cursor.execute(
                'INSERT INTO transactions (user_id, ticker, type, quantity, price, total) VALUES (?, ?, ?, ?, ?, ?)',
                (user_id, ticker, 'SELL', quantity, price, total_proceeds)
            )
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Sell error: {str(e)}")
            return False
    
    def get_portfolio(self, user_id: int) -> List[Dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = cursor.cursor()
            cursor.execute('SELECT ticker, quantity, avg_price, current_price FROM holdings WHERE user_id = ?', (user_id,))
            
            holdings = []
            for row in cursor.fetchall():
                holdings.append({
                    'ticker': row[0],
                    'quantity': row[1],
                    'avg_price': row[2],
                    'current_price': row[3] or row[2]
                })
            
            conn.close()
            return holdings
        except:
            return []
    
    def get_transactions(self, user_id: int, limit: int = 50) -> List[Dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                'SELECT ticker, type, quantity, price, total, timestamp FROM transactions WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?',
                (user_id, limit)
            )
            
            transactions = []
            for row in cursor.fetchall():
                transactions.append({
                    'ticker': row[0],
                    'type': row[1],
                    'quantity': row[2],
                    'price': row[3],
                    'total': row[4],
                    'timestamp': row[5]
                })
            
            conn.close()
            return transactions
        except:
            return []
    
    def calculate_portfolio_value(self, user_id: int, current_prices: Dict[str, float]) -> Dict:
        portfolio = self.get_portfolio(user_id)
        balance = self.get_user_balance(user_id)
        
        total_investment = 0
        current_value = 0
        
        for holding in portfolio:
            ticker = holding['ticker']
            quantity = holding['quantity']
            avg_price = holding['avg_price']
            current_price = current_prices.get(ticker, holding['current_price'])
            
            total_investment += quantity * avg_price
            current_value += quantity * current_price
        
        total_pl = current_value - total_investment
        pl_percent = (total_pl / total_investment * 100) if total_investment > 0 else 0
        
        return {
            'balance': balance,
            'invested': total_investment,
            'current_value': current_value,
            'total_value': balance + current_value,
            'pl': total_pl,
            'pl_percent': pl_percent
        }
