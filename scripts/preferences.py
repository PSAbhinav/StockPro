"""
User Preferences and Theme Manager
Handles theme preferences and user settings
"""

import sqlite3
import json
import os
from typing import Optional, Dict


class PreferencesManager:
    """Manages user preferences and theme settings."""
    
    THEMES = ['light', 'dark', 'blue', 'purple', 'green']
    
    def __init__(self, db_path: str = None):
        """Initialize preferences manager."""
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, 'stock_platform.db')
        
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create preferences table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id INTEGER,
                preference_key TEXT,
                preference_value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, preference_key)
            )
        ''')
        
        # Guest preferences (user_id = 0)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS guest_preferences (
                preference_key TEXT PRIMARY KEY,
                preference_value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def set_preference(self, key: str, value: str, user_id: int = 0):
        """Set user preference."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if user_id == 0:  # Guest
            cursor.execute('''
                INSERT OR REPLACE INTO guest_preferences (preference_key, preference_value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, value))
        else:
            cursor.execute('''
                INSERT OR REPLACE INTO user_preferences (user_id, preference_key, preference_value, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, key, value))
        
        conn.commit()
        conn.close()
    
    def get_preference(self, key: str, default: str = None, user_id: int = 0) -> Optional[str]:
        """Get user preference."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if user_id == 0:  # Guest
            cursor.execute('SELECT preference_value FROM guest_preferences WHERE preference_key = ?', (key,))
        else:
            cursor.execute(
                'SELECT preference_value FROM user_preferences WHERE user_id = ? AND preference_key = ?',
                (user_id, key)
            )
        
        row = cursor.fetchone()
        conn.close()
        
        return row[0] if row else default
    
    def get_all_preferences(self, user_id: int = 0) -> Dict:
        """Get all preferences for user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if user_id == 0:
            cursor.execute('SELECT preference_key, preference_value FROM guest_preferences')
        else:
            cursor.execute(
                'SELECT preference_key, preference_value FROM user_preferences WHERE user_id = ?',
                (user_id,)
            )
        
        preferences = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        
        return preferences


if __name__ == '__main__':
    print("Testing PreferencesManager...")
    
    manager = PreferencesManager()
    
    # Guest preferences
    manager.set_preference('theme', 'dark', user_id=0)
    manager.set_preference('mode', 'educational', user_id=0)
    
    theme = manager.get_preference('theme', 'light', user_id=0)
    print(f"Guest theme: {theme}")
    
    all_prefs = manager.get_all_preferences(user_id=0)
    print(f"All guest preferences: {all_prefs}")
    
    print("✓ PreferencesManager working!")
