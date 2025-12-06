"""
Launch script for Advanced Stock Market Platform
"""

import os
import sys
import time
import webbrowser
import logging
from pathlib import Path

# Add scripts to path
scripts_dir = Path(__file__).parent / 'scripts'
sys.path.insert(0, str(scripts_dir))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check required packages."""
    required = ['flask', 'flask_cors', 'flask_socketio', 'yfinance', 'pandas', 'numpy', 'sklearn']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.error("Run: pip install -r requirements_realtime.txt")
        return False
    
    return True


def main():
    """Launch the platform."""
    print("=" * 70)
    print("🚀 ADVANCED STOCK MARKET PLATFORM")
    print("=" * 70)
    print()
    print("Features:")
    print("  🔍 Search ANY stock globally")
    print("  🤖 AI predictions with LSTM")
    print("  📊 50+ market data fields")
    print("  💾 Watchlist management")
    print("  📰 Live news integration")
    print()
    print("=" * 70)
    print()
    
    if not check_dependencies():
        logger.error("Dependency check failed!")
        sys.exit(1)
    
    logger.info("All dependencies OK!")
    
    # Import server
    try:
        from realtime_server import start_server
    except ImportError as e:
        logger.error(f"Failed to import server: {e}")
        sys.exit(1)
    
    # Configuration
    host = '127.0.0.1'
    port = 5000
    url = f'http://{host}:{port}'
    
    print(f"Starting server on {url}")
    print(f"Press Ctrl+C to stop")
    print("=" * 70)
    print()
    
    # Open browser
    def open_browser():
        time.sleep(2)
        logger.info(f"Opening browser at {url}")
        try:
            webbrowser.open(url)
        except:
            logger.info(f"Please open {url} in your browser")
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start server
    try:
        start_server(host=host, port=port, debug=False)
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("Server stopped")
        print("=" * 70)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
