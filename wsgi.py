import os
import sys
import threading
from pathlib import Path

# Add scripts directory to path to allow imports
scripts_dir = Path(__file__).parent / 'scripts'
sys.path.insert(0, str(scripts_dir))

# Import the server components
try:
    from realtime_server import app, socketio, realtime_updater, background_updater
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def start_background_tasks():
    """Start the background data acquisition threads."""
    print("Starting background threads...")
    
    # 1. Start background updater (watchlist updates)
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()
    print("Background updater thread started")
    
    # 2. Start real-time price updater (yfinance fetching)
    realtime_updater.start()
    print("Real-time price updater started")

# We want to ensure background tasks start whether running as main or via Gunicorn
# Gunicorn with eventlet works best with tasks started in the master process or once per worker
start_background_tasks()

if __name__ == "__main__":
    # Get port from environment or default to 5000
    port = int(os.environ.get("PORT", 5000))
    # Run using socketio wrapper for local testing
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
