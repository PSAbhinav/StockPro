import sys
import os

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

from realtime_server import app

# This is required for Vercel
# Vercel will look for 'app' by default in the entry point
# Alternatively, we can use the 'handler' variable
handler = app
