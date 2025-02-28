#!/bin/bash

# Start mask_api.py in background
uvicorn mask_api:app --host 0.0.0.0 --port 8000 &

# Start live_feed_api.py in background  
uvicorn live_feed_api:app --host 0.0.0.0 --port 8001 &

# Start HTTP server for HTML files
python -m http.server 9000

# Keep container running
wait