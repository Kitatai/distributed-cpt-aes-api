#!/bin/bash
# Start server in background (survives SSH disconnect)
cd "$(dirname "$0")"

# Kill existing server if running
pkill -f "uvicorn main:app" 2>/dev/null
sleep 1

# Start server
nohup uv run uvicorn main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
PID=$!
echo $PID > server.pid

echo "Server started with PID: $PID"
echo "Log: $(pwd)/server.log"
echo "Stop: pkill -f 'uvicorn main:app'"
