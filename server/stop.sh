#!/bin/bash
# Stop background server
# Usage: bash stop.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/server.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping server (PID: $PID)..."
        kill "$PID"
        rm "$PID_FILE"
        echo "Server stopped."
    else
        echo "Server not running (stale PID file)."
        rm "$PID_FILE"
    fi
else
    echo "No PID file found. Server may not be running."
    echo "Try: pkill -f 'uvicorn main:app'"
fi
