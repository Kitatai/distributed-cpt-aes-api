#!/bin/bash
# Stop background server
# Usage: bash stop.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/server.pid"

# Try PID file first
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping server (PID: $PID)..."
        kill "$PID"
        rm "$PID_FILE"
        echo "Server stopped."
        exit 0
    else
        echo "Stale PID file, removing..."
        rm "$PID_FILE"
    fi
fi

# Fallback: find and kill uvicorn process
PIDS=$(pgrep -f "uvicorn main:app" 2>/dev/null)
if [ -n "$PIDS" ]; then
    echo "Found running server process(es): $PIDS"
    pkill -f "uvicorn main:app"
    echo "Server stopped."
else
    echo "No server running."
fi
