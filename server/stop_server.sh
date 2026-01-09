#!/bin/bash
# Stop the API server
# Usage: ./stop_server.sh

cd "$(dirname "$0")"

PID_FILE="server.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No PID file found. Server may not be running."
    exit 0
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping server (PID: $PID)..."
    kill "$PID"
    sleep 2
    if kill -0 "$PID" 2>/dev/null; then
        echo "Force killing..."
        kill -9 "$PID"
    fi
    rm -f "$PID_FILE"
    echo "Server stopped"
else
    echo "Server not running (stale PID file)"
    rm -f "$PID_FILE"
fi
