#!/bin/bash
# Stop background worker
# Usage: bash stop.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/worker.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping worker (PID: $PID)..."
        kill "$PID"
        rm "$PID_FILE"
        echo "Worker stopped."
    else
        echo "Worker not running (stale PID file)."
        rm "$PID_FILE"
    fi
else
    echo "No PID file found. Worker may not be running."
    echo "Try: pkill -f 'python worker.py'"
fi
