#!/bin/bash
# Stop background worker
# Usage: bash stop.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/worker.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping worker (PID: $PID) and child processes..."
        # Kill the entire process tree (SIGTERM for graceful shutdown)
        pkill -TERM -P "$PID" 2>/dev/null
        kill -TERM "$PID" 2>/dev/null

        # Wait for graceful shutdown (up to 10 seconds)
        for i in {1..10}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done

        # Force kill if still running
        if kill -0 "$PID" 2>/dev/null; then
            echo "Force killing..."
            pkill -KILL -P "$PID" 2>/dev/null
            kill -KILL "$PID" 2>/dev/null
        fi

        rm -f "$PID_FILE"
        echo "Worker stopped."
    else
        echo "Worker not running (stale PID file)."
        rm -f "$PID_FILE"
    fi
else
    echo "No PID file found. Worker may not be running."
    echo "Checking for orphaned worker processes..."
    if pgrep -f "python.*worker.py" > /dev/null; then
        echo "Found orphaned worker. Stopping..."
        pkill -TERM -f "python.*worker.py"
        sleep 2
        pkill -KILL -f "python.*worker.py" 2>/dev/null
        echo "Done."
    else
        echo "No worker processes found."
    fi
fi
