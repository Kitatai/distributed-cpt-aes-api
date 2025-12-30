#!/bin/bash
# Stop few-shot experiment

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/fewshot.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Stopping few-shot worker (PID: $PID)..."
        kill "$PID"
        sleep 2
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Force killing..."
            kill -9 "$PID"
        fi
        echo "Stopped"
    else
        echo "Process not running"
    fi
    rm -f "$PID_FILE"
else
    echo "No PID file found"
    # Try to find and kill any running worker
    pkill -f "worker_fewshot.py" && echo "Killed worker_fewshot.py processes"
fi
