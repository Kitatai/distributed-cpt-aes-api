#!/bin/bash
# Run the API server in background
# Usage: ./run_server.sh [--foreground]

cd "$(dirname "$0")"

LOG_FILE="server.log"
PID_FILE="server.pid"

if [[ "$1" == "--foreground" ]]; then
    echo "Starting server in foreground..."
    uv run python main.py
else
    if [[ -f "$PID_FILE" ]]; then
        OLD_PID=$(cat "$PID_FILE")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            echo "Server already running (PID: $OLD_PID)"
            echo "Use ./stop_server.sh to stop it first"
            exit 1
        fi
    fi

    echo "Starting server in background..."
    nohup uv run python main.py > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Server started (PID: $(cat $PID_FILE))"
    echo "Log: $LOG_FILE"
    echo "Use ./stop_server.sh to stop"
fi
