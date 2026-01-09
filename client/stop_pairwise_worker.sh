#!/bin/bash
# Stop pairwise comparison worker
# Usage: ./stop_pairwise_worker.sh

cd "$(dirname "$0")"

PID_FILE="pairwise_worker.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No PID file found. Worker may not be running."
    exit 0
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping worker (PID: $PID)..."
    # Send SIGINT for graceful shutdown
    kill -INT "$PID"
    echo "Waiting for graceful shutdown (will finish current task)..."

    # Wait up to 60 seconds for graceful shutdown
    for i in {1..60}; do
        if ! kill -0 "$PID" 2>/dev/null; then
            break
        fi
        sleep 1
    done

    if kill -0 "$PID" 2>/dev/null; then
        echo "Force killing..."
        kill -9 "$PID"
    fi
    rm -f "$PID_FILE"
    echo "Worker stopped"
else
    echo "Worker not running (stale PID file)"
    rm -f "$PID_FILE"
fi
