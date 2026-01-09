#!/bin/bash
# Stop pairwise comparison worker

cd "$(dirname "$0")"

PID_FILE="logs/pairwise_v1ckpt/worker.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping worker with PID: $PID"
        kill "$PID"
        rm -f "$PID_FILE"
        echo "Worker stopped"
    else
        echo "Worker with PID $PID is not running"
        rm -f "$PID_FILE"
    fi
else
    echo "No PID file found. Worker may not be running."
    echo "Try: pkill -f 'worker_pairwise_distributed.py'"
fi
