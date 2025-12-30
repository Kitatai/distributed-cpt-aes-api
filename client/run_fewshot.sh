#!/bin/bash
# Run few-shot experiment in background
# Usage: ./run_fewshot.sh [task_id]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$(dirname "$SCRIPT_DIR")/server/data"
LOG_FILE="$SCRIPT_DIR/fewshot.log"
PID_FILE="$SCRIPT_DIR/fewshot.pid"

cd "$SCRIPT_DIR"

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Few-shot worker already running (PID: $PID)"
        echo "Use ./stop_fewshot.sh to stop it first"
        exit 1
    fi
fi

if [ -n "$1" ]; then
    echo "Starting few-shot experiment for task: $1"
    nohup uv run python worker_fewshot.py --server-dir "$SERVER_DIR" --task-id "$1" > "$LOG_FILE" 2>&1 &
else
    echo "Starting all few-shot experiments"
    nohup uv run python worker_fewshot.py --server-dir "$SERVER_DIR" > "$LOG_FILE" 2>&1 &
fi

echo $! > "$PID_FILE"
echo "Started with PID: $(cat $PID_FILE)"
echo "Log file: $LOG_FILE"
echo ""
echo "To monitor: tail -f $LOG_FILE"
echo "To stop: ./stop_fewshot.sh"
