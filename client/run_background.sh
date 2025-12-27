#!/bin/bash
# Run worker in background (survives SSH disconnect)
# Usage: bash run_background.sh --server http://SERVER_IP:8000 [--dataset toefl11|asap]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="$SCRIPT_DIR/worker.log"

echo "Starting CPT-AES worker in background..."
echo "Log file: $LOG_FILE"

nohup uv run python worker.py "$@" > "$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "$SCRIPT_DIR/worker.pid"

echo "Worker started with PID: $PID"
echo ""
echo "Commands:"
echo "  View logs:    tail -f $LOG_FILE"
echo "  Stop worker:  bash stop.sh"
