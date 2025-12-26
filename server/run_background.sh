#!/bin/bash
# Run server in background (survives SSH disconnect)
# Usage: bash run_background.sh [PORT]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=${1:-8000}
LOG_FILE="$SCRIPT_DIR/server.log"

echo "Starting CPT-AES server on port $PORT in background..."
echo "Log file: $LOG_FILE"

nohup uv run uvicorn main:app --host 0.0.0.0 --port "$PORT" > "$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "$SCRIPT_DIR/server.pid"

echo "Server started with PID: $PID"
echo ""
echo "Commands:"
echo "  View logs:    tail -f $LOG_FILE"
echo "  Stop server:  bash stop.sh"
echo "  Check status: curl http://localhost:$PORT/health"
