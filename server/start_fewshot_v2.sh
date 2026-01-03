#!/bin/bash
# Start server for few-shot v2 distributed experiment
# Generates tasks (if needed) and starts the server in background
#
# Usage:
#   ./start_fewshot_v2.sh                    # Default port 8000
#   ./start_fewshot_v2.sh --port 8080        # Specify port

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
PORT=8000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--port PORT]"
            exit 1
            ;;
    esac
done

echo "=== Few-shot v2 Distributed Server ==="
echo "Port: $PORT"
echo ""

# Check required directories
if [ ! -d "data/checkpoints" ]; then
    echo "Error: data/checkpoints not found"
    echo "Please ensure checkpoints from zero-shot training exist"
    exit 1
fi
echo "Checkpoints: OK"

if [ ! -f "data/sample_patterns_v2.json" ]; then
    echo "Error: data/sample_patterns_v2.json not found"
    echo "Please run: python generate_sample_patterns.py"
    exit 1
fi
echo "Patterns: OK"

if [ ! -d "data/asap" ] || [ ! -f "data/asap/training_set_rel3.tsv" ]; then
    echo "Error: ASAP data not found in data/asap/"
    exit 1
fi
echo "ASAP data: OK"

# Generate tasks if not exist
if [ ! -d "data/tasks_fewshot_v2" ] || [ -z "$(ls -A data/tasks_fewshot_v2 2>/dev/null)" ]; then
    echo ""
    echo "Generating tasks..."
    python generate_fewshot_v2_tasks.py
else
    TASK_COUNT=$(ls -1 data/tasks_fewshot_v2/*.json 2>/dev/null | wc -l)
    echo "Tasks: $TASK_COUNT found"
fi

# Create results directory if not exist
mkdir -p data/results_fewshot_v2

echo ""

# Stop existing server if running
if [ -f "server.pid" ]; then
    OLD_PID=$(cat server.pid)
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping existing server (PID: $OLD_PID)..."
        kill "$OLD_PID"
        sleep 2
    fi
    rm -f server.pid
fi

# Start server in background
LOG_FILE="$SCRIPT_DIR/server_fewshot_v2.log"
echo "Starting server on port $PORT..."
nohup uv run uvicorn main:app --host 0.0.0.0 --port "$PORT" > "$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "$SCRIPT_DIR/server.pid"

# Wait for server to be ready
echo "Waiting for server to start..."
for i in {1..10}; do
    sleep 1
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        break
    fi
done

# Check if server is running
if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo ""
    echo "=== Server started successfully ==="
    echo "PID: $PID"
    echo "URL: http://0.0.0.0:$PORT"
    echo "Log: $LOG_FILE"
    echo ""

    # Show task status
    STATUS=$(curl -s "http://localhost:$PORT/fewshot_v2/tasks")
    PENDING=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('pending',0))" 2>/dev/null || echo "?")
    COMPLETED=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('completed',0))" 2>/dev/null || echo "?")
    TOTAL=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_tasks',0))" 2>/dev/null || echo "?")
    echo "Tasks: $PENDING pending, $COMPLETED completed (total: $TOTAL)"
    echo ""

    echo "Commands:"
    echo "  View logs:    tail -f $LOG_FILE"
    echo "  Check status: curl http://localhost:$PORT/fewshot_v2/tasks"
    echo "  Get summary:  curl http://localhost:$PORT/fewshot_v2/summary"
    echo "  Stop server:  bash stop.sh"
    echo ""
    echo "To start workers on client machines:"
    echo "  ./run_fewshot_v2_distributed.sh --server http://YOUR_IP:$PORT"
else
    echo "Error: Server failed to start"
    echo "Check logs: $LOG_FILE"
    exit 1
fi
