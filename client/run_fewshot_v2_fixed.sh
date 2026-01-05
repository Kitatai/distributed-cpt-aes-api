#!/bin/bash
# Distributed worker script for few-shot v2 fixed epoch experiment
# Runs the worker in background with logging
#
# Usage:
#   ./run_fewshot_v2_fixed.sh --server http://SERVER_IP:8000 --epoch 20
#   ./run_fewshot_v2_fixed.sh --server http://192.168.100.10:8000 --epoch 20 --k 1
#   ./run_fewshot_v2_fixed.sh --server http://192.168.100.10:8000 --epoch 20 --model llama8b

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
SERVER=""
EPOCH=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --server)
            SERVER="$2"
            shift 2
            ;;
        --epoch)
            EPOCH="$2"
            shift 2
            ;;
        --k)
            EXTRA_ARGS="$EXTRA_ARGS --k $2"
            shift 2
            ;;
        --model)
            EXTRA_ARGS="$EXTRA_ARGS --model $2"
            shift 2
            ;;
        --prompt)
            EXTRA_ARGS="$EXTRA_ARGS --prompt $2"
            shift 2
            ;;
        --once)
            EXTRA_ARGS="$EXTRA_ARGS --once"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$SERVER" ]; then
    echo "Error: --server is required"
    echo "Usage: $0 --server http://SERVER:8000 --epoch 20"
    exit 1
fi

if [ -z "$EPOCH" ]; then
    echo "Error: --epoch is required"
    echo "Usage: $0 --server http://SERVER:8000 --epoch 20"
    exit 1
fi

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/fewshot_v2_e${EPOCH}_${TIMESTAMP}.log"

echo "=== Few-shot v2 Fixed Epoch Worker (e=$EPOCH) ==="
echo "Server: $SERVER"
echo "Fixed epoch: $EPOCH"
echo "Log file: $LOG_FILE"
echo ""

# Build command
CMD="uv run python worker_fewshot_v2_distributed.py --server $SERVER --fixed-epoch $EPOCH $EXTRA_ARGS"

echo "Running: $CMD"
echo "Starting in background..."
echo ""

# Run in background
nohup $CMD > "$LOG_FILE" 2>&1 &
PID=$!

# Save PID
echo $PID > "$SCRIPT_DIR/worker_e${EPOCH}.pid"

echo "Worker started with PID: $PID"
echo "Log file: $LOG_FILE"
echo ""
echo "To monitor:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To stop:"
echo "  kill $PID"
echo "  # or: kill \$(cat $SCRIPT_DIR/worker_e${EPOCH}.pid)"
