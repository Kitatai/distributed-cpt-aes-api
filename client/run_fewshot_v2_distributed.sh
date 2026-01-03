#!/bin/bash
# Distributed worker script for few-shot v2 experiment
# Runs the worker in background with logging
#
# Usage:
#   ./run_fewshot_v2_distributed.sh --server http://SERVER_IP:8000
#   ./run_fewshot_v2_distributed.sh --server http://192.168.100.10:8000 --k 1
#   ./run_fewshot_v2_distributed.sh --server http://192.168.100.10:8000 --model llama8b

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/fewshot_v2_distributed_${TIMESTAMP}.log"

echo "=== Few-shot v2 Distributed Worker ==="
echo "Log file: $LOG_FILE"
echo ""

# Build command with all arguments
CMD="uv run python worker_fewshot_v2_distributed.py"

while [[ $# -gt 0 ]]; do
    case $1 in
        --server)
            CMD="$CMD --server $2"
            echo "Server: $2"
            shift 2
            ;;
        --k)
            CMD="$CMD --k $2"
            echo "Filter k: $2"
            shift 2
            ;;
        --model)
            CMD="$CMD --model $2"
            echo "Filter model: $2"
            shift 2
            ;;
        --prompt)
            CMD="$CMD --prompt $2"
            echo "Filter prompt: $2"
            shift 2
            ;;
        --once)
            CMD="$CMD --once"
            echo "Run once: yes"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "Running: $CMD"
echo "Starting in background..."
echo ""

# Run in background
nohup $CMD > "$LOG_FILE" 2>&1 &
PID=$!

echo "Worker started with PID: $PID"
echo "Log file: $LOG_FILE"
echo ""
echo "To monitor:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To stop:"
echo "  kill $PID"
