#!/bin/bash
# Worker script for few-shot v2 experiment
# Runs the worker in background with logging

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Data directory (backup_zeroshot_v1)
DATA_DIR="$SCRIPT_DIR/../server/data/backup_zeroshot_v1"

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/fewshot_v2_${TIMESTAMP}.log"

echo "=== Few-shot v2 Experiment Worker ==="
echo "Data directory: $DATA_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Check if venv exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment"
fi

# Check if checkpoints exist
if [ ! -d "$DATA_DIR/checkpoints" ]; then
    echo "Error: Checkpoints not found at $DATA_DIR/checkpoints"
    echo "Please run the setup script first (from server directory):"
    echo "  ./run_fewshot_v2_setup.sh"
    exit 1
fi

# Check if tasks exist
if [ ! -d "$DATA_DIR/tasks_fewshot_v2" ] || [ -z "$(ls -A $DATA_DIR/tasks_fewshot_v2 2>/dev/null)" ]; then
    echo "Error: No tasks found at $DATA_DIR/tasks_fewshot_v2"
    echo "Please run the setup script first (from server directory):"
    echo "  ./run_fewshot_v2_setup.sh"
    exit 1
fi

# Parse optional arguments
TASK_ID="${1:-}"
PROMPT="${2:-}"

CMD="python worker_fewshot_v2.py --data-dir $DATA_DIR"

if [ -n "$TASK_ID" ]; then
    CMD="$CMD --task-id $TASK_ID"
elif [ -n "$PROMPT" ]; then
    CMD="$CMD --prompt $PROMPT"
fi

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
