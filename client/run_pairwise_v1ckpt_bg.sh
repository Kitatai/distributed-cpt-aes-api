#!/bin/bash
# Run pairwise comparison worker with v1 checkpoints in background

cd "$(dirname "$0")"

LOG_DIR="logs/pairwise_v1ckpt"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/worker_${TIMESTAMP}.log"

echo "Starting pairwise comparison worker in background..."
echo "  Server: http://localhost:8000"
echo "  Checkpoint source: backup_zeroshot_v1"
echo "  Log file: $LOG_FILE"

nohup uv run python worker_pairwise_distributed.py \
  --server http://localhost:8000 \
  --exp-name pairwise \
  --checkpoint-source backup_zeroshot_v1 \
  --log-dir "$LOG_DIR" \
  "$@" > "$LOG_FILE" 2>&1 &

PID=$!
echo "Worker started with PID: $PID"
echo "$PID" > "$LOG_DIR/worker.pid"
echo "To monitor: tail -f $LOG_FILE"
echo "To stop: kill $PID"
