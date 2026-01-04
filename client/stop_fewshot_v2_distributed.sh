#!/bin/bash
# Stop few-shot v2 distributed worker(s)
# Usage: ./stop_fewshot_v2_distributed.sh

echo "=== Stopping Few-shot v2 Workers ==="

# Find worker processes
PIDS=$(pgrep -f "worker_fewshot_v2_distributed.py" 2>/dev/null)

if [ -z "$PIDS" ]; then
    echo "No workers running."
else
    echo "Found worker process(es): $PIDS"
    for PID in $PIDS; do
        echo "Stopping PID $PID..."
        kill "$PID" 2>/dev/null
    done
    echo "Workers stopped."
fi
