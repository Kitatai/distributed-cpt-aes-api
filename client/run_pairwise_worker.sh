#!/bin/bash
# Run pairwise comparison worker in background
# Usage: ./run_pairwise_worker.sh --server http://SERVER:8000 [--foreground]

cd "$(dirname "$0")"

SERVER=""
FOREGROUND=false
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --server)
            SERVER="$2"
            shift 2
            ;;
        --foreground)
            FOREGROUND=true
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

if [[ -z "$SERVER" ]]; then
    echo "Error: --server is required"
    echo "Usage: $0 --server http://SERVER:8000 [--foreground]"
    exit 1
fi

LOG_FILE="pairwise_worker.log"
PID_FILE="pairwise_worker.pid"

if [[ "$FOREGROUND" == "true" ]]; then
    echo "Starting pairwise worker in foreground..."
    echo "Server: $SERVER"
    uv run python worker_pairwise_distributed.py --server "$SERVER" $EXTRA_ARGS
else
    if [[ -f "$PID_FILE" ]]; then
        OLD_PID=$(cat "$PID_FILE")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            echo "Worker already running (PID: $OLD_PID)"
            echo "Use ./stop_pairwise_worker.sh to stop it first"
            exit 1
        fi
    fi

    echo "Starting pairwise worker in background..."
    echo "Server: $SERVER"
    nohup uv run python worker_pairwise_distributed.py --server "$SERVER" $EXTRA_ARGS > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Worker started (PID: $(cat $PID_FILE))"
    echo "Log: $LOG_FILE"
    echo "Use ./stop_pairwise_worker.sh to stop"
fi
