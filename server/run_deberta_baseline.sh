#!/bin/bash
# Run DeBERTa baseline experiment
# Usage:
#   ./run_deberta_baseline.sh                    # Run all
#   ./run_deberta_baseline.sh --prompt 1         # Run specific prompt
#   ./run_deberta_baseline.sh --split 5:10       # Run specific split

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/deberta_baseline_${TIMESTAMP}.log"

echo "=== DeBERTa Baseline Experiment ==="
echo "Log file: $LOG_FILE"
echo ""

# Check data
if [ ! -f "data/sample_patterns_v2.json" ]; then
    echo "Error: data/sample_patterns_v2.json not found"
    exit 1
fi

if [ ! -f "data/asap/training_set_rel3.tsv" ]; then
    echo "Error: ASAP data not found"
    exit 1
fi

# Build command
CMD="uv run python run_deberta_baseline.py"

while [[ $# -gt 0 ]]; do
    case $1 in
        --prompt)
            CMD="$CMD --prompt $2"
            echo "Prompt: $2"
            shift 2
            ;;
        --pattern)
            CMD="$CMD --pattern $2"
            echo "Pattern: $2"
            shift 2
            ;;
        --split)
            CMD="$CMD --split $2"
            echo "Split: $2"
            shift 2
            ;;
        --max-epochs)
            CMD="$CMD --max-epochs $2"
            echo "Max epochs: $2"
            shift 2
            ;;
        --fg)
            FG=1
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

if [ "$FG" == "1" ]; then
    # Foreground mode
    $CMD 2>&1 | tee "$LOG_FILE"
else
    # Background mode
    echo "Starting in background..."
    nohup $CMD > "$LOG_FILE" 2>&1 &
    PID=$!

    echo ""
    echo "Started with PID: $PID"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "To monitor:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "To stop:"
    echo "  kill $PID"
fi
