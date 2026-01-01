#!/bin/bash
# Setup script for few-shot v2 experiment
# Generates tasks for the experiment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Few-shot v2 Experiment Setup ==="
echo "Data directory: $SCRIPT_DIR/data/backup_zeroshot_v1"
echo ""

# Check if checkpoints exist
if [ ! -d "data/backup_zeroshot_v1/checkpoints" ]; then
    echo "Error: Checkpoints not found!"
    echo "Please extract checkpoints.zip first:"
    echo "  cd data && unzip -q checkpoints.zip -d backup_zeroshot_v1/"
    exit 1
fi

# Parse arguments (default: all prompts, 1 pattern, llama8b)
PROMPTS="${1:-1,2,3,4,5,6,7,8}"
PATTERNS="${2:-1}"
MODEL="${3:-llama8b}"

echo "Generating tasks..."
echo "  Prompts: $PROMPTS"
echo "  Patterns: $PATTERNS"
echo "  Model: $MODEL"
echo ""

python generate_fewshot_v2_tasks.py \
    --prompts "$PROMPTS" \
    --patterns "$PATTERNS" \
    --model "$MODEL"

echo ""
echo "=== Setup complete ==="
echo "Tasks created in: data/backup_zeroshot_v1/tasks_fewshot_v2/"
echo ""
echo "To run the experiment, go to the client directory and run:"
echo "  ./run_fewshot_v2_worker.sh"
