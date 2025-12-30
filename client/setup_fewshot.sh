#!/bin/bash
# Setup client for few-shot experiment
# Run this after git pull on the client machine

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up few-shot experiment client..."

# Ensure venv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Sync dependencies
echo "Syncing dependencies..."
uv sync

echo ""
echo "Setup complete."
echo ""
echo "Usage:"
echo "  ./run_fewshot.sh              # Run all tasks"
echo "  ./run_fewshot.sh <task_id>    # Run specific task"
echo "  ./stop_fewshot.sh             # Stop running worker"
echo "  tail -f fewshot.log           # Monitor progress"
