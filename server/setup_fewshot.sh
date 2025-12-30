#!/bin/bash
# Setup few-shot experiment tasks
# Run this on the server before starting client workers

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Generating few-shot tasks..."
uv run python generate_fewshot_tasks.py

echo ""
echo "Tasks created in: data/tasks_fewshot/"
ls -la data/tasks_fewshot/

echo ""
echo "Setup complete. Now run the client worker on GPU machines."
