#!/bin/bash
# Run worker
# Usage: bash run.sh --server http://SERVER_IP:8000 [--dataset toefl11|asap] [--task-id TASK_ID] [--single]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting CPT-AES worker..."
uv run python worker.py "$@"
