#!/bin/bash
# Run server
# Usage: bash run.sh [--port PORT]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=${1:-8000}

echo "Starting CPT-AES server on port $PORT..."
uv run uvicorn main:app --host 0.0.0.0 --port "$PORT"
