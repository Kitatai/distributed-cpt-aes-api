#!/bin/bash
# Server setup script
# Usage: bash setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "CPT-AES Server Setup"
echo "=========================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed."
    echo "Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "uv version: $(uv --version)"

# Create Python version file
if [ ! -f ".python-version" ]; then
    echo "3.12" > .python-version
    echo "Created .python-version (Python 3.12)"
fi

# Create data directories
mkdir -p data/asap data/tasks data/checkpoints data/results

# Sync environment
echo ""
echo "Installing dependencies..."
uv sync

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy ASAP data to data/asap/training_set_rel3.tsv"
echo "2. Initialize tasks: uv run python main.py (then POST /tasks/init)"
echo "3. Start server: bash run.sh"
echo ""
