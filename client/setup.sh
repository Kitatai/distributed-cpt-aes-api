#!/bin/bash
# Client setup script (NO sudo required)
# Usage: bash setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "CPT-AES Client Setup"
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

# Sync environment with optional CUDA dependencies (includes flash-attn)
echo ""
echo "Installing dependencies (including Flash Attention if CUDA available)..."
if uv sync --extra cuda; then
    echo "All dependencies installed successfully!"
else
    echo "WARNING: Some optional dependencies failed to install."
    echo "Falling back to base dependencies only..."
    uv sync
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start worker:"
echo "  bash run.sh --server http://SERVER_IP:8000"
echo ""
