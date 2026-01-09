#!/bin/bash
# Run pairwise comparison worker with v1 checkpoints

cd "$(dirname "$0")"

echo "Starting pairwise comparison worker..."
echo "  Server: http://localhost:8000"
echo "  Checkpoint source: backup_zeroshot_v1"
echo "  Log dir: logs/pairwise_v1ckpt"

uv run python worker_pairwise_distributed.py \
  --server http://localhost:8000 \
  --exp-name pairwise \
  --checkpoint-source backup_zeroshot_v1 \
  --log-dir logs/pairwise_v1ckpt \
  "$@"
