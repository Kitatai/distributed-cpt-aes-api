#!/bin/bash
# Reset experiment: clear results/checkpoints and reset tasks
# Usage: bash reset_experiment.sh [commit_message]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

echo "=== Resetting Experiment ==="

# Clear checkpoints and results
echo "Clearing checkpoints..."
rm -rf "$DATA_DIR/checkpoints/"*
echo "Clearing results..."
rm -rf "$DATA_DIR/results/"*

# Reset all tasks
echo "Resetting tasks..."
python3 -c "
import json
from pathlib import Path

tasks_dir = Path('$DATA_DIR/tasks')
count = 0
for f in tasks_dir.glob('*.json'):
    with open(f) as fp:
        t = json.load(fp)
    t['status'] = 'pending'
    t['worker_id'] = None
    t['started_at'] = None
    t['completed_at'] = None
    t['last_completed_epoch'] = -1
    t['error_message'] = None
    with open(f, 'w') as fp:
        json.dump(t, fp, indent=2)
    count += 1
print(f'Reset {count} tasks')
"

# Show current config
echo ""
echo "=== Current Config ==="
grep -E "lr:|grad_accum_steps:" "$SCRIPT_DIR/../client/src/config.py" | head -2

# Commit and push if message provided
if [ -n "$1" ]; then
    echo ""
    echo "=== Committing & Pushing ==="
    cd "$SCRIPT_DIR/.."
    git add -A
    git commit -m "$1

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
    git push origin main
    echo "Pushed to origin/main"
else
    echo ""
    echo "No commit message provided. To commit, run:"
    echo "  git add -A && git commit -m 'your message' && git push origin main"
fi

echo ""
echo "=== Done ==="
echo "Workers need to: git pull && bash run_background.sh --server http://SERVER_IP:8000"
