#!/bin/bash
# Show pairwise comparison experiment status
# Usage: ./status_pairwise.sh [--server URL]

SERVER="${1:-http://localhost:8000}"

# Remove --server prefix if present
if [[ "$1" == "--server" ]]; then
    SERVER="$2"
fi

echo "=== Pairwise Comparison Status ==="
echo ""

# Get task counts
curl -s "$SERVER/pairwise/tasks" | python3 -c "
import sys, json
data = json.load(sys.stdin)
total = data.get('total_tasks', 0)
pending = data.get('pending', 0)
running = data.get('running', 0)
completed = data.get('completed', 0)
failed = data.get('failed', 0)

print(f'Tasks: {completed}/{total} completed ({100*completed/total:.1f}%)' if total > 0 else 'No tasks')
print(f'  Pending:   {pending}')
print(f'  Running:   {running}')
print(f'  Completed: {completed}')
print(f'  Failed:    {failed}')
"

echo ""

# Get summary by epoch
curl -s "$SERVER/pairwise/summary" | python3 -c "
import sys, json
data = json.load(sys.stdin)
n = data.get('n_results', 0)
summary = data.get('summary', {})
by_pm = summary.get('by_prompt_model', {})

if not by_pm:
    print('No results yet')
    sys.exit(0)

# Show Spearman by epoch for each prompt/model
print('=== Spearman by Epoch ===')
for key, pm_data in sorted(by_pm.items()):
    prompt_id = pm_data.get('prompt_id')
    model_short = pm_data.get('model_short')
    n_patterns = pm_data.get('n_patterns')
    epoch_spearman = pm_data.get('epoch_spearman', {})

    print(f'\n{key} (n={n_patterns}):')
    epochs = sorted([int(e) for e in epoch_spearman.keys()])

    for epoch in epochs:
        stats = epoch_spearman[str(epoch)] if str(epoch) in epoch_spearman else epoch_spearman.get(epoch, {})
        mean = stats.get('mean', 0)
        std = stats.get('std', 0)
        print(f'  E{epoch:2d}: {mean:.3f} ± {std:.3f}')
"
