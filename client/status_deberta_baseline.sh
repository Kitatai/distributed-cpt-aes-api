#!/bin/bash
# Show DeBERTa baseline experiment status
# Usage: ./status_deberta_baseline.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="$SCRIPT_DIR/../server/data/results_deberta_baseline"

echo "=== DeBERTa Baseline Status ==="
echo ""

if [ ! -d "$RESULTS_DIR" ]; then
    echo "No results yet (directory not found)"
    exit 0
fi

# Count completed tasks
TOTAL_EXPECTED=$((8 * 10 * 4))  # 8 prompts × 10 patterns × 4 splits = 320
COMPLETED=$(ls -1 "$RESULTS_DIR"/*.json 2>/dev/null | grep -v summary | wc -l)

if [ "$COMPLETED" -eq 0 ]; then
    echo "Tasks: 0/$TOTAL_EXPECTED completed (0.0%)"
    echo ""
    echo "No results yet"
    exit 0
fi

PERCENT=$(echo "scale=1; $COMPLETED * 100 / $TOTAL_EXPECTED" | bc)
echo "Tasks: $COMPLETED/$TOTAL_EXPECTED completed ($PERCENT%)"
echo ""

# Generate summary from individual results
python3 -c "
import json
from pathlib import Path
import numpy as np

results_dir = Path('$RESULTS_DIR')
results = []
for f in results_dir.glob('*.json'):
    if f.name == 'summary.json':
        continue
    try:
        with open(f) as fp:
            results.append(json.load(fp))
    except:
        pass

if not results:
    print('No results yet')
    exit(0)

# Group by split
splits_data = {}
for r in results:
    split = r.get('split_ratio', 'unknown')
    if split not in splits_data:
        splits_data[split] = []
    splits_data[split].append(r)

# Header
print('Results by train:val split (QWK mean±std):')
print(f\"{'Split':<10} {'QWK':>15} {'Spearman(raw)':>18} {'Spearman(round)':>18} {'N':>6}\")
print('-' * 70)

split_order = ['5:7', '7:5', '9:3', '11:1']
for split in split_order:
    if split in splits_data:
        rs = splits_data[split]
        qwks = [r['best_epoch_metrics']['qwk'] for r in rs]
        sp_raws = [r['best_epoch_metrics']['spearman_raw'] for r in rs]
        sp_rounds = [r['best_epoch_metrics']['spearman_rounded'] for r in rs]

        qwk_str = f'{np.mean(qwks):.3f}±{np.std(qwks):.3f}'
        sp_raw_str = f'{np.mean(sp_raws):.3f}±{np.std(sp_raws):.3f}'
        sp_round_str = f'{np.mean(sp_rounds):.3f}±{np.std(sp_rounds):.3f}'
        print(f'{split:<10} {qwk_str:>15} {sp_raw_str:>18} {sp_round_str:>18} {len(rs):>6}')

# Per-prompt breakdown for the first split that has data
print('')
print('Per-prompt breakdown (QWK, first available split):')
first_split = None
for split in split_order:
    if split in splits_data and len(splits_data[split]) >= 8:
        first_split = split
        break

if first_split:
    print(f\"{'Prompt':<10}  {first_split} (train:val)\")
    print('-' * 30)

    for prompt_id in range(1, 9):
        prompt_results = [r for r in splits_data[first_split] if r['prompt_id'] == prompt_id]
        if prompt_results:
            qwks = [r['best_epoch_metrics']['qwk'] for r in prompt_results]
            print(f'Prompt {prompt_id:<3} {np.mean(qwks):.3f}±{np.std(qwks):.3f}')
"
