#!/bin/bash
# Show DeBERTa baseline experiment status
# Usage: ./status_deberta_baseline.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="data/results_deberta_baseline"

echo "=== DeBERTa Baseline Status ==="
echo ""

if [ ! -d "$RESULTS_DIR" ]; then
    echo "No results yet (directory not found)"
    exit 0
fi

# Count completed tasks
TOTAL_EXPECTED=$((8 * 10 * 5))  # 8 prompts × 10 patterns × 5 splits = 400
COMPLETED=$(ls -1 "$RESULTS_DIR"/*.json 2>/dev/null | grep -v summary | wc -l)

echo "Tasks: $COMPLETED/$TOTAL_EXPECTED completed ($(echo "scale=1; $COMPLETED * 100 / $TOTAL_EXPECTED" | bc)%)"
echo ""

# Show summary if exists
if [ -f "$RESULTS_DIR/summary.json" ]; then
    python3 -c "
import json
with open('$RESULTS_DIR/summary.json') as f:
    summary = json.load(f)

by_split = summary.get('by_split', {})
if not by_split:
    print('No summary data yet')
else:
    print('Results by train:val split (QWK mean±std):')
    print(f\"{'Split':<10} {'QWK':>15} {'Spearman(raw)':>18} {'Spearman(round)':>18} {'N':>6}\")
    print('-' * 70)

    for split in ['5:10', '7:8', '9:6', '11:4', '13:2']:
        if split in by_split:
            d = by_split[split]
            qwk = f\"{d['qwk_mean']:.3f}±{d['qwk_std']:.3f}\"
            sp_raw = f\"{d['spearman_raw_mean']:.3f}±{d['spearman_raw_std']:.3f}\"
            sp_round = f\"{d['spearman_rounded_mean']:.3f}±{d['spearman_rounded_std']:.3f}\"
            print(f'{split:<10} {qwk:>15} {sp_raw:>18} {sp_round:>18} {d[\"n_tasks\"]:>6}')
"
else
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
    with open(f) as fp:
        results.append(json.load(fp))

if not results:
    print('No results yet')
else:
    # Group by split
    splits = {}
    for r in results:
        split = r['split_ratio']
        if split not in splits:
            splits[split] = []
        splits[split].append(r)

    print('Results by train:val split (QWK mean±std):')
    print(f\"{'Split':<10} {'QWK':>15} {'Spearman(raw)':>18} {'N':>6}\")
    print('-' * 55)

    for split in ['5:10', '7:8', '9:6', '11:4', '13:2']:
        if split in splits:
            rs = splits[split]
            qwks = [r['best_epoch_metrics']['qwk'] for r in rs]
            sp_raws = [r['best_epoch_metrics']['spearman_raw'] for r in rs]
            qwk = f'{np.mean(qwks):.3f}±{np.std(qwks):.3f}'
            sp_raw = f'{np.mean(sp_raws):.3f}±{np.std(sp_raws):.3f}'
            print(f'{split:<10} {qwk:>15} {sp_raw:>18} {len(rs):>6}')
"
fi
