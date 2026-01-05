#!/bin/bash
# Show few-shot v2 fixed epoch experiment status
# Usage: ./status_fewshot_v2_fixed.sh --epoch 20 [--server URL]

SERVER="http://localhost:8000"
EPOCH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --server)
            SERVER="$2"
            shift 2
            ;;
        --epoch)
            EPOCH="$2"
            shift 2
            ;;
        *)
            # Try to parse as epoch if numeric
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                EPOCH="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$EPOCH" ]; then
    echo "Error: --epoch is required"
    echo "Usage: $0 --epoch 20 [--server http://localhost:8000]"
    exit 1
fi

echo "=== Few-shot v2 Fixed Epoch (e=$EPOCH) Status ==="
echo ""

# Get task counts
curl -s "$SERVER/fewshot_v2_e${EPOCH}/tasks" | python3 -c "
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

# Get summary by k and model
curl -s "$SERVER/fewshot_v2_e${EPOCH}/summary" | python3 -c "
import sys, json
data = json.load(sys.stdin)
n = data.get('n_results', 0)
summary = data.get('summary', {})
by_k_model = summary.get('by_k_model', {})

if not by_k_model:
    print('No results yet')
    sys.exit(0)

k_values = sorted(by_k_model.keys(), key=int)
models = ['llama8b', 'llama3b', 'mistral']

# Header
print('Results by k-shot and model (QWK mean±std):')
print(f\"{'Model':<10}\", end='')
for k in k_values:
    print(f'   {k}-shot E0        {k}-shot Best   ', end='')
print()
print('-' * (10 + 40 * len(k_values)))

# Data rows
for model in models:
    print(f'{model:<10}', end='')
    for k in k_values:
        if model in by_k_model.get(k, {}):
            m = by_k_model[k][model]
            e0 = m['e0_qwk']
            e0_std = m.get('e0_qwk_std', 0)
            best = m['best_qwk']
            best_std = m.get('best_qwk_std', 0)
            print(f'  {e0:.3f}±{e0_std:.3f}       {best:.3f}±{best_std:.3f}  ', end='')
        else:
            print(f'       -             -        ', end='')
    print()

# Delta row
print('-' * (10 + 40 * len(k_values)))
print(f\"{'Δ (avg)':<10}\", end='')
for k in k_values:
    deltas = [(by_k_model[k][m]['delta_qwk'], by_k_model[k][m].get('delta_qwk_std', 0))
              for m in models if m in by_k_model.get(k, {})]
    if deltas:
        avg_delta = sum(d[0] for d in deltas) / len(deltas)
        avg_std = sum(d[1] for d in deltas) / len(deltas)
        print(f'           {avg_delta:+.3f}±{avg_std:.3f}       ', end='')
    else:
        print(f'                  -              ', end='')
print()
"
