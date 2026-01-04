#!/usr/bin/env python3
"""
Generate LaTeX table row for DeBERTa baseline results.

Selects the split with the best average QWK and outputs per-prompt values.

Usage:
    python generate_latex_deberta.py
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np


def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / "data" / "results_deberta_baseline"

    if not results_dir.exists():
        print("Error: Results directory not found")
        return

    # Load all results
    results = []
    for f in results_dir.glob("*.json"):
        if f.name == "summary.json":
            continue
        with open(f) as fp:
            results.append(json.load(fp))

    if not results:
        print("Error: No results found")
        return

    # Group by split
    splits_data = defaultdict(list)
    for r in results:
        split = r.get('split_ratio', 'unknown')
        splits_data[split].append(r)

    # Find best split by average QWK
    best_split = None
    best_avg_qwk = -1

    for split, rs in splits_data.items():
        qwks = [r['best_epoch_metrics']['qwk'] for r in rs]
        avg_qwk = np.mean(qwks)
        if avg_qwk > best_avg_qwk:
            best_avg_qwk = avg_qwk
            best_split = split

    print(f"% DeBERTa baseline results (best split: {best_split})")
    print(f"% Average QWK: {best_avg_qwk:.3f}")
    print()

    # Calculate per-prompt QWK for best split
    best_results = splits_data[best_split]
    prompt_qwks = {}

    for prompt_id in range(1, 9):
        prompt_results = [r for r in best_results if r['prompt_id'] == prompt_id]
        if prompt_results:
            qwks = [r['best_epoch_metrics']['qwk'] for r in prompt_results]
            prompt_qwks[prompt_id] = np.mean(qwks)
        else:
            prompt_qwks[prompt_id] = 0

    # Calculate average
    avg_qwk = np.mean(list(prompt_qwks.values()))

    # Format output (without leading zero)
    def fmt_qwk(v):
        return f"\\qwk{{{v:.3f}}}".replace("0.", ".")

    qwk_strs = [fmt_qwk(prompt_qwks[p]) for p in range(1, 9)]
    avg_str = fmt_qwk(avg_qwk)

    line = f"& DeBERTa & Sup. & {' & '.join(qwk_strs)} & {avg_str} \\\\"
    print(line)


if __name__ == "__main__":
    main()
