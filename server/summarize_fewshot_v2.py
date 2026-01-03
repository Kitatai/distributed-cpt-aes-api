#!/usr/bin/env python3
"""
Summarize few-shot v2 experiment results.

Aggregates results across patterns and generates per-model summaries.

Usage:
    python summarize_fewshot_v2.py [--results-dir DIR]
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Summarize few-shot v2 results")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Results directory (default: data/backup_zeroshot_v3/results_fewshot_v2)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: results_dir/summary.json)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = script_dir / "data" / "backup_zeroshot_v3" / "results_fewshot_v2"

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Load all results
    all_results = []
    for task_dir in sorted(results_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        summary_file = task_dir / "summary.json"
        if summary_file.exists():
            result = load_json(summary_file)
            all_results.append(result)

    if not all_results:
        print("No results found")
        return

    print(f"Loaded {len(all_results)} results")

    # Group by k value
    k_values = sorted(set(r['k'] for r in all_results))
    models = ["llama8b", "llama3b", "mistral"]

    summary = {
        'metadata': {
            'n_results': len(all_results),
            'k_values': k_values,
            'generated_at': datetime.now().isoformat(),
        },
        'by_k': {},
        'by_k_model': {},
        'by_k_model_prompt': {},
    }

    for k in k_values:
        k_results = [r for r in all_results if r['k'] == k]
        summary['by_k'][k] = {}
        summary['by_k_model'][k] = {}
        summary['by_k_model_prompt'][k] = {}

        # Overall for this k
        e0_qwks = [r['epoch_0']['qwk'] for r in k_results]
        best_qwks = [r['best_epoch_metrics']['qwk'] for r in k_results]
        summary['by_k'][k] = {
            'e0_qwk': float(np.mean(e0_qwks)),
            'best_qwk': float(np.mean(best_qwks)),
            'delta_qwk': float(np.mean(best_qwks) - np.mean(e0_qwks)),
            'n_tasks': len(k_results),
        }

        # Per model for this k
        for model in models:
            model_results = [r for r in k_results if r['model_short_name'] == model]
            if not model_results:
                continue

            e0_qwks = [r['epoch_0']['qwk'] for r in model_results]
            best_qwks = [r['best_epoch_metrics']['qwk'] for r in model_results]

            summary['by_k_model'][k][model] = {
                'e0_qwk': float(np.mean(e0_qwks)),
                'best_qwk': float(np.mean(best_qwks)),
                'delta_qwk': float(np.mean(best_qwks) - np.mean(e0_qwks)),
                'n_tasks': len(model_results),
            }

            # Per prompt for this k and model
            summary['by_k_model_prompt'][k][model] = {}
            for prompt_id in range(1, 9):
                prompt_results = [r for r in model_results if r['prompt_id'] == prompt_id]
                if prompt_results:
                    e0_qwks = [r['epoch_0']['qwk'] for r in prompt_results]
                    best_qwks = [r['best_epoch_metrics']['qwk'] for r in prompt_results]
                    summary['by_k_model_prompt'][k][model][f'prompt{prompt_id}'] = {
                        'e0_qwk': float(np.mean(e0_qwks)),
                        'best_qwk': float(np.mean(best_qwks)),
                        'delta_qwk': float(np.mean(best_qwks) - np.mean(e0_qwks)),
                        'n_patterns': len(prompt_results),
                    }

    # Save summary
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = results_dir / "summary.json"

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {output_file}")

    # Print summary tables
    print("\n" + "=" * 90)
    print("Few-shot v2 Results Summary (model averages across 8 prompts)")
    print("=" * 90)

    for k in k_values:
        print(f"\n--- {k}-shot ---")
        print(f"{'Model':<12} {'E0 QWK':>10} {'Best QWK':>10} {'Δ QWK':>10} {'N':>6}")
        print("-" * 50)

        for model in models:
            if model in summary['by_k_model'][k]:
                s = summary['by_k_model'][k][model]
                print(f"{model:<12} {s['e0_qwk']:>10.4f} {s['best_qwk']:>10.4f} {s['delta_qwk']:>+10.4f} {s['n_tasks']:>6}")

        print("-" * 50)
        s = summary['by_k'][k]
        print(f"{'Overall':<12} {s['e0_qwk']:>10.4f} {s['best_qwk']:>10.4f} {s['delta_qwk']:>+10.4f} {s['n_tasks']:>6}")

    # Comparison across k values
    print("\n" + "=" * 90)
    print("Comparison: Best QWK by k and model")
    print("=" * 90)
    print(f"{'Model':<12}", end="")
    for k in k_values:
        print(f"  {k}-shot", end="")
    print()
    print("-" * (12 + 10 * len(k_values)))

    for model in models:
        print(f"{model:<12}", end="")
        for k in k_values:
            if model in summary['by_k_model'][k]:
                qwk = summary['by_k_model'][k][model]['best_qwk']
                print(f"  {qwk:.4f}", end="")
            else:
                print(f"  {'N/A':>6}", end="")
        print()

    print("-" * (12 + 10 * len(k_values)))
    print(f"{'Overall':<12}", end="")
    for k in k_values:
        qwk = summary['by_k'][k]['best_qwk']
        print(f"  {qwk:.4f}", end="")
    print()


if __name__ == "__main__":
    main()
