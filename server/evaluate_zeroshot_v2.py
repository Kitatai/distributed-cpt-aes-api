#!/usr/bin/env python3
"""
Evaluate zero-shot results using new sample patterns (v2).

Uses existing prediction files from zero-shot training to:
1. Select best epoch based on MSE on dev_ids
2. Evaluate QWK on test_ids for epoch 0 and best epoch

Usage:
    python evaluate_zeroshot_v2.py [--n-patterns N] [--output DIR]
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(y_true, y_pred, min_score, max_score):
    """Calculate QWK."""
    y_true = np.clip(y_true, min_score, max_score).astype(int)
    y_pred = np.clip(y_pred, min_score, max_score).astype(int)
    return cohen_kappa_score(y_true, y_pred, weights='quadratic',
                             labels=list(range(min_score, max_score + 1)))


def load_predictions(results_dir: Path, task_id: str, epoch: int) -> pd.DataFrame:
    """Load predictions for a specific epoch."""
    pred_file = results_dir / task_id / f"predictions_epoch_{epoch}.csv"
    if not pred_file.exists():
        return None
    return pd.read_csv(pred_file)


def get_score_range(prompt_id: int) -> tuple:
    """Get min and max scores for a prompt."""
    score_ranges = {
        1: (2, 12), 2: (1, 6), 3: (0, 3), 4: (0, 3),
        5: (0, 4), 6: (0, 4), 7: (0, 30), 8: (0, 60),
    }
    return score_ranges[prompt_id]


def evaluate_on_ids(df: pd.DataFrame, essay_ids: list, prompt_id: int) -> dict:
    """Evaluate predictions on specific essay IDs."""
    subset = df[df['essay_id'].isin(essay_ids)].copy()
    if len(subset) == 0:
        return None

    y_true = subset['y_true'].values
    y_pred = subset['y_hat_greedy'].values

    # Remove NaN predictions
    valid_mask = ~np.isnan(y_pred)
    if valid_mask.sum() == 0:
        return None

    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    min_score, max_score = get_score_range(prompt_id)

    mse = np.mean((y_true - y_pred) ** 2)
    qwk = quadratic_weighted_kappa(y_true, y_pred, min_score, max_score)
    spearman = spearmanr(y_true, y_pred)[0]

    return {
        'mse': float(mse),
        'qwk': float(qwk),
        'spearman': float(spearman),
        'n_samples': int(len(y_true)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate zero-shot with new patterns")
    parser.add_argument("--n-patterns", type=int, default=10,
                        help="Number of patterns to evaluate (default: 10)")
    parser.add_argument("--n-dev", type=int, default=10,
                        help="Number of dev samples to use for epoch selection (default: 10, max: 10)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: data/results_zeroshot_v2)")
    parser.add_argument("--max-epochs", type=int, default=30,
                        help="Maximum epoch number (default: 30)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # Load patterns
    patterns_path = script_dir / "data" / "sample_patterns_v2.json"
    print(f"Loading patterns from {patterns_path}")
    with open(patterns_path) as f:
        patterns_data = json.load(f)
    patterns = patterns_data['patterns']

    # Results directory (existing zero-shot results)
    results_dir = script_dir / "data" / "backup_zeroshot_v3" / "results"

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = script_dir / "data" / "results_zeroshot_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Models
    models = [
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", "llama8b"),
        ("meta-llama/Llama-3.2-3B-Instruct", "llama3b"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
    ]

    all_results = []

    for model_name, model_short in models:
        for prompt_id in range(1, 9):
            task_id = f"prompt{prompt_id}_{model_short}"
            prompt_key = f"prompt{prompt_id}"

            print(f"\nProcessing {task_id}...")

            for pattern_idx in range(args.n_patterns):
                pattern = patterns[prompt_key][pattern_idx]
                test_ids = pattern['test_ids']
                dev_ids = pattern['dev_ids'][:args.n_dev]  # Use first n_dev samples

                # Evaluate all epochs on dev_ids to find best epoch
                epoch_mse = {}
                for epoch in range(0, args.max_epochs + 1):
                    df = load_predictions(results_dir, task_id, epoch)
                    if df is None:
                        continue

                    metrics = evaluate_on_ids(df, dev_ids, prompt_id)
                    if metrics:
                        epoch_mse[epoch] = metrics['mse']

                if not epoch_mse:
                    print(f"  Pattern {pattern_idx}: No predictions found")
                    continue

                # Select best epoch (lowest MSE, earliest if tie)
                best_epoch = min(epoch_mse.keys(), key=lambda e: (epoch_mse[e], e))

                # Evaluate epoch 0 and best epoch on test_ids
                df_e0 = load_predictions(results_dir, task_id, 0)
                df_best = load_predictions(results_dir, task_id, best_epoch)

                metrics_e0 = evaluate_on_ids(df_e0, test_ids, prompt_id) if df_e0 is not None else None
                metrics_best = evaluate_on_ids(df_best, test_ids, prompt_id) if df_best is not None else None

                result = {
                    'task_id': task_id,
                    'prompt_id': prompt_id,
                    'model_short': model_short,
                    'pattern_idx': pattern_idx,
                    'n_test': len(test_ids),
                    'n_dev': len(dev_ids),
                    'selected_epoch': best_epoch,
                    'selected_epoch_mse': epoch_mse[best_epoch],
                    'epoch_0': metrics_e0,
                    'best_epoch': metrics_best,
                }
                all_results.append(result)

                if metrics_e0 and metrics_best:
                    delta = metrics_best['qwk'] - metrics_e0['qwk']
                    print(f"  Pattern {pattern_idx}: epoch={best_epoch}, "
                          f"E0 QWK={metrics_e0['qwk']:.4f}, Best QWK={metrics_best['qwk']:.4f}, Δ={delta:+.4f}")

    # Calculate summary statistics
    summary = {
        'by_model': {},
        'by_model_prompt': {},
        'overall': {},
    }

    # Per-model summary (averaging across 8 prompts × n patterns)
    for model_short in ["llama8b", "llama3b", "mistral"]:
        model_results = [r for r in all_results if r['model_short'] == model_short]
        if not model_results:
            continue

        e0_qwks = [r['epoch_0']['qwk'] for r in model_results if r['epoch_0']]
        best_qwks = [r['best_epoch']['qwk'] for r in model_results if r['best_epoch']]

        if e0_qwks and best_qwks:
            summary['by_model'][model_short] = {
                'e0_qwk': float(np.mean(e0_qwks)),
                'best_qwk': float(np.mean(best_qwks)),
                'delta_qwk': float(np.mean(best_qwks) - np.mean(e0_qwks)),
                'n_tasks': len(model_results),
            }

        # Per-prompt breakdown for each model
        summary['by_model_prompt'][model_short] = {}
        for prompt_id in range(1, 9):
            prompt_results = [r for r in model_results if r['prompt_id'] == prompt_id]
            if prompt_results:
                e0_qwks = [r['epoch_0']['qwk'] for r in prompt_results if r['epoch_0']]
                best_qwks = [r['best_epoch']['qwk'] for r in prompt_results if r['best_epoch']]
                if e0_qwks and best_qwks:
                    summary['by_model_prompt'][model_short][f'prompt{prompt_id}'] = {
                        'e0_qwk': float(np.mean(e0_qwks)),
                        'best_qwk': float(np.mean(best_qwks)),
                        'delta_qwk': float(np.mean(best_qwks) - np.mean(e0_qwks)),
                        'n_patterns': len(prompt_results),
                    }

    # Overall summary
    e0_qwks = [r['epoch_0']['qwk'] for r in all_results if r['epoch_0']]
    best_qwks = [r['best_epoch']['qwk'] for r in all_results if r['best_epoch']]
    if e0_qwks and best_qwks:
        summary['overall'] = {
            'e0_qwk': float(np.mean(e0_qwks)),
            'best_qwk': float(np.mean(best_qwks)),
            'delta_qwk': float(np.mean(best_qwks) - np.mean(e0_qwks)),
            'n_tasks': len(all_results),
        }

    # Save all results with summary
    output_file = output_dir / "all_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'n_patterns': args.n_patterns,
                'max_epochs': args.max_epochs,
                'generated_at': datetime.now().isoformat(),
            },
            'summary': summary,
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved results to {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary: Zero-shot with new patterns (model averages across 8 prompts)")
    print("=" * 80)
    print(f"{'Model':<12} {'E0 QWK':>10} {'Best QWK':>10} {'Δ QWK':>10} {'N':>6}")
    print("-" * 50)

    for model_short in ["llama8b", "llama3b", "mistral"]:
        if model_short in summary['by_model']:
            s = summary['by_model'][model_short]
            print(f"{model_short:<12} {s['e0_qwk']:>10.4f} {s['best_qwk']:>10.4f} {s['delta_qwk']:>+10.4f} {s['n_tasks']:>6}")

    print("-" * 50)
    if summary['overall']:
        s = summary['overall']
        print(f"{'Overall':<12} {s['e0_qwk']:>10.4f} {s['best_qwk']:>10.4f} {s['delta_qwk']:>+10.4f} {s['n_tasks']:>6}")


if __name__ == "__main__":
    main()
