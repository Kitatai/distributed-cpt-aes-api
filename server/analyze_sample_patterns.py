#!/usr/bin/env python3
"""
Analyze experiment results using sample patterns.

For each pattern (10 essays):
1. Calculate MSE on the 10 samples for each epoch (0-30)
2. Select best epoch (lowest MSE)
3. Evaluate on remaining essays (QWK, Spearman) at best epoch and epoch 0

Usage:
    python analyze_sample_patterns.py [--models MODEL1,MODEL2,...] [--prompts 1,2,...]
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


def load_sample_patterns(path: Path) -> dict:
    """Load sample patterns from JSON."""
    with open(path) as f:
        return json.load(f)


def load_predictions(results_dir: Path, task_id: str, epoch: int) -> Optional[pd.DataFrame]:
    """Load predictions for a specific task and epoch."""
    pred_path = results_dir / task_id / f"predictions_epoch_{epoch}.csv"
    if not pred_path.exists():
        return None
    return pd.read_csv(pred_path)


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def calculate_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Quadratic Weighted Kappa."""
    return float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))


def calculate_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Spearman correlation."""
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr) if not np.isnan(corr) else 0.0


def analyze_pattern(
    results_dir: Path,
    task_id: str,
    sample_ids: List[int],
    max_epoch: int = 30,
) -> Optional[Dict]:
    """
    Analyze a single pattern.

    Args:
        results_dir: Path to results directory
        task_id: Task identifier (e.g., "prompt1_llama3b")
        sample_ids: List of 10 essay IDs for this pattern
        max_epoch: Maximum epoch to consider

    Returns:
        Dictionary with analysis results or None if data not available
    """
    sample_set = set(sample_ids)

    # Find best epoch using MSE on sample
    best_epoch = 0
    best_mse = float('inf')
    epoch_mses = {}

    for epoch in range(max_epoch + 1):
        df = load_predictions(results_dir, task_id, epoch)
        if df is None:
            continue

        # Filter to sample essays
        sample_df = df[df['essay_id'].isin(sample_set)]
        if len(sample_df) < len(sample_ids):
            continue

        y_true = sample_df['y_true'].values
        y_pred = sample_df['y_hat_greedy'].values

        mse = calculate_mse(y_true, y_pred)
        epoch_mses[epoch] = mse

        if mse <= best_mse:
            best_mse = mse
            best_epoch = epoch

    if not epoch_mses:
        return None

    # Evaluate at epoch 0 on non-sample essays
    df_epoch0 = load_predictions(results_dir, task_id, 0)
    if df_epoch0 is None:
        return None

    non_sample_df_0 = df_epoch0[~df_epoch0['essay_id'].isin(sample_set)]
    y_true_0 = non_sample_df_0['y_true'].values
    y_pred_0 = non_sample_df_0['y_hat_greedy'].values

    qwk_epoch0 = calculate_qwk(y_true_0, y_pred_0)
    spearman_epoch0 = calculate_spearman(y_true_0, y_pred_0)

    # Evaluate at best epoch on non-sample essays
    df_best = load_predictions(results_dir, task_id, best_epoch)
    if df_best is None:
        return None

    non_sample_df_best = df_best[~df_best['essay_id'].isin(sample_set)]
    y_true_best = non_sample_df_best['y_true'].values
    y_pred_best = non_sample_df_best['y_hat_greedy'].values

    qwk_best = calculate_qwk(y_true_best, y_pred_best)
    spearman_best = calculate_spearman(y_true_best, y_pred_best)

    return {
        'best_epoch': best_epoch,
        'best_mse': best_mse,
        'epoch0': {
            'qwk': qwk_epoch0,
            'spearman': spearman_epoch0,
        },
        'best': {
            'qwk': qwk_best,
            'spearman': spearman_best,
        },
    }


def analyze_task(
    results_dir: Path,
    patterns: dict,
    task_id: str,
    prompt_id: int,
    max_epoch: int = 30,
) -> Optional[Dict]:
    """
    Analyze all patterns for a task.

    Args:
        results_dir: Path to results directory
        patterns: Sample patterns dictionary
        task_id: Task identifier
        prompt_id: Prompt ID (1-8)
        max_epoch: Maximum epoch

    Returns:
        Dictionary with aggregated results
    """
    prompt_key = f"prompt{prompt_id}"
    prompt_patterns = patterns['patterns'].get(prompt_key, [])

    if not prompt_patterns:
        return None

    # Check if task has results
    task_dir = results_dir / task_id
    if not task_dir.exists():
        return None

    pattern_results = []

    for pattern_idx, sample_ids in enumerate(prompt_patterns):
        result = analyze_pattern(results_dir, task_id, sample_ids, max_epoch)
        if result:
            result['pattern_idx'] = pattern_idx
            pattern_results.append(result)

    if not pattern_results:
        return None

    # Aggregate results
    n_patterns = len(pattern_results)

    # Epoch 0 metrics (same for all patterns since it's the same predictions)
    epoch0_qwk = pattern_results[0]['epoch0']['qwk']
    epoch0_spearman = pattern_results[0]['epoch0']['spearman']

    # Best epoch metrics (varies per pattern)
    best_qwks = [r['best']['qwk'] for r in pattern_results]
    best_spearmans = [r['best']['spearman'] for r in pattern_results]
    best_epochs = [r['best_epoch'] for r in pattern_results]

    return {
        'task_id': task_id,
        'prompt_id': prompt_id,
        'n_patterns': n_patterns,
        'epoch0': {
            'qwk': epoch0_qwk,
            'spearman': epoch0_spearman,
        },
        'best_epoch_stats': {
            'mean': float(np.mean(best_epochs)),
            'std': float(np.std(best_epochs)),
            'min': int(np.min(best_epochs)),
            'max': int(np.max(best_epochs)),
        },
        'best': {
            'qwk_mean': float(np.mean(best_qwks)),
            'qwk_std': float(np.std(best_qwks)),
            'spearman_mean': float(np.mean(best_spearmans)),
            'spearman_std': float(np.std(best_spearmans)),
        },
        'improvement': {
            'qwk': float(np.mean(best_qwks)) - epoch0_qwk,
            'spearman': float(np.mean(best_spearmans)) - epoch0_spearman,
        },
        'pattern_details': pattern_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze sample patterns")
    parser.add_argument(
        "--models",
        type=str,
        default="llama3b,llama8b,mistral",
        help="Comma-separated model names (default: llama3b,llama8b,mistral)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="1,2,3,4,5,6,7,8",
        help="Comma-separated prompt IDs (default: 1,2,3,4,5,6,7,8)",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=30,
        help="Maximum epoch to consider (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/analysis_results.json)",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        default=None,
        help="Sample patterns file path (default: data/sample_patterns.json)",
    )

    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(',')]
    prompts = [int(p.strip()) for p in args.prompts.split(',')]

    # Paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / "data" / "results"

    if args.patterns:
        patterns_path = Path(args.patterns)
    else:
        patterns_path = script_dir / "data" / "sample_patterns.json"

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = script_dir / "data" / "analysis_results.json"

    # Load patterns
    print(f"Loading sample patterns from {patterns_path}")
    patterns = load_sample_patterns(patterns_path)
    print(f"Loaded {patterns['metadata']['n_patterns']} patterns per prompt")

    # Analyze each task
    all_results = {}
    summary = []

    for prompt_id in prompts:
        for model in models:
            task_id = f"prompt{prompt_id}_{model}"
            print(f"\nAnalyzing {task_id}...")

            result = analyze_task(
                results_dir, patterns, task_id, prompt_id, args.max_epoch
            )

            if result:
                all_results[task_id] = result

                # Summary row
                summary.append({
                    'task_id': task_id,
                    'prompt': prompt_id,
                    'model': model,
                    'epoch0_qwk': result['epoch0']['qwk'],
                    'epoch0_spearman': result['epoch0']['spearman'],
                    'best_qwk_mean': result['best']['qwk_mean'],
                    'best_qwk_std': result['best']['qwk_std'],
                    'best_spearman_mean': result['best']['spearman_mean'],
                    'best_spearman_std': result['best']['spearman_std'],
                    'improvement_qwk': result['improvement']['qwk'],
                    'improvement_spearman': result['improvement']['spearman'],
                    'best_epoch_mean': result['best_epoch_stats']['mean'],
                })

                print(f"  Epoch 0: QWK={result['epoch0']['qwk']:.4f}, "
                      f"Spearman={result['epoch0']['spearman']:.4f}")
                print(f"  Best:    QWK={result['best']['qwk_mean']:.4f} (±{result['best']['qwk_std']:.4f}), "
                      f"Spearman={result['best']['spearman_mean']:.4f} (±{result['best']['spearman_std']:.4f})")
                print(f"  Improvement: QWK={result['improvement']['qwk']:+.4f}, "
                      f"Spearman={result['improvement']['spearman']:+.4f}")
            else:
                print(f"  No results available")

    # Create output
    output = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'models': models,
            'prompts': prompts,
            'max_epoch': args.max_epoch,
            'n_patterns': patterns['metadata']['n_patterns'],
            'n_samples_per_pattern': patterns['metadata']['n_samples_per_pattern'],
            'pattern_seed': patterns['metadata']['seed'],
        },
        'summary': summary,
        'details': all_results,
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Task':<20} {'E0 QWK':>8} {'Best QWK':>10} {'Δ QWK':>8} {'E0 Spear':>9} {'Best Spear':>11} {'Δ Spear':>8}")
    print("-" * 80)

    for row in summary:
        print(f"{row['task_id']:<20} "
              f"{row['epoch0_qwk']:>8.4f} "
              f"{row['best_qwk_mean']:>7.4f}±{row['best_qwk_std']:.2f} "
              f"{row['improvement_qwk']:>+8.4f} "
              f"{row['epoch0_spearman']:>9.4f} "
              f"{row['best_spearman_mean']:>8.4f}±{row['best_spearman_std']:.2f} "
              f"{row['improvement_spearman']:>+8.4f}")

    # Model-wise aggregation (average across prompts)
    print(f"\n{'='*60}")
    print("MODEL-WISE AVERAGE (across prompts)")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'E0 QWK':>8} {'Best QWK':>10} {'Δ QWK':>8} {'E0 Spear':>9} {'Best Spear':>11} {'Δ Spear':>8}")
    print("-" * 70)

    model_aggregates = {}
    for model in models:
        model_rows = [r for r in summary if r['model'] == model]
        if model_rows:
            model_aggregates[model] = {
                'epoch0_qwk': np.mean([r['epoch0_qwk'] for r in model_rows]),
                'best_qwk_mean': np.mean([r['best_qwk_mean'] for r in model_rows]),
                'improvement_qwk': np.mean([r['improvement_qwk'] for r in model_rows]),
                'epoch0_spearman': np.mean([r['epoch0_spearman'] for r in model_rows]),
                'best_spearman_mean': np.mean([r['best_spearman_mean'] for r in model_rows]),
                'improvement_spearman': np.mean([r['improvement_spearman'] for r in model_rows]),
                'n_prompts': len(model_rows),
            }
            agg = model_aggregates[model]
            print(f"{model:<12} "
                  f"{agg['epoch0_qwk']:>8.4f} "
                  f"{agg['best_qwk_mean']:>10.4f} "
                  f"{agg['improvement_qwk']:>+8.4f} "
                  f"{agg['epoch0_spearman']:>9.4f} "
                  f"{agg['best_spearman_mean']:>11.4f} "
                  f"{agg['improvement_spearman']:>+8.4f}")

    # Overall average (all models)
    print("-" * 70)
    if summary:
        overall = {
            'epoch0_qwk': np.mean([r['epoch0_qwk'] for r in summary]),
            'best_qwk_mean': np.mean([r['best_qwk_mean'] for r in summary]),
            'improvement_qwk': np.mean([r['improvement_qwk'] for r in summary]),
            'epoch0_spearman': np.mean([r['epoch0_spearman'] for r in summary]),
            'best_spearman_mean': np.mean([r['best_spearman_mean'] for r in summary]),
            'improvement_spearman': np.mean([r['improvement_spearman'] for r in summary]),
        }
        print(f"{'ALL':<12} "
              f"{overall['epoch0_qwk']:>8.4f} "
              f"{overall['best_qwk_mean']:>10.4f} "
              f"{overall['improvement_qwk']:>+8.4f} "
              f"{overall['epoch0_spearman']:>9.4f} "
              f"{overall['best_spearman_mean']:>11.4f} "
              f"{overall['improvement_spearman']:>+8.4f}")

    # Add model aggregates to output
    output['model_averages'] = model_aggregates
    output['overall_average'] = overall if summary else {}

    # Re-save with aggregates
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results (with aggregates) saved to {output_path}")


if __name__ == "__main__":
    main()
