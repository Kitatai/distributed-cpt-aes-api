#!/usr/bin/env python3
"""
Generate LaTeX table for zero-shot v2 evaluation results.

Outputs:
- No-CP: Epoch 0 (no continual pretraining)
- Best: Best epoch selected by MSE on dev_ids
- Oracle: Best possible epoch selected by QWK on test_ids (cheating upper bound)

Usage:
    python generate_latex_zeroshot_v2.py [--n-dev N]
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(y_true, y_pred, min_score, max_score):
    """Calculate QWK."""
    y_true = np.clip(y_true, min_score, max_score).astype(int)
    y_pred = np.clip(y_pred, min_score, max_score).astype(int)
    return cohen_kappa_score(y_true, y_pred, weights='quadratic',
                             labels=list(range(min_score, max_score + 1)))


def get_score_range(prompt_id: int) -> tuple:
    """Get min and max scores for a prompt."""
    score_ranges = {
        1: (2, 12), 2: (1, 6), 3: (0, 3), 4: (0, 3),
        5: (0, 4), 6: (0, 4), 7: (0, 30), 8: (0, 60),
    }
    return score_ranges[prompt_id]


def load_predictions(results_dir: Path, task_id: str, epoch: int) -> pd.DataFrame:
    """Load predictions for a specific epoch."""
    pred_file = results_dir / task_id / f"predictions_epoch_{epoch}.csv"
    if not pred_file.exists():
        return None
    return pd.read_csv(pred_file)


def evaluate_on_ids(df: pd.DataFrame, essay_ids: list, prompt_id: int, metric: str = 'mse') -> float:
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

    if metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'qwk':
        return quadratic_weighted_kappa(y_true, y_pred, min_score, max_score)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX table for zero-shot v2")
    parser.add_argument("--n-patterns", type=int, default=10,
                        help="Number of patterns to evaluate (default: 10)")
    parser.add_argument("--n-dev", type=int, default=10,
                        help="Number of dev samples for epoch selection (default: 10)")
    parser.add_argument("--max-epochs", type=int, default=30,
                        help="Maximum epoch number (default: 30)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # Load patterns
    patterns_path = script_dir / "data" / "sample_patterns_v2.json"
    with open(patterns_path) as f:
        patterns_data = json.load(f)
    patterns = patterns_data['patterns']

    # Results directory
    results_dir = script_dir / "data" / "backup_zeroshot_v3" / "results"

    # Models
    models = [
        ("meta-llama/Llama-3.2-3B-Instruct", "llama3b", "Llama3B"),
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", "llama8b", "Llama8B"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "mistral", "Mistral"),
    ]

    # Collect results: results[model][prompt_id] = {'no_cp': [...], 'best': [...], 'oracle': [...]}
    results = defaultdict(lambda: defaultdict(lambda: {'no_cp': [], 'best': [], 'oracle': []}))

    for model_name, model_short, model_display in models:
        for prompt_id in range(1, 9):
            task_id = f"prompt{prompt_id}_{model_short}"
            prompt_key = f"prompt{prompt_id}"

            for pattern_idx in range(args.n_patterns):
                pattern = patterns[prompt_key][pattern_idx]
                test_ids = pattern['test_ids']
                dev_ids = pattern['dev_ids'][:args.n_dev]

                # Collect epoch results
                epoch_dev_mse = {}
                epoch_test_qwk = {}

                for epoch in range(0, args.max_epochs + 1):
                    df = load_predictions(results_dir, task_id, epoch)
                    if df is None:
                        continue

                    # Dev MSE for epoch selection
                    dev_mse = evaluate_on_ids(df, dev_ids, prompt_id, 'mse')
                    if dev_mse is not None:
                        epoch_dev_mse[epoch] = dev_mse

                    # Test QWK for evaluation
                    test_qwk = evaluate_on_ids(df, test_ids, prompt_id, 'qwk')
                    if test_qwk is not None:
                        epoch_test_qwk[epoch] = test_qwk

                if not epoch_test_qwk:
                    continue

                # No-CP: Epoch 0
                no_cp_qwk = epoch_test_qwk.get(0, 0)

                # Best: Select by dev MSE, evaluate on test
                if epoch_dev_mse:
                    best_epoch = min(epoch_dev_mse.keys(), key=lambda e: (epoch_dev_mse[e], e))
                    best_qwk = epoch_test_qwk.get(best_epoch, 0)
                else:
                    best_qwk = no_cp_qwk

                # Oracle: Best possible QWK on test (cheating)
                oracle_qwk = max(epoch_test_qwk.values())

                results[model_short][prompt_id]['no_cp'].append(no_cp_qwk)
                results[model_short][prompt_id]['best'].append(best_qwk)
                results[model_short][prompt_id]['oracle'].append(oracle_qwk)

    # Generate LaTeX output
    print("% Zero-shot v2 results (n_dev={})".format(args.n_dev))
    print("% Generated by generate_latex_zeroshot_v2.py")
    print()

    model_order = [("llama3b", "Llama3B"), ("llama8b", "Llama8B"), ("mistral", "Mistral")]
    setting_order = [('no_cp', 'No-CP'), ('best', 'Best'), ('oracle', 'Oracle')]

    for model_short, model_display in model_order:
        for idx, (setting_key, setting_name) in enumerate(setting_order):
            # Collect values for each prompt
            prompt_values = []
            for prompt_id in range(1, 9):
                values = results[model_short][prompt_id][setting_key]
                if values:
                    mean_qwk = np.mean(values)
                    prompt_values.append(mean_qwk)
                else:
                    prompt_values.append(0)

            # Calculate average
            avg_qwk = np.mean(prompt_values) if prompt_values else 0

            # Format line
            if idx == 0:
                model_col = f"& \\multirow{{3}}{{*}}{{{model_display}}}"
            else:
                model_col = "&"

            qwk_strs = [f"\\qwk{{{v:.3f}}}" for v in prompt_values]
            avg_str = f"\\qwk{{{avg_qwk:.3f}}}"

            line = f"{model_col} & {setting_name:<7} & {' & '.join(qwk_strs)} & {avg_str} \\\\"
            print(line)

        print("\\cline{2-12}")
        print()


if __name__ == "__main__":
    main()
