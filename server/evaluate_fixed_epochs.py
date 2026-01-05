#!/usr/bin/env python3
"""
Evaluate zero-shot performance at fixed epochs (0, 5, 10, 15).
No epoch selection - just fixed checkpoint evaluation.

Usage:
    python evaluate_fixed_epochs.py [--epochs 0,5,10,15] [--latex]
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
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


def load_predictions(results_dir: Path, epoch: int) -> pd.DataFrame:
    """Load predictions for a specific epoch."""
    pred_file = results_dir / f"predictions_epoch_{epoch}.csv"
    if not pred_file.exists():
        return None
    return pd.read_csv(pred_file)


def evaluate_on_ids(df: pd.DataFrame, essay_ids: list, prompt_id: int) -> float:
    """Evaluate QWK on specific essay IDs."""
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
    return quadratic_weighted_kappa(y_true, y_pred, min_score, max_score)


def main():
    parser = argparse.ArgumentParser(description="Evaluate at fixed epochs")
    parser.add_argument("--epochs", type=str, default="0,5,10,15",
                        help="Comma-separated epochs to evaluate (default: 0,5,10,15)")
    parser.add_argument("--n-patterns", type=int, default=10,
                        help="Number of patterns to use (default: 10)")
    parser.add_argument("--no-latex", action="store_true",
                        help="Disable LaTeX output")
    parser.add_argument("--models", type=str, default="llama8b,llama3b,mistral",
                        help="Comma-separated model names")
    parser.add_argument("--results-dir", type=str, default="backup_zeroshot_v3",
                        help="Results directory name under server/data/ (default: backup_zeroshot_v3)")
    args = parser.parse_args()

    epochs = [int(e) for e in args.epochs.split(",")]
    models = args.models.split(",")

    script_dir = Path(__file__).parent
    results_base = script_dir / "data" / args.results_dir / "results"

    # Load patterns (always use v2)
    patterns_path = script_dir / "data" / "sample_patterns_v2.json"
    with open(patterns_path) as f:
        patterns_data = json.load(f)

    print(f"Results: {args.results_dir}, Patterns: sample_patterns_v2.json")

    # results[model][prompt_id][epoch] = list of QWKs
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    print("Evaluating fixed epochs...")

    for model in models:
        for prompt_id in range(1, 9):
            task_id = f"prompt{prompt_id}_{model}"
            results_dir = results_base / task_id

            if not results_dir.exists():
                continue

            patterns = patterns_data['patterns'][f"prompt{prompt_id}"]

            for pattern_idx in range(min(args.n_patterns, len(patterns))):
                pattern = patterns[pattern_idx]
                test_ids = pattern['test_ids']

                for epoch in epochs:
                    df = load_predictions(results_dir, epoch)
                    if df is None:
                        continue

                    qwk = evaluate_on_ids(df, test_ids, prompt_id)
                    if qwk is not None:
                        results[model][prompt_id][epoch].append(qwk)

    # Print results
    print("\n" + "=" * 80)
    print("Fixed Epoch Zero-shot Results (QWK mean±std)")
    print("=" * 80)

    for model in models:
        print(f"\n### {model.upper()} ###")
        print(f"{'Prompt':<8}", end="")
        for epoch in epochs:
            print(f"  E{epoch:<6}", end="")
        print()
        print("-" * (8 + 9 * len(epochs)))

        for prompt_id in range(1, 9):
            print(f"P{prompt_id:<7}", end="")
            for epoch in epochs:
                qwks = results[model][prompt_id][epoch]
                if qwks:
                    mean = np.mean(qwks)
                    std = np.std(qwks)
                    print(f"  {mean:.3f}±{std:.3f}", end="")
                else:
                    print(f"  {'---':^9}", end="")
            print()

        # Average row
        print("-" * (8 + 9 * len(epochs)))
        print(f"{'Avg':<8}", end="")
        for epoch in epochs:
            all_means = []
            for prompt_id in range(1, 9):
                qwks = results[model][prompt_id][epoch]
                if qwks:
                    all_means.append(np.mean(qwks))
            if all_means:
                print(f"  {np.mean(all_means):.3f}    ", end="")
            else:
                print(f"  {'---':^9}", end="")
        print()

    # LaTeX output (default)
    if not args.no_latex:
        # Format: \qwk{.XXX}
        def fmt_qwk(val):
            if val is None:
                return "---"
            s = f"{val:.3f}"
            if s.startswith("0."):
                s = s[1:]  # Remove leading zero
            return f"\\qwk{{{s}}}"

        # Combined table (all models, average across prompts)
        print("\n\\begin{table}[t]")
        print("\\centering")
        print("\\caption{Zero-shot QWK at fixed epochs (average across 8 prompts)}")
        print("\\begin{tabular}{l" + "c" * len(epochs) + "}")
        print("\\hline")
        header = "Model & " + " & ".join([f"$e={e}$" for e in epochs]) + " \\\\"
        print(header)
        print("\\hline")

        for model in models:
            model_display = {"llama8b": "Llama-8B", "llama3b": "Llama-3B",
                           "mistral": "Mistral-7B", "qwen": "Qwen"}
            row = model_display.get(model, model)
            for epoch in epochs:
                all_means = []
                for prompt_id in range(1, 9):
                    qwks = results[model][prompt_id][epoch]
                    if qwks:
                        all_means.append(np.mean(qwks))
                if all_means:
                    row += f" & {fmt_qwk(np.mean(all_means))}"
                else:
                    row += " & ---"
            row += " \\\\"
            print(row)

        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")


if __name__ == "__main__":
    main()
