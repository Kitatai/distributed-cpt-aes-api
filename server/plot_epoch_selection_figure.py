#!/usr/bin/env python3
"""
Generate figure showing:
- Top: QWK vs epoch (thin lines: individual patterns, thick line: mean)
- Bottom: Histogram of selected epochs based on dev MSE

Usage:
    python plot_epoch_selection_figure.py --prompt 3 --model mistral [--n-dev 7]
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def evaluate_on_ids(df: pd.DataFrame, essay_ids: list, prompt_id: int, metric: str = 'qwk') -> float:
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
    parser = argparse.ArgumentParser(description="Plot epoch selection figure")
    parser.add_argument("--prompt", type=int, required=True, help="Prompt ID (1-8)")
    parser.add_argument("--model", type=str, required=True,
                        choices=["llama3b", "llama8b", "mistral", "qwen"],
                        help="Model short name")
    parser.add_argument("--n-dev", type=int, default=7,
                        help="Number of dev samples for epoch selection (default: 7)")
    parser.add_argument("--n-patterns", type=int, default=50,
                        help="Number of patterns to use (default: 50)")
    parser.add_argument("--max-epochs", type=int, default=30,
                        help="Maximum epoch number (default: 30)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: figure_prompt{P}_{model}.pdf)")
    parser.add_argument("--y-min", type=float, default=None,
                        help="Y-axis minimum (default: auto)")
    parser.add_argument("--y-max", type=float, default=None,
                        help="Y-axis maximum (default: auto)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # Load patterns
    patterns_path = script_dir / "data" / "sample_patterns_v2.json"
    with open(patterns_path) as f:
        patterns_data = json.load(f)
    patterns = patterns_data['patterns'][f"prompt{args.prompt}"]

    # Results directory
    task_id = f"prompt{args.prompt}_{args.model}"
    results_dir = script_dir / "data" / "backup_zeroshot_v3" / "results" / task_id

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Collect data for each pattern
    n_patterns = min(args.n_patterns, len(patterns))
    epochs = list(range(0, args.max_epochs + 1))

    # epoch_qwks[pattern_idx][epoch] = qwk on test_ids
    epoch_qwks = defaultdict(dict)
    # epoch_dev_mse[pattern_idx][epoch] = mse on dev_ids
    epoch_dev_mse = defaultdict(dict)

    print(f"Processing {task_id} with {n_patterns} patterns...")

    for pattern_idx in range(n_patterns):
        pattern = patterns[pattern_idx]
        test_ids = pattern['test_ids']
        dev_ids = pattern['dev_ids'][:args.n_dev]

        for epoch in epochs:
            df = load_predictions(results_dir, epoch)
            if df is None:
                continue

            # QWK on test_ids (for top panel)
            qwk = evaluate_on_ids(df, test_ids, args.prompt, 'qwk')
            if qwk is not None:
                epoch_qwks[pattern_idx][epoch] = qwk

            # MSE on dev_ids (for epoch selection)
            mse = evaluate_on_ids(df, dev_ids, args.prompt, 'mse')
            if mse is not None:
                epoch_dev_mse[pattern_idx][epoch] = mse

    # Select best epoch for each pattern (based on dev MSE)
    selected_epochs = []
    for pattern_idx in range(n_patterns):
        if epoch_dev_mse[pattern_idx]:
            # Select epoch with lowest MSE (earliest if tie)
            best_epoch = min(epoch_dev_mse[pattern_idx].keys(),
                           key=lambda e: (epoch_dev_mse[pattern_idx][e], e))
            selected_epochs.append(best_epoch)

    # Calculate mean QWK per epoch
    mean_qwks = {}
    for epoch in epochs:
        qwks = [epoch_qwks[p][epoch] for p in range(n_patterns) if epoch in epoch_qwks[p]]
        if qwks:
            mean_qwks[epoch] = np.mean(qwks)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[2, 1], sharex=True)

    # Top panel: QWK vs epoch
    ax1 = axes[0]

    # Plot individual patterns (thin lines)
    for pattern_idx in range(n_patterns):
        if epoch_qwks[pattern_idx]:
            eps = sorted(epoch_qwks[pattern_idx].keys())
            qwks = [epoch_qwks[pattern_idx][e] for e in eps]
            ax1.plot(eps, qwks, color='dimgray', alpha=0.6, linewidth=1.0)

    # Plot mean (thick line)
    if mean_qwks:
        eps = sorted(mean_qwks.keys())
        qwks = [mean_qwks[e] for e in eps]
        ax1.plot(eps, qwks, color='black', linewidth=2.5, label='Mean')

    ax1.set_ylabel('QWK on $D_{eval}$')

    # Y-axis range: auto or manual
    if args.y_min is not None or args.y_max is not None:
        y_min = args.y_min if args.y_min is not None else 0
        y_max = args.y_max if args.y_max is not None else 1
        ax1.set_ylim(y_min, y_max)
    else:
        # Auto-scale with padding
        all_qwks = [epoch_qwks[p][e] for p in range(n_patterns) for e in epoch_qwks[p]]
        if all_qwks:
            data_min, data_max = min(all_qwks), max(all_qwks)
            padding = (data_max - data_min) * 0.1
            ax1.set_ylim(max(0, data_min - padding), min(1, data_max + padding))

    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Histogram of selected epochs
    ax2 = axes[1]

    if selected_epochs:
        bins = np.arange(-0.5, args.max_epochs + 1.5, 1)
        ax2.hist(selected_epochs, bins=bins, color='gray', edgecolor='black', alpha=0.8)

    ax2.set_xlabel('Epoch $e$')
    ax2.set_ylabel('Count of $\\hat{e}$')
    ax2.set_xlim(-0.5, args.max_epochs + 0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    # Title
    model_display = {
        "llama3b": "Llama-3B",
        "llama8b": "Llama-8B",
        "mistral": "Mistral-7B",
        "qwen": "Qwen"
    }
    fig.suptitle(f'Prompt {args.prompt}, {model_display.get(args.model, args.model)} '
                 f'(n={n_patterns}, $|D_{{sel}}|$={args.n_dev})', fontsize=12)

    plt.tight_layout()

    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = script_dir / f"figure_prompt{args.prompt}_{args.model}.pdf"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Also save PNG
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {png_path}")

    # Print summary
    print(f"\nSummary:")
    print(f"  Selected epochs: mean={np.mean(selected_epochs):.1f}, "
          f"median={np.median(selected_epochs):.0f}, "
          f"mode={max(set(selected_epochs), key=selected_epochs.count)}")
    print(f"  E0 QWK: {mean_qwks.get(0, 0):.4f}")
    if mean_qwks:
        best_mean_epoch = max(mean_qwks.keys(), key=lambda e: mean_qwks[e])
        print(f"  Best mean QWK: {mean_qwks[best_mean_epoch]:.4f} (epoch {best_mean_epoch})")


if __name__ == "__main__":
    main()
