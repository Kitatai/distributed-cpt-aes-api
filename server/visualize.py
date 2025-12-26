#!/usr/bin/env python3
"""
Visualize completed experiment results.
Generates plots for completed tasks and saves them to output folder.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_task_metrics(results_dir: Path, task_id: str) -> dict:
    """Load all epoch metrics for a task."""
    task_dir = results_dir / task_id
    if not task_dir.exists():
        return {}

    metrics = {}
    for f in task_dir.glob("metrics_epoch_*.json"):
        epoch = int(f.stem.split("_")[-1])
        with open(f) as fp:
            metrics[epoch] = json.load(fp)
    return metrics


def load_task_predictions(results_dir: Path, task_id: str, epoch: int) -> Optional[pd.DataFrame]:
    """Load predictions for a specific epoch."""
    pred_file = results_dir / task_id / f"predictions_epoch_{epoch}.csv"
    if pred_file.exists():
        return pd.read_csv(pred_file)
    return None


def plot_metrics_over_epochs(metrics: dict, task_id: str, output_dir: Path):
    """Plot QWK, Pearson, Spearman over epochs."""
    if not metrics:
        return

    epochs = sorted(metrics.keys())

    # Extract metrics for test set
    qwk_values = []
    pearson_values = []
    spearman_values = []

    for epoch in epochs:
        m = metrics[epoch].get("greedy", {}).get("test", {})
        qwk_values.append(m.get("qwk", 0))
        pearson_values.append(m.get("pearson", 0))
        spearman_values.append(m.get("spearman", 0))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # QWK
    axes[0].plot(epochs, qwk_values, 'o-', linewidth=2, markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('QWK')
    axes[0].set_title(f'{task_id} - QWK over Epochs')
    axes[0].set_ylim(-0.1, 1.0)
    best_epoch = epochs[np.argmax(qwk_values)]
    best_qwk = max(qwk_values)
    axes[0].axhline(y=best_qwk, color='r', linestyle='--', alpha=0.5)
    axes[0].annotate(f'Best: {best_qwk:.4f} (e={best_epoch})',
                     xy=(best_epoch, best_qwk), xytext=(5, 10),
                     textcoords='offset points', fontsize=9)

    # Pearson
    axes[1].plot(epochs, pearson_values, 'o-', linewidth=2, markersize=4, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Pearson r')
    axes[1].set_title(f'{task_id} - Pearson over Epochs')
    axes[1].set_ylim(-0.1, 1.0)

    # Spearman
    axes[2].plot(epochs, spearman_values, 'o-', linewidth=2, markersize=4, color='orange')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Spearman ρ')
    axes[2].set_title(f'{task_id} - Spearman over Epochs')
    axes[2].set_ylim(-0.1, 1.0)

    plt.tight_layout()
    plt.savefig(output_dir / f"{task_id}_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_predictions_scatter(predictions: pd.DataFrame, task_id: str, epoch: int, output_dir: Path):
    """Plot predicted vs actual scatter plot."""
    if predictions is None or predictions.empty:
        return

    y_true = predictions['y_true'].values
    y_pred = predictions['y_hat_greedy'].values

    # Remove NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot with jitter
    jitter = 0.1
    y_true_jitter = y_true + np.random.uniform(-jitter, jitter, len(y_true))
    y_pred_jitter = y_pred + np.random.uniform(-jitter, jitter, len(y_pred))

    ax.scatter(y_true_jitter, y_pred_jitter, alpha=0.3, s=20)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

    ax.set_xlabel('Actual Score', fontsize=12)
    ax.set_ylabel('Predicted Score', fontsize=12)
    ax.set_title(f'{task_id} - Epoch {epoch}\nPredicted vs Actual', fontsize=14)
    ax.legend()

    # Add correlation info
    from scipy.stats import pearsonr, spearmanr
    r, _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    ax.text(0.05, 0.95, f'Pearson r = {r:.4f}\nSpearman ρ = {rho:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / f"{task_id}_scatter_epoch{epoch}.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(predictions: pd.DataFrame, task_id: str, epoch: int, output_dir: Path):
    """Plot confusion matrix."""
    if predictions is None or predictions.empty:
        return

    y_true = predictions['y_true'].values
    y_pred = predictions['y_hat_greedy'].values

    # Remove NaN and convert to int
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask].astype(int)
    y_pred = y_pred[mask].astype(int)

    if len(y_true) == 0:
        return

    # Get unique labels
    labels = sorted(set(y_true) | set(y_pred))

    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'{task_id} - Epoch {epoch}\nConfusion Matrix', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / f"{task_id}_confusion_epoch{epoch}.png", dpi=150, bbox_inches='tight')
    plt.close()


def get_completed_tasks(tasks_dir: Path) -> list:
    """Get list of completed tasks from individual task files."""
    if not tasks_dir.exists():
        return []

    completed = []
    for task_file in tasks_dir.glob("*.json"):
        if task_file.name == "tasks.json":
            continue
        with open(task_file) as f:
            task = json.load(f)
            if task.get("status") == "completed":
                completed.append(task["task_id"])
    return completed


def plot_model_comparison(results_dir: Path, output_dir: Path, completed_tasks: list):
    """Plot comparison across models for each prompt."""
    # Group by prompt
    prompts = {}
    for task_id in completed_tasks:
        parts = task_id.split("_")
        prompt = parts[0]
        model = "_".join(parts[1:])

        if prompt not in prompts:
            prompts[prompt] = {}

        metrics = load_task_metrics(results_dir, task_id)
        if metrics:
            # Get best QWK
            best_qwk = 0
            best_epoch = 0
            for epoch, m in metrics.items():
                qwk = m.get("greedy", {}).get("test", {}).get("qwk", 0)
                if qwk > best_qwk:
                    best_qwk = qwk
                    best_epoch = epoch
            prompts[prompt][model] = {"best_qwk": best_qwk, "best_epoch": best_epoch}

    if not prompts:
        return

    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    prompt_names = sorted(prompts.keys())
    model_names = sorted(set(m for p in prompts.values() for m in p.keys()))

    x = np.arange(len(prompt_names))
    width = 0.8 / len(model_names)

    for i, model in enumerate(model_names):
        values = [prompts[p].get(model, {}).get("best_qwk", 0) for p in prompt_names]
        bars = ax.bar(x + i * width - 0.4 + width/2, values, width, label=model)

        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_xlabel('Prompt')
    ax.set_ylabel('Best QWK')
    ax.set_title('Model Comparison - Best QWK per Prompt')
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_names)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def generate_summary_table(results_dir: Path, output_dir: Path, completed_tasks: list):
    """Generate a summary table of all completed experiments."""
    rows = []

    for task_id in completed_tasks:
        metrics = load_task_metrics(results_dir, task_id)
        if not metrics:
            continue

        parts = task_id.split("_")
        prompt = parts[0]
        model = "_".join(parts[1:])

        # Find best epoch by QWK
        best_qwk = 0
        best_epoch = 0
        best_metrics = {}

        for epoch, m in metrics.items():
            qwk = m.get("greedy", {}).get("test", {}).get("qwk", 0)
            if qwk > best_qwk:
                best_qwk = qwk
                best_epoch = epoch
                best_metrics = m.get("greedy", {}).get("test", {})

        rows.append({
            "prompt": prompt,
            "model": model,
            "best_epoch": best_epoch,
            "qwk": best_metrics.get("qwk", 0),
            "pearson": best_metrics.get("pearson", 0),
            "spearman": best_metrics.get("spearman", 0),
            "n_samples": best_metrics.get("n_samples", 0),
        })

    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(["prompt", "model"])
        df.to_csv(output_dir / "summary.csv", index=False)
        print("\nSummary Table:")
        print(df.to_string(index=False))
        return df
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize completed experiment results")
    parser.add_argument("--results-dir", type=str, default="data/results",
                       help="Path to results directory")
    parser.add_argument("--tasks-dir", type=str, default="data/tasks",
                       help="Path to tasks directory")
    parser.add_argument("--output-dir", type=str, default="data/visualizations",
                       help="Path to output directory for plots")
    parser.add_argument("--task", type=str, default=None,
                       help="Specific task to visualize (optional)")
    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / args.results_dir
    tasks_dir = script_dir / args.tasks_dir
    output_dir = script_dir / args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get completed tasks
    if args.task:
        completed_tasks = [args.task]
    else:
        completed_tasks = get_completed_tasks(tasks_dir)

    if not completed_tasks:
        print("No completed tasks found.")
        return

    print(f"Found {len(completed_tasks)} completed tasks:")
    for t in completed_tasks:
        print(f"  - {t}")

    # Generate visualizations for each task
    for task_id in completed_tasks:
        print(f"\nProcessing {task_id}...")

        metrics = load_task_metrics(results_dir, task_id)
        if not metrics:
            print(f"  No metrics found for {task_id}")
            continue

        # Plot metrics over epochs
        plot_metrics_over_epochs(metrics, task_id, output_dir)
        print(f"  Created {task_id}_metrics.png")

        # Find best epoch and plot scatter/confusion for it
        best_epoch = max(metrics.keys(), key=lambda e: metrics[e].get("greedy", {}).get("test", {}).get("qwk", 0))
        predictions = load_task_predictions(results_dir, task_id, best_epoch)

        if predictions is not None:
            plot_predictions_scatter(predictions, task_id, best_epoch, output_dir)
            print(f"  Created {task_id}_scatter_epoch{best_epoch}.png")

            plot_confusion_matrix(predictions, task_id, best_epoch, output_dir)
            print(f"  Created {task_id}_confusion_epoch{best_epoch}.png")

    # Generate comparison plot if multiple tasks
    if len(completed_tasks) > 1:
        plot_model_comparison(results_dir, output_dir, completed_tasks)
        print("\nCreated model_comparison.png")

    # Generate summary table
    generate_summary_table(results_dir, output_dir, completed_tasks)

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
