"""
Visualization module for essay scoring results.

Generates plots for:
- True vs predicted score scatter plots
- Confusion matrices
- Score distributions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_true_vs_predicted(
    y_true: List[int],
    y_pred: List[int],
    y_min: int,
    y_max: int,
    title: str = "True vs Predicted Scores",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Create scatter plot of true vs predicted scores.

    Args:
        y_true: True scores
        y_pred: Predicted scores
        y_min: Minimum score
        y_max: Maximum score
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create DataFrame for plotting
    df = pd.DataFrame({'True': y_true, 'Predicted': y_pred})

    # Count occurrences for each (true, pred) pair
    counts = df.groupby(['True', 'Predicted']).size().reset_index(name='Count')

    # Scatter plot with size based on count
    scatter = ax.scatter(
        counts['True'],
        counts['Predicted'],
        s=counts['Count'] * 50,
        alpha=0.6,
        c=counts['Count'],
        cmap='viridis',
        edgecolors='black',
        linewidth=0.5,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Count', fontsize=12)

    # Add diagonal line (perfect predictions)
    ax.plot([y_min, y_max], [y_min, y_max], 'r--', linewidth=2, label='Perfect prediction')

    # Set limits and labels
    ax.set_xlim(y_min - 0.5, y_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    ax.set_xlabel('True Score', fontsize=14)
    ax.set_ylabel('Predicted Score', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='upper left')

    # Set integer ticks
    ax.set_xticks(range(y_min, y_max + 1))
    ax.set_yticks(range(y_min, y_max + 1))

    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved scatter plot to {save_path}")

    return fig


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    y_min: int,
    y_max: int,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = False,
) -> plt.Figure:
    """
    Create confusion matrix heatmap.

    Args:
        y_true: True scores
        y_pred: Predicted scores
        y_min: Minimum score
        y_max: Maximum score
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        normalize: Whether to normalize by true labels

    Returns:
        matplotlib Figure
    """
    labels = list(range(y_min, y_max + 1))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # Handle division by zero
        fmt = '.2f'
        title = title + " (Normalized)"
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
    )

    ax.set_xlabel('Predicted Score', fontsize=14)
    ax.set_ylabel('True Score', fontsize=14)
    ax.set_title(title, fontsize=16)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")

    return fig


def plot_score_distribution(
    y_true: List[int],
    y_pred: List[int],
    y_min: int,
    y_max: int,
    title: str = "Score Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Create bar plot comparing true and predicted score distributions.

    Args:
        y_true: True scores
        y_pred: Predicted scores
        y_min: Minimum score
        y_max: Maximum score
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    labels = list(range(y_min, y_max + 1))
    x = np.arange(len(labels))
    width = 0.35

    # Count occurrences
    true_counts = [y_true.count(s) for s in labels]
    pred_counts = [y_pred.count(s) for s in labels]

    # Create bars
    bars1 = ax.bar(x - width/2, true_counts, width, label='True', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, pred_counts, width, label='Predicted', color='coral', alpha=0.8)

    ax.set_xlabel('Score', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved distribution plot to {save_path}")

    return fig


def plot_epoch_metrics(
    epoch_metrics: Dict[int, Dict],
    metric_name: str = "spearman",
    title: str = "Metric by Epoch",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    pred_type: str = "greedy",
) -> plt.Figure:
    """
    Plot metric values across epochs.

    Args:
        epoch_metrics: Dict mapping epoch -> metrics dict
        metric_name: Name of metric to plot
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        pred_type: 'greedy' or 'expected'

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = sorted(epoch_metrics.keys())
    dev_values = []
    test_values = []

    for e in epochs:
        metrics = epoch_metrics[e]
        dev_val = metrics.get(pred_type, {}).get('dev', {}).get(metric_name, None)
        test_val = metrics.get(pred_type, {}).get('test', {}).get(metric_name, None)
        dev_values.append(dev_val)
        test_values.append(test_val)

    ax.plot(epochs, dev_values, 'o-', label='Dev', linewidth=2, markersize=8)
    ax.plot(epochs, test_values, 's--', label='Test', linewidth=2, markersize=8)

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel(metric_name.capitalize(), fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark best epoch
    if dev_values:
        best_epoch = epochs[np.argmax([v if v else -np.inf for v in dev_values])]
        ax.axvline(x=best_epoch, color='red', linestyle=':', alpha=0.7, label=f'Best (e*={best_epoch})')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved epoch metrics plot to {save_path}")

    return fig


def plot_all_epoch_metrics(
    epoch_results: Dict[int, Dict],
    output_dir: str,
    pred_type: str = "greedy",
) -> Dict[str, str]:
    """
    Create comprehensive plots for all metrics across epochs.

    Args:
        epoch_results: Dict mapping epoch -> result dict with 'metrics' key
        output_dir: Directory to save plots
        pred_type: 'greedy' or 'expected'

    Returns:
        Dict mapping plot name to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = sorted(epoch_results.keys())
    metrics_list = ['pearson', 'spearman', 'qwk']

    saved_paths = {}

    # Extract metrics for each epoch
    epoch_metrics = {}
    for e in epochs:
        epoch_metrics[e] = epoch_results[e].get('metrics', {})

    # 1. Individual metric plots
    for metric_name in metrics_list:
        save_path = output_dir / f"epoch_{pred_type}_{metric_name}.png"
        plot_epoch_metrics(
            epoch_metrics,
            metric_name=metric_name,
            title=f"{metric_name.upper()} by Epoch ({pred_type})",
            save_path=str(save_path),
            pred_type=pred_type,
        )
        saved_paths[f'{metric_name}'] = str(save_path)
        plt.close()

    # 2. Combined plot with all metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, metric_name in enumerate(metrics_list):
        ax = axes[idx]
        dev_values = []
        test_values = []

        for e in epochs:
            metrics = epoch_metrics[e]
            dev_val = metrics.get(pred_type, {}).get('dev', {}).get(metric_name, None)
            test_val = metrics.get(pred_type, {}).get('test', {}).get(metric_name, None)
            dev_values.append(dev_val if dev_val is not None else np.nan)
            test_values.append(test_val if test_val is not None else np.nan)

        ax.plot(epochs, dev_values, 'o-', label='Dev', linewidth=2, markersize=6, color='steelblue')
        ax.plot(epochs, test_values, 's--', label='Test', linewidth=2, markersize=6, color='coral')

        # Mark best dev epoch
        valid_dev = [(e, v) for e, v in zip(epochs, dev_values) if not np.isnan(v)]
        if valid_dev:
            best_epoch = max(valid_dev, key=lambda x: x[1])[0]
            ax.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.7, linewidth=2)
            ax.scatter([best_epoch], [dev_values[epochs.index(best_epoch)]],
                      color='green', s=100, zorder=5, marker='*', label=f'e*={best_epoch}')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_name.upper(), fontsize=12)
        ax.set_title(metric_name.upper(), fontsize=14)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(epochs)

    plt.suptitle(f'Metrics by Epoch ({pred_type})', fontsize=16, y=1.02)
    plt.tight_layout()

    combined_path = output_dir / f"epoch_{pred_type}_all_metrics.png"
    fig.savefig(str(combined_path), dpi=150, bbox_inches='tight')
    saved_paths['combined'] = str(combined_path)
    plt.close()

    # 3. Dev vs Test comparison table plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for table-like visualization
    x = np.arange(len(epochs))
    width = 0.15

    colors = {'pearson': ['#1f77b4', '#aec7e8'],
              'spearman': ['#ff7f0e', '#ffbb78'],
              'qwk': ['#2ca02c', '#98df8a']}

    for idx, metric_name in enumerate(metrics_list):
        dev_values = []
        test_values = []
        for e in epochs:
            metrics = epoch_metrics[e]
            dev_val = metrics.get(pred_type, {}).get('dev', {}).get(metric_name, 0)
            test_val = metrics.get(pred_type, {}).get('test', {}).get(metric_name, 0)
            dev_values.append(dev_val if dev_val else 0)
            test_values.append(test_val if test_val else 0)

        offset = (idx - 1) * width * 2
        ax.bar(x + offset - width/2, dev_values, width, label=f'{metric_name} (dev)',
               color=colors[metric_name][0], alpha=0.8)
        ax.bar(x + offset + width/2, test_values, width, label=f'{metric_name} (test)',
               color=colors[metric_name][1], alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'All Metrics by Epoch ({pred_type})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(epochs)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    bar_path = output_dir / f"epoch_{pred_type}_metrics_bar.png"
    fig.savefig(str(bar_path), dpi=150, bbox_inches='tight')
    saved_paths['bar_chart'] = str(bar_path)
    plt.close()

    logger.info(f"Created {len(saved_paths)} epoch metric plots")
    return saved_paths


def create_evaluation_plots(
    results: List[Dict],
    y_min: int,
    y_max: int,
    output_dir: str,
    epoch: int,
    split_name: str = "test",
    pred_column: str = "y_hat_greedy",
) -> Dict[str, str]:
    """
    Create all evaluation plots for a set of results.

    Args:
        results: List of result dicts with y_true and predictions
        y_min: Minimum score
        y_max: Maximum score
        output_dir: Directory to save plots
        epoch: Epoch number
        split_name: Name of the split (dev/test)
        pred_column: Column name for predictions

    Returns:
        Dict mapping plot type to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract valid predictions
    y_true = []
    y_pred = []
    for r in results:
        if r.get('y_true') is not None and r.get(pred_column) is not None:
            y_true.append(int(r['y_true']))
            y_pred.append(int(r[pred_column]))

    if not y_true:
        logger.warning("No valid predictions to plot")
        return {}

    saved_paths = {}
    prefix = f"epoch_{epoch}_{split_name}_{pred_column}"

    # Scatter plot
    scatter_path = output_dir / f"{prefix}_scatter.png"
    plot_true_vs_predicted(
        y_true, y_pred, y_min, y_max,
        title=f"True vs Predicted ({split_name}, epoch {epoch})",
        save_path=str(scatter_path),
    )
    saved_paths['scatter'] = str(scatter_path)
    plt.close()

    # Confusion matrix
    cm_path = output_dir / f"{prefix}_confusion.png"
    plot_confusion_matrix(
        y_true, y_pred, y_min, y_max,
        title=f"Confusion Matrix ({split_name}, epoch {epoch})",
        save_path=str(cm_path),
    )
    saved_paths['confusion'] = str(cm_path)
    plt.close()

    # Normalized confusion matrix
    cm_norm_path = output_dir / f"{prefix}_confusion_norm.png"
    plot_confusion_matrix(
        y_true, y_pred, y_min, y_max,
        title=f"Confusion Matrix ({split_name}, epoch {epoch})",
        save_path=str(cm_norm_path),
        normalize=True,
    )
    saved_paths['confusion_normalized'] = str(cm_norm_path)
    plt.close()

    # Score distribution
    dist_path = output_dir / f"{prefix}_distribution.png"
    plot_score_distribution(
        y_true, y_pred, y_min, y_max,
        title=f"Score Distribution ({split_name}, epoch {epoch})",
        save_path=str(dist_path),
    )
    saved_paths['distribution'] = str(dist_path)
    plt.close()

    logger.info(f"Created {len(saved_paths)} plots for {split_name} epoch {epoch}")
    return saved_paths


def plot_metrics_progress(
    epochs: List[int],
    pearson_values: List[float],
    spearman_values: List[float],
    qwk_values: List[float],
    save_path: str,
    title: str = "Metrics Progress (All Essays)",
    max_epochs: int = None,
) -> None:
    """
    Plot Pearson, Spearman, and QWK progression on a single graph.
    Designed to be called after each epoch for live updates.

    Args:
        epochs: List of epoch numbers
        pearson_values: Pearson correlation values
        spearman_values: Spearman correlation values
        qwk_values: QWK values
        save_path: Path to save the figure
        title: Plot title
        max_epochs: Maximum epochs (for x-axis range)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot all three metrics
    ax.plot(epochs, pearson_values, 'o-', label='Pearson', linewidth=2, markersize=8, color='#1f77b4')
    ax.plot(epochs, spearman_values, 's-', label='Spearman', linewidth=2, markersize=8, color='#ff7f0e')
    ax.plot(epochs, qwk_values, '^-', label='QWK', linewidth=2, markersize=8, color='#2ca02c')

    # Mark best epochs for each metric
    if spearman_values:
        best_spearman_idx = np.argmax(spearman_values)
        ax.axvline(x=epochs[best_spearman_idx], color='#ff7f0e', linestyle=':', alpha=0.5)
        ax.scatter([epochs[best_spearman_idx]], [spearman_values[best_spearman_idx]],
                  color='#ff7f0e', s=150, zorder=5, marker='*', edgecolors='black')

    # Formatting
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Set x-axis range
    if max_epochs:
        ax.set_xlim(-0.5, max_epochs + 0.5)
    ax.set_ylim(0, 1.0)

    # Add epoch 0 baseline annotation
    if len(epochs) > 0 and epochs[0] == 0:
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        ax.annotate('Baseline', xy=(0, spearman_values[0]), xytext=(1, spearman_values[0] - 0.05),
                   fontsize=10, color='gray')

    # Show current best values in legend
    if spearman_values:
        best_spearman = max(spearman_values)
        best_epoch = epochs[np.argmax(spearman_values)]
        ax.text(0.02, 0.98, f'Best Spearman: {best_spearman:.4f} (epoch {best_epoch})',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Updated metrics progress plot: {save_path}")
