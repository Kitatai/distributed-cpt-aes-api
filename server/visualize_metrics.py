#!/usr/bin/env python3
"""
Visualize QWK and Spearman metrics over epochs for completed tasks.

Usage:
    python visualize_metrics.py

Output:
    data/visualizations/prompt{N}_metrics.png - Per-prompt plots
    data/visualizations/all_tasks_metrics.png - Combined plot
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt


# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "data" / "results"
VIS_DIR = SCRIPT_DIR / "data" / "visualizations"

# Color scheme for models
MODEL_COLORS = {
    "llama3b": "#1f77b4",  # blue
    "llama8b": "#ff7f0e",  # orange
    "mistral": "#2ca02c",  # green
    "qwen": "#d62728",     # red
}

# Line styles for different prompts
PROMPT_STYLES = {
    "prompt1": "-",
    "prompt2": "--",
    "prompt3": ":",
    "prompt4": "-.",
    "prompt5": "-",
    "prompt6": "--",
    "prompt7": ":",
    "prompt8": "-.",
}


def find_tasks_with_results(min_epochs: int = 1) -> list:
    """Find tasks with at least min_epochs of results (including in-progress)."""
    tasks = []
    for task_dir in RESULTS_DIR.iterdir():
        if task_dir.is_dir() and not task_dir.name.startswith("toefl11"):
            metrics_files = list(task_dir.glob("metrics_epoch_*.json"))
            if len(metrics_files) >= min_epochs:
                tasks.append(task_dir.name)
    return sorted(tasks)


def load_task_metrics(task_id: str) -> dict:
    """Load metrics for all epochs of a task."""
    task_dir = RESULTS_DIR / task_id
    epochs = []
    qwk_values = []
    spearman_values = []

    for epoch in range(31):  # 0-30
        metrics_file = task_dir / f"metrics_epoch_{epoch}.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
            test_metrics = data.get("greedy", {}).get("test", {})
            qwk = test_metrics.get("qwk")
            spearman = test_metrics.get("spearman")
            if qwk is not None and spearman is not None:
                epochs.append(epoch)
                qwk_values.append(qwk)
                spearman_values.append(spearman)

    return {
        "epochs": epochs,
        "qwk": qwk_values,
        "spearman": spearman_values,
    }


def create_prompt_plot(prompt: str, tasks: dict, output_path: Path):
    """Create a plot for a single prompt with all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{prompt.upper()} - Metrics over Epochs", fontsize=14)

    # QWK plot
    ax1 = axes[0]
    for task_id, metrics in sorted(tasks.items()):
        model = task_id.split("_")[1]
        color = MODEL_COLORS.get(model, "#333333")
        n_epochs = len(metrics["epochs"])
        is_complete = n_epochs >= 31
        linestyle = "-" if is_complete else "--"
        label = model if is_complete else f"{model} ({n_epochs}ep)"
        ax1.plot(metrics["epochs"], metrics["qwk"],
                 label=label, color=color, marker='o', markersize=3,
                 linewidth=1.5, linestyle=linestyle)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("QWK")
    ax1.set_title("Quadratic Weighted Kappa (QWK)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 30)

    # Spearman plot
    ax2 = axes[1]
    for task_id, metrics in sorted(tasks.items()):
        model = task_id.split("_")[1]
        color = MODEL_COLORS.get(model, "#333333")
        n_epochs = len(metrics["epochs"])
        is_complete = n_epochs >= 31
        linestyle = "-" if is_complete else "--"
        label = model if is_complete else f"{model} ({n_epochs}ep)"
        ax2.plot(metrics["epochs"], metrics["spearman"],
                 label=label, color=color, marker='o', markersize=3,
                 linewidth=1.5, linestyle=linestyle)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Spearman ρ")
    ax2.set_title("Spearman Rank Correlation")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_combined_plot(task_metrics: dict, output_path: Path):
    """Create a combined plot for all tasks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("All Tasks - Metrics over Epochs", fontsize=14)

    ax1, ax2 = axes
    for task_id, metrics in sorted(task_metrics.items()):
        prompt = task_id.split("_")[0]
        model = task_id.split("_")[1]
        color = MODEL_COLORS.get(model, "#333333")
        n_epochs = len(metrics["epochs"])
        is_complete = n_epochs >= 31
        style = "-" if is_complete else "--"
        label = f"{prompt}_{model}" if is_complete else f"{prompt}_{model} ({n_epochs}ep)"

        ax1.plot(metrics["epochs"], metrics["qwk"],
                 label=label, color=color, linestyle=style, marker='o', markersize=2, linewidth=1.2)
        ax2.plot(metrics["epochs"], metrics["spearman"],
                 label=label, color=color, linestyle=style, marker='o', markersize=2, linewidth=1.2)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("QWK")
    ax1.set_title("Quadratic Weighted Kappa (QWK)")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 30)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Spearman ρ")
    ax2.set_title("Spearman Rank Correlation")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_summary(task_metrics: dict):
    """Print summary of final epoch results."""
    print("\n=== Final Epoch Results ===")
    print(f"{'Task':<25} {'QWK':>8} {'Spearman':>10}")
    print("-" * 45)

    # Sort by QWK descending
    sorted_tasks = sorted(
        task_metrics.items(),
        key=lambda x: x[1]["qwk"][-1] if x[1]["qwk"] else 0,
        reverse=True
    )

    for task_id, metrics in sorted_tasks:
        if metrics["qwk"] and metrics["spearman"]:
            qwk = metrics["qwk"][-1]
            spearman = metrics["spearman"][-1]
            print(f"{task_id:<25} {qwk:>8.4f} {spearman:>10.4f}")


def main():
    # Create output directory
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # Find all tasks with results (including in-progress)
    tasks_with_results = find_tasks_with_results()
    print(f"Found {len(tasks_with_results)} tasks with results")

    if not tasks_with_results:
        print("No tasks with results found.")
        return

    # Load metrics for all tasks
    task_metrics = {}
    for task_id in tasks_with_results:
        metrics = load_task_metrics(task_id)
        if metrics["epochs"]:
            task_metrics[task_id] = metrics
            n_epochs = len(metrics['epochs'])
            status = "✓" if n_epochs >= 31 else f"({n_epochs}/31)"
            print(f"  {task_id}: {n_epochs} epochs {status}, "
                  f"latest QWK={metrics['qwk'][-1]:.4f}, "
                  f"Spearman={metrics['spearman'][-1]:.4f}")

    # Group by prompt
    prompts = {}
    for task_id, metrics in task_metrics.items():
        prompt = task_id.split("_")[0]
        if prompt not in prompts:
            prompts[prompt] = {}
        prompts[prompt][task_id] = metrics

    # Create per-prompt plots
    for prompt, tasks in sorted(prompts.items()):
        output_path = VIS_DIR / f"{prompt}_metrics.png"
        create_prompt_plot(prompt, tasks, output_path)
        print(f"Saved: {output_path}")

    # Create combined plot
    output_path = VIS_DIR / "all_tasks_metrics.png"
    create_combined_plot(task_metrics, output_path)
    print(f"Saved: {output_path}")

    # Print summary
    print_summary(task_metrics)

    print(f"\nVisualization complete. Output: {VIS_DIR}")


if __name__ == "__main__":
    main()
