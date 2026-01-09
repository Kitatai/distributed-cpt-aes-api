#!/usr/bin/env python3
"""
Analyze pairwise comparison experiment results and generate figures.

Usage:
    python analyze_pairwise_results.py [--output-dir figures/]
"""

import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SERVER_DIR = Path(__file__).parent
RESULTS_DIR = SERVER_DIR / "data" / "results_pairwise"


def load_all_results():
    """Load all pairwise experiment results."""
    results = []
    for task_dir in RESULTS_DIR.iterdir():
        if not task_dir.is_dir():
            continue
        result_file = task_dir / "result.json"
        if result_file.exists():
            with open(result_file) as f:
                results.append(json.load(f))
    return results


def aggregate_by_prompt_model(results):
    """Aggregate results by prompt and model."""
    aggregated = defaultdict(lambda: {"epoch_spearman": defaultdict(list), "n_patterns": 0})

    for result in results:
        prompt_id = result.get("prompt_id")
        model_short = result.get("model_short")
        results_by_epoch = result.get("results_by_epoch", {})

        key = f"prompt{prompt_id}_{model_short}"
        aggregated[key]["prompt_id"] = prompt_id
        aggregated[key]["model_short"] = model_short
        aggregated[key]["n_patterns"] += 1

        for epoch_str, epoch_result in results_by_epoch.items():
            epoch = int(epoch_str)
            spearman = epoch_result.get("spearman", 0)
            aggregated[key]["epoch_spearman"][epoch].append(spearman)

    return dict(aggregated)


def generate_figure(aggregated, output_dir: Path):
    """Generate figure showing Spearman correlation across epochs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, data in aggregated.items():
        prompt_id = data["prompt_id"]
        model_short = data["model_short"]
        epoch_spearman = data["epoch_spearman"]

        if not epoch_spearman:
            continue

        # Sort epochs
        epochs = sorted(epoch_spearman.keys())
        means = [np.mean(epoch_spearman[e]) for e in epochs]
        stds = [np.std(epoch_spearman[e]) for e in epochs]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.errorbar(epochs, means, yerr=stds, fmt='b-o', linewidth=2, markersize=6, capsize=3)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Spearman Correlation", fontsize=12)
        ax.set_title(f"Pairwise Comparison - Prompt {prompt_id}, {model_short}", fontsize=14)
        ax.grid(True, alpha=0.3)

        if epochs:
            ax.set_xticks(range(0, max(epochs) + 1, 5))

        # Highlight best epoch
        best_idx = np.argmax(means)
        best_epoch = epochs[best_idx]
        best_spearman = means[best_idx]
        ax.scatter([best_epoch], [best_spearman], c='red', s=100, zorder=5,
                   label=f"Best: E{best_epoch} ({best_spearman:.3f})")
        ax.legend()

        plt.tight_layout()
        fig_path = output_dir / f"figure_pairwise_{key}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logger.info(f"Saved figure to {fig_path}")


def generate_comparison_figure(aggregated, output_dir: Path):
    """Generate comparison figure for all models on same prompt."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by prompt
    by_prompt = defaultdict(dict)
    for key, data in aggregated.items():
        prompt_id = data["prompt_id"]
        model_short = data["model_short"]
        by_prompt[prompt_id][model_short] = data

    # Color map for models
    colors = {"llama3b": "blue", "llama8b": "green", "mistral": "orange"}

    for prompt_id, models_data in by_prompt.items():
        if len(models_data) < 1:
            continue

        fig, ax = plt.subplots(figsize=(12, 7))

        for model_short, data in models_data.items():
            epoch_spearman = data["epoch_spearman"]
            if not epoch_spearman:
                continue

            epochs = sorted(epoch_spearman.keys())
            means = [np.mean(epoch_spearman[e]) for e in epochs]
            stds = [np.std(epoch_spearman[e]) for e in epochs]

            color = colors.get(model_short, "gray")
            ax.errorbar(epochs, means, yerr=stds, fmt='-o', color=color,
                        linewidth=2, markersize=5, capsize=3, label=model_short)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Spearman Correlation", fontsize=12)
        ax.set_title(f"Pairwise Comparison - Prompt {prompt_id} (All Models)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        if epochs:
            ax.set_xticks(range(0, max(epochs) + 1, 5))

        plt.tight_layout()
        fig_path = output_dir / f"figure_pairwise_prompt{prompt_id}_comparison.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logger.info(f"Saved comparison figure to {fig_path}")


def print_summary(aggregated):
    """Print summary of results."""
    print("\n" + "=" * 60)
    print("Pairwise Comparison Results Summary")
    print("=" * 60)

    for key, data in sorted(aggregated.items()):
        prompt_id = data["prompt_id"]
        model_short = data["model_short"]
        n_patterns = data["n_patterns"]
        epoch_spearman = data["epoch_spearman"]

        print(f"\n{key} (n={n_patterns}):")

        if not epoch_spearman:
            print("  No results")
            continue

        epochs = sorted(epoch_spearman.keys())
        means = [np.mean(epoch_spearman[e]) for e in epochs]

        # Find best epoch
        best_idx = np.argmax(means)
        best_epoch = epochs[best_idx]
        best_spearman = means[best_idx]

        # E0 spearman
        e0_spearman = np.mean(epoch_spearman.get(0, [0]))

        print(f"  E0 Spearman: {e0_spearman:.3f}")
        print(f"  Best Epoch: E{best_epoch} ({best_spearman:.3f})")
        print(f"  Improvement: {best_spearman - e0_spearman:+.3f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze pairwise comparison results")
    parser.add_argument("--output-dir", type=str, default="figures",
                       help="Output directory for figures")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    logger.info("Loading results...")
    results = load_all_results()
    logger.info(f"Loaded {len(results)} results")

    if not results:
        logger.warning("No results found")
        return

    logger.info("Aggregating results...")
    aggregated = aggregate_by_prompt_model(results)

    print_summary(aggregated)

    logger.info("Generating figures...")
    generate_figure(aggregated, output_dir)
    generate_comparison_figure(aggregated, output_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()
