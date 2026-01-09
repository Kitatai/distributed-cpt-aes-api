#!/usr/bin/env python3
"""
Generate tasks for pairwise comparison experiments.

Creates tasks with:
- Comparison pairs (each essay participates exactly k times)
- Epochs to evaluate
- Model and prompt configuration

Usage:
    python generate_pairwise_tasks.py --server http://localhost:8000 --model llama3b --prompt 1
"""

import json
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SERVER_DIR = Path(__file__).parent
DATA_DIR = SERVER_DIR / "data"

MODEL_CONFIGS = {
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}


class ComparisonPairGenerator:
    """Generate comparison pairs where each essay participates exactly k times."""

    def __init__(self, essay_ids: List[int], comparisons_per_essay: int = 5, seed: int = 42):
        self.essay_ids = list(essay_ids)
        self.comparisons_per_essay = comparisons_per_essay
        self.seed = seed
        random.seed(seed)

    def generate_pairs(self) -> List[Tuple[int, int]]:
        """Generate comparison pairs with fair participation."""
        n = len(self.essay_ids)
        k = self.comparisons_per_essay

        if n < 2:
            raise ValueError("Need at least 2 essays to compare")

        total_comparisons = (n * k) // 2
        usage_count = {eid: 0 for eid in self.essay_ids}
        pairs = []

        shuffled_ids = self.essay_ids.copy()
        random.shuffle(shuffled_ids)

        max_iterations = n * k * 10
        iteration = 0

        while len(pairs) < total_comparisons and iteration < max_iterations:
            iteration += 1

            available = [eid for eid in shuffled_ids if usage_count[eid] < k]

            if len(available) < 2:
                break

            available.sort(key=lambda x: usage_count[x])

            found_pair = False
            for i in range(len(available)):
                for j in range(i + 1, len(available)):
                    a, b = available[i], available[j]
                    if (a, b) not in pairs and (b, a) not in pairs:
                        # Randomize order
                        if random.random() < 0.5:
                            pairs.append((a, b))
                        else:
                            pairs.append((b, a))
                        usage_count[a] += 1
                        usage_count[b] += 1
                        found_pair = True
                        break
                if found_pair:
                    break

            if not found_pair:
                random.shuffle(shuffled_ids)

        return pairs


def generate_tasks(
    server_url: str,
    exp_name: str,
    model_short: str,
    prompts: List[int],
    patterns: List[int],
    epochs: List[int],
    comparisons_per_essay: int = 5,
    seed: int = 42,
    epoch_specific_pairs: bool = False,
):
    """Generate and register tasks on the server.

    Args:
        epoch_specific_pairs: If True, generate different comparison pairs for each epoch.
                             This allows testing rater-specific BT with independent comparisons.
    """

    # Load sample patterns
    patterns_path = DATA_DIR / "backup_zeroshot_v3" / "sample_patterns_v2.json"
    with open(patterns_path) as f:
        patterns_data = json.load(f)

    model_name = MODEL_CONFIGS[model_short]
    tasks = []

    for prompt_id in prompts:
        prompt_patterns = patterns_data["patterns"].get(f"prompt{prompt_id}", [])

        for pattern_idx in patterns:
            if pattern_idx >= len(prompt_patterns):
                logger.warning(f"Pattern {pattern_idx} not found for prompt {prompt_id}")
                continue

            pattern = prompt_patterns[pattern_idx]
            test_ids = pattern["test_ids"]

            if epoch_specific_pairs:
                # Generate different pairs for each epoch
                comparison_pairs_by_epoch = {}
                for epoch in epochs:
                    generator = ComparisonPairGenerator(
                        essay_ids=test_ids,
                        comparisons_per_essay=comparisons_per_essay,
                        seed=seed + epoch * 1000,  # Different seed per epoch
                    )
                    comparison_pairs_by_epoch[epoch] = generator.generate_pairs()

                task_id = f"prompt{prompt_id}_{model_short}_p{pattern_idx}_epochvar"
                first_epoch_pairs = comparison_pairs_by_epoch[epochs[0]]

                task = {
                    "task_id": task_id,
                    "prompt_id": prompt_id,
                    "model_name": model_name,
                    "model_short_name": model_short,
                    "pattern_idx": pattern_idx,
                    "test_ids": test_ids,
                    "comparison_pairs_by_epoch": comparison_pairs_by_epoch,
                    "epochs": epochs,
                    "comparisons_per_essay": comparisons_per_essay,
                    "seed": seed,
                    "epoch_specific_pairs": True,
                }

                logger.info(f"Generated task: {task_id} (epoch-specific pairs)")
                logger.info(f"  Test essays: {len(test_ids)}, Pairs/epoch: {len(first_epoch_pairs)}")
            else:
                # Generate same pairs for all epochs (original behavior)
                generator = ComparisonPairGenerator(
                    essay_ids=test_ids,
                    comparisons_per_essay=comparisons_per_essay,
                    seed=seed,
                )
                comparison_pairs = generator.generate_pairs()

                task_id = f"prompt{prompt_id}_{model_short}_p{pattern_idx}"

                task = {
                    "task_id": task_id,
                    "prompt_id": prompt_id,
                    "model_name": model_name,
                    "model_short_name": model_short,
                    "pattern_idx": pattern_idx,
                    "test_ids": test_ids,
                    "comparison_pairs": comparison_pairs,
                    "epochs": epochs,
                    "comparisons_per_essay": comparisons_per_essay,
                    "seed": seed,
                }

                logger.info(f"Generated task: {task_id}")
                logger.info(f"  Test essays: {len(test_ids)}, Pairs: {len(comparison_pairs)}")

            tasks.append(task)

    # Register tasks on server
    logger.info(f"\nRegistering {len(tasks)} tasks on server...")

    for task in tasks:
        response = requests.post(
            f"{server_url}/{exp_name}/tasks",
            json=task,
        )
        if response.status_code == 200:
            logger.info(f"Registered: {task['task_id']}")
        else:
            logger.error(f"Failed to register {task['task_id']}: {response.text}")

    logger.info(f"\nDone! Registered {len(tasks)} tasks.")


def main():
    parser = argparse.ArgumentParser(description="Generate pairwise comparison tasks")
    parser.add_argument("--server", type=str, default="http://localhost:8000",
                       help="API server URL")
    parser.add_argument("--exp-name", type=str, default="pairwise",
                       help="Experiment name")
    parser.add_argument("--model", type=str, default="llama3b",
                       choices=["llama3b", "llama8b", "mistral"],
                       help="Model to use")
    parser.add_argument("--prompts", type=str, default="1",
                       help="Prompt IDs (comma-separated, e.g., '1,2,3')")
    parser.add_argument("--patterns", type=str, default="0",
                       help="Pattern indices (comma-separated, e.g., '0,1,2')")
    parser.add_argument("--epochs", type=str, default="0-30",
                       help="Epochs to evaluate (e.g., '0-30' or '0,5,10,20,30')")
    parser.add_argument("--comparisons", type=int, default=5,
                       help="Number of comparisons per essay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epoch-specific-pairs", action="store_true",
                       help="Generate different comparison pairs for each epoch")

    args = parser.parse_args()

    # Parse prompts
    prompts = [int(p) for p in args.prompts.split(",")]

    # Parse patterns
    patterns = [int(p) for p in args.patterns.split(",")]

    # Parse epochs
    if "-" in args.epochs:
        start, end = map(int, args.epochs.split("-"))
        epochs = list(range(start, end + 1))
    else:
        epochs = [int(e) for e in args.epochs.split(",")]

    generate_tasks(
        server_url=args.server,
        exp_name=args.exp_name,
        model_short=args.model,
        prompts=prompts,
        patterns=patterns,
        epochs=epochs,
        comparisons_per_essay=args.comparisons,
        seed=args.seed,
        epoch_specific_pairs=args.epoch_specific_pairs,
    )


if __name__ == "__main__":
    main()
