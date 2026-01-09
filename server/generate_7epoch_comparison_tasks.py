#!/usr/bin/env python3
"""
Generate tasks for 7-epoch comparison experiment.

Creates two tasks:
1. 7-epoch task: epochs 0,5,10,15,20,25,30 with epoch-specific pairs (1 comparison/essay/epoch)
2. Baseline task: epoch 0 only, but runs ALL comparison pairs from the 7 epochs

This allows fair comparison between P2c (multi-epoch) and plain BT (single epoch, same total comparisons).
"""

import json
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SERVER_DIR = Path(__file__).parent
DATA_DIR = SERVER_DIR / "data"

MODEL_CONFIGS = {
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

EPOCHS_7 = [0, 5, 10, 15, 20, 25, 30]


class ComparisonPairGenerator:
    """Generate comparison pairs where each essay participates exactly k times."""

    def __init__(self, essay_ids: List[int], comparisons_per_essay: int = 1, seed: int = 42):
        self.essay_ids = list(essay_ids)
        self.comparisons_per_essay = comparisons_per_essay
        random.seed(seed)

    def generate_pairs(self) -> List[Tuple[int, int]]:
        n = len(self.essay_ids)
        k = self.comparisons_per_essay
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
    prompt_id: int,
    pattern_idx: int = 0,
    seed: int = 42,
):
    """Generate two tasks for the 7-epoch comparison experiment."""

    # Load sample patterns
    patterns_path = DATA_DIR / "backup_zeroshot_v3" / "sample_patterns_v2.json"
    with open(patterns_path) as f:
        patterns_data = json.load(f)

    model_name = MODEL_CONFIGS[model_short]
    prompt_patterns = patterns_data["patterns"].get(f"prompt{prompt_id}", [])

    if pattern_idx >= len(prompt_patterns):
        raise ValueError(f"Pattern {pattern_idx} not found for prompt {prompt_id}")

    pattern = prompt_patterns[pattern_idx]
    test_ids = pattern["test_ids"]

    # Generate epoch-specific pairs for 7 epochs (1 comparison per essay per epoch)
    comparison_pairs_by_epoch = {}
    all_pairs_combined = []  # All pairs for baseline task

    for epoch in EPOCHS_7:
        generator = ComparisonPairGenerator(
            essay_ids=test_ids,
            comparisons_per_essay=1,  # 1 comparison per essay per epoch
            seed=seed + epoch * 1000,
        )
        pairs = generator.generate_pairs()
        comparison_pairs_by_epoch[epoch] = pairs
        all_pairs_combined.extend(pairs)
        logger.info(f"Epoch {epoch}: {len(pairs)} pairs")

    logger.info(f"Total pairs across 7 epochs: {len(all_pairs_combined)}")

    tasks = []

    # Task 1: 7-epoch P2c task
    task1_id = f"prompt{prompt_id}_{model_short}_p{pattern_idx}_7epoch"
    task1 = {
        "task_id": task1_id,
        "prompt_id": prompt_id,
        "model_name": model_name,
        "model_short_name": model_short,
        "pattern_idx": pattern_idx,
        "test_ids": test_ids,
        "comparison_pairs_by_epoch": comparison_pairs_by_epoch,
        "epochs": EPOCHS_7,
        "comparisons_per_essay": 1,
        "seed": seed,
        "epoch_specific_pairs": True,
    }
    tasks.append(task1)
    logger.info(f"Task 1: {task1_id} (7 epochs, {len(test_ids)} essays, {len(comparison_pairs_by_epoch[0])} pairs/epoch)")

    # Task 2: Baseline (epoch 0 only, but with ALL pairs from 7 epochs)
    task2_id = f"prompt{prompt_id}_{model_short}_p{pattern_idx}_baseline7"
    task2 = {
        "task_id": task2_id,
        "prompt_id": prompt_id,
        "model_name": model_name,
        "model_short_name": model_short,
        "pattern_idx": pattern_idx,
        "test_ids": test_ids,
        "comparison_pairs": all_pairs_combined,  # All pairs from 7 epochs
        "epochs": [0],  # Only epoch 0
        "comparisons_per_essay": 7,  # Total 7 comparisons per essay
        "seed": seed,
        "epoch_specific_pairs": False,
    }
    tasks.append(task2)
    logger.info(f"Task 2: {task2_id} (epoch 0 only, {len(all_pairs_combined)} total pairs)")

    # Register tasks
    logger.info(f"\nRegistering {len(tasks)} tasks on server...")
    for task in tasks:
        response = requests.post(f"{server_url}/{exp_name}/tasks", json=task)
        if response.status_code == 200:
            logger.info(f"Registered: {task['task_id']}")
        else:
            logger.error(f"Failed to register {task['task_id']}: {response.text}")

    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--exp-name", default="pairwise")
    parser.add_argument("--model", choices=["llama3b", "llama8b", "mistral"], default="llama8b")
    parser.add_argument("--prompt", type=int, default=1)
    parser.add_argument("--pattern", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_tasks(
        server_url=args.server,
        exp_name=args.exp_name,
        model_short=args.model,
        prompt_id=args.prompt,
        pattern_idx=args.pattern,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
