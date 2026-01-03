#!/usr/bin/env python3
"""
Generate few-shot v2 experiment tasks.

Creates tasks for few-shot experiments using new sample patterns:
- 3 models × 8 prompts × 3 k values × 10 patterns = 720 tasks
- Uses sample_patterns_v2.json with test_ids, dev_ids, fewshot_ids

Usage:
    python generate_fewshot_v2_tasks.py [--n-patterns N] [--k-values K1,K2,K3]
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Generate few-shot v2 tasks")
    parser.add_argument("--n-patterns", type=int, default=10,
                        help="Number of patterns to use (default: 10)")
    parser.add_argument("--k-values", type=str, default="1,3,5",
                        help="Comma-separated k values for few-shot (default: 1,3,5)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Base data directory (default: data)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: DATA_DIR/tasks_fewshot_v2)")
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]

    script_dir = Path(__file__).parent

    # Data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            data_dir = script_dir / data_dir
    else:
        data_dir = script_dir / "data"

    # Load patterns
    patterns_path = data_dir / "sample_patterns_v2.json"
    print(f"Loading patterns from {patterns_path}")
    with open(patterns_path) as f:
        patterns_data = json.load(f)
    patterns = patterns_data['patterns']

    # Load experiment config
    config_path = data_dir / "experiment_config.json"
    if not config_path.exists():
        config_path = script_dir / "data" / "experiment_config.json"
    print(f"Loading experiment config from {config_path}")
    with open(config_path) as f:
        exp_config = json.load(f)

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = data_dir / "tasks_fewshot_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Models
    models = [
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", "llama8b"),
        ("meta-llama/Llama-3.2-3B-Instruct", "llama3b"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
    ]

    tasks_created = []

    for model_name, model_short in models:
        for prompt_id in range(1, 9):
            prompt_key = f"prompt{prompt_id}"

            for k in k_values:
                for pattern_idx in range(args.n_patterns):
                    task_id = f"fewshot_k{k}_prompt{prompt_id}_{model_short}_p{pattern_idx}"

                    pattern = patterns[prompt_key][pattern_idx]
                    test_ids = pattern['test_ids']
                    dev_ids = pattern['dev_ids']
                    fewshot_ids = pattern['fewshot_ids']

                    # Use first k examples from fewshot_ids
                    example_ids = fewshot_ids[:k]

                    task_config = {
                        "task_id": task_id,
                        "prompt_id": prompt_id,
                        "model_name": model_name,
                        "model_short_name": model_short,
                        "k": k,
                        "pattern_idx": pattern_idx,
                        "test_ids": test_ids,
                        "dev_ids": dev_ids,
                        "example_ids": example_ids,
                        "dataset": "asap",
                        # Hyperparameters
                        "lr": exp_config.get("lr", 1e-5),
                        "lora_r": exp_config.get("lora_r", 16),
                        "lora_alpha": exp_config.get("lora_alpha", 32),
                        "max_seq_len": exp_config.get("max_seq_len", 2048),
                        "max_epochs": exp_config.get("max_epochs", 30),
                        # Status
                        "status": "pending",
                        "created_at": datetime.now().isoformat(),
                    }

                    task_path = output_dir / f"{task_id}.json"
                    with open(task_path, 'w') as f:
                        json.dump(task_config, f, indent=2)

                    tasks_created.append(task_id)

    print(f"\n{len(tasks_created)} tasks created in {output_dir}")
    print(f"  Models: {[m[1] for m in models]}")
    print(f"  Prompts: 1-8")
    print(f"  K values: {k_values}")
    print(f"  Patterns: 0-{args.n_patterns - 1}")

    # Show example
    print(f"\nExample task: {tasks_created[0]}")
    with open(output_dir / f"{tasks_created[0]}.json") as f:
        example = json.load(f)
    print(f"  k: {example['k']}")
    print(f"  pattern_idx: {example['pattern_idx']}")
    print(f"  n_test: {len(example['test_ids'])}")
    print(f"  n_dev: {len(example['dev_ids'])}")
    print(f"  example_ids: {example['example_ids']}")


if __name__ == "__main__":
    main()
