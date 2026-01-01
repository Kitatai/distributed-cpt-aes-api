#!/usr/bin/env python3
"""
Generate few-shot v2 experiment tasks.

Creates tasks for the new few-shot approach:
- Split 10 dev samples: 3 shot + 7 eval
- Search all epochs 0-30 using 7-sample MSE
- Evaluate test set at best epoch

Usage:
    python generate_fewshot_v2_tasks.py --prompts 1,2,3,4,5,6,7,8 --patterns 1
    python generate_fewshot_v2_tasks.py --prompts 1 --patterns 50  # All 50 patterns
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: Path):
    """Save JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate few-shot v2 tasks")
    parser.add_argument(
        "--prompts",
        type=str,
        default="1,2,3,4,5,6,7,8",
        help="Comma-separated prompt IDs (default: 1,2,3,4,5,6,7,8)",
    )
    parser.add_argument(
        "--patterns",
        type=int,
        default=1,
        help="Number of patterns to use (1-50, default: 1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama8b",
        choices=["llama8b", "llama3b", "mistral"],
        help="Model to use (default: llama8b)",
    )
    parser.add_argument(
        "--n-shot",
        type=int,
        default=3,
        help="Number of few-shot examples (default: 3)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test sample ratio (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (default: data/backup_zeroshot_v1)",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = script_dir / "data" / "backup_zeroshot_v1"

    # Model mapping
    model_configs = {
        "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
        "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    }

    model_short = args.model
    model_name = model_configs[model_short]

    # Parse prompts
    prompt_ids = [int(p.strip()) for p in args.prompts.split(",")]
    n_patterns = args.patterns

    # Load sample patterns
    patterns_path = data_dir / "sample_patterns.json"
    print(f"Loading sample patterns from {patterns_path}")
    patterns = load_json(patterns_path)

    # Output directory
    tasks_dir = data_dir / "tasks_fewshot_v2"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    tasks_created = []

    for prompt_id in prompt_ids:
        prompt_key = f"prompt{prompt_id}"
        prompt_patterns = patterns['patterns'].get(prompt_key, [])

        if len(prompt_patterns) < n_patterns:
            print(f"Warning: Only {len(prompt_patterns)} patterns available for {prompt_key}")
            n_patterns_actual = len(prompt_patterns)
        else:
            n_patterns_actual = n_patterns

        for pattern_idx in range(n_patterns_actual):
            sample_ids = prompt_patterns[pattern_idx]

            if len(sample_ids) < 10:
                print(f"Warning: Pattern {pattern_idx} has only {len(sample_ids)} samples, skipping")
                continue

            task_id = f"fewshot_v2_prompt{prompt_id}_{model_short}_p{pattern_idx}"

            task_config = {
                "task_id": task_id,
                "prompt_id": prompt_id,
                "model_name": model_name,
                "model_short": model_short,
                "dataset": "asap",
                "pattern_idx": pattern_idx,
                "sample_ids": sample_ids,
                "n_shot": args.n_shot,
                "split_seed": 42,
                "test_sample_ratio": args.test_ratio,
                "max_epochs": 30,
                "status": "pending",
                "created_at": datetime.now().isoformat(),
            }

            # Save task
            task_path = tasks_dir / f"{task_id}.json"
            save_json(task_config, task_path)
            tasks_created.append(task_id)

            print(f"Created {task_id}:")
            print(f"  sample_ids: {sample_ids}")
            print(f"  n_shot: {args.n_shot}, test_ratio: {args.test_ratio}")

    print(f"\n{len(tasks_created)} tasks created in {tasks_dir}")
    print("Tasks:", tasks_created)

    # Print run command
    print("\nTo run the experiment:")
    print(f"  cd {script_dir.parent / 'client'}")
    print(f"  python worker_fewshot_v2.py --data-dir {data_dir}")


if __name__ == "__main__":
    main()
