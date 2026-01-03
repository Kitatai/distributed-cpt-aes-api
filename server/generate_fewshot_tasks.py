#!/usr/bin/env python3
"""
Generate few-shot experiment tasks.

Creates tasks for few-shot scoring experiments using:
- Best epochs from analysis_results.json
- Sample patterns from sample_patterns.json
- 5 randomly selected examples from each 10-sample pattern (5-shot)
- All 3 models: llama8b, llama3b, mistral

Usage:
    python generate_fewshot_tasks.py
"""

import json
import random
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
    script_dir = Path(__file__).parent
    
    # Load analysis results (to get best epochs)
    analysis_path = script_dir / "data" / "analysis_results.json"
    print(f"Loading analysis results from {analysis_path}")
    analysis = load_json(analysis_path)
    
    # Load sample patterns (to get 10-sample essay IDs)
    patterns_path = script_dir / "data" / "sample_patterns.json"
    print(f"Loading sample patterns from {patterns_path}")
    patterns = load_json(patterns_path)
    
    # Load experiment config (for hyperparameters)
    config_path = script_dir / "data" / "experiment_config.json"
    print(f"Loading experiment config from {config_path}")
    exp_config = load_json(config_path)
    
    # Output directory
    tasks_dir = script_dir / "data" / "tasks_fewshot"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    
    # Model configs
    models = [
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", "llama8b"),
        ("meta-llama/Llama-3.2-3B-Instruct", "llama3b"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
    ]

    # Few-shot config
    n_fewshot = 5  # 5-shot experiment
    n_dev = 10  # 10 samples for epoch selection (total = n_fewshot + n_dev = 15)
    fewshot_seed = 42
    pattern_idx = 0  # Use first pattern only

    # Load ASAP data for additional sample selection
    asap_path = script_dir / "data" / "asap" / "training_set_rel3.tsv"
    import pandas as pd
    asap_df = pd.read_csv(asap_path, sep='\t', encoding='latin-1')

    tasks_created = []

    for model_name, model_short in models:
        for prompt_id in range(1, 9):
            task_id = f"fewshot_prompt{prompt_id}_{model_short}"

            # Get best epoch from analysis results
            zeroshot_task_id = f"prompt{prompt_id}_{model_short}"
            task_details = analysis.get('details', {}).get(zeroshot_task_id, {})
            pattern_details = task_details.get('pattern_details', [])

            if not pattern_details or len(pattern_details) <= pattern_idx:
                print(f"Warning: No pattern details for {zeroshot_task_id}, skipping")
                continue

            best_epoch = pattern_details[pattern_idx]['best_epoch']

            # Get 10 sample IDs from patterns
            prompt_key = f"prompt{prompt_id}"
            base_sample_ids = patterns['patterns'].get(prompt_key, [])[pattern_idx]

            if len(base_sample_ids) < 10:
                print(f"Warning: Not enough samples for {prompt_key}, skipping")
                continue

            # Select n_fewshot examples with fixed seed
            rng = random.Random(fewshot_seed + prompt_id)
            example_ids = rng.sample(base_sample_ids, n_fewshot)

            # Get additional samples for dev (to reach n_fewshot + n_dev total)
            # Select from essays not in base_sample_ids
            prompt_essays = asap_df[asap_df['essay_set'] == prompt_id]['essay_id'].tolist()
            available_ids = [eid for eid in prompt_essays if eid not in base_sample_ids]
            n_additional = (n_fewshot + n_dev) - len(base_sample_ids)  # 15 - 10 = 5
            additional_ids = rng.sample(available_ids, n_additional)

            # Combine: 10 from patterns + 5 additional = 15 total
            sample_ids = base_sample_ids + additional_ids

            # Create task config
            task_config = {
                "task_id": task_id,
                "prompt_id": prompt_id,
                "model_name": model_name,
                "model_short_name": model_short,
                "dataset": "asap",
                "best_epoch": best_epoch,
                "sample_ids": sample_ids,  # All 10 samples (excluded from evaluation)
                "example_ids": example_ids,  # n_fewshot examples for few-shot
                "pattern_idx": pattern_idx,
                "n_fewshot": n_fewshot,
                "fewshot_seed": fewshot_seed,
                # Hyperparameters from experiment config
                "lr": exp_config.get("lr", 1e-5),
                "lora_r": exp_config.get("lora_r", 16),
                "lora_alpha": exp_config.get("lora_alpha", 32),
                "max_seq_len": exp_config.get("max_seq_len", 2048),
                "batch_size": exp_config.get("batch_size", 1),
                "grad_accum_steps": exp_config.get("grad_accum_steps", 4),
                "seed": exp_config.get("seed", 42),
                # Task status
                "status": "pending",
                "created_at": datetime.now().isoformat(),
            }

            # Save task
            task_path = tasks_dir / f"{task_id}.json"
            save_json(task_config, task_path)
            tasks_created.append(task_id)

            # Calculate dev_ids for logging
            dev_ids = [sid for sid in sample_ids if sid not in example_ids]
            print(f"Created {task_id}:")
            print(f"  best_epoch: {best_epoch}")
            print(f"  sample_ids ({len(sample_ids)}): {sample_ids}")
            print(f"  example_ids ({len(example_ids)} 5-shot): {example_ids}")
            print(f"  dev_ids ({len(dev_ids)}): {dev_ids}")
    
    print(f"\n{len(tasks_created)} tasks created in {tasks_dir}")
    print("Tasks:", tasks_created)


if __name__ == "__main__":
    main()
