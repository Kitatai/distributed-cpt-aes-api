#!/usr/bin/env python3
"""
Generate few-shot experiment tasks.

Creates tasks for few-shot scoring experiments using:
- Best epochs from analysis_results.json
- Sample patterns from sample_patterns.json
- 5 randomly selected examples from each 10-sample pattern

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
    
    # Model config
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model_short = "llama8b"
    
    # Few-shot config
    n_fewshot = 5
    fewshot_seed = 42
    pattern_idx = 0  # Use first pattern only
    
    tasks_created = []
    
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
        sample_ids = patterns['patterns'].get(prompt_key, [])[pattern_idx]
        
        if len(sample_ids) < 10:
            print(f"Warning: Not enough samples for {prompt_key}, skipping")
            continue
        
        # Select 5 examples with fixed seed
        rng = random.Random(fewshot_seed + prompt_id)
        example_ids = rng.sample(sample_ids, n_fewshot)
        
        # Create task config
        task_config = {
            "task_id": task_id,
            "prompt_id": prompt_id,
            "model_name": model_name,
            "dataset": "asap",
            "best_epoch": best_epoch,
            "sample_ids": sample_ids,  # All 10 samples (excluded from evaluation)
            "example_ids": example_ids,  # 5 examples for few-shot
            "pattern_idx": pattern_idx,
            "n_fewshot": n_fewshot,
            "fewshot_seed": fewshot_seed,
            # Hyperparameters from experiment config
            "lr": exp_config["lr"],
            "lora_r": exp_config["lora_r"],
            "lora_alpha": exp_config["lora_alpha"],
            "max_seq_len": exp_config["max_seq_len"],
            "batch_size": exp_config["batch_size"],
            "grad_accum_steps": exp_config["grad_accum_steps"],
            "seed": exp_config["seed"],
            # Task status
            "status": "pending",
            "created_at": datetime.now().isoformat(),
        }
        
        # Save task
        task_path = tasks_dir / f"{task_id}.json"
        save_json(task_config, task_path)
        tasks_created.append(task_id)
        
        print(f"Created {task_id}:")
        print(f"  best_epoch: {best_epoch}")
        print(f"  sample_ids: {sample_ids}")
        print(f"  example_ids: {example_ids}")
    
    print(f"\n{len(tasks_created)} tasks created in {tasks_dir}")
    print("Tasks:", tasks_created)


if __name__ == "__main__":
    main()
