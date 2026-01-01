#!/usr/bin/env python3
"""
Few-shot v2 worker for CPT-AES experiments.

New approach:
- Split 10 dev samples into 3 (shot) + 7 (eval)
- For each epoch 0-30, score 7 eval samples with few-shot prompting
- Select best epoch based on MSE on 7 samples
- Evaluate test set (10% of remaining essays) at best epoch
- Report QWK and Spearman correlation

Uses checkpoints from backup_zeroshot_v1 (instruction model training).
"""

import os
import sys
import json
import logging
import argparse
import tempfile
import zipfile
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure src is in path
CLIENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CLIENT_DIR / "src"))


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: Path):
    """Save JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def save_csv(data: List[dict], path: Path):
    """Save list of dicts to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def split_samples(sample_ids: List[int], n_shot: int = 3, seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Split 10 sample IDs into shot examples and dev evaluation.

    Args:
        sample_ids: List of 10 sample essay IDs
        n_shot: Number of examples for few-shot (default: 3)
        seed: Random seed for reproducible split

    Returns:
        (shot_ids, dev_ids): Tuple of shot IDs (3) and dev IDs (7)
    """
    rng = random.Random(seed)
    shuffled = sample_ids.copy()
    rng.shuffle(shuffled)
    shot_ids = shuffled[:n_shot]
    dev_ids = shuffled[n_shot:]
    return shot_ids, dev_ids


def run_fewshot_v2_experiment(
    task_config: dict,
    data_path: Path,
    checkpoints_dir: Path,
    output_dir: Path,
):
    """
    Run few-shot v2 experiment for a single task.

    Args:
        task_config: Task configuration with sample_ids, prompt_id, etc.
        data_path: Path to ASAP data file
        checkpoints_dir: Path to checkpoints directory (backup_zeroshot_v1/checkpoints)
        output_dir: Path to save results
    """
    import torch
    import gc
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score, mean_squared_error

    from config import ASAP_SCORE_RANGES
    from models.prompts_fewshot import (
        create_fewshot_prompt_builder,
        FewShotExample,
    )

    task_id = task_config["task_id"]
    prompt_id = task_config["prompt_id"]
    model_name = task_config["model_name"]
    model_short = task_config.get("model_short", "llama8b")
    sample_ids = task_config["sample_ids"]
    pattern_idx = task_config.get("pattern_idx", 0)
    n_shot = task_config.get("n_shot", 3)
    split_seed = task_config.get("split_seed", 42)
    test_sample_ratio = task_config.get("test_sample_ratio", 0.1)
    max_epochs = task_config.get("max_epochs", 30)

    y_min, y_max = ASAP_SCORE_RANGES[prompt_id]

    logger.info(f"=" * 60)
    logger.info(f"Task: {task_id}")
    logger.info(f"Prompt: {prompt_id}, Model: {model_name}")
    logger.info(f"Pattern: {pattern_idx}, Samples: {sample_ids}")
    logger.info(f"Split: {n_shot} shot + {len(sample_ids) - n_shot} dev")
    logger.info(f"=" * 60)

    # Split samples into shot and dev
    shot_ids, dev_ids = split_samples(sample_ids, n_shot=n_shot, seed=split_seed)
    logger.info(f"Shot IDs: {shot_ids}")
    logger.info(f"Dev IDs: {dev_ids}")

    # Load ASAP data
    logger.info(f"Loading ASAP data from {data_path}")
    df = pd.read_csv(data_path, sep='\t', encoding='latin-1')
    prompt_df = df[df['essay_set'] == prompt_id].copy()
    logger.info(f"Loaded {len(prompt_df)} essays for prompt {prompt_id}")

    # Prepare few-shot examples (from shot_ids)
    shot_essays = prompt_df[prompt_df['essay_id'].isin(shot_ids)]
    fewshot_examples = [
        FewShotExample(
            essay_text=row['essay'],
            score=int(row['domain1_score']),
            essay_id=int(row['essay_id']),
        )
        for _, row in shot_essays.iterrows()
    ]
    logger.info(f"Prepared {len(fewshot_examples)} few-shot examples")

    # Prepare dev essays (7 samples for epoch selection)
    dev_df = prompt_df[prompt_df['essay_id'].isin(dev_ids)]
    logger.info(f"Dev essays for epoch selection: {len(dev_df)}")

    # Prepare test essays (remaining essays, sample 10%)
    test_df = prompt_df[~prompt_df['essay_id'].isin(sample_ids)]
    if test_sample_ratio < 1.0:
        test_df = test_df.sample(frac=test_sample_ratio, random_state=42)
    logger.info(f"Test essays ({test_sample_ratio*100:.0f}%): {len(test_df)}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check for Flash Attention
    attn_impl = None
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        logger.info("Flash Attention 2 enabled")
    except ImportError:
        logger.info("Flash Attention not available")

    # Create few-shot prompt builder
    prompt_builder = create_fewshot_prompt_builder(
        prompt_id=prompt_id,
        y_min=y_min,
        y_max=y_max,
        examples=fewshot_examples,
    )

    def score_essays_fewshot(model, essays_df: pd.DataFrame, desc: str, show_progress: bool = False) -> List[dict]:
        """Score essays using few-shot prompting."""
        model.eval()
        results = []

        iterator = essays_df.iterrows()
        if show_progress:
            iterator = tqdm(list(iterator), desc=desc, ncols=80)

        for idx, row in iterator:
            essay_id = int(row['essay_id'])
            essay_text = row['essay']
            y_true = int(row['domain1_score'])

            # Build few-shot prompt
            messages = prompt_builder.to_messages(essay_text, use_prefill=True)

            # Tokenize
            prompt_text = tokenizer.apply_chat_template(
                messages[:-1],  # system + user
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_text += messages[-1]["content"]  # assistant prefill

            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Extract score from output
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Parse score
            y_pred = None
            for char in generated_text:
                if char.isdigit():
                    potential_score = int(char)
                    # Check for two-digit scores
                    idx_char = generated_text.find(char)
                    if idx_char + 1 < len(generated_text) and generated_text[idx_char + 1].isdigit():
                        potential_score = int(generated_text[idx_char:idx_char + 2])
                    if y_min <= potential_score <= y_max:
                        y_pred = potential_score
                        break

            if y_pred is None:
                y_pred = y_min  # Fallback

            results.append({
                'essay_id': essay_id,
                'y_true': y_true,
                'y_pred': y_pred,
                'generated_text': generated_text[:50],
            })

        return results

    def load_model_with_adapter(epoch: int):
        """Load model with LoRA adapter for specified epoch."""
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )

        if epoch == 0:
            # Epoch 0: no adapter
            return model

        # Load LoRA adapter
        zeroshot_task_id = f"prompt{prompt_id}_{model_short}"
        checkpoint_path = checkpoints_dir / zeroshot_task_id / f"epoch_{epoch}" / "adapter.zip"

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return model

        # Extract and load adapter
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with zipfile.ZipFile(checkpoint_path, 'r') as zf:
                zf.extractall(tmp_path)

            model = PeftModel.from_pretrained(model, str(tmp_path), is_trainable=False)
            logger.info(f"Loaded LoRA adapter from {checkpoint_path}")

        return model

    def unload_model():
        """Clear GPU memory."""
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()

    # Create output directory
    task_output_dir = output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Find best epoch using 7 dev samples
    logger.info("=" * 60)
    logger.info("Phase 1: Epoch selection using dev samples")
    logger.info("=" * 60)

    epoch_results = []

    for epoch in range(0, max_epochs + 1):
        logger.info(f"Evaluating epoch {epoch}/{max_epochs}")

        # Load model
        model = load_model_with_adapter(epoch)

        # Score dev samples
        dev_results = score_essays_fewshot(model, dev_df, f"Epoch {epoch} dev")

        # Calculate MSE
        y_true = np.array([r['y_true'] for r in dev_results])
        y_pred = np.array([r['y_pred'] for r in dev_results])
        mse = float(mean_squared_error(y_true, y_pred))

        epoch_results.append({
            'epoch': epoch,
            'mse': mse,
            'predictions': dev_results,
        })

        logger.info(f"  Epoch {epoch}: MSE = {mse:.4f}")

        # Save epoch results
        save_json({
            'epoch': epoch,
            'mse': mse,
            'predictions': dev_results,
        }, task_output_dir / f"dev_epoch_{epoch}.json")

        # Unload model
        del model
        unload_model()

    # Find best epoch (lowest MSE)
    best_epoch_result = min(epoch_results, key=lambda x: x['mse'])
    best_epoch = best_epoch_result['epoch']
    best_mse = best_epoch_result['mse']

    logger.info("=" * 60)
    logger.info(f"Best epoch: {best_epoch} (MSE = {best_mse:.4f})")
    logger.info("=" * 60)

    # Phase 2: Evaluate test set at epoch 0 (baseline) and best epoch
    logger.info("=" * 60)
    logger.info("Phase 2: Test evaluation")
    logger.info("=" * 60)

    def evaluate_test_at_epoch(epoch: int, desc: str) -> dict:
        """Evaluate test set at a specific epoch."""
        model = load_model_with_adapter(epoch)
        results = score_essays_fewshot(model, test_df, desc, show_progress=True)

        y_true = np.array([r['y_true'] for r in results])
        y_pred = np.array([r['y_pred'] for r in results])

        qwk = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
        spearman_val, _ = spearmanr(y_true, y_pred)
        spearman_val = float(spearman_val) if not np.isnan(spearman_val) else 0.0
        mse = float(mean_squared_error(y_true, y_pred))

        del model
        unload_model()

        return {
            'epoch': epoch,
            'qwk': qwk,
            'spearman': spearman_val,
            'mse': mse,
            'n_samples': len(results),
            'predictions': results,
        }

    # Evaluate at epoch 0 (baseline - no continual learning)
    logger.info("Evaluating test set at epoch 0 (baseline)...")
    test_epoch0 = evaluate_test_at_epoch(0, "Test Epoch 0")
    logger.info(f"Epoch 0 (baseline): QWK={test_epoch0['qwk']:.4f}, Spearman={test_epoch0['spearman']:.4f}, MSE={test_epoch0['mse']:.4f}")

    # Evaluate at best epoch
    if best_epoch > 0:
        logger.info(f"Evaluating test set at epoch {best_epoch} (best)...")
        test_best = evaluate_test_at_epoch(best_epoch, f"Test Epoch {best_epoch}")
        logger.info(f"Epoch {best_epoch} (best): QWK={test_best['qwk']:.4f}, Spearman={test_best['spearman']:.4f}, MSE={test_best['mse']:.4f}")

        # Calculate improvement
        improvement = {
            'qwk': test_best['qwk'] - test_epoch0['qwk'],
            'spearman': test_best['spearman'] - test_epoch0['spearman'],
            'mse': test_epoch0['mse'] - test_best['mse'],  # Lower MSE is better
        }
        logger.info(f"Improvement: QWK={improvement['qwk']:+.4f}, Spearman={improvement['spearman']:+.4f}, MSE={improvement['mse']:+.4f}")
    else:
        test_best = test_epoch0
        improvement = {'qwk': 0.0, 'spearman': 0.0, 'mse': 0.0}

    # Save results
    summary = {
        'task_id': task_id,
        'prompt_id': prompt_id,
        'model_name': model_name,
        'model_short': model_short,
        'pattern_idx': pattern_idx,
        'sample_ids': sample_ids,
        'shot_ids': shot_ids,
        'dev_ids': dev_ids,
        'n_shot': n_shot,
        'split_seed': split_seed,
        'test_sample_ratio': test_sample_ratio,
        'max_epochs': max_epochs,
        'epoch_selection': {
            'best_epoch': best_epoch,
            'best_mse': best_mse,
            'all_epochs': [{'epoch': r['epoch'], 'mse': r['mse']} for r in epoch_results],
        },
        'test_baseline': {
            'epoch': 0,
            'qwk': test_epoch0['qwk'],
            'spearman': test_epoch0['spearman'],
            'mse': test_epoch0['mse'],
            'n_samples': test_epoch0['n_samples'],
        },
        'test_best': {
            'epoch': best_epoch,
            'qwk': test_best['qwk'],
            'spearman': test_best['spearman'],
            'mse': test_best['mse'],
            'n_samples': test_best['n_samples'],
        },
        'improvement': improvement,
        'completed_at': datetime.now().isoformat(),
    }

    save_json(summary, task_output_dir / "summary.json")
    save_csv(test_epoch0['predictions'], task_output_dir / "test_predictions_epoch0.csv")
    if best_epoch > 0:
        save_csv(test_best['predictions'], task_output_dir / "test_predictions_best.csv")

    # Save epoch MSE curve
    epoch_curve = pd.DataFrame([{'epoch': r['epoch'], 'mse': r['mse']} for r in epoch_results])
    epoch_curve.to_csv(task_output_dir / "epoch_mse_curve.csv", index=False)

    logger.info("=" * 60)
    logger.info(f"Task {task_id} completed")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Baseline (epoch 0): QWK={test_epoch0['qwk']:.4f}, Spearman={test_epoch0['spearman']:.4f}")
    logger.info(f"Best (epoch {best_epoch}): QWK={test_best['qwk']:.4f}, Spearman={test_best['spearman']:.4f}")
    logger.info(f"Improvement: QWK={improvement['qwk']:+.4f}, Spearman={improvement['spearman']:+.4f}")
    logger.info("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Few-shot v2 CPT-AES Worker")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to backup_zeroshot_v1 data directory",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Specific task ID to run (default: run all pending tasks)",
    )
    parser.add_argument(
        "--prompt",
        type=int,
        default=None,
        help="Specific prompt ID to run (1-8)",
    )
    parser.add_argument(
        "--pattern",
        type=int,
        default=0,
        help="Pattern index to use (0-49, default: 0)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tasks_dir = data_dir / "tasks_fewshot_v2"
    checkpoints_dir = data_dir / "checkpoints"
    results_dir = data_dir / "results_fewshot_v2"
    asap_data_path = data_dir.parent / "asap" / "training_set_rel3.tsv"

    # Check paths
    if not checkpoints_dir.exists():
        logger.error(f"Checkpoints not found: {checkpoints_dir}")
        logger.error("Make sure checkpoints.zip is extracted to backup_zeroshot_v1/checkpoints/")
        sys.exit(1)

    if not asap_data_path.exists():
        logger.error(f"ASAP data not found: {asap_data_path}")
        sys.exit(1)

    # Find tasks to run
    if args.task_id:
        task_files = [tasks_dir / f"{args.task_id}.json"]
    elif args.prompt:
        task_files = sorted(tasks_dir.glob(f"fewshot_v2_prompt{args.prompt}_*.json"))
    else:
        task_files = sorted(tasks_dir.glob("*.json"))

    logger.info(f"Found {len(task_files)} task(s) to process")

    for task_file in task_files:
        if not task_file.exists():
            logger.warning(f"Task file not found: {task_file}")
            continue

        task_config = load_json(task_file)
        task_id = task_config['task_id']

        # Check if already completed
        summary_path = results_dir / task_id / "summary.json"
        if summary_path.exists():
            logger.info(f"Task {task_id} already completed, skipping")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting task: {task_id}")
        logger.info(f"{'='*60}")

        try:
            run_fewshot_v2_experiment(
                task_config=task_config,
                data_path=asap_data_path,
                checkpoints_dir=checkpoints_dir,
                output_dir=results_dir,
            )
        except Exception as e:
            logger.exception(f"Task {task_id} failed: {e}")
            continue

    logger.info("\nAll tasks completed")


if __name__ == "__main__":
    main()
