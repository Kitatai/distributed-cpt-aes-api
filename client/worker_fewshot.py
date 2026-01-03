#!/usr/bin/env python3
"""
Few-shot worker for distributed CPT-AES experiments.

Experiment flow:
1. Use 3 examples from 13 samples as few-shot examples
2. Use remaining 10 samples as dev data for epoch selection
3. For each epoch (0-30), score dev samples with few-shot prompting
4. Select best epoch by MSE on dev data (earliest if tie)
5. Score 10% of remaining data with best epoch checkpoint
6. Calculate QWK and Spearman correlation
"""

import os
import sys
import json
import logging
import argparse
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import gc
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


def run_fewshot_experiment(
    task_config: dict,
    data_path: Path,
    checkpoints_dir: Path,
    output_dir: Path,
):
    """
    Run few-shot experiment for a single task.

    Args:
        task_config: Task configuration with sample_ids, example_ids
        data_path: Path to ASAP data
        checkpoints_dir: Path to checkpoints directory
        output_dir: Path to save results
    """
    import torch
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
    model_short = task_config.get("model_short_name", "llama8b")
    sample_ids = set(task_config["sample_ids"])  # All 10 samples
    example_ids = set(task_config["example_ids"])  # 3 few-shot examples
    dev_ids = sample_ids - example_ids  # 7 dev samples
    max_epochs = 30

    y_min, y_max = ASAP_SCORE_RANGES[prompt_id]

    logger.info(f"Task: {task_id}")
    logger.info(f"Prompt: {prompt_id}, Model: {model_name}")
    logger.info(f"Sample IDs (10): {task_config['sample_ids']}")
    logger.info(f"Example IDs (3 few-shot): {task_config['example_ids']}")
    logger.info(f"Dev IDs (7): {sorted(dev_ids)}")

    # Load ASAP data
    logger.info(f"Loading ASAP data from {data_path}")
    df = pd.read_csv(data_path, sep='\t', encoding='latin-1')
    prompt_df = df[df['essay_set'] == prompt_id].copy()
    logger.info(f"Loaded {len(prompt_df)} essays for prompt {prompt_id}")

    # Prepare few-shot examples (from example_ids)
    example_essays = prompt_df[prompt_df['essay_id'].isin(example_ids)]
    fewshot_examples = [
        FewShotExample(
            essay_text=row['essay'],
            score=int(row['domain1_score']),
            essay_id=int(row['essay_id']),
        )
        for _, row in example_essays.iterrows()
    ]
    logger.info(f"Prepared {len(fewshot_examples)} few-shot examples")

    # Prepare dev data for epoch selection
    dev_df = prompt_df[prompt_df['essay_id'].isin(dev_ids)]
    logger.info(f"Dev data: {len(dev_df)} essays for epoch selection")

    # Prepare evaluation essays (exclude all sample_ids)
    # Use 10% of total essays (including dev) for evaluation
    eval_full_df = prompt_df[~prompt_df['essay_id'].isin(sample_ids)]
    total_essays = len(prompt_df)  # Total including dev
    eval_size = int(total_essays * 0.1)
    eval_df = eval_full_df.sample(n=min(eval_size, len(eval_full_df)), random_state=42)
    logger.info(f"Evaluation essays: {len(eval_df)} (10% of {total_essays} total)")

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

    def score_essays_fewshot(model, essays_df: pd.DataFrame, desc: str) -> List[dict]:
        """Score essays using few-shot prompting."""
        model.eval()
        results = []

        for idx, row in tqdm(essays_df.iterrows(), total=len(essays_df), desc=desc):
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
                'y_hat_greedy': y_pred,
                'generated_text': generated_text[:50],
            })

        return results

    def calculate_mse(results: List[dict]) -> float:
        """Calculate MSE."""
        y_true = np.array([r['y_true'] for r in results])
        y_pred = np.array([r['y_hat_greedy'] for r in results])
        return float(mean_squared_error(y_true, y_pred))

    def calculate_metrics(results: List[dict]) -> dict:
        """Calculate QWK and Spearman."""
        y_true = np.array([r['y_true'] for r in results])
        y_pred = np.array([r['y_hat_greedy'] for r in results])

        qwk = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
        spearman, _ = spearmanr(y_true, y_pred)
        spearman = float(spearman) if not np.isnan(spearman) else 0.0
        mse = float(mean_squared_error(y_true, y_pred))

        return {
            'qwk': qwk,
            'spearman': spearman,
            'mse': mse,
            'n_samples': len(results),
        }

    def load_model_with_lora(base_model, epoch: int) -> Tuple[any, bool]:
        """Load LoRA adapter for given epoch. Returns (model, success)."""
        if epoch == 0:
            return base_model, True

        zeroshot_task_id = f"prompt{prompt_id}_{model_short}"
        checkpoint_path = checkpoints_dir / zeroshot_task_id / f"epoch_{epoch}" / "adapter.zip"

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return base_model, False

        # Extract and load adapter
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with zipfile.ZipFile(checkpoint_path, 'r') as zf:
                zf.extractall(tmp_path)

            model_with_lora = PeftModel.from_pretrained(base_model, str(tmp_path), is_trainable=False)
            return model_with_lora, True

    # Create output directory
    task_output_dir = output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================
    # Phase 1: Evaluate all epochs on dev data
    # ========================================
    logger.info("=" * 60)
    logger.info(f"Phase 1: Evaluating all epochs on dev data ({len(dev_df)} samples)")
    logger.info("=" * 60)

    epoch_mse_results = {}

    # Load base model once
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    for epoch in range(0, max_epochs + 1):
        logger.info(f"Evaluating epoch {epoch} on dev data...")

        if epoch == 0:
            # Use base model directly for epoch 0
            model = base_model
        else:
            # Load LoRA adapter
            zeroshot_task_id = f"prompt{prompt_id}_{model_short}"
            checkpoint_path = checkpoints_dir / zeroshot_task_id / f"epoch_{epoch}" / "adapter.zip"

            if not checkpoint_path.exists():
                logger.warning(f"Skipping epoch {epoch} - checkpoint not found")
                continue

            # Extract and load adapter
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                with zipfile.ZipFile(checkpoint_path, 'r') as zf:
                    zf.extractall(tmp_path)

                model = PeftModel.from_pretrained(base_model, str(tmp_path), is_trainable=False)

        # Score dev data
        dev_results = score_essays_fewshot(model, dev_df, f"Epoch {epoch} dev")
        mse = calculate_mse(dev_results)
        epoch_mse_results[epoch] = {
            'mse': mse,
            'results': dev_results,
        }

        logger.info(f"  Epoch {epoch}: MSE = {mse:.4f}")

        # Clean up LoRA model (but keep base_model)
        if epoch > 0:
            # Unload adapter to restore base_model
            base_model = model.unload()
            del model
            gc.collect()
            torch.cuda.empty_cache()

    # Clean up base model after Phase 1
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # ========================================
    # Phase 2: Select best epoch (lowest MSE, earliest if tie)
    # ========================================
    logger.info("=" * 60)
    logger.info("Phase 2: Selecting best epoch")
    logger.info("=" * 60)

    # Find best epoch
    best_epoch = 0
    best_mse = float('inf')

    for epoch in sorted(epoch_mse_results.keys()):
        mse = epoch_mse_results[epoch]['mse']
        if mse < best_mse:
            best_mse = mse
            best_epoch = epoch

    logger.info(f"Best epoch: {best_epoch} (MSE = {best_mse:.4f})")

    # Log all epoch MSEs
    logger.info("All epoch MSEs:")
    for epoch in sorted(epoch_mse_results.keys()):
        marker = " <-- BEST" if epoch == best_epoch else ""
        logger.info(f"  Epoch {epoch}: MSE = {epoch_mse_results[epoch]['mse']:.4f}{marker}")

    # ========================================
    # Phase 3: Evaluate on full test data
    # ========================================
    logger.info("=" * 60)
    logger.info("Phase 3: Evaluating on full test data")
    logger.info("=" * 60)

    results_summary = {
        'task_id': task_id,
        'prompt_id': prompt_id,
        'model_name': model_name,
        'model_short_name': model_short,
        'n_fewshot_examples': len(fewshot_examples),
        'n_dev': len(dev_df),
        'n_eval': len(eval_df),
        'example_ids': list(example_ids),
        'dev_ids': list(dev_ids),
        'epoch_mse': {str(e): epoch_mse_results[e]['mse'] for e in epoch_mse_results},
        'selected_epoch': best_epoch,
        'selected_epoch_mse': best_mse,
    }

    # Evaluate epoch 0 on full test data
    logger.info("Evaluating epoch 0 on full test data...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    results_epoch0 = score_essays_fewshot(base_model, eval_df, "Epoch 0 full")
    metrics_epoch0 = calculate_metrics(results_epoch0)

    logger.info(f"Epoch 0: QWK={metrics_epoch0['qwk']:.4f}, Spearman={metrics_epoch0['spearman']:.4f}")

    save_csv(results_epoch0, task_output_dir / "predictions_epoch_0.csv")
    save_json({'fewshot': metrics_epoch0}, task_output_dir / "metrics_epoch_0.json")
    results_summary['epoch_0'] = metrics_epoch0

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate best epoch on full test data (if different from 0)
    if best_epoch > 0:
        logger.info(f"Evaluating epoch {best_epoch} on full test data...")

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )

        model_best, success = load_model_with_lora(base_model, best_epoch)

        if success:
            results_best = score_essays_fewshot(model_best, eval_df, f"Epoch {best_epoch} full")
            metrics_best = calculate_metrics(results_best)

            logger.info(f"Epoch {best_epoch}: QWK={metrics_best['qwk']:.4f}, Spearman={metrics_best['spearman']:.4f}")

            save_csv(results_best, task_output_dir / f"predictions_epoch_{best_epoch}.csv")
            save_json({'fewshot': metrics_best}, task_output_dir / f"metrics_epoch_{best_epoch}.json")
            results_summary['best_epoch_metrics'] = metrics_best
        else:
            results_summary['best_epoch_metrics'] = metrics_epoch0

        del model_best
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        results_summary['best_epoch_metrics'] = metrics_epoch0

    # Calculate improvement
    results_summary['improvement'] = {
        'qwk': results_summary['best_epoch_metrics']['qwk'] - metrics_epoch0['qwk'],
        'spearman': results_summary['best_epoch_metrics']['spearman'] - metrics_epoch0['spearman'],
    }

    # Save summary
    results_summary['completed_at'] = datetime.now().isoformat()
    save_json(results_summary, task_output_dir / "summary.json")

    logger.info("=" * 60)
    logger.info(f"Task {task_id} completed")
    logger.info(f"Selected epoch: {best_epoch} (MSE on dev: {best_mse:.4f})")
    logger.info(f"Epoch 0:         QWK={metrics_epoch0['qwk']:.4f}, Spearman={metrics_epoch0['spearman']:.4f}")
    logger.info(f"Epoch {best_epoch}:        QWK={results_summary['best_epoch_metrics']['qwk']:.4f}, Spearman={results_summary['best_epoch_metrics']['spearman']:.4f}")
    logger.info(f"Improvement:     QWK={results_summary['improvement']['qwk']:+.4f}, Spearman={results_summary['improvement']['spearman']:+.4f}")
    logger.info("=" * 60)

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Few-shot CPT-AES Worker")
    parser.add_argument(
        "--server-dir",
        type=str,
        required=True,
        help="Path to server data directory",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Specific task ID to run (default: run all pending tasks)",
    )

    args = parser.parse_args()

    server_dir = Path(args.server_dir)
    tasks_dir = server_dir / "tasks_fewshot"
    checkpoints_dir = server_dir / "checkpoints"
    results_dir = server_dir / "results_fewshot"
    data_path = server_dir / "asap" / "training_set_rel3.tsv"

    # Find tasks to run
    if args.task_id:
        task_files = [tasks_dir / f"{args.task_id}.json"]
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
            run_fewshot_experiment(
                task_config=task_config,
                data_path=data_path,
                checkpoints_dir=checkpoints_dir,
                output_dir=results_dir,
            )
        except Exception as e:
            logger.exception(f"Task {task_id} failed: {e}")
        finally:
            # Force cleanup between tasks
            import torch
            gc.collect()
            torch.cuda.empty_cache()

    logger.info("\nAll tasks completed")


if __name__ == "__main__":
    main()
