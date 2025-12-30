#!/usr/bin/env python3
"""
Few-shot worker for distributed CPT-AES experiments.

Loads pre-trained checkpoints and evaluates using few-shot prompting.
Evaluates at epoch 0 (baseline) and best_epoch (from zero-shot analysis).
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
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

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
        task_config: Task configuration with best_epoch, sample_ids, example_ids
        data_path: Path to ASAP data
        checkpoints_dir: Path to checkpoints directory
        output_dir: Path to save results
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score

    from config import ASAP_SCORE_RANGES
    from models.prompts_fewshot import (
        create_fewshot_prompt_builder,
        FewShotExample,
    )

    task_id = task_config["task_id"]
    prompt_id = task_config["prompt_id"]
    model_name = task_config["model_name"]
    best_epoch = task_config["best_epoch"]
    sample_ids = set(task_config["sample_ids"])
    example_ids = task_config["example_ids"]

    y_min, y_max = ASAP_SCORE_RANGES[prompt_id]

    logger.info(f"Task: {task_id}")
    logger.info(f"Prompt: {prompt_id}, Model: {model_name}")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Sample IDs (excluded): {task_config['sample_ids']}")
    logger.info(f"Example IDs (few-shot): {example_ids}")

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
    # Sort by score ascending
    fewshot_examples.sort(key=lambda x: x.score)
    logger.info(f"Prepared {len(fewshot_examples)} few-shot examples (sorted by score ascending)")

    # Prepare evaluation essays (exclude all sample_ids)
    eval_df = prompt_df[~prompt_df['essay_id'].isin(sample_ids)]

    # Optional: sample a fraction of evaluation essays for quick testing
    eval_sample_ratio = task_config.get("eval_sample_ratio", 1.0)
    if eval_sample_ratio < 1.0:
        eval_df = eval_df.sample(frac=eval_sample_ratio, random_state=42)
    logger.info(f"Evaluation essays: {len(eval_df)} (ratio: {eval_sample_ratio})")

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

        for idx, row in essays_df.iterrows():
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

            if len(results) % 50 == 0:
                logger.info(f"  {desc}: {len(results)}/{len(essays_df)} scored")

        return results

    def calculate_metrics(results: List[dict]) -> dict:
        """Calculate QWK and Spearman."""
        y_true = np.array([r['y_true'] for r in results])
        y_pred = np.array([r['y_hat_greedy'] for r in results])

        qwk = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
        spearman, _ = spearmanr(y_true, y_pred)
        spearman = float(spearman) if not np.isnan(spearman) else 0.0

        return {
            'qwk': qwk,
            'spearman': spearman,
            'n_samples': len(results),
        }

    results_summary = {
        'task_id': task_id,
        'prompt_id': prompt_id,
        'model_name': model_name,
        'best_epoch': best_epoch,
        'n_examples': len(fewshot_examples),
        'n_eval': len(eval_df),
        'example_ids': example_ids,
    }

    # Create output directory
    task_output_dir = output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate at epoch 0 (baseline model, no LoRA)
    logger.info("=" * 50)
    logger.info("Evaluating at epoch 0 (baseline)")
    logger.info("=" * 50)

    model_epoch0 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    results_epoch0 = score_essays_fewshot(model_epoch0, eval_df, "Epoch 0")
    metrics_epoch0 = calculate_metrics(results_epoch0)

    logger.info(f"Epoch 0: QWK={metrics_epoch0['qwk']:.4f}, Spearman={metrics_epoch0['spearman']:.4f}")

    # Save epoch 0 results
    save_csv(results_epoch0, task_output_dir / "predictions_epoch_0.csv")
    save_json({'fewshot': metrics_epoch0}, task_output_dir / "metrics_epoch_0.json")

    results_summary['epoch_0'] = metrics_epoch0

    # Clean up
    del model_epoch0
    torch.cuda.empty_cache()

    # Evaluate at best_epoch (if not 0)
    if best_epoch > 0:
        logger.info("=" * 50)
        logger.info(f"Evaluating at epoch {best_epoch} (best)")
        logger.info("=" * 50)

        # Load base model
        model_best = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )

        # Load LoRA adapter
        zeroshot_task_id = f"prompt{prompt_id}_llama8b"
        checkpoint_path = checkpoints_dir / zeroshot_task_id / f"epoch_{best_epoch}" / "adapter.zip"

        if checkpoint_path.exists():
            # Extract adapter
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                with zipfile.ZipFile(checkpoint_path, 'r') as zf:
                    zf.extractall(tmp_path)

                # Load LoRA
                model_best = PeftModel.from_pretrained(model_best, str(tmp_path), is_trainable=False)
                logger.info(f"Loaded LoRA adapter from {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            logger.warning("Using base model for best epoch")

        results_best = score_essays_fewshot(model_best, eval_df, f"Epoch {best_epoch}")
        metrics_best = calculate_metrics(results_best)

        logger.info(f"Epoch {best_epoch}: QWK={metrics_best['qwk']:.4f}, Spearman={metrics_best['spearman']:.4f}")

        # Save best epoch results
        save_csv(results_best, task_output_dir / f"predictions_epoch_{best_epoch}.csv")
        save_json({'fewshot': metrics_best}, task_output_dir / f"metrics_epoch_{best_epoch}.json")

        results_summary['best_epoch_metrics'] = metrics_best

        # Calculate improvement
        results_summary['improvement'] = {
            'qwk': metrics_best['qwk'] - metrics_epoch0['qwk'],
            'spearman': metrics_best['spearman'] - metrics_epoch0['spearman'],
        }

        # Clean up
        del model_best
        torch.cuda.empty_cache()
    else:
        # best_epoch is 0, no improvement
        results_summary['best_epoch_metrics'] = metrics_epoch0
        results_summary['improvement'] = {'qwk': 0.0, 'spearman': 0.0}

    # Save summary
    results_summary['completed_at'] = datetime.now().isoformat()
    save_json(results_summary, task_output_dir / "summary.json")

    logger.info("=" * 50)
    logger.info(f"Task {task_id} completed")
    logger.info(f"Epoch 0:    QWK={results_summary['epoch_0']['qwk']:.4f}, Spearman={results_summary['epoch_0']['spearman']:.4f}")
    if best_epoch > 0:
        logger.info(f"Epoch {best_epoch}: QWK={results_summary['best_epoch_metrics']['qwk']:.4f}, Spearman={results_summary['best_epoch_metrics']['spearman']:.4f}")
        logger.info(f"Improvement: QWK={results_summary['improvement']['qwk']:+.4f}, Spearman={results_summary['improvement']['spearman']:+.4f}")
    logger.info("=" * 50)

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
            continue

    logger.info("\nAll tasks completed")


if __name__ == "__main__":
    main()
