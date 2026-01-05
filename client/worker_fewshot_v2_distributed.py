#!/usr/bin/env python3
"""
Distributed few-shot v2 worker for CPT-AES experiments.

Connects to the API server to get tasks, runs experiments locally,
and uploads results back to the server.

Usage:
    python worker_fewshot_v2_distributed.py --server http://SERVER_IP:8000

Uses new sample_patterns_v2.json format with explicit:
- test_ids: Essays for final evaluation
- dev_ids: Essays for epoch selection (MSE-based)
- example_ids: Essays for few-shot examples (k=1,3,5)
"""

import os
import sys
import json
import time
import signal
import socket
import logging
import argparse
import tempfile
import zipfile
import gc
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

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

from api_client import APIClient


# Global state for graceful shutdown
class WorkerState:
    """Global state for worker shutdown handling."""
    shutdown_requested = False
    current_task_id = None
    client = None
    lock = threading.Lock()


_state = WorkerState()


def signal_handler(signum, frame):
    """Handle shutdown signals (SIGINT, SIGTERM)."""
    with _state.lock:
        if _state.shutdown_requested:
            logger.warning("Force shutdown requested, exiting immediately")
            sys.exit(1)

        _state.shutdown_requested = True
        logger.info("Shutdown requested, will stop after current task completes")
        logger.info("Press Ctrl+C again to force immediate exit")


def run_fewshot_v2_experiment(
    task_config: dict,
    data_path: Path,
    client: APIClient,
    e0_results_dir_name: Optional[str] = None,
):
    """
    Run few-shot v2 experiment for a single task.

    Args:
        task_config: Task configuration with test_ids, dev_ids, example_ids
        data_path: Path to ASAP data file
        client: API client for downloading checkpoints
        e0_results_dir_name: Name of results directory to reuse E0 from (e.g., "results_fewshot_v2_dev10")
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
    k = task_config["k"]
    pattern_idx = task_config["pattern_idx"]
    test_ids = task_config["test_ids"]
    dev_ids = task_config["dev_ids"]
    example_ids = task_config["example_ids"]
    max_epochs = task_config.get("max_epochs", 30)
    fixed_epoch = task_config.get("fixed_epoch", None)  # None means use epoch selection

    y_min, y_max = ASAP_SCORE_RANGES[prompt_id]

    logger.info(f"=" * 60)
    logger.info(f"Task: {task_id}")
    logger.info(f"Prompt: {prompt_id}, Model: {model_name}")
    logger.info(f"K: {k}, Pattern: {pattern_idx}")
    logger.info(f"Test: {len(test_ids)}, Dev: {len(dev_ids)}, Examples: {len(example_ids)}")
    if fixed_epoch is not None:
        logger.info(f"Fixed epoch mode: e={fixed_epoch} (no epoch selection)")
    logger.info(f"=" * 60)

    # Load ASAP data
    logger.info(f"Loading ASAP data from {data_path}")
    df = pd.read_csv(data_path, sep='\t', encoding='latin-1')
    prompt_df = df[df['essay_set'] == prompt_id].copy()
    logger.info(f"Loaded {len(prompt_df)} essays for prompt {prompt_id}")

    # Prepare few-shot examples
    example_essays = prompt_df[prompt_df['essay_id'].isin(example_ids)]
    fewshot_examples = [
        FewShotExample(
            essay_text=row['essay'],
            score=int(row['domain1_score']),
            essay_id=int(row['essay_id']),
        )
        for _, row in example_essays.iterrows()
    ]
    logger.info(f"Prepared {len(fewshot_examples)} few-shot examples: {example_ids}")

    # Prepare dev data for epoch selection
    dev_df = prompt_df[prompt_df['essay_id'].isin(dev_ids)]
    logger.info(f"Dev essays for epoch selection: {len(dev_df)}")

    # Prepare test data for final evaluation
    test_df = prompt_df[prompt_df['essay_id'].isin(test_ids)]
    logger.info(f"Test essays for evaluation: {len(test_df)}")

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

    def score_essays_fewshot(model, essays_df: pd.DataFrame, desc: str, show_progress: bool = True) -> List[dict]:
        """Score essays using few-shot prompting."""
        model.eval()
        results = []

        iterator = list(essays_df.iterrows())
        if show_progress:
            iterator = tqdm(iterator, desc=desc, ncols=80)

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

    def download_and_load_adapter(base_model, epoch: int):
        """Download checkpoint from server and load as LoRA adapter."""
        if epoch == 0:
            return base_model, True

        zeroshot_task_id = f"prompt{prompt_id}_{model_short}"

        # Download checkpoint from server
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            success = client.download_checkpoint(zeroshot_task_id, epoch, tmp_path)
            if not success:
                logger.warning(f"Checkpoint not found on server: {zeroshot_task_id}/epoch_{epoch}")
                return base_model, False

            model_with_lora = PeftModel.from_pretrained(base_model, str(tmp_path), is_trainable=False)
            return model_with_lora, True

    # ========================================
    # Phase 1: Evaluate all epochs on dev data
    # ========================================
    logger.info("=" * 60)
    logger.info(f"Phase 1: Epoch selection using dev data ({len(dev_df)} samples)")
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

    zeroshot_task_id = f"prompt{prompt_id}_{model_short}"

    for epoch in range(0, max_epochs + 1):
        # Check for shutdown request
        if _state.shutdown_requested:
            logger.info("Shutdown requested, stopping early")
            break

        logger.info(f"Evaluating epoch {epoch}/{max_epochs} on dev...")

        if epoch == 0:
            model = base_model
        else:
            # Download checkpoint from server
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)

                if not client.check_checkpoint(zeroshot_task_id, epoch):
                    logger.warning(f"Skipping epoch {epoch} - checkpoint not found on server")
                    continue

                success = client.download_checkpoint(zeroshot_task_id, epoch, tmp_path)
                if not success:
                    logger.warning(f"Failed to download checkpoint for epoch {epoch}")
                    continue

                model = PeftModel.from_pretrained(base_model, str(tmp_path), is_trainable=False)

        # Score dev data
        dev_results = score_essays_fewshot(model, dev_df, f"E{epoch} dev", show_progress=False)

        y_true = np.array([r['y_true'] for r in dev_results])
        y_pred = np.array([r['y_pred'] for r in dev_results])
        mse = float(mean_squared_error(y_true, y_pred))

        epoch_mse_results[epoch] = mse
        logger.info(f"  Epoch {epoch}: MSE = {mse:.4f}")

        # Clean up LoRA model
        if epoch > 0:
            base_model = model.unload()
            del model
            gc.collect()
            torch.cuda.empty_cache()

    # Clean up base model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    if not epoch_mse_results:
        raise RuntimeError("No epochs could be evaluated")

    # ========================================
    # Phase 2: Select best epoch (or use fixed epoch)
    # ========================================
    logger.info("=" * 60)
    if fixed_epoch is not None:
        logger.info(f"Phase 2: Using fixed epoch {fixed_epoch}")
        best_epoch = fixed_epoch
        best_mse = epoch_mse_results.get(fixed_epoch, 0.0)
    else:
        logger.info("Phase 2: Selecting best epoch")
        # Find best epoch (lowest MSE, earliest if tie)
        best_epoch = min(epoch_mse_results.keys(), key=lambda e: (epoch_mse_results[e], e))
        best_mse = epoch_mse_results[best_epoch]
    logger.info("=" * 60)

    logger.info(f"Best epoch: {best_epoch} (MSE = {best_mse:.4f})")

    # ========================================
    # Phase 3: Evaluate on test data
    # ========================================
    logger.info("=" * 60)
    logger.info(f"Phase 3: Evaluating on test data ({len(test_df)} samples)")
    logger.info("=" * 60)

    def evaluate_epoch_on_test(epoch: int, desc: str) -> dict:
        """Evaluate specific epoch on test data."""
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )

        if epoch == 0:
            model = base_model
        else:
            model, success = download_and_load_adapter(base_model, epoch)
            if not success:
                model = base_model

        results = score_essays_fewshot(model, test_df, desc)

        y_true = np.array([r['y_true'] for r in results])
        y_pred = np.array([r['y_pred'] for r in results])

        qwk = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
        spearman_val, _ = spearmanr(y_true, y_pred)
        spearman_val = float(spearman_val) if not np.isnan(spearman_val) else 0.0
        mse = float(mean_squared_error(y_true, y_pred))

        del model
        if epoch > 0:
            del base_model
        gc.collect()
        torch.cuda.empty_cache()

        return {
            'qwk': qwk,
            'spearman': spearman_val,
            'mse': mse,
            'n_samples': len(results),
            'predictions': results,
        }

    # Evaluate epoch 0 (or reuse from existing results)
    metrics_e0 = None
    if e0_results_dir_name is not None:
        e0_data = client.get_e0_results(e0_results_dir_name, task_id)
        if e0_data and "epoch_0" in e0_data:
            metrics_e0 = e0_data["epoch_0"]
            logger.info(f"Epoch 0 (reused from {e0_results_dir_name}): QWK={metrics_e0['qwk']:.4f}, Spearman={metrics_e0['spearman']:.4f}")

    if metrics_e0 is None:
        logger.info("Evaluating epoch 0 on test...")
        metrics_e0 = evaluate_epoch_on_test(0, "Test E0")
        logger.info(f"Epoch 0: QWK={metrics_e0['qwk']:.4f}, Spearman={metrics_e0['spearman']:.4f}")

    # Evaluate best epoch
    if best_epoch > 0:
        logger.info(f"Evaluating epoch {best_epoch} on test...")
        metrics_best = evaluate_epoch_on_test(best_epoch, f"Test E{best_epoch}")
        logger.info(f"Epoch {best_epoch}: QWK={metrics_best['qwk']:.4f}, Spearman={metrics_best['spearman']:.4f}")
    else:
        metrics_best = metrics_e0

    # Calculate improvement
    improvement = {
        'qwk': metrics_best['qwk'] - metrics_e0['qwk'],
        'spearman': metrics_best['spearman'] - metrics_e0['spearman'],
    }

    # Build summary (without predictions for JSON serialization)
    summary = {
        'task_id': task_id,
        'prompt_id': prompt_id,
        'model_name': model_name,
        'model_short_name': model_short,
        'k': k,
        'pattern_idx': pattern_idx,
        'n_test': len(test_ids),
        'n_dev': len(dev_ids),
        'n_examples': len(example_ids),
        'example_ids': example_ids,
        'fixed_epoch': fixed_epoch,  # None if epoch selection was used
        'epoch_mse': {str(e): epoch_mse_results[e] for e in sorted(epoch_mse_results.keys())},
        'selected_epoch': best_epoch,
        'selected_epoch_mse': best_mse,
        'epoch_0': {
            'qwk': metrics_e0['qwk'],
            'spearman': metrics_e0['spearman'],
            'mse': metrics_e0['mse'],
            'n_samples': metrics_e0['n_samples'],
        },
        'best_epoch_metrics': {
            'qwk': metrics_best['qwk'],
            'spearman': metrics_best['spearman'],
            'mse': metrics_best['mse'],
            'n_samples': metrics_best['n_samples'],
        },
        'improvement': improvement,
        'completed_at': datetime.now().isoformat(),
    }

    logger.info("=" * 60)
    logger.info(f"Task {task_id} completed")
    logger.info(f"Selected epoch: {best_epoch} (MSE on dev: {best_mse:.4f})")
    logger.info(f"Epoch 0:    QWK={metrics_e0['qwk']:.4f}, Spearman={metrics_e0['spearman']:.4f}")
    logger.info(f"Epoch {best_epoch}:   QWK={metrics_best['qwk']:.4f}, Spearman={metrics_best['spearman']:.4f}")
    logger.info(f"Improvement: QWK={improvement['qwk']:+.4f}, Spearman={improvement['spearman']:+.4f}")
    logger.info("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Distributed Few-shot v2 CPT-AES Worker")
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8000",
        help="Server URL (e.g., http://192.168.100.10:8000)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Filter tasks by k value (1, 3, or 5)",
    )
    parser.add_argument(
        "--prompt",
        type=int,
        default=None,
        help="Filter tasks by prompt ID (1-8)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter tasks by model (llama8b, llama3b, mistral)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run only one task and exit",
    )
    parser.add_argument(
        "--fixed-epoch",
        type=int,
        default=None,
        help="Fixed epoch mode (e.g., --fixed-epoch 20)",
    )
    parser.add_argument(
        "--reuse-e0-from",
        type=str,
        default=None,
        help="Reuse E0 results from existing results directory (e.g., results_fewshot_v2_dev10)",
    )

    args = parser.parse_args()

    # Generate worker ID
    hostname = socket.gethostname()
    worker_id = f"{hostname}-fewshot-{datetime.now().strftime('%H%M%S')}"
    logger.info(f"Worker ID: {worker_id}")

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create API client
    client = APIClient(args.server, worker_id)
    _state.client = client

    # Check server health
    if not client.health_check():
        logger.error(f"Cannot connect to server at {args.server}")
        sys.exit(1)
    logger.info(f"Connected to server: {args.server}")

    # Download ASAP data if needed
    data_dir = CLIENT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "training_set_rel3.tsv"

    if not data_path.exists():
        logger.info("Downloading ASAP data from server...")
        if not client.download_asap_data(data_path):
            logger.error("Failed to download ASAP data")
            sys.exit(1)

    # Get initial status
    if args.fixed_epoch is not None:
        status = client.get_fewshot_v2_fixed_status(args.fixed_epoch)
        logger.info(f"Fixed epoch mode: e={args.fixed_epoch}")
    else:
        status = client.get_fewshot_v2_status()
    if status:
        logger.info(f"Few-shot v2 status: {status.get('pending', 0)} pending, "
                   f"{status.get('running', 0)} running, "
                   f"{status.get('completed', 0)} completed")

    # Setup E0 reuse
    e0_results_dir_name = args.reuse_e0_from
    if e0_results_dir_name:
        logger.info(f"E0 results will be reused from server: {e0_results_dir_name}")

    # Main loop
    tasks_completed = 0
    consecutive_failures = 0
    max_failures = 3

    while not _state.shutdown_requested:
        # Get next task
        if args.fixed_epoch is not None:
            task = client.get_next_fewshot_v2_fixed_task(
                epoch=args.fixed_epoch,
                k=args.k,
                model=args.model,
                prompt=args.prompt,
            )
        else:
            task = client.get_next_fewshot_v2_task(
                k=args.k,
                model=args.model,
                prompt=args.prompt,
            )

        if task is None:
            if consecutive_failures > 0:
                logger.info("No more tasks available")
            else:
                logger.info("No pending tasks, waiting...")
            if args.once:
                break
            time.sleep(30)
            continue

        task_id = task["task_id"]
        _state.current_task_id = task_id
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting task: {task_id}")
        logger.info(f"{'='*60}")

        try:
            # Run experiment
            summary = run_fewshot_v2_experiment(
                task_config=task,
                data_path=data_path,
                client=client,
                e0_results_dir_name=e0_results_dir_name,
            )

            # Report completion
            if args.fixed_epoch is not None:
                success = client.complete_fewshot_v2_fixed_task(args.fixed_epoch, task_id, summary)
            else:
                success = client.complete_fewshot_v2_task(task_id, summary)

            if success:
                logger.info(f"Task {task_id} completed and reported to server")
                tasks_completed += 1
                consecutive_failures = 0
            else:
                logger.error(f"Failed to report task completion for {task_id}")

        except Exception as e:
            logger.exception(f"Task {task_id} failed: {e}")
            if args.fixed_epoch is not None:
                client.fail_fewshot_v2_fixed_task(args.fixed_epoch, task_id, str(e))
            else:
                client.fail_fewshot_v2_task(task_id, str(e))
            consecutive_failures += 1

            if consecutive_failures >= max_failures:
                logger.error(f"Too many consecutive failures ({max_failures}), stopping")
                break

        finally:
            _state.current_task_id = None

            # Clear GPU memory
            import torch
            gc.collect()
            torch.cuda.empty_cache()

        if args.once:
            break

    logger.info(f"\nWorker finished. Tasks completed: {tasks_completed}")


if __name__ == "__main__":
    main()
