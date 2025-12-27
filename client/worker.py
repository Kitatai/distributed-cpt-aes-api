#!/usr/bin/env python3
"""
Worker script for distributed CPT-AES experiments.

Connects to the API server, gets tasks, runs experiments locally,
and uploads results back to the server.
"""

import os
import sys
import time
import signal
import socket
import logging
import argparse
import tempfile
import shutil
import threading
from pathlib import Path
from datetime import datetime

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
    sig_name = signal.Signals(signum).name
    logger.info(f"\nReceived {sig_name}, initiating graceful shutdown...")

    with _state.lock:
        _state.shutdown_requested = True

        if _state.current_task_id and _state.client:
            logger.info(f"Releasing current task: {_state.current_task_id}")
            try:
                _state.client.release_task(_state.current_task_id)
            except Exception as e:
                logger.error(f"Failed to release task: {e}")


def get_worker_id() -> str:
    """Generate a unique worker ID."""
    hostname = socket.gethostname()
    pid = os.getpid()
    return f"{hostname}_{pid}"


def cleanup_gpu_memory():
    """Clear GPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
    except Exception as e:
        logger.warning(f"GPU cleanup failed: {e}")


def run_experiment_for_task(
    client: APIClient,
    task_id: str,
    config: dict,
    data_path: Path,
    work_dir: Path,
) -> dict:
    """
    Run experiment for a single task.

    Args:
        client: API client
        task_id: Task identifier
        config: Experiment configuration
        data_path: Path to ASAP data
        work_dir: Working directory for this task

    Returns:
        Experiment summary dict
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, LoraConfig, get_peft_model, TaskType

    from config import ExperimentConfig, create_default_config, ASAP_SCORE_RANGES
    from data.data_loader import load_asap_for_experiment
    from models.scorer import ZeroShotScorer
    from models.logit_extractor import create_logit_extractor
    from training.simple_trainer import SimpleLoRATrainer, SimpleTrainerConfig
    from evaluation.metrics import evaluate_scoring_results, select_best_epoch, EvaluationResult
    from evaluation.visualize import plot_metrics_progress
    from utils.utils import set_seed, save_json, save_csv, ensure_dir, Timer

    # Setup directories
    checkpoint_dir = work_dir / "checkpoints"
    output_dir = work_dir / "results"
    splits_dir = work_dir / "splits"

    for d in [checkpoint_dir, output_dir, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create experiment config
    prompt_id = config["prompt_id"]
    model_name = config["model_name"]
    max_epochs = config.get("max_epochs", 30)
    seed = config.get("seed", 42)

    exp_config = create_default_config(prompt_id, model_name)
    exp_config.data.data_path = str(data_path)
    exp_config.cpt.max_epochs = max_epochs
    exp_config.seed = seed
    exp_config.cpt.lr = config.get("lr", 1e-6)
    exp_config.cpt.lora_r = config.get("lora_r", 4)
    exp_config.cpt.lora_alpha = config.get("lora_alpha", 16)
    exp_config.cpt.max_seq_len = config.get("max_seq_len", 256)
    exp_config.cpt.batch_size = config.get("batch_size", 1)
    exp_config.cpt.grad_accum_steps = config.get("grad_accum_steps", 8)

    set_seed(seed)

    # Load data
    logger.info(f"Loading ASAP data for prompt {prompt_id}...")
    dev_split, test_split, full_split, _ = load_asap_for_experiment(
        data_path=str(data_path),
        prompt_id=prompt_id,
        dev_M=config.get("dev_M", 5),
        seed=config.get("dev_seed", 42),
        splits_dir=str(splits_dir),
    )

    y_min, y_max = exp_config.data.y_min, exp_config.data.y_max
    logger.info(f"Loaded {len(full_split)} essays, score range: [{y_min}, {y_max}]")

    # Check for existing progress on server
    last_results_epoch = client.get_last_results(task_id)
    logger.info(f"Last completed epoch on server: {last_results_epoch}")

    # Load model
    logger.info(f"Loading model: {model_name}")
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try Flash Attention
    attn_impl = None
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        logger.info("Flash Attention 2 enabled")
    except ImportError:
        logger.info("Flash Attention not available")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    if exp_config.cpt.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    torch.backends.cudnn.benchmark = True

    # Initialize scorer
    scorer = ZeroShotScorer(
        model_name=model_name,
        y_min=y_min,
        y_max=y_max,
        prompt_id=prompt_id,
        device="cuda",
        dtype="bfloat16",
    )
    scorer.set_model(model, tokenizer)

    # Initialize logit extractor
    logit_extractor = None
    if exp_config.logit_extraction.enabled:
        logit_extractor = create_logit_extractor(
            model=model,
            tokenizer=tokenizer,
            y_min=y_min,
            y_max=y_max,
            max_steps=exp_config.logit_extraction.max_steps,
            prob_threshold=exp_config.logit_extraction.prob_threshold,
            device="cuda",
        )

    # Results tracking
    epoch_results = {}
    full_metrics_history = {'epochs': [], 'pearson': [], 'spearman': [], 'qwk': []}

    def evaluate_epoch(epoch: int) -> dict:
        """Evaluate a single epoch."""
        scorer.model.eval()

        essays = [
            {'essay_id': e.essay_id, 'essay_text': e.essay_text, 'score': e.score}
            for e in full_split.essays
        ]

        results = scorer.score_essays(
            essays,
            compute_logit_expected=exp_config.logit_extraction.enabled,
            logit_extractor=logit_extractor,
            show_progress=True,
        )
        results = [r.to_dict() for r in results]

        # Save predictions locally
        predictions_path = output_dir / f"predictions_epoch_{epoch}.csv"
        save_csv(results, str(predictions_path))

        # Compute metrics
        dev_ids = set(dev_split.essay_ids)
        test_ids = set(test_split.essay_ids)

        dev_results = [r for r in results if r['essay_id'] in dev_ids]
        test_results = [r for r in results if r['essay_id'] in test_ids]

        greedy_dev = evaluate_scoring_results(dev_results, 'y_hat_greedy')
        greedy_test = evaluate_scoring_results(test_results, 'y_hat_greedy')
        greedy_full = evaluate_scoring_results(results, 'y_hat_greedy')

        metrics = {
            'greedy': {
                'dev': greedy_dev.to_dict(),
                'test': greedy_test.to_dict(),
                'full': greedy_full.to_dict(),
            }
        }

        if exp_config.logit_extraction.enabled:
            expected_dev = evaluate_scoring_results(dev_results, 'y_tilde_round')
            expected_test = evaluate_scoring_results(test_results, 'y_tilde_round')
            metrics['expected'] = {
                'dev': expected_dev.to_dict(),
                'test': expected_test.to_dict(),
            }

        # Save metrics locally
        metrics_path = output_dir / f"metrics_epoch_{epoch}.json"
        save_json(metrics, str(metrics_path))

        # Upload to server
        client.upload_metrics(task_id, epoch, metrics)
        client.upload_predictions(task_id, epoch, predictions_path)

        return {'metrics': metrics, 'predictions_path': str(predictions_path)}

    # Epoch 0: Baseline
    if last_results_epoch < 0:
        logger.info("=" * 50)
        logger.info("Evaluating baseline (epoch 0)")
        logger.info("=" * 50)

        epoch_results[0] = evaluate_epoch(0)
        full_m = epoch_results[0]['metrics'].get('greedy', {}).get('full', {})
        full_metrics_history['epochs'].append(0)
        full_metrics_history['pearson'].append(full_m.get('pearson', 0))
        full_metrics_history['spearman'].append(full_m.get('spearman', 0))
        full_metrics_history['qwk'].append(full_m.get('qwk', 0))
        logger.info(f"Epoch 0: Spearman={full_m.get('spearman', 0):.4f}, QWK={full_m.get('qwk', 0):.4f}")
    else:
        # Load existing epoch 0 results from server
        logger.info("Loading existing epoch 0 results from server...")
        epoch_0_metrics = client.download_metrics(task_id, 0)
        if epoch_0_metrics:
            epoch_results[0] = {'metrics': epoch_0_metrics}
            full_m = epoch_0_metrics.get('greedy', {}).get('full', {})
            full_metrics_history['epochs'].append(0)
            full_metrics_history['pearson'].append(full_m.get('pearson', 0))
            full_metrics_history['spearman'].append(full_m.get('spearman', 0))
            full_metrics_history['qwk'].append(full_m.get('qwk', 0))

    # Continual pre-training
    if max_epochs > 0:
        logger.info("=" * 50)
        logger.info("Starting continual pre-training")
        logger.info("=" * 50)

        # Determine start epoch
        start_epoch = max(1, last_results_epoch + 1)

        # Initialize trainer
        trainer_config = SimpleTrainerConfig(
            lr=exp_config.cpt.lr,
            lora_r=exp_config.cpt.lora_r,
            lora_alpha=exp_config.cpt.lora_alpha,
            lora_dropout=0.0,
            target_modules=exp_config.cpt.target_modules,
            max_seq_len=exp_config.cpt.max_seq_len,
            batch_size=exp_config.cpt.batch_size,
            grad_accum_steps=exp_config.cpt.grad_accum_steps,
            seed=seed,
        )

        trainer = SimpleLoRATrainer(
            model=model,
            tokenizer=tokenizer,
            config=trainer_config,
            output_dir=str(checkpoint_dir),
        )

        # Resume from checkpoint if needed
        if start_epoch > 1:
            logger.info(f"Resuming from epoch {start_epoch - 1}")
            # Download checkpoint from server
            adapter_dir = checkpoint_dir / f"epoch_{start_epoch - 1}" / "adapter"
            if client.download_checkpoint(task_id, start_epoch - 1, adapter_dir):
                trainer.model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
                trainer._lora_applied = True
                from torch.optim import AdamW
                trainer.optimizer = AdamW(trainer.model.parameters(), lr=trainer_config.lr)
                scorer.set_model(trainer.model, tokenizer)
                if logit_extractor:
                    logit_extractor = create_logit_extractor(
                        model=trainer.model, tokenizer=tokenizer,
                        y_min=y_min, y_max=y_max,
                        max_steps=exp_config.logit_extraction.max_steps,
                        prob_threshold=exp_config.logit_extraction.prob_threshold,
                        device="cuda",
                    )
            else:
                logger.warning("Failed to download checkpoint, starting from scratch")
                start_epoch = 1
                trainer.apply_lora()
                scorer.set_model(trainer.model, tokenizer)
        else:
            trainer.apply_lora()
            scorer.set_model(trainer.model, tokenizer)

        if logit_extractor and start_epoch == 1:
            logit_extractor = create_logit_extractor(
                model=trainer.model, tokenizer=tokenizer,
                y_min=y_min, y_max=y_max,
                max_steps=exp_config.logit_extraction.max_steps,
                prob_threshold=exp_config.logit_extraction.prob_threshold,
                device="cuda",
            )

        essay_texts = full_split.get_texts()

        # Train and evaluate each epoch
        for epoch in range(start_epoch, max_epochs + 1):
            logger.info("=" * 50)
            logger.info(f"Epoch {epoch}/{max_epochs}")
            logger.info("=" * 50)

            # Train
            train_result = trainer.train_epoch(essay_texts, epoch)
            logger.info(f"Training loss: {train_result['avg_loss']:.4f}")

            # Upload checkpoint
            adapter_path = Path(train_result['adapter_path'])
            client.upload_checkpoint(task_id, epoch, adapter_path)

            # Evaluate
            epoch_results[epoch] = evaluate_epoch(epoch)

            # Update progress on server
            client.update_progress(task_id, epoch)

            # Log metrics
            full_m = epoch_results[epoch]['metrics'].get('greedy', {}).get('full', {})
            full_metrics_history['epochs'].append(epoch)
            full_metrics_history['pearson'].append(full_m.get('pearson', 0))
            full_metrics_history['spearman'].append(full_m.get('spearman', 0))
            full_metrics_history['qwk'].append(full_m.get('qwk', 0))
            logger.info(f"Epoch {epoch}: Spearman={full_m.get('spearman', 0):.4f}, QWK={full_m.get('qwk', 0):.4f}")

            # Plot progress
            plot_metrics_progress(
                epochs=full_metrics_history['epochs'],
                pearson_values=full_metrics_history['pearson'],
                spearman_values=full_metrics_history['spearman'],
                qwk_values=full_metrics_history['qwk'],
                save_path=str(output_dir / "metrics_progress.png"),
                title=f"Metrics Progress - Prompt {prompt_id} ({model_name.split('/')[-1]})",
                max_epochs=max_epochs,
            )

    # Select best epoch
    metric_key = 'greedy'
    dev_eval_results = {}
    for ep, result in epoch_results.items():
        metrics = result['metrics'].get(metric_key, {})
        dev_metrics = metrics.get('dev', {})
        dev_eval_results[ep] = EvaluationResult(
            pearson=dev_metrics.get('pearson', 0),
            pearson_pvalue=dev_metrics.get('pearson_pvalue', 1),
            spearman=dev_metrics.get('spearman', 0),
            spearman_pvalue=dev_metrics.get('spearman_pvalue', 1),
            qwk=dev_metrics.get('qwk', 0),
            n_samples=dev_metrics.get('n_samples', 0),
            n_valid=dev_metrics.get('n_valid', 0),
            mean_y_true=dev_metrics.get('mean_y_true', 0),
            mean_y_pred=dev_metrics.get('mean_y_pred', 0),
            std_y_true=dev_metrics.get('std_y_true', 0),
            std_y_pred=dev_metrics.get('std_y_pred', 0),
        )

    e_star, _ = select_best_epoch(dev_eval_results, metric='qwk')

    # Create summary
    summary = {
        'task_id': task_id,
        'prompt_id': prompt_id,
        'model_name': model_name,
        'score_range': [y_min, y_max],
        'n_essays': len(full_split),
        'n_dev': len(dev_split),
        'n_test': len(test_split),
        'e_star': e_star,
        'completed_at': datetime.now().isoformat(),
    }

    if 0 in epoch_results:
        summary['baseline'] = epoch_results[0]['metrics'].get(metric_key, {})

    if e_star in epoch_results:
        summary['best'] = epoch_results[e_star]['metrics'].get(metric_key, {})

    save_json(summary, str(output_dir / "summary.json"))

    logger.info("=" * 50)
    logger.info(f"Experiment completed: e* = {e_star}")
    logger.info("=" * 50)

    return summary


def worker_loop(client: APIClient, data_dir: Path, single: bool = False):
    """
    Main worker loop.

    Args:
        client: API client
        data_dir: Directory for local data
        single: If True, process only one task then exit
    """
    global _state

    # Store client in global state for signal handler
    _state.client = client

    # Check server health
    if not client.health_check():
        logger.error("Server is not healthy!")
        return

    logger.info(f"Worker {client.worker_id} starting...")
    logger.info("Press Ctrl+C to stop gracefully")

    # Download ASAP data if not present
    asap_path = data_dir / "asap" / "training_set_rel3.tsv"
    if not asap_path.exists():
        logger.info("Downloading ASAP data from server...")
        if not client.download_asap_data(asap_path):
            logger.error("Failed to download ASAP data!")
            return

    while not _state.shutdown_requested:
        # Get next task
        task = client.get_next_task()

        if not task:
            logger.info("No pending tasks available")
            if single:
                break
            logger.info("Waiting 60 seconds before checking again...")
            # Check for shutdown during wait
            for _ in range(60):
                if _state.shutdown_requested:
                    break
                time.sleep(1)
            continue

        task_id = task['task_id']
        logger.info(f"Got task: {task_id}")

        # Track current task for graceful shutdown
        with _state.lock:
            _state.current_task_id = task_id

        # Get task config
        config = client.get_task_config(task_id)
        if not config:
            logger.error(f"Failed to get config for {task_id}")
            client.fail_task(task_id, "Failed to get task configuration")
            with _state.lock:
                _state.current_task_id = None
            continue

        # Create work directory
        work_dir = data_dir / "work" / task_id
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Run experiment
            summary = run_experiment_for_task(
                client=client,
                task_id=task_id,
                config=config,
                data_path=asap_path,
                work_dir=work_dir,
            )

            # Mark task as completed (only if not shutdown)
            if not _state.shutdown_requested:
                client.complete_task(task_id, summary)
                logger.info(f"Task {task_id} completed successfully")

        except Exception as e:
            if _state.shutdown_requested:
                logger.info(f"Task {task_id} interrupted by shutdown")
                # Task already released by signal handler
            else:
                logger.exception(f"Task {task_id} failed")
                client.fail_task(task_id, str(e))

        finally:
            # Clear current task
            with _state.lock:
                _state.current_task_id = None
            cleanup_gpu_memory()

        if single or _state.shutdown_requested:
            break

    logger.info("Worker stopped")


def main():
    parser = argparse.ArgumentParser(description="CPT-AES Worker")
    parser.add_argument(
        "--server",
        type=str,
        required=True,
        help="Server URL (e.g., http://192.168.100.10:8000)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Local data directory (default: ~/.cpt-aes-worker)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Process only one task then exit",
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Worker ID (default: auto-generated from hostname)",
    )

    args = parser.parse_args()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Setup data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path.home() / ".cpt-aes-worker"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create API client
    worker_id = args.worker_id or get_worker_id()
    client = APIClient(args.server, worker_id)

    logger.info(f"Worker ID: {worker_id}")
    logger.info(f"Server: {args.server}")
    logger.info(f"Data directory: {data_dir}")

    # Run worker loop
    worker_loop(client, data_dir, single=args.single)


if __name__ == "__main__":
    main()
