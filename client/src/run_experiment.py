"""
Main experiment runner for zero-shot essay scoring with continual pre-training.

This script implements the full experimental pipeline:
1. Load ASAP data and create dev split
2. Load model once and apply LoRA
3. Evaluate baseline (epoch 0)
4. For each epoch: train then evaluate (same model instance)
5. Select best epoch and report results
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import hashlib
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import (
    ExperimentConfig,
    ASAP_SCORE_RANGES,
    create_default_config,
)
from data.data_loader import (
    ASAPDataLoader,
    load_asap_for_experiment,
    DatasetSplit,
)
from models.scorer import ZeroShotScorer, load_scorer
from models.logit_extractor import create_logit_extractor
from models.prompts import create_prompt_builder
from training.simple_trainer import SimpleLoRATrainer, SimpleTrainerConfig
from evaluation.metrics import (
    evaluate_predictions,
    evaluate_scoring_results,
    select_best_epoch,
    EvaluationResult,
    compute_p_invalid_stats,
)
from evaluation.visualize import create_evaluation_plots, plot_all_epoch_metrics, plot_metrics_progress
from utils.utils import (
    set_seed,
    setup_logging,
    save_json,
    save_csv,
    save_jsonl,
    ensure_dir,
    Timer,
    ExperimentTracker,
    cleanup_gpu_memory,
)

logger = logging.getLogger(__name__)


def get_training_config_dict(config: 'ExperimentConfig') -> Dict:
    """Extract training-related config for comparison."""
    return {
        'model_name': config.model_name,
        'prompt_id': config.data.prompt_id,
        'lr': config.cpt.lr,
        'lora_r': config.cpt.lora_r,
        'lora_alpha': config.cpt.lora_alpha,
        'target_modules': sorted(config.cpt.target_modules or []),
        'max_seq_len': config.cpt.max_seq_len,
        'batch_size': config.cpt.batch_size,
        'grad_accum_steps': config.cpt.grad_accum_steps,
        'max_epochs': config.cpt.max_epochs,
    }


def config_to_hash(config_dict: Dict) -> str:
    """Create a hash from training config for identification."""
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def check_existing_adapters(checkpoint_dir: Path, max_epochs: int) -> Tuple[bool, Optional[Dict]]:
    """
    Check if compatible adapters already exist.

    Returns:
        Tuple of (all_exist, saved_config)
    """
    config_file = checkpoint_dir / "training_config.json"

    if not config_file.exists():
        return False, None

    with open(config_file, 'r') as f:
        saved_config = json.load(f)

    # Check if all epoch adapters exist
    for epoch in range(1, max_epochs + 1):
        adapter_path = checkpoint_dir / f"epoch_{epoch}" / "adapter"
        if not (adapter_path / "adapter_config.json").exists():
            logger.info(f"Missing adapter for epoch {epoch}")
            return False, saved_config

    return True, saved_config


def find_last_completed_epoch(checkpoint_dir: Path, output_dir: Path, max_epochs: int) -> int:
    """
    Find the last completed epoch (both training and evaluation done).

    Returns:
        Last completed epoch number (0 if none completed, 0 means baseline only)
    """
    last_completed = 0

    for epoch in range(1, max_epochs + 1):
        # Check if adapter exists
        adapter_path = checkpoint_dir / f"epoch_{epoch}" / "adapter"
        if not (adapter_path / "adapter_config.json").exists():
            break

        # Check if evaluation results exist
        metrics_path = output_dir / f"metrics_epoch_{epoch}.json"
        if not metrics_path.exists():
            break

        last_completed = epoch

    return last_completed


def load_existing_results(output_dir: Path, last_epoch: int) -> Dict[int, Dict]:
    """
    Load existing evaluation results from previous epochs.

    Returns:
        Dictionary mapping epoch to result dict
    """
    epoch_results = {}

    for epoch in range(0, last_epoch + 1):
        metrics_path = output_dir / f"metrics_epoch_{epoch}.json"
        predictions_path = output_dir / f"predictions_epoch_{epoch}.csv"

        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            epoch_results[epoch] = {'metrics': metrics}
            logger.info(f"Loaded existing results for epoch {epoch}")

    return epoch_results


def save_training_config(checkpoint_dir: Path, config_dict: Dict):
    """Save training config to checkpoint directory."""
    config_file = checkpoint_dir / "training_config.json"
    config_dict['saved_at'] = datetime.now().isoformat()
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Saved training config to {config_file}")


def run_scoring(
    scorer: ZeroShotScorer,
    essays: List[Dict],
    compute_logit: bool = False,
    logit_extractor=None,
) -> List[Dict]:
    """
    Run scoring on essays.

    Args:
        scorer: ZeroShotScorer instance
        essays: List of essay dicts with 'essay_id', 'essay_text', 'score'
        compute_logit: Whether to compute logit-based extraction
        logit_extractor: Logit extractor instance

    Returns:
        List of scoring result dicts
    """
    results = scorer.score_essays(
        essays,
        compute_logit_expected=compute_logit,
        logit_extractor=logit_extractor,
        show_progress=True,
    )
    return [r.to_dict() for r in results]


def evaluate_epoch(
    scorer: ZeroShotScorer,
    dev_split: DatasetSplit,
    test_split: DatasetSplit,
    full_split: DatasetSplit,
    epoch: int,
    output_dir: Path,
    compute_logit: bool = True,
    logit_extractor=None,
) -> Dict:
    """
    Evaluate a single epoch.

    Args:
        scorer: Scorer instance
        dev_split: Development data split (for epoch selection)
        test_split: Test data split (for final evaluation, excludes dev)
        full_split: Full data split (for continual pre-training)
        epoch: Epoch number
        output_dir: Output directory
        compute_logit: Whether to compute logit extraction
        logit_extractor: Logit extractor instance

    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating epoch {epoch}")

    # Set model to eval mode
    scorer.model.eval()

    # Prepare essays for scoring (score all essays in full_split)
    essays = [
        {
            'essay_id': e.essay_id,
            'essay_text': e.essay_text,
            'score': e.score,
        }
        for e in full_split.essays
    ]

    # Run scoring
    with Timer(f"Scoring epoch {epoch}"):
        results = run_scoring(
            scorer, essays,
            compute_logit=compute_logit,
            logit_extractor=logit_extractor,
        )

    # Save predictions
    predictions_path = output_dir / f"predictions_epoch_{epoch}.csv"
    save_csv(results, str(predictions_path))

    # Save score distribution logs
    if compute_logit:
        dist_data = [
            {
                'essay_id': r['essay_id'],
                'range': [scorer.y_min, scorer.y_max],
                'logp_by_score': r.get('logp_by_score', {}),
                'p_valid_sum': r.get('p_valid_sum'),
                'p_invalid': r.get('p_invalid'),
            }
            for r in results
        ]
        dist_path = output_dir / f"score_dist_epoch_{epoch}.jsonl"
        save_jsonl(dist_data, str(dist_path))

    # Split results by dataset
    dev_ids = set(dev_split.essay_ids)
    test_ids = set(test_split.essay_ids)

    dev_results = [r for r in results if r['essay_id'] in dev_ids]
    test_results = [r for r in results if r['essay_id'] in test_ids]

    # Compute metrics for both greedy and expected value
    metrics = {}

    # Greedy predictions
    greedy_dev = evaluate_scoring_results(dev_results, 'y_hat_greedy')
    greedy_test = evaluate_scoring_results(test_results, 'y_hat_greedy')
    greedy_full = evaluate_scoring_results(results, 'y_hat_greedy')  # All essays

    metrics['greedy'] = {
        'dev': greedy_dev.to_dict(),
        'test': greedy_test.to_dict(),
        'full': greedy_full.to_dict(),  # Add full metrics
    }

    # Expected value predictions (if computed)
    if compute_logit:
        expected_dev = evaluate_scoring_results(dev_results, 'y_tilde_round')
        expected_test = evaluate_scoring_results(test_results, 'y_tilde_round')

        metrics['expected'] = {
            'dev': expected_dev.to_dict(),
            'test': expected_test.to_dict(),
        }

        # p_invalid statistics
        metrics['p_invalid_stats'] = {
            'dev': compute_p_invalid_stats(dev_results),
            'test': compute_p_invalid_stats(test_results),
        }

    # Save metrics
    metrics_path = output_dir / f"metrics_epoch_{epoch}.json"
    save_json(metrics, str(metrics_path))

    # Create visualization plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Greedy prediction plots for test set
    test_plots = create_evaluation_plots(
        results=test_results,
        y_min=scorer.y_min,
        y_max=scorer.y_max,
        output_dir=str(plots_dir),
        epoch=epoch,
        split_name="test",
        pred_column="y_hat_greedy",
    )

    # Expected value plots if computed
    if compute_logit:
        test_expected_plots = create_evaluation_plots(
            results=test_results,
            y_min=scorer.y_min,
            y_max=scorer.y_max,
            output_dir=str(plots_dir),
            epoch=epoch,
            split_name="test",
            pred_column="y_tilde_round",
        )

    return {
        'epoch': epoch,
        'metrics': metrics,
        'predictions_path': str(predictions_path),
        'plots_dir': str(plots_dir),
        'n_essays': len(results),
        'n_dev': len(dev_results),
        'n_test': len(test_results),
    }


def run_experiment(config: ExperimentConfig):
    """
    Run the full experiment.

    Uses a single model instance for both training and evaluation.

    Args:
        config: Experiment configuration
    """
    # Setup
    set_seed(config.seed)

    # Create output directories
    output_dir = config.output.get_output_dir(
        config.model_name, config.data.dataset, config.data.prompt_id
    )
    checkpoint_dir = Path(config.output.checkpoint_dir) / \
        config.model_name.split("/")[-1] / \
        config.data.dataset / \
        f"prompt_{config.data.prompt_id}"

    ensure_dir(output_dir)
    ensure_dir(checkpoint_dir)

    # Setup logging
    setup_logging(str(output_dir), log_file="experiment.log")
    logger.info(f"Starting experiment for prompt {config.data.prompt_id}")
    logger.info(f"Model: {config.model_name}")

    # Initialize tracker
    tracker = ExperimentTracker(str(output_dir))
    tracker.set_config(config.to_dict())

    # Load data
    logger.info("Loading data...")
    splits_dir = config.output.get_splits_dir(config.data.dataset, config.data.prompt_id)
    dev_split, test_split, full_split, dev_ids_path = load_asap_for_experiment(
        data_path=config.data.data_path,
        prompt_id=config.data.prompt_id,
        dev_M=config.dev_split.M,
        seed=config.dev_split.seed,
        splits_dir=str(splits_dir),
    )

    logger.info(f"Loaded {len(full_split)} essays: {len(dev_split)} for dev, {len(test_split)} for test")

    # Get score range
    y_min, y_max = config.data.y_min, config.data.y_max
    logger.info(f"Score range: [{y_min}, {y_max}]")

    # ============================================
    # Load model ONCE
    # ============================================
    logger.info("=" * 50)
    logger.info("Loading model (single instance for train & eval)")
    logger.info("=" * 50)

    dtype = getattr(torch, config.dtype) if isinstance(config.dtype, str) else config.dtype

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with optimizations
    # Try Flash Attention 2 if available
    attn_impl = None
    if torch.cuda.is_available():
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            logger.info("Flash Attention 2 enabled")
        except ImportError:
            logger.info("Flash Attention not available, using default attention")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map=config.device,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    # Enable gradient checkpointing if configured
    if config.cpt.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Enable cuDNN benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True

    logger.info(f"Model loaded on {config.device}")

    # ============================================
    # Initialize scorer (uses same model)
    # ============================================
    scorer = ZeroShotScorer(
        model_name=config.model_name,
        y_min=y_min,
        y_max=y_max,
        prompt_id=config.data.prompt_id,
        device=config.device,
        dtype=config.dtype,
    )
    scorer.set_model(model, tokenizer)

    # Initialize logit extractor if enabled
    logit_extractor = None
    if config.logit_extraction.enabled:
        logger.info("Initializing logit extractor...")
        logit_extractor = create_logit_extractor(
            model=model,
            tokenizer=tokenizer,
            y_min=y_min,
            y_max=y_max,
            max_steps=config.logit_extraction.max_steps,
            prob_threshold=config.logit_extraction.prob_threshold,
            device=config.device,
        )

    # ============================================
    # Epoch 0: Baseline (no continual pre-training)
    # ============================================
    epoch_results = {}

    # Track full metrics for live plotting
    full_metrics_history = {
        'epochs': [],
        'pearson': [],
        'spearman': [],
        'qwk': [],
    }
    metrics_plot_path = output_dir / "metrics_progress.png"

    # Check if epoch 0 results already exist (for resume)
    epoch_0_metrics_path = output_dir / "metrics_epoch_0.json"
    if epoch_0_metrics_path.exists():
        logger.info("=" * 50)
        logger.info("Loading existing baseline (epoch 0)")
        logger.info("=" * 50)
        with open(epoch_0_metrics_path, 'r') as f:
            epoch_0_metrics = json.load(f)
        epoch_results[0] = {'metrics': epoch_0_metrics}
        logger.info("Loaded existing epoch 0 results")
    else:
        logger.info("=" * 50)
        logger.info("Evaluating baseline (epoch 0)")
        logger.info("=" * 50)

        epoch_0_result = evaluate_epoch(
            scorer=scorer,
            dev_split=dev_split,
            test_split=test_split,
            full_split=full_split,
            epoch=0,
            output_dir=output_dir,
            compute_logit=config.logit_extraction.enabled,
            logit_extractor=logit_extractor,
        )
        epoch_results[0] = epoch_0_result
        tracker.add_epoch_result(0, epoch_0_result['metrics'])

    # Update metrics history and plot for epoch 0
    full_m = epoch_results[0]['metrics'].get('greedy', {}).get('full', {})
    full_metrics_history['epochs'].append(0)
    full_metrics_history['pearson'].append(full_m.get('pearson', 0))
    full_metrics_history['spearman'].append(full_m.get('spearman', 0))
    full_metrics_history['qwk'].append(full_m.get('qwk', 0))

    plot_metrics_progress(
        epochs=full_metrics_history['epochs'],
        pearson_values=full_metrics_history['pearson'],
        spearman_values=full_metrics_history['spearman'],
        qwk_values=full_metrics_history['qwk'],
        save_path=str(metrics_plot_path),
        title=f"Metrics Progress - Prompt {config.data.prompt_id} ({config.model_name.split('/')[-1]})",
        max_epochs=config.cpt.max_epochs,
    )
    logger.info(f"Full metrics (epoch 0): Spearman={full_m.get('spearman', 0):.4f}, QWK={full_m.get('qwk', 0):.4f}")

    # ============================================
    # Continual Pre-training (if epochs > 0)
    # ============================================
    if config.cpt.max_epochs > 0:
        logger.info("=" * 50)
        logger.info("Starting continual pre-training")
        logger.info("=" * 50)

        # Check for existing compatible adapters
        training_config = get_training_config_dict(config)
        adapters_exist, saved_config = check_existing_adapters(checkpoint_dir, config.cpt.max_epochs)

        # Compare configs (excluding timestamp)
        use_existing = False
        if adapters_exist and saved_config:
            saved_config_compare = {k: v for k, v in saved_config.items() if k != 'saved_at'}
            if saved_config_compare == training_config:
                use_existing = True
                logger.info("=" * 50)
                logger.info("Found existing compatible adapters - skipping training")
                logger.info(f"Config hash: {config_to_hash(training_config)}")
                logger.info("=" * 50)
            else:
                logger.info("Existing adapters have different config - will retrain")
                logger.info(f"Saved config: {saved_config_compare}")
                logger.info(f"Current config: {training_config}")

        # Check for partial progress (resume capability)
        last_completed_epoch = find_last_completed_epoch(checkpoint_dir, output_dir, config.cpt.max_epochs)

        if last_completed_epoch > 0:
            logger.info("=" * 50)
            logger.info(f"Found existing progress: epochs 1-{last_completed_epoch} completed")
            logger.info(f"Will resume from epoch {last_completed_epoch + 1}")
            logger.info("=" * 50)

            # Load existing results
            epoch_results.update(load_existing_results(output_dir, last_completed_epoch))

            # Update metrics history from loaded results
            for ep in range(0, last_completed_epoch + 1):
                if ep in epoch_results:
                    full_m = epoch_results[ep]['metrics'].get('greedy', {}).get('full', {})
                    if ep not in full_metrics_history['epochs']:
                        full_metrics_history['epochs'].append(ep)
                        full_metrics_history['pearson'].append(full_m.get('pearson', 0))
                        full_metrics_history['spearman'].append(full_m.get('spearman', 0))
                        full_metrics_history['qwk'].append(full_m.get('qwk', 0))

        if use_existing:
            # Load existing adapters for evaluation (all epochs complete)
            for epoch in range(1, config.cpt.max_epochs + 1):
                logger.info("=" * 50)
                logger.info(f"Epoch {epoch}/{config.cpt.max_epochs} (loading existing adapter)")
                logger.info("=" * 50)

                adapter_path = checkpoint_dir / f"epoch_{epoch}" / "adapter"
                logger.info(f"Loading adapter from: {adapter_path}")

                # Load adapter onto base model
                if epoch == 1:
                    lora_model = PeftModel.from_pretrained(model, str(adapter_path))
                else:
                    # For subsequent epochs, we need to reload the base model with new adapter
                    # But since we're just evaluating, we can load fresh each time
                    lora_model = PeftModel.from_pretrained(model, str(adapter_path))

                lora_model.eval()
                scorer.set_model(lora_model, tokenizer)

                # Reinitialize logit extractor with LoRA model
                if config.logit_extraction.enabled:
                    logit_extractor = create_logit_extractor(
                        model=lora_model,
                        tokenizer=tokenizer,
                        y_min=y_min,
                        y_max=y_max,
                        max_steps=config.logit_extraction.max_steps,
                        prob_threshold=config.logit_extraction.prob_threshold,
                        device=config.device,
                    )

                # Evaluate
                epoch_result = evaluate_epoch(
                    scorer=scorer,
                    dev_split=dev_split,
                    test_split=test_split,
                    full_split=full_split,
                    epoch=epoch,
                    output_dir=output_dir,
                    compute_logit=config.logit_extraction.enabled,
                    logit_extractor=logit_extractor,
                )
                epoch_results[epoch] = epoch_result
                tracker.add_epoch_result(
                    epoch, epoch_result['metrics'],
                    adapter_path=str(adapter_path)
                )

                # Log metrics and update progress plot
                full_m = epoch_result['metrics'].get('greedy', {}).get('full', {})
                full_metrics_history['epochs'].append(epoch)
                full_metrics_history['pearson'].append(full_m.get('pearson', 0))
                full_metrics_history['spearman'].append(full_m.get('spearman', 0))
                full_metrics_history['qwk'].append(full_m.get('qwk', 0))

                plot_metrics_progress(
                    epochs=full_metrics_history['epochs'],
                    pearson_values=full_metrics_history['pearson'],
                    spearman_values=full_metrics_history['spearman'],
                    qwk_values=full_metrics_history['qwk'],
                    save_path=str(metrics_plot_path),
                    title=f"Metrics Progress - Prompt {config.data.prompt_id} ({config.model_name.split('/')[-1]})",
                    max_epochs=config.cpt.max_epochs,
                )
                logger.info(f"Full metrics (epoch {epoch}): Spearman={full_m.get('spearman', 0):.4f}, QWK={full_m.get('qwk', 0):.4f}")

                # Clean up for next iteration
                del lora_model
                cleanup_gpu_memory()

        else:
            # Train new adapters (or resume from checkpoint)
            # Get essay texts for CPT
            essay_texts = full_split.get_texts()

            # Determine start epoch
            start_epoch = last_completed_epoch + 1

            # Initialize simple trainer config
            trainer_config = SimpleTrainerConfig(
                lr=config.cpt.lr,
                lora_r=config.cpt.lora_r,
                lora_alpha=config.cpt.lora_alpha,
                lora_dropout=0.0,
                target_modules=config.cpt.target_modules,
                max_seq_len=config.cpt.max_seq_len,
                batch_size=config.cpt.batch_size,
                grad_accum_steps=config.cpt.grad_accum_steps,
                seed=config.seed,  # For reproducible shuffling
            )

            # Initialize trainer with the SAME model
            trainer = SimpleLoRATrainer(
                model=model,
                tokenizer=tokenizer,
                config=trainer_config,
                output_dir=str(checkpoint_dir),
            )

            # Resume from checkpoint or start fresh
            if last_completed_epoch > 0:
                # Load the last completed adapter to resume training
                last_adapter_path = checkpoint_dir / f"epoch_{last_completed_epoch}" / "adapter"
                logger.info(f"Resuming from epoch {last_completed_epoch}, loading adapter from {last_adapter_path}")
                trainer.model = PeftModel.from_pretrained(model, str(last_adapter_path), is_trainable=True)
                trainer._lora_applied = True
                # Reinitialize optimizer for resumed training
                from torch.optim import AdamW
                trainer.optimizer = AdamW(trainer.model.parameters(), lr=trainer_config.lr)
            else:
                # Apply LoRA (only once for fresh start)
                trainer.apply_lora()

            # Save training config
            save_training_config(checkpoint_dir, training_config)

            # Update scorer to use the LoRA model
            scorer.set_model(trainer.model, tokenizer)

            # Reinitialize logit extractor with LoRA model
            if config.logit_extraction.enabled:
                logit_extractor = create_logit_extractor(
                    model=trainer.model,
                    tokenizer=tokenizer,
                    y_min=y_min,
                    y_max=y_max,
                    max_steps=config.logit_extraction.max_steps,
                    prob_threshold=config.logit_extraction.prob_threshold,
                    device=config.device,
                )

            # Train and evaluate each epoch (resume from start_epoch)
            for epoch in range(start_epoch, config.cpt.max_epochs + 1):
                logger.info("=" * 50)
                logger.info(f"Epoch {epoch}/{config.cpt.max_epochs}")
                logger.info("=" * 50)

                # Train for one epoch (model.train() is called inside)
                train_result = trainer.train_epoch(essay_texts, epoch)
                logger.info(f"Training loss: {train_result['avg_loss']:.4f}")
                logger.info(f"Adapter saved to: {train_result['adapter_path']}")

                # Evaluate (model.eval() is called inside evaluate_epoch)
                epoch_result = evaluate_epoch(
                    scorer=scorer,
                    dev_split=dev_split,
                    test_split=test_split,
                    full_split=full_split,
                    epoch=epoch,
                    output_dir=output_dir,
                    compute_logit=config.logit_extraction.enabled,
                    logit_extractor=logit_extractor,
                )
                epoch_results[epoch] = epoch_result
                tracker.add_epoch_result(
                    epoch, epoch_result['metrics'],
                    adapter_path=train_result['adapter_path']
                )

                # Log metrics and update progress plot
                full_m = epoch_result['metrics'].get('greedy', {}).get('full', {})
                full_metrics_history['epochs'].append(epoch)
                full_metrics_history['pearson'].append(full_m.get('pearson', 0))
                full_metrics_history['spearman'].append(full_m.get('spearman', 0))
                full_metrics_history['qwk'].append(full_m.get('qwk', 0))

                plot_metrics_progress(
                    epochs=full_metrics_history['epochs'],
                    pearson_values=full_metrics_history['pearson'],
                    spearman_values=full_metrics_history['spearman'],
                    qwk_values=full_metrics_history['qwk'],
                    save_path=str(metrics_plot_path),
                    title=f"Metrics Progress - Prompt {config.data.prompt_id} ({config.model_name.split('/')[-1]})",
                    max_epochs=config.cpt.max_epochs,
                )
                logger.info(f"Full metrics (epoch {epoch}): Spearman={full_m.get('spearman', 0):.4f}, QWK={full_m.get('qwk', 0):.4f}")

    # ============================================
    # Select best epoch
    # ============================================
    logger.info("=" * 50)
    logger.info("Selecting best epoch")
    logger.info("=" * 50)

    # Build evaluation results for epoch selection
    selection_target = config.selection.selection_target
    metric_key = 'greedy' if selection_target == 'y_hat_greedy' else 'expected'

    dev_eval_results = {}
    for epoch, result in epoch_results.items():
        metrics = result['metrics'].get(metric_key, result['metrics'].get('greedy', {}))
        dev_metrics = metrics.get('dev', {})
        dev_eval_results[epoch] = EvaluationResult(
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

    # Select e* (best on dev)
    e_star, e_star_result = select_best_epoch(
        dev_eval_results,
        metric=config.selection.metric_for_e_star,
    )

    # Find e_oracle (best on test data - oracle upper bound)
    test_eval_results = {}
    for epoch, result in epoch_results.items():
        metrics = result['metrics'].get(metric_key, result['metrics'].get('greedy', {}))
        test_metrics = metrics.get('test', {})
        test_eval_results[epoch] = EvaluationResult(
            pearson=test_metrics.get('pearson', 0),
            pearson_pvalue=test_metrics.get('pearson_pvalue', 1),
            spearman=test_metrics.get('spearman', 0),
            spearman_pvalue=test_metrics.get('spearman_pvalue', 1),
            qwk=test_metrics.get('qwk', 0),
            n_samples=test_metrics.get('n_samples', 0),
            n_valid=test_metrics.get('n_valid', 0),
            mean_y_true=test_metrics.get('mean_y_true', 0),
            mean_y_pred=test_metrics.get('mean_y_pred', 0),
            std_y_true=test_metrics.get('std_y_true', 0),
            std_y_pred=test_metrics.get('std_y_pred', 0),
        )

    e_oracle, e_oracle_result = select_best_epoch(
        test_eval_results,
        metric=config.selection.metric_for_e_star,
    )

    logger.info(f"e* (dev-selected): epoch {e_star}, {config.selection.metric_for_e_star}={getattr(e_star_result, config.selection.metric_for_e_star):.4f}")
    logger.info(f"e_oracle (test): epoch {e_oracle}, {config.selection.metric_for_e_star}={getattr(e_oracle_result, config.selection.metric_for_e_star):.4f}")

    tracker.set_best_epoch(e_star, config.selection.metric_for_e_star, getattr(e_star_result, config.selection.metric_for_e_star))

    # ============================================
    # Final summary
    # ============================================
    summary = {
        'prompt_id': config.data.prompt_id,
        'model_name': config.model_name,
        'score_range': [y_min, y_max],
        'n_essays': len(full_split),
        'n_dev': len(dev_split),
        'n_test': len(test_split),
        'baseline': {
            'epoch': 0,
            'dev': epoch_results[0]['metrics'].get(metric_key, epoch_results[0]['metrics'].get('greedy', {})).get('dev', {}),
            'test': epoch_results[0]['metrics'].get(metric_key, epoch_results[0]['metrics'].get('greedy', {})).get('test', {}),
        },
        'e_star': {
            'epoch': e_star,
            'dev': epoch_results[e_star]['metrics'].get(metric_key, epoch_results[e_star]['metrics'].get('greedy', {})).get('dev', {}),
            'test': epoch_results[e_star]['metrics'].get(metric_key, epoch_results[e_star]['metrics'].get('greedy', {})).get('test', {}),
        },
        'e_oracle': {
            'epoch': e_oracle,
            'dev': epoch_results[e_oracle]['metrics'].get(metric_key, epoch_results[e_oracle]['metrics'].get('greedy', {})).get('dev', {}),
            'test': epoch_results[e_oracle]['metrics'].get(metric_key, epoch_results[e_oracle]['metrics'].get('greedy', {})).get('test', {}),
        },
        'selection_metric': config.selection.metric_for_e_star,
        'selection_target': selection_target,
    }

    save_json(summary, str(output_dir / "summary.json"))
    tracker.finish()

    # ============================================
    # Create epoch-wise metric plots
    # ============================================
    if len(epoch_results) > 1:
        logger.info("Creating epoch-wise metric plots...")
        plots_dir = output_dir / "plots"

        # Greedy prediction metrics
        plot_all_epoch_metrics(
            epoch_results=epoch_results,
            output_dir=str(plots_dir),
            pred_type="greedy",
        )

        # Expected value metrics (if computed)
        if config.logit_extraction.enabled:
            plot_all_epoch_metrics(
                epoch_results=epoch_results,
                output_dir=str(plots_dir),
                pred_type="expected",
            )

    logger.info("=" * 50)
    logger.info("Experiment completed")
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 50)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run zero-shot essay scoring with continual pre-training"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--prompt_id", type=int, default=1,
        help="ASAP prompt ID (1-8)"
    )
    parser.add_argument(
        "--data_path", type=str, default="data/asap/training_set_rel3.tsv",
        help="Path to ASAP data file"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=30,
        help="Maximum epochs for continual pre-training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--disable_logit", action="store_true",
        help="Disable logit-based extraction"
    )

    args = parser.parse_args()

    # Create or load config
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = create_default_config(args.prompt_id, args.model)
        config.data.data_path = args.data_path
        config.cpt.max_epochs = args.max_epochs
        config.cpt.lr = args.lr
        config.device = args.device
        config.seed = args.seed
        config.logit_extraction.enabled = not args.disable_logit

    # Run experiment
    run_experiment(config)


if __name__ == "__main__":
    main()
