#!/usr/bin/env python3
"""
DeBERTa baseline experiment for CPT-AES comparison.

Uses the same sample patterns as few-shot experiments:
- test_ids: 10% for evaluation
- fewshot_ids (5) + dev_ids (10) = 15 samples for train/val

Train:Val ratios tested: 5:10, 7:8, 9:6, 11:4, 13:2

Model: DeBERTa + Linear + Sigmoid (regression to 0-1 scaled scores)

Usage:
    python run_deberta_baseline.py [--prompt PROMPT_ID] [--pattern PATTERN_IDX]
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from scipy.stats import spearmanr
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Score ranges for each prompt
ASAP_SCORE_RANGES = {
    1: (2, 12), 2: (1, 6), 3: (0, 3), 4: (0, 3),
    5: (0, 4), 6: (0, 4), 7: (0, 30), 8: (0, 60),
}

# Train:Val split configurations
SPLIT_CONFIGS = [
    (5, 10),   # Same as few-shot: 5 train, 10 val
    (7, 8),
    (9, 6),
    (11, 4),
    (13, 2),
]


class EssayDataset(Dataset):
    """Dataset for essay scoring."""

    def __init__(self, essays: List[str], scores: List[float], tokenizer, max_length: int = 512):
        self.essays = essays
        self.scores = scores  # Already scaled to 0-1
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.essays[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'score': torch.tensor(self.scores[idx], dtype=torch.float32)
        }


class DeBERTaRegressor(nn.Module):
    """DeBERTa with linear head for regression."""

    def __init__(self, model_name: str = "microsoft/deberta-v3-base"):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.deberta.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        score = self.regressor(cls_output).squeeze(-1)
        return score


def scale_score(score: float, y_min: int, y_max: int) -> float:
    """Scale score to 0-1 range."""
    return (score - y_min) / (y_max - y_min)


def unscale_score(scaled: float, y_min: int, y_max: int) -> float:
    """Unscale score from 0-1 to original range."""
    return scaled * (y_max - y_min) + y_min


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_min: int, y_max: int) -> Dict:
    """Calculate QWK and Spearman correlation."""
    # Round predictions to integers
    y_pred_rounded = np.clip(np.round(y_pred), y_min, y_max).astype(int)
    y_true_int = y_true.astype(int)

    # QWK (with rounded predictions)
    qwk = cohen_kappa_score(
        y_true_int, y_pred_rounded,
        weights='quadratic',
        labels=list(range(y_min, y_max + 1))
    )

    # Spearman with raw predictions
    spearman_raw, _ = spearmanr(y_true, y_pred)

    # Spearman with rounded predictions
    spearman_rounded, _ = spearmanr(y_true_int, y_pred_rounded)

    # MSE
    mse = mean_squared_error(y_true, y_pred)

    return {
        'qwk': float(qwk),
        'spearman_raw': float(spearman_raw) if not np.isnan(spearman_raw) else 0.0,
        'spearman_rounded': float(spearman_rounded) if not np.isnan(spearman_rounded) else 0.0,
        'mse': float(mse),
    }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    criterion = nn.MSELoss()

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        scores = batch['score'].to(device)

        optimizer.zero_grad()
        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions, scores)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model and return predictions."""
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score']

            predictions = model(input_ids, attention_mask)
            all_preds.extend(predictions.cpu().numpy())
            all_true.extend(scores.numpy())

    return np.array(all_preds), np.array(all_true)


def run_experiment(
    prompt_id: int,
    pattern_idx: int,
    n_train: int,
    n_val: int,
    df: pd.DataFrame,
    pattern: Dict,
    output_dir: Path,
    max_epochs: int = 50,
    lr: float = 2e-5,
    batch_size: int = 4,
    model_name: str = "microsoft/deberta-v3-base",
):
    """Run a single experiment configuration."""

    y_min, y_max = ASAP_SCORE_RANGES[prompt_id]

    # Get IDs
    test_ids = pattern['test_ids']
    fewshot_ids = pattern['fewshot_ids']  # 5 samples
    dev_ids = pattern['dev_ids']  # 10 samples

    # Combine fewshot + dev for train/val pool (15 total)
    pool_ids = fewshot_ids + dev_ids
    train_ids = pool_ids[:n_train]
    val_ids = pool_ids[n_train:n_train + n_val]

    # Get data
    train_df = df[df['essay_id'].isin(train_ids)]
    val_df = df[df['essay_id'].isin(val_ids)]
    test_df = df[df['essay_id'].isin(test_ids)]

    # Scale scores to 0-1
    train_essays = train_df['essay'].tolist()
    train_scores = [scale_score(s, y_min, y_max) for s in train_df['domain1_score'].tolist()]

    val_essays = val_df['essay'].tolist()
    val_scores = [scale_score(s, y_min, y_max) for s in val_df['domain1_score'].tolist()]

    test_essays = test_df['essay'].tolist()
    test_scores_raw = test_df['domain1_score'].values  # Keep original for evaluation
    test_scores_scaled = [scale_score(s, y_min, y_max) for s in test_scores_raw]

    logger.info(f"Data: train={len(train_essays)}, val={len(val_essays)}, test={len(test_essays)}")

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DeBERTaRegressor(model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create datasets
    train_dataset = EssayDataset(train_essays, train_scores, tokenizer)
    val_dataset = EssayDataset(val_essays, val_scores, tokenizer)
    test_dataset = EssayDataset(test_essays, test_scores_scaled, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # Training loop
    best_val_mse = float('inf')
    best_epoch = 0
    epoch_results = []

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)

        # Validate
        val_preds_scaled, val_true_scaled = evaluate(model, val_loader, device)
        val_preds = np.array([unscale_score(p, y_min, y_max) for p in val_preds_scaled])
        val_true = np.array([unscale_score(t, y_min, y_max) for t in val_true_scaled])
        val_mse = mean_squared_error(val_true, val_preds)

        epoch_results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_mse': val_mse,
        })

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            # Save best model state
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_mse={val_mse:.4f}")

    logger.info(f"Best epoch: {best_epoch} (val_mse={best_val_mse:.4f})")

    # Load best model and evaluate on test
    model.load_state_dict(best_state)

    # Evaluate epoch 0 (initial model) - need to retrain briefly
    model_e0 = DeBERTaRegressor(model_name).to(device)
    test_preds_e0_scaled, _ = evaluate(model_e0, test_loader, device)
    test_preds_e0 = np.array([unscale_score(p, y_min, y_max) for p in test_preds_e0_scaled])
    metrics_e0 = calculate_metrics(test_scores_raw, test_preds_e0, y_min, y_max)
    del model_e0

    # Evaluate best epoch
    test_preds_scaled, _ = evaluate(model, test_loader, device)
    test_preds = np.array([unscale_score(p, y_min, y_max) for p in test_preds_scaled])
    metrics_best = calculate_metrics(test_scores_raw, test_preds, y_min, y_max)

    # Clean up
    del model
    torch.cuda.empty_cache()

    result = {
        'prompt_id': prompt_id,
        'pattern_idx': pattern_idx,
        'n_train': n_train,
        'n_val': n_val,
        'split_ratio': f"{n_train}:{n_val}",
        'best_epoch': best_epoch,
        'best_val_mse': best_val_mse,
        'epoch_0': metrics_e0,
        'best_epoch_metrics': metrics_best,
        'improvement': {
            'qwk': metrics_best['qwk'] - metrics_e0['qwk'],
            'spearman_raw': metrics_best['spearman_raw'] - metrics_e0['spearman_raw'],
            'spearman_rounded': metrics_best['spearman_rounded'] - metrics_e0['spearman_rounded'],
        },
        'epoch_results': epoch_results,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="DeBERTa baseline experiment")
    parser.add_argument("--prompt", type=int, default=None, help="Specific prompt (1-8)")
    parser.add_argument("--pattern", type=int, default=None, help="Specific pattern (0-9)")
    parser.add_argument("--split", type=str, default=None, help="Specific split ratio (e.g., '5:10')")
    parser.add_argument("--max-epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = data_dir / "results_deberta_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load patterns
    patterns_path = data_dir / "sample_patterns_v2.json"
    logger.info(f"Loading patterns from {patterns_path}")
    with open(patterns_path) as f:
        patterns_data = json.load(f)
    patterns = patterns_data['patterns']

    # Load ASAP data
    asap_path = data_dir / "asap" / "training_set_rel3.tsv"
    logger.info(f"Loading ASAP data from {asap_path}")
    df = pd.read_csv(asap_path, sep='\t', encoding='latin-1')

    # Determine what to run
    prompts = [args.prompt] if args.prompt else list(range(1, 9))
    pattern_indices = [args.pattern] if args.pattern is not None else list(range(10))

    if args.split:
        n_train, n_val = map(int, args.split.split(':'))
        split_configs = [(n_train, n_val)]
    else:
        split_configs = SPLIT_CONFIGS

    all_results = []

    for prompt_id in prompts:
        prompt_df = df[df['essay_set'] == prompt_id].copy()
        prompt_key = f"prompt{prompt_id}"

        for pattern_idx in pattern_indices:
            pattern = patterns[prompt_key][pattern_idx]

            for n_train, n_val in split_configs:
                task_id = f"deberta_prompt{prompt_id}_p{pattern_idx}_t{n_train}v{n_val}"
                result_file = output_dir / f"{task_id}.json"

                # Skip if already done
                if result_file.exists():
                    logger.info(f"Skipping {task_id} (already completed)")
                    with open(result_file) as f:
                        all_results.append(json.load(f))
                    continue

                logger.info(f"\n{'='*60}")
                logger.info(f"Running: {task_id}")
                logger.info(f"Prompt {prompt_id}, Pattern {pattern_idx}, Split {n_train}:{n_val}")
                logger.info(f"{'='*60}")

                try:
                    result = run_experiment(
                        prompt_id=prompt_id,
                        pattern_idx=pattern_idx,
                        n_train=n_train,
                        n_val=n_val,
                        df=prompt_df,
                        pattern=pattern,
                        output_dir=output_dir,
                        max_epochs=args.max_epochs,
                        lr=args.lr,
                    )
                    result['task_id'] = task_id
                    result['completed_at'] = datetime.now().isoformat()

                    # Save individual result
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2)

                    all_results.append(result)

                    logger.info(f"Result: QWK={result['best_epoch_metrics']['qwk']:.4f}, "
                               f"Spearman(raw)={result['best_epoch_metrics']['spearman_raw']:.4f}")

                except Exception as e:
                    logger.exception(f"Failed: {task_id}: {e}")

                # Clear GPU
                torch.cuda.empty_cache()

    # Generate summary
    if all_results:
        summary = generate_summary(all_results)
        summary_file = output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print_summary(summary)


def generate_summary(results: List[Dict]) -> Dict:
    """Generate summary statistics."""
    summary = {
        'n_results': len(results),
        'by_split': {},
        'by_prompt_split': {},
    }

    # Group by split ratio
    split_ratios = sorted(set(r['split_ratio'] for r in results))

    for split in split_ratios:
        split_results = [r for r in results if r['split_ratio'] == split]

        qwks = [r['best_epoch_metrics']['qwk'] for r in split_results]
        spearman_raws = [r['best_epoch_metrics']['spearman_raw'] for r in split_results]
        spearman_rounds = [r['best_epoch_metrics']['spearman_rounded'] for r in split_results]

        summary['by_split'][split] = {
            'qwk_mean': float(np.mean(qwks)),
            'qwk_std': float(np.std(qwks)),
            'spearman_raw_mean': float(np.mean(spearman_raws)),
            'spearman_raw_std': float(np.std(spearman_raws)),
            'spearman_rounded_mean': float(np.mean(spearman_rounds)),
            'spearman_rounded_std': float(np.std(spearman_rounds)),
            'n_tasks': len(split_results),
        }

        # Per-prompt breakdown
        summary['by_prompt_split'][split] = {}
        for prompt_id in range(1, 9):
            prompt_results = [r for r in split_results if r['prompt_id'] == prompt_id]
            if prompt_results:
                qwks = [r['best_epoch_metrics']['qwk'] for r in prompt_results]
                summary['by_prompt_split'][split][f'prompt{prompt_id}'] = {
                    'qwk_mean': float(np.mean(qwks)),
                    'qwk_std': float(np.std(qwks)),
                    'n_patterns': len(prompt_results),
                }

    return summary


def print_summary(summary: Dict):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("DeBERTa Baseline Results Summary")
    print("=" * 80)

    print(f"\n{'Split':<10} {'QWK':>15} {'Spearman(raw)':>18} {'Spearman(round)':>18} {'N':>6}")
    print("-" * 70)

    for split, data in summary['by_split'].items():
        qwk = f"{data['qwk_mean']:.3f}±{data['qwk_std']:.3f}"
        sp_raw = f"{data['spearman_raw_mean']:.3f}±{data['spearman_raw_std']:.3f}"
        sp_round = f"{data['spearman_rounded_mean']:.3f}±{data['spearman_rounded_std']:.3f}"
        print(f"{split:<10} {qwk:>15} {sp_raw:>18} {sp_round:>18} {data['n_tasks']:>6}")


if __name__ == "__main__":
    main()
