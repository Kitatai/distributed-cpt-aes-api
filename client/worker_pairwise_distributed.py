#!/usr/bin/env python3
"""
Distributed pairwise comparison worker for essay scoring experiments.

Connects to the API server to get tasks, runs pairwise comparisons locally,
and uploads results back to the server.

Usage:
    python worker_pairwise_distributed.py --server http://SERVER_IP:8000
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
import gc
import threading
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

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
from config import ASAP_SCORE_RANGES


# Load prompts directly from files
_PROMPTS_BASE_DIR = CLIENT_DIR / "exp" / "llm_prompts"


def _load_prompt_text(prompt_id: int) -> str:
    """Load essay prompt text from file."""
    prompt_file = _PROMPTS_BASE_DIR / "essay_prompts" / f"prompt_{prompt_id}.md"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8").strip()
    return ""


def _load_rubric_text(prompt_id: int) -> str:
    """Load rubric text from file."""
    rubric_file = _PROMPTS_BASE_DIR / "overall" / f"Overall_{prompt_id}.md"
    if rubric_file.exists():
        return rubric_file.read_text(encoding="utf-8").strip()
    return ""


# Model configurations
MODEL_CONFIGS = {
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}


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


class PairwiseComparisonPromptBuilder:
    """Build pairwise comparison prompts for essay scoring."""

    def __init__(
        self,
        prompt_id: int,
        y_min: int,
        y_max: int,
        prompt_text: str,
        rubric_text: str,
    ):
        self.prompt_id = prompt_id
        self.y_min = y_min
        self.y_max = y_max
        self.prompt_text = prompt_text
        self.rubric_text = rubric_text

    def build_system_message(self) -> str:
        """Build the system message."""
        return """You are a strict automated essay comparison engine.
Output ONLY "A" or "B", then a newline.
Do not output any other words, explanations, or punctuation."""

    def build_user_message(self, essay_a: str, essay_b: str) -> str:
        """Build the user message with both essays."""
        return f"""[Task]
Compare two student essays written for the following prompt and decide which one is better according to the scoring rubric.

[Writing Prompt]
{self.prompt_text}

[Scoring Rubric]
{self.rubric_text}

[Essay A]
{essay_a}

[Essay B]
{essay_b}

[Output Format]
The better essay is: <A or B>"""

    def build_assistant_prefill(self) -> str:
        """Build the assistant prefill."""
        return "The better essay is: "

    def to_messages(self, essay_a: str, essay_b: str) -> List[Dict[str, str]]:
        """Convert to chat messages format (without assistant prefill)."""
        return [
            {"role": "system", "content": self.build_system_message()},
            {"role": "user", "content": self.build_user_message(essay_a, essay_b)},
        ]


def create_pairwise_prompt_builder(prompt_id: int) -> PairwiseComparisonPromptBuilder:
    """Create a pairwise comparison prompt builder."""
    y_min, y_max = ASAP_SCORE_RANGES[prompt_id]
    prompt_text = _load_prompt_text(prompt_id)
    rubric_text = _load_rubric_text(prompt_id)

    return PairwiseComparisonPromptBuilder(
        prompt_id=prompt_id,
        y_min=y_min,
        y_max=y_max,
        prompt_text=prompt_text,
        rubric_text=rubric_text,
    )


def extract_token_probabilities(
    model,
    tokenizer,
    prompt_text: str,
    device: str = "cuda",
    debug: bool = False,
) -> Tuple[float, float]:
    """
    Extract token probabilities for "A" and "B" responses.

    Returns:
        Tuple of (P(A), P(B)) where P(A) = P("A") + P(" A"), P(B) = P("B") + P(" B")
    """
    import torch
    import torch.nn.functional as F

    # Tokenize input
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    # Get logits for next token
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]  # Shape: (1, vocab_size)
        probs = F.softmax(logits, dim=-1).squeeze(0)  # Shape: (vocab_size,)

    # Get token IDs for "A", " A", "B", " B"
    tokens_a = ["A", " A"]
    tokens_b = ["B", " B"]

    def get_token_prob(token_str: str) -> float:
        try:
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if len(token_ids) == 1:
                return probs[token_ids[0]].item()
            return 0.0
        except:
            return 0.0

    p_a = sum(get_token_prob(t) for t in tokens_a)
    p_b = sum(get_token_prob(t) for t in tokens_b)

    # Debug: show top tokens
    if debug:
        top_k = 10
        top_probs, top_indices = torch.topk(probs, top_k)
        top_tokens = [tokenizer.decode([idx]) for idx in top_indices.tolist()]
        logger.info(f"Top {top_k} tokens: {list(zip(top_tokens, top_probs.tolist()))}")
        logger.info(f"P(a)={p_a:.4f}, P(b)={p_b:.4f}, sum={p_a+p_b:.4f}")

    return p_a, p_b


def run_pairwise_comparison(
    model,
    tokenizer,
    prompt_builder: PairwiseComparisonPromptBuilder,
    essay_a: str,
    essay_b: str,
    device: str = "cuda",
    debug: bool = False,
) -> Dict:
    """Run a single pairwise comparison."""
    import torch

    # Build messages
    messages = prompt_builder.to_messages(essay_a, essay_b)

    # Apply chat template and add assistant prefill
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_text += prompt_builder.build_assistant_prefill()

    # Get token probabilities
    p_a, p_b = extract_token_probabilities(model, tokenizer, prompt_text, device, debug=debug)

    # Check validity (P(A) + P(B) >= 0.9)
    total_prob = p_a + p_b
    is_valid = total_prob >= 0.9

    # Normalize probabilities
    if total_prob > 0:
        p_a_norm = p_a / total_prob
        p_b_norm = p_b / total_prob
    else:
        p_a_norm = 0.5
        p_b_norm = 0.5

    # Determine winner
    winner = "A" if p_a > p_b else "B"

    return {
        "p_a": p_a,
        "p_b": p_b,
        "p_a_normalized": p_a_norm,
        "p_b_normalized": p_b_norm,
        "total_prob": total_prob,
        "is_valid": is_valid,
        "winner": winner,
    }


def run_pairwise_experiment(
    task_config: dict,
    data_path: Path,
    client: APIClient,
    checkpoint_source: Optional[str] = None,
    log_dir: Optional[Path] = None,
):
    """
    Run pairwise comparison experiment for a single task.

    Args:
        task_config: Task configuration with comparison pairs and epochs
        data_path: Path to ASAP data file
        client: API client for downloading checkpoints
        checkpoint_source: Source directory for checkpoints
        log_dir: Directory to save epoch-level logs (JSON files)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from scipy.stats import spearmanr
    from scipy.optimize import minimize

    task_id = task_config["task_id"]
    prompt_id = task_config["prompt_id"]
    model_name = task_config["model_name"]
    model_short = task_config.get("model_short_name", "llama3b")
    pattern_idx = task_config["pattern_idx"]
    test_ids = task_config["test_ids"]
    epochs = task_config["epochs"]

    # Check for epoch-specific pairs or shared pairs
    epoch_specific_pairs = task_config.get("epoch_specific_pairs", False)
    if epoch_specific_pairs:
        comparison_pairs_by_epoch = task_config["comparison_pairs_by_epoch"]
        # Convert string keys to int if needed
        comparison_pairs_by_epoch = {int(k): v for k, v in comparison_pairs_by_epoch.items()}
        sample_pairs = comparison_pairs_by_epoch[epochs[0]]
    else:
        comparison_pairs = task_config["comparison_pairs"]
        comparison_pairs_by_epoch = None
        sample_pairs = comparison_pairs

    y_min, y_max = ASAP_SCORE_RANGES[prompt_id]

    logger.info(f"=" * 60)
    logger.info(f"Task: {task_id}")
    logger.info(f"Prompt: {prompt_id}, Model: {model_name}")
    logger.info(f"Pattern: {pattern_idx}, Epochs: {epochs}")
    if epoch_specific_pairs:
        logger.info(f"Test essays: {len(test_ids)}, Pairs/epoch: {len(sample_pairs)} (epoch-specific)")
    else:
        logger.info(f"Test essays: {len(test_ids)}, Comparison pairs: {len(sample_pairs)}")
    logger.info(f"=" * 60)

    # Load ASAP data
    logger.info(f"Loading ASAP data from {data_path}")
    df = pd.read_csv(data_path, sep='\t', encoding='latin-1')
    prompt_df = df[df['essay_set'] == prompt_id].copy()
    logger.info(f"Loaded {len(prompt_df)} essays for prompt {prompt_id}")

    # Get test essays
    test_df = prompt_df[prompt_df['essay_id'].isin(test_ids)]
    logger.info(f"Test essays: {len(test_df)}")

    # Create essay lookup
    essay_lookup = {int(row['essay_id']): row['essay'] for _, row in test_df.iterrows()}
    true_scores = {int(row['essay_id']): int(row['domain1_score']) for _, row in test_df.iterrows()}

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

    # Create prompt builder
    prompt_builder = create_pairwise_prompt_builder(prompt_id)

    # Load base model
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    def download_and_load_adapter(epoch: int):
        """Download checkpoint from server and load as LoRA adapter."""
        if epoch == 0:
            return base_model, True

        zeroshot_task_id = f"prompt{prompt_id}_{model_short}"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            success = client.download_checkpoint(zeroshot_task_id, epoch, tmp_path, source=checkpoint_source)
            if not success:
                logger.warning(f"Checkpoint not found: {zeroshot_task_id}/epoch_{epoch}")
                return base_model, False

            model_with_lora = PeftModel.from_pretrained(base_model, str(tmp_path), is_trainable=False)
            return model_with_lora, True

    # Bradley-Terry model fitting
    def fit_bradley_terry(comparisons: List[Tuple[int, int, float]], items: List[int]) -> Tuple[Dict[int, float], Dict]:
        """
        Fit Bradley-Terry model and return scores.

        Returns:
            Tuple of (scores dict, diagnostics dict)
        """
        n = len(items)
        item_to_idx = {item: idx for idx, item in enumerate(items)}
        idx_to_item = {idx: item for item, idx in item_to_idx.items()}

        # Build comparison matrix
        W = np.zeros((n, n))
        for winner, loser, weight in comparisons:
            i = item_to_idx.get(winner)
            j = item_to_idx.get(loser)
            if i is not None and j is not None:
                W[i, j] += weight

        # Log W matrix statistics
        w_nonzero = W[W > 0]
        logger.info(f"BT comparison matrix: shape={W.shape}, non-zero={len(w_nonzero)}")
        logger.info(f"  W values: mean={w_nonzero.mean():.4f}, min={w_nonzero.min():.4f}, max={w_nonzero.max():.4f}")

        # Check connectivity (each item should have comparisons)
        items_with_comps = np.sum(W > 0, axis=1) + np.sum(W > 0, axis=0)
        logger.info(f"  Comparisons per item: mean={items_with_comps.mean():.1f}, min={items_with_comps.min()}, max={items_with_comps.max()}")

        def negative_log_likelihood(strengths):
            # Fix first item to 0 for identifiability (instead of mean centering)
            s = strengths.copy()
            s[0] = 0.0
            nll = 0.0
            for i in range(n):
                for j in range(n):
                    if W[i, j] > 0:
                        # log P(i > j) = s_i - log(exp(s_i) + exp(s_j))
                        log_prob = s[i] - np.logaddexp(s[i], s[j])
                        nll -= W[i, j] * log_prob
            # Add small regularization to prevent extreme values
            nll += 0.001 * np.sum(s ** 2)
            return nll

        def gradient(strengths):
            s = strengths.copy()
            s[0] = 0.0
            grad = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        # p_ij = P(i > j) = exp(s_i) / (exp(s_i) + exp(s_j))
                        diff = s[i] - s[j]
                        if diff > 100:
                            p_ij = 1.0
                        elif diff < -100:
                            p_ij = 0.0
                        else:
                            p_ij = 1.0 / (1.0 + np.exp(-diff))
                        # d/ds_i of -W[i,j] * log(p_ij) = -W[i,j] * (1 - p_ij)
                        grad[i] -= W[i, j] * (1.0 - p_ij)
                        # d/ds_i of -W[j,i] * log(p_ji) = W[j,i] * p_ij
                        grad[i] += W[j, i] * p_ij
            # Regularization gradient
            grad += 0.002 * s
            # First item is fixed
            grad[0] = 0.0
            return grad

        # Optimize from zeros (BT likelihood is concave, so unique solution)
        result = minimize(
            negative_log_likelihood,
            np.zeros(n),
            method='L-BFGS-B',
            jac=gradient,
            options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        logger.info(f"BT optimization: success={result.success}, nit={result.nit}, nll={result.fun:.4f}")
        if not result.success:
            logger.warning(f"BT optimization warning: {result.message}")

        # Normalize scores (mean=0)
        strengths = result.x - result.x.mean()

        # Log score distribution
        logger.info(f"BT scores: mean={strengths.mean():.4f}, std={strengths.std():.4f}, min={strengths.min():.4f}, max={strengths.max():.4f}")

        scores = {item: strengths[idx] for item, idx in item_to_idx.items()}

        diagnostics = {
            "success": result.success,
            "message": str(result.message),
            "nit": result.nit,
            "nll": float(result.fun),
            "score_std": float(strengths.std()),
            "score_range": float(strengths.max() - strengths.min()),
        }

        return scores, diagnostics

    # Run experiment for each epoch
    results = {}
    for epoch in epochs:
        if _state.shutdown_requested:
            logger.info("Shutdown requested, stopping early")
            break

        logger.info(f"\n{'='*40}")
        logger.info(f"Epoch {epoch}")
        logger.info(f"{'='*40}")

        # Load adapter
        model, success = download_and_load_adapter(epoch)
        if not success:
            logger.warning(f"Skipping epoch {epoch} (adapter not found)")
            continue

        model.eval()

        # Get comparison pairs for this epoch
        if comparison_pairs_by_epoch is not None:
            current_pairs = comparison_pairs_by_epoch[epoch]
        else:
            current_pairs = comparison_pairs

        # Run bidirectional comparisons
        comparison_results = []
        n_agree_a = 0  # Both directions say A wins
        n_agree_b = 0  # Both directions say B wins
        n_disagree = 0  # Directions disagree (draw)

        for idx, (essay_id_a, essay_id_b) in enumerate(tqdm(current_pairs, desc=f"E{epoch} comparisons", ncols=80)):
            essay_a = essay_lookup.get(essay_id_a)
            essay_b = essay_lookup.get(essay_id_b)

            if essay_a is None or essay_b is None:
                continue

            # Debug first comparison of each epoch
            debug = (idx == 0)

            # Forward: A vs B (A is first)
            result_forward = run_pairwise_comparison(
                model, tokenizer, prompt_builder,
                essay_a, essay_b, device="cuda", debug=debug
            )

            # Backward: B vs A (B is first)
            result_backward = run_pairwise_comparison(
                model, tokenizer, prompt_builder,
                essay_b, essay_a, device="cuda", debug=False
            )

            # Determine winner from each direction
            # Forward: if p_a > p_b, A wins
            # Backward: if p_a > p_b (meaning B wins in original terms)
            forward_winner = "A" if result_forward["p_a"] > result_forward["p_b"] else "B"
            backward_winner = "B" if result_backward["p_a"] > result_backward["p_b"] else "A"

            # Check agreement
            if forward_winner == backward_winner:
                agreed_winner = forward_winner
                if agreed_winner == "A":
                    n_agree_a += 1
                else:
                    n_agree_b += 1
            else:
                agreed_winner = "draw"
                n_disagree += 1

            result = {
                "essay_id_a": essay_id_a,
                "essay_id_b": essay_id_b,
                "forward_p_a": result_forward["p_a"],
                "forward_p_b": result_forward["p_b"],
                "backward_p_a": result_backward["p_a"],
                "backward_p_b": result_backward["p_b"],
                "forward_winner": forward_winner,
                "backward_winner": backward_winner,
                "agreed_winner": agreed_winner,
            }
            comparison_results.append(result)

        # Log statistics
        n_total = len(comparison_results)
        logger.info(f"Bidirectional comparison results:")
        logger.info(f"  A wins (agreed): {n_agree_a} ({100*n_agree_a/n_total:.1f}%)")
        logger.info(f"  B wins (agreed): {n_agree_b} ({100*n_agree_b/n_total:.1f}%)")
        logger.info(f"  Draw (disagreed): {n_disagree} ({100*n_disagree/n_total:.1f}%)")

        # Convert to Bradley-Terry format using averaged probabilities
        bt_comparisons = []
        for result in comparison_results:
            essay_id_a = result["essay_id_a"]
            essay_id_b = result["essay_id_b"]

            # Forward: P(A>B) = p_a / (p_a + p_b)
            fwd_total = result["forward_p_a"] + result["forward_p_b"]
            p_a_wins_forward = result["forward_p_a"] / fwd_total if fwd_total > 0 else 0.5

            # Backward: P(A>B) = p_b / (p_a + p_b) (since B is first, "B" means A wins)
            bwd_total = result["backward_p_a"] + result["backward_p_b"]
            p_a_wins_backward = result["backward_p_b"] / bwd_total if bwd_total > 0 else 0.5

            # Average probability that A > B
            p_a_wins = (p_a_wins_forward + p_a_wins_backward) / 2

            result["p_a_wins_avg"] = p_a_wins
            result["p_a_wins_forward"] = p_a_wins_forward
            result["p_a_wins_backward"] = p_a_wins_backward

            # Debug first 3 comparisons
            if len(comparison_results) <= 3:
                logger.info(f"  Pair {len(comparison_results)}: essay_{essay_id_a} vs essay_{essay_id_b}")
                logger.info(f"    Forward:  p_a={result['forward_p_a']:.4f}, p_b={result['forward_p_b']:.4f} -> P(A>B)={p_a_wins_forward:.4f}")
                logger.info(f"    Backward: p_a={result['backward_p_a']:.4f}, p_b={result['backward_p_b']:.4f} -> P(A>B)={p_a_wins_backward:.4f}")
                logger.info(f"    Average P(A>B)={p_a_wins:.4f}")

            # Add to BT: both directions with their probabilities
            # This is NOT redundant - BT needs both W[a,b] and W[b,a] for the likelihood
            bt_comparisons.append((essay_id_a, essay_id_b, p_a_wins))      # A beats B
            bt_comparisons.append((essay_id_b, essay_id_a, 1 - p_a_wins))  # B beats A

        # Log detailed probability statistics
        p_a_wins_list = [r["p_a_wins_avg"] for r in comparison_results]
        fwd_totals = [r["forward_p_a"] + r["forward_p_b"] for r in comparison_results]
        bwd_totals = [r["backward_p_a"] + r["backward_p_b"] for r in comparison_results]

        logger.info(f"Probability statistics:")
        logger.info(f"  Forward sum (p_a+p_b):  mean={np.mean(fwd_totals):.4f}, min={np.min(fwd_totals):.4f}, max={np.max(fwd_totals):.4f}")
        logger.info(f"  Backward sum (p_a+p_b): mean={np.mean(bwd_totals):.4f}, min={np.min(bwd_totals):.4f}, max={np.max(bwd_totals):.4f}")
        logger.info(f"  P(A>B) after debiasing: mean={np.mean(p_a_wins_list):.4f}, std={np.std(p_a_wins_list):.4f}")
        logger.info(f"  P(A>B) range: min={np.min(p_a_wins_list):.4f}, max={np.max(p_a_wins_list):.4f}")

        # Count how many are far from 0.5
        n_decisive = sum(1 for p in p_a_wins_list if abs(p - 0.5) > 0.1)
        logger.info(f"  Decisive comparisons (|P-0.5|>0.1): {n_decisive}/{len(p_a_wins_list)} ({100*n_decisive/len(p_a_wins_list):.1f}%)")

        # Fit Bradley-Terry and get scores
        estimated_scores, bt_diagnostics = fit_bradley_terry(bt_comparisons, test_ids)

        # Calculate Spearman correlation
        common_ids = set(estimated_scores.keys()) & set(true_scores.keys())
        est_list = [estimated_scores[eid] for eid in common_ids]
        true_list = [true_scores[eid] for eid in common_ids]
        spearman_corr, p_value = spearmanr(est_list, true_list)

        logger.info(f"Spearman correlation: {spearman_corr:.4f}")

        epoch_result = {
            "epoch": epoch,
            "spearman": spearman_corr,
            "spearman_pvalue": p_value,
            "n_essays": len(common_ids),
            "n_comparisons": len(comparison_results),
            "n_agree_a": n_agree_a,
            "n_agree_b": n_agree_b,
            "n_disagree": n_disagree,
            "comparison_results": comparison_results,
            "estimated_scores": {str(k): v for k, v in estimated_scores.items()},
            "true_scores": {str(k): v for k, v in true_scores.items()},
            "bt_comparisons": [(a, b, float(w)) for a, b, w in bt_comparisons],
            "bt_diagnostics": bt_diagnostics,
        }

        results[epoch] = epoch_result

        # Save epoch log to file
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            epoch_log_file = log_dir / f"{task_id}_epoch{epoch}.json"
            with open(epoch_log_file, "w") as f:
                json.dump(epoch_result, f, indent=2)
            logger.info(f"Epoch log saved to {epoch_log_file}")

        # Unload adapter
        if epoch > 0:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    result = {
        "task_id": task_id,
        "prompt_id": prompt_id,
        "model_short": model_short,
        "pattern_idx": pattern_idx,
        "epochs": epochs,
        "epoch_specific_pairs": epoch_specific_pairs,
        "results_by_epoch": results,
        "true_scores": {str(k): v for k, v in true_scores.items()},
    }

    if epoch_specific_pairs:
        result["comparison_pairs_by_epoch"] = comparison_pairs_by_epoch
    else:
        result["comparison_pairs"] = comparison_pairs

    return result


def main():
    parser = argparse.ArgumentParser(description="Distributed pairwise comparison worker")
    parser.add_argument("--server", type=str, default="http://localhost:8000",
                       help="API server URL")
    parser.add_argument("--exp-name", type=str, default="pairwise",
                       help="Experiment name for API endpoints")
    parser.add_argument("--data-path", type=str, default=None,
                       help="Path to ASAP data file")
    parser.add_argument("--checkpoint-source", type=str, default="backup_zeroshot_v3",
                       help="Source directory for checkpoints")
    parser.add_argument("--worker-id", type=str, default=None,
                       help="Worker ID (default: hostname)")
    parser.add_argument("--max-tasks", type=int, default=0,
                       help="Maximum number of tasks to process (0=unlimited)")
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Directory to save epoch logs (default: logs/pairwise)")

    args = parser.parse_args()

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Default data path
    if args.data_path is None:
        data_path = CLIENT_DIR / "data" / "training_set_rel3.tsv"
    else:
        data_path = Path(args.data_path)

    # Worker ID
    worker_id = args.worker_id or socket.gethostname()

    # Log directory
    if args.log_dir is None:
        log_dir = CLIENT_DIR / "logs" / "pairwise"
    else:
        log_dir = Path(args.log_dir)

    logger.info(f"=" * 60)
    logger.info(f"Pairwise Comparison Worker")
    logger.info(f"Server: {args.server}")
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Worker ID: {worker_id}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Checkpoint source: {args.checkpoint_source}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"=" * 60)

    # Create API client
    client = APIClient(args.server, worker_id)
    _state.client = client

    tasks_completed = 0

    while True:
        if _state.shutdown_requested:
            logger.info("Shutdown requested, exiting")
            break

        if args.max_tasks > 0 and tasks_completed >= args.max_tasks:
            logger.info(f"Completed {tasks_completed} tasks, exiting")
            break

        # Get next task
        task = client.get_task(args.exp_name, worker_id)

        if task is None:
            logger.info("No tasks available, waiting 30 seconds...")
            time.sleep(30)
            continue

        task_id = task.get("task_id")
        _state.current_task_id = task_id

        logger.info(f"Got task: {task_id}")

        try:
            # Run experiment
            results = run_pairwise_experiment(
                task_config=task,
                data_path=data_path,
                client=client,
                checkpoint_source=args.checkpoint_source,
                log_dir=log_dir,
            )

            # Upload results
            success = client.upload_result(args.exp_name, task_id, results)
            if success:
                logger.info(f"Results uploaded for {task_id}")
                tasks_completed += 1
            else:
                logger.error(f"Failed to upload results for {task_id}")

        except Exception as e:
            logger.exception(f"Error processing task {task_id}: {e}")
            # Mark task as failed
            try:
                client.fail_task(args.exp_name, task_id, str(e))
            except:
                pass

        _state.current_task_id = None

    logger.info(f"Worker finished. Completed {tasks_completed} tasks.")


if __name__ == "__main__":
    main()
