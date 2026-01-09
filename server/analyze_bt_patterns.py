#!/usr/bin/env python3
"""
Analyze Bradley-Terry patterns for epoch selection.

Pattern 1: Select epoch with minimum deviance
Pattern 2: Rater-specific discrimination BT model (all epochs, all comparisons)
Pattern 3: Rater-specific discrimination BT model (split comparisons across epochs)
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


def load_epoch_logs(log_dir: Path, task_id: str) -> Dict[int, Dict]:
    """Load all epoch logs for a task."""
    import re
    epoch_data = {}
    # Use regex to match exactly: {task_id}_epoch{number}.json
    pattern = re.compile(rf"^{re.escape(task_id)}_epoch(\d+)\.json$")

    for log_file in sorted(log_dir.iterdir()):
        match = pattern.match(log_file.name)
        if match:
            with open(log_file) as f:
                data = json.load(f)
                epoch_data[data["epoch"]] = data
    return epoch_data


def compute_deviance(bt_comparisons: List, estimated_scores: Dict) -> float:
    """Compute BT deviance from comparisons and scores."""
    deviance = 0.0
    for winner, loser, weight in bt_comparisons:
        s_i = estimated_scores.get(str(winner), 0.0)
        s_j = estimated_scores.get(str(loser), 0.0)
        diff = s_i - s_j
        if diff > 100:
            log_p = 0.0
        elif diff < -100:
            log_p = diff
        else:
            log_p = diff - np.log(1 + np.exp(diff))
        deviance -= 2 * weight * log_p
    return deviance


def pattern1_min_deviance(epoch_data: Dict[int, Dict]) -> Tuple[int, Dict, float, float]:
    """Pattern 1: Select epoch with minimum deviance."""
    best_epoch, best_deviance, best_scores = None, float('inf'), None

    for epoch, data in epoch_data.items():
        bt_comparisons = data.get("bt_comparisons", [])
        estimated_scores = data.get("estimated_scores", {})
        if not bt_comparisons or not estimated_scores:
            continue
        deviance = compute_deviance(bt_comparisons, estimated_scores)
        if deviance < best_deviance:
            best_deviance, best_epoch, best_scores = deviance, epoch, estimated_scores

    true_scores = epoch_data[best_epoch].get("true_scores", {})
    common_ids = set(best_scores.keys()) & set(true_scores.keys())
    est_list = [best_scores[eid] for eid in common_ids]
    true_list = [true_scores[eid] for eid in common_ids]
    spearman, _ = spearmanr(est_list, true_list)

    return best_epoch, best_scores, spearman, best_deviance


def fit_simple_bt(bt_comparisons: List) -> Dict:
    """Fit simple BT model (no rater discrimination) and return scores. Vectorized."""
    items = set()
    for winner, loser, _ in bt_comparisons:
        items.add(winner)
        items.add(loser)
    items = sorted(items)
    n_items = len(items)
    item_to_idx = {item: i for i, item in enumerate(items)}

    # Pre-convert to arrays for speed
    winners = np.array([item_to_idx[w] for w, _, _ in bt_comparisons])
    losers = np.array([item_to_idx[l] for _, l, _ in bt_comparisons])
    weights = np.array([wt for _, _, wt in bt_comparisons])

    def nll(params):
        s = params.copy()
        s[0] = 0.0
        diff = np.clip(s[winners] - s[losers], -30, 30)
        p = 1 / (1 + np.exp(-diff))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        loss = -np.sum(weights * np.log(p) + (1 - weights) * np.log(1 - p))
        return loss + 0.001 * np.sum(s ** 2)

    x0 = np.zeros(n_items)
    result = minimize(nll, x0, method='L-BFGS-B', options={'maxiter': 500})
    return {items[i]: result.x[i] for i in range(n_items)}


def pattern1c_min_deviance_scaled(epoch_data: Dict[int, Dict]) -> Tuple[int, Dict, float, float]:
    """Pattern 1c: Select epoch with minimum deviance using scale-normalized weights."""
    best_epoch, best_deviance, best_scores = None, float('inf'), None

    for epoch, data in epoch_data.items():
        bt_comparisons = data.get("bt_comparisons", [])
        if not bt_comparisons:
            continue

        # Scale normalize the weights
        weights = [w for _, _, w in bt_comparisons]
        norm_weights = scale_normalize_probabilities(weights)

        # Create normalized comparisons
        norm_comparisons = [(w, l, nw) for (w, l, _), nw in zip(bt_comparisons, norm_weights)]

        # Fit BT with normalized weights
        estimated_scores = fit_simple_bt(norm_comparisons)

        # Compute deviance with normalized weights
        deviance = compute_deviance(norm_comparisons, {str(k): v for k, v in estimated_scores.items()})

        if deviance < best_deviance:
            best_deviance, best_epoch, best_scores = deviance, epoch, estimated_scores

    true_scores = epoch_data[best_epoch].get("true_scores", {})
    common_ids = set(str(k) for k in best_scores.keys()) & set(true_scores.keys())
    est_list = [best_scores[int(eid)] for eid in common_ids]
    true_list = [true_scores[eid] for eid in common_ids]
    spearman, _ = spearmanr(est_list, true_list)

    return best_epoch, best_scores, spearman, best_deviance


def fit_rater_specific_bt(
    all_comparisons: List[Tuple],
    items: List,
    epochs: List,
) -> Tuple[Dict, Dict, object]:
    """
    Fit rater-specific discrimination BT model.
    P(i > j | rater r) = exp(alpha_r * s_i) / (exp(alpha_r * s_i) + exp(alpha_r * s_j))

    Identifiability: Score variance is fixed to 1 after optimization.
    Alpha is free to vary (no regularization on alpha).
    """
    n_items, n_raters = len(items), len(epochs)
    item_to_idx = {item: idx for idx, item in enumerate(items)}
    epoch_to_idx = {epoch: idx for idx, epoch in enumerate(epochs)}
    n_params = n_items + n_raters

    def negative_log_likelihood(params):
        s = params[:n_items].copy()
        s[0] = 0.0
        log_alpha = params[n_items:]
        alpha = np.exp(np.clip(log_alpha, -5, 5))

        nll = 0.0
        for rater, winner, loser, weight in all_comparisons:
            i, j, r = item_to_idx.get(winner), item_to_idx.get(loser), epoch_to_idx.get(rater)
            if i is None or j is None or r is None:
                continue
            diff = alpha[r] * (s[i] - s[j])
            if diff > 100:
                log_p = 0.0
            elif diff < -100:
                log_p = diff
            else:
                log_p = diff - np.log(1 + np.exp(diff))
            nll -= weight * log_p

        # Only regularize scores lightly for numerical stability (no alpha regularization)
        nll += 0.0001 * np.sum(s ** 2)
        return nll

    def gradient(params):
        s = params[:n_items].copy()
        s[0] = 0.0
        log_alpha = params[n_items:]
        alpha = np.exp(np.clip(log_alpha, -5, 5))

        grad_s = np.zeros(n_items)
        grad_log_alpha = np.zeros(n_raters)

        for rater, winner, loser, weight in all_comparisons:
            i, j, r = item_to_idx.get(winner), item_to_idx.get(loser), epoch_to_idx.get(rater)
            if i is None or j is None or r is None:
                continue
            diff = alpha[r] * (s[i] - s[j])
            if diff > 100:
                p = 1.0
            elif diff < -100:
                p = 0.0
            else:
                p = 1.0 / (1.0 + np.exp(-diff))

            grad_s[i] -= weight * alpha[r] * (1.0 - p)
            grad_s[j] += weight * alpha[r] * (1.0 - p)
            grad_log_alpha[r] -= weight * (s[i] - s[j]) * (1.0 - p) * alpha[r]

        grad_s += 0.0002 * s
        grad_s[0] = 0.0
        return np.concatenate([grad_s, grad_log_alpha])

    x0 = np.zeros(n_params)
    result = minimize(
        negative_log_likelihood, x0,
        method='L-BFGS-B', jac=gradient,
        options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-12}
    )
    s = result.x[:n_items]
    log_alpha = result.x[n_items:]
    alpha = np.exp(log_alpha)

    # Normalize scores to have variance = 1, adjust alpha accordingly
    s = s - s.mean()
    s_std = s.std()
    if s_std > 0:
        s = s / s_std
        alpha = alpha * s_std  # Compensate alpha for score scaling
    log_alpha = np.log(alpha)
    alpha = np.exp(log_alpha)

    estimated_scores = {items[i]: s[i] for i in range(n_items)}
    alpha_by_epoch = {epochs[r]: alpha[r] for r in range(n_raters)}

    return estimated_scores, alpha_by_epoch, result


def rank_normalize_probabilities(probabilities: List[float]) -> List[float]:
    """
    Transform probabilities to match BT theoretical distribution.

    BT model assumes: P(A>B) = sigmoid(s_A - s_B) where s_A, s_B ~ N(0,1)
    So s_A - s_B ~ N(0, sqrt(2)), meaning z-scores should have std = sqrt(2).

    Steps:
    1. Convert probabilities to ranks
    2. Map ranks to normal quantiles (z-scores with std=1)
    3. Scale z-scores by sqrt(2) to match BT theoretical distribution
    4. Convert back to probabilities via sigmoid
    """
    from scipy.stats import norm

    n = len(probabilities)
    if n == 0:
        return []

    # Get ranks (1 to n)
    ranks = np.argsort(np.argsort(probabilities)) + 1

    # Map ranks to normal quantiles: rank -> (rank - 0.5) / n -> Φ^(-1)
    quantiles = [(r - 0.5) / n for r in ranks]
    normalized = [norm.ppf(q) for q in quantiles]

    # Scale by sqrt(2) to match BT theoretical distribution
    scale = np.sqrt(2)
    normalized_probs = [1 / (1 + np.exp(-z * scale)) for z in normalized]

    return normalized_probs


def scale_normalize_probabilities(probabilities: List[float], target_std: float = 1.0) -> List[float]:
    """
    Scale probabilities while preserving 0.5 as center point.

    Method:
    1. Convert to logit space: logit(p) = log(p/(1-p))
       - 0.5 maps to 0
       - Order is preserved
    2. Scale logits to have target standard deviation
    3. Convert back via sigmoid

    This preserves:
    - 0.5 stays at 0.5
    - Order is preserved (no flipping around 0.5)
    - Variance is normalized across raters
    """
    n = len(probabilities)
    if n == 0:
        return []

    # Clip to avoid log(0) or log(inf)
    probs = np.clip(probabilities, 1e-6, 1 - 1e-6)

    # Convert to logit space
    logits = np.log(probs / (1 - probs))

    # Scale to target std (keeping mean at 0 which corresponds to p=0.5)
    current_std = np.std(logits)
    if current_std > 0:
        scaled_logits = logits * (target_std / current_std)
    else:
        scaled_logits = logits

    # Convert back to probability
    scaled_probs = 1 / (1 + np.exp(-scaled_logits))

    return scaled_probs.tolist()


def pattern2_rater_specific_bt(epoch_data: Dict[int, Dict], normalize_mode: str = None) -> Tuple[Dict, Dict, float]:
    """
    Pattern 2: Rater-specific BT with ALL comparisons from ALL epochs.

    normalize_mode:
        None: No normalization (original P2)
        "rank": Rank-based normalization (P2b) - may flip values around 0.5
        "scale": Scale normalization (P2c) - preserves 0.5, adjusts variance
    """
    all_comparisons, all_items = [], set()
    epochs = sorted(epoch_data.keys())

    for epoch in epochs:
        bt_comparisons = epoch_data[epoch].get("bt_comparisons", [])

        if normalize_mode == "rank" and bt_comparisons:
            weights = [w for _, _, w in bt_comparisons]
            norm_weights = rank_normalize_probabilities(weights)
            for (winner, loser, _), norm_w in zip(bt_comparisons, norm_weights):
                all_comparisons.append((epoch, winner, loser, norm_w))
                all_items.update([winner, loser])
        elif normalize_mode == "scale" and bt_comparisons:
            weights = [w for _, _, w in bt_comparisons]
            norm_weights = scale_normalize_probabilities(weights)
            for (winner, loser, _), norm_w in zip(bt_comparisons, norm_weights):
                all_comparisons.append((epoch, winner, loser, norm_w))
                all_items.update([winner, loser])
        else:
            for winner, loser, weight in bt_comparisons:
                all_comparisons.append((epoch, winner, loser, weight))
                all_items.update([winner, loser])

    items = sorted(all_items)
    print(f"Pattern 2: {len(items)} items, {len(epochs)} raters, {len(all_comparisons)} comparisons")

    estimated_scores, alpha_by_epoch, result = fit_rater_specific_bt(all_comparisons, items, epochs)
    print(f"Optimization: success={result.success}, nit={result.nit}, nll={result.fun:.4f}")

    true_scores = epoch_data[epochs[0]].get("true_scores", {})
    common_ids = set(str(k) for k in estimated_scores.keys()) & set(true_scores.keys())
    est_list = [estimated_scores[int(eid)] for eid in common_ids]
    true_list = [true_scores[eid] for eid in common_ids]
    spearman, _ = spearmanr(est_list, true_list)

    return estimated_scores, alpha_by_epoch, spearman


def pattern3_split_comparisons_bt(epoch_data: Dict[int, Dict], seed: int = 42) -> Tuple[Dict, Dict, float]:
    """Pattern 3: Rater-specific BT with SPLIT comparisons (each comparison used once)."""
    import random
    random.seed(seed)

    epochs = sorted(epoch_data.keys())
    n_epochs = len(epochs)

    # Get unique pairs from first epoch
    bt_first = epoch_data[epochs[0]].get("bt_comparisons", [])
    pair_to_comps = {}
    for winner, loser, weight in bt_first:
        pair_key = tuple(sorted([winner, loser]))
        if pair_key not in pair_to_comps:
            pair_to_comps[pair_key] = []
        pair_to_comps[pair_key].append((winner, loser, weight))

    unique_pairs = list(pair_to_comps.keys())
    random.shuffle(unique_pairs)

    # Assign each pair to one epoch (round-robin)
    pair_to_epoch = {pair: epochs[i % n_epochs] for i, pair in enumerate(unique_pairs)}

    # Collect comparisons only from assigned epoch
    all_comparisons, all_items = [], set()
    for epoch in epochs:
        for winner, loser, weight in epoch_data[epoch].get("bt_comparisons", []):
            pair_key = tuple(sorted([winner, loser]))
            if pair_to_epoch.get(pair_key) == epoch:
                all_comparisons.append((epoch, winner, loser, weight))
                all_items.update([winner, loser])

    items = sorted(all_items)
    print(f"Pattern 3: {len(items)} items, {n_epochs} raters, {len(all_comparisons)} comparisons (split)")

    estimated_scores, alpha_by_epoch, result = fit_rater_specific_bt(all_comparisons, items, epochs)
    print(f"Optimization: success={result.success}, nit={result.nit}, nll={result.fun:.4f}")

    true_scores = epoch_data[epochs[0]].get("true_scores", {})
    common_ids = set(str(k) for k in estimated_scores.keys()) & set(true_scores.keys())
    est_list = [estimated_scores[int(eid)] for eid in common_ids]
    true_list = [true_scores[eid] for eid in common_ids]
    spearman, _ = spearmanr(est_list, true_list)

    return estimated_scores, alpha_by_epoch, spearman


def plot_results(epoch_data, p1_epoch, p1_spearman, p2_spearman, p2b_spearman, p3_spearman,
                 alpha_p2, alpha_p2b, alpha_p3, output_path, task_id):
    """Plot comparison of patterns."""
    epochs = sorted(epoch_data.keys())
    spearman_by_epoch = [epoch_data[e].get("spearman", np.nan) for e in epochs]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Spearman comparison
    ax1 = axes[0]
    ax1.plot(epochs, spearman_by_epoch, 'o-', color='blue', lw=2, ms=6, label='Per-epoch')
    ax1.axhline(p1_spearman, color='green', ls='--', lw=2, label=f'P1 (E{p1_epoch}): {p1_spearman:.4f}')
    ax1.axhline(p2_spearman, color='red', ls='--', lw=2, label=f'P2: {p2_spearman:.4f}')
    ax1.axhline(p2b_spearman, color='orange', ls='--', lw=2, label=f'P2b (RankNorm): {p2b_spearman:.4f}')
    ax1.axvline(p1_epoch, color='green', ls=':', alpha=0.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Spearman Correlation')
    ax1.set_title(f'Pattern Comparison: {task_id}')
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Discrimination parameters (P2 vs P2b)
    ax2 = axes[1]
    alpha_p2_vals = [alpha_p2[e] for e in epochs]
    alpha_p2b_vals = [alpha_p2b[e] for e in epochs]
    x = np.arange(len(epochs))
    w = 0.35
    ax2.bar(x - w/2, alpha_p2_vals, w, label='P2 (Original)', color='red', alpha=0.7)
    ax2.bar(x + w/2, alpha_p2b_vals, w, label='P2b (RankNorm)', color='orange', alpha=0.7)
    ax2.axhline(1.0, color='black', ls='--', lw=1)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Discrimination (α)')
    ax2.set_xticks(x[::5])
    ax2.set_xticklabels(epochs[::5])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    epoch_data = load_epoch_logs(log_dir, args.task_id)
    if not epoch_data:
        print(f"Error: No epoch logs found")
        return

    print(f"Loaded {len(epoch_data)} epochs")

    # Pattern 1
    print("\n" + "=" * 60)
    print("Pattern 1: Minimum Deviance")
    print("=" * 60)
    p1_epoch, _, p1_spearman, p1_deviance = pattern1_min_deviance(epoch_data)
    print(f"Best epoch: {p1_epoch}, Deviance: {p1_deviance:.2f}, Spearman: {p1_spearman:.4f}")

    # Pattern 1c: Scale normalized deviance
    print("\n" + "=" * 60)
    print("Pattern 1c: Minimum Deviance (Scale Normalized)")
    print("=" * 60)
    p1c_epoch, _, p1c_spearman, p1c_deviance = pattern1c_min_deviance_scaled(epoch_data)
    print(f"Best epoch: {p1c_epoch}, Deviance: {p1c_deviance:.2f}, Spearman: {p1c_spearman:.4f}")

    # Pattern 2
    print("\n" + "=" * 60)
    print("Pattern 2: Rater-Specific BT (All Comparisons)")
    print("=" * 60)
    _, alpha_p2, p2_spearman = pattern2_rater_specific_bt(epoch_data, normalize_mode=None)
    print(f"Spearman: {p2_spearman:.4f}")

    # Pattern 2b: With rank normalization
    print("\n" + "=" * 60)
    print("Pattern 2b: Rater-Specific BT (Rank Normalized)")
    print("=" * 60)
    _, alpha_p2b, p2b_spearman = pattern2_rater_specific_bt(epoch_data, normalize_mode="rank")
    print(f"Spearman: {p2b_spearman:.4f}")

    # Pattern 2c: With scale normalization (preserves 0.5)
    print("\n" + "=" * 60)
    print("Pattern 2c: Rater-Specific BT (Scale Normalized)")
    print("=" * 60)
    _, alpha_p2c, p2c_spearman = pattern2_rater_specific_bt(epoch_data, normalize_mode="scale")
    print(f"Spearman: {p2c_spearman:.4f}")

    # Pattern 3
    print("\n" + "=" * 60)
    print("Pattern 3: Rater-Specific BT (Split Comparisons)")
    print("=" * 60)
    _, alpha_p3, p3_spearman = pattern3_split_comparisons_bt(epoch_data)
    print(f"Spearman: {p3_spearman:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    best_epoch = max(epoch_data.keys(), key=lambda e: epoch_data[e].get("spearman", 0))
    best_spearman = epoch_data[best_epoch].get("spearman", 0)
    print(f"{'Pattern':<35} {'Spearman':<12}")
    print("-" * 50)
    print(f"{'Pattern 1 (Min Dev E' + str(p1_epoch) + ')':<35} {p1_spearman:<12.4f}")
    print(f"{'Pattern 1c (Scale Dev E' + str(p1c_epoch) + ')':<35} {p1c_spearman:<12.4f}")
    print(f"{'Pattern 2 (Rater-Specific)':<35} {p2_spearman:<12.4f}")
    print(f"{'Pattern 2b (Rank Normalized)':<35} {p2b_spearman:<12.4f}")
    print(f"{'Pattern 2c (Scale Normalized)':<35} {p2c_spearman:<12.4f}")
    print(f"{'Pattern 3 (Split)':<35} {p3_spearman:<12.4f}")
    print(f"{'Best Per-Epoch (E' + str(best_epoch) + ')':<35} {best_spearman:<12.4f}")

    # Alpha values
    print("\nDiscrimination (α) by epoch:")
    print(f"{'Epoch':<6} {'P2 α':<10} {'P2b α':<10} {'P2c α':<10} {'P3 α':<10}")
    print("-" * 46)
    for e in sorted(alpha_p2.keys()):
        print(f"{e:<6} {alpha_p2[e]:<10.4f} {alpha_p2b[e]:<10.4f} {alpha_p2c[e]:<10.4f} {alpha_p3[e]:<10.4f}")

    # Plot
    output_path = Path(args.output) if args.output else Path(f"{args.task_id}_patterns.png")
    plot_results(epoch_data, p1_epoch, p1_spearman, p2_spearman, p2b_spearman, p3_spearman,
                 alpha_p2, alpha_p2b, alpha_p3, output_path, args.task_id)


if __name__ == "__main__":
    main()
