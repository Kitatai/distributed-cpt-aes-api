"""
Evaluation metrics module for essay scoring.

Implements Pearson, Spearman, and QWK metrics for scoring evaluation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    pearson: float
    pearson_pvalue: float
    spearman: float
    spearman_pvalue: float
    qwk: float
    n_samples: int
    n_valid: int
    mean_y_true: float
    mean_y_pred: float
    std_y_true: float
    std_y_pred: float

    def to_dict(self) -> dict:
        return {
            'pearson': self.pearson,
            'pearson_pvalue': self.pearson_pvalue,
            'spearman': self.spearman,
            'spearman_pvalue': self.spearman_pvalue,
            'qwk': self.qwk,
            'n_samples': self.n_samples,
            'n_valid': self.n_valid,
            'mean_y_true': self.mean_y_true,
            'mean_y_pred': self.mean_y_pred,
            'std_y_true': self.std_y_true,
            'std_y_pred': self.std_y_pred,
        }


def compute_pearson(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Pearson correlation coefficient.

    Args:
        y_true: Ground truth scores
        y_pred: Predicted scores

    Returns:
        Tuple of (correlation, p-value)
    """
    if len(y_true) < 2:
        return 0.0, 1.0

    try:
        r, p = stats.pearsonr(y_true, y_pred)
        return float(r), float(p)
    except Exception as e:
        logger.warning(f"Error computing Pearson: {e}")
        return 0.0, 1.0


def compute_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation coefficient.

    Args:
        y_true: Ground truth scores
        y_pred: Predicted scores

    Returns:
        Tuple of (correlation, p-value)
    """
    if len(y_true) < 2:
        return 0.0, 1.0

    try:
        r, p = stats.spearmanr(y_true, y_pred)
        return float(r), float(p)
    except Exception as e:
        logger.warning(f"Error computing Spearman: {e}")
        return 0.0, 1.0


def compute_qwk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_min: Optional[int] = None,
    y_max: Optional[int] = None,
) -> float:
    """
    Compute Quadratic Weighted Kappa (QWK).

    Args:
        y_true: Ground truth scores (integers)
        y_pred: Predicted scores (integers)
        y_min: Minimum score (for label set)
        y_max: Maximum score (for label set)

    Returns:
        QWK score
    """
    if len(y_true) < 2:
        return 0.0

    try:
        # Ensure integer types
        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)

        # Compute QWK
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        return float(qwk)
    except Exception as e:
        logger.warning(f"Error computing QWK: {e}")
        return 0.0


def evaluate_predictions(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_min: Optional[int] = None,
    y_max: Optional[int] = None,
) -> EvaluationResult:
    """
    Evaluate predictions with all metrics.

    Args:
        y_true: Ground truth scores
        y_pred: Predicted scores
        y_min: Minimum score
        y_max: Maximum score

    Returns:
        EvaluationResult with all metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Filter out invalid predictions (None values)
    valid_mask = ~(np.isnan(y_true.astype(float)) | np.isnan(y_pred.astype(float)))
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    n_samples = len(y_true)
    n_valid = len(y_true_valid)

    if n_valid < 2:
        logger.warning(f"Too few valid samples ({n_valid}) for evaluation")
        return EvaluationResult(
            pearson=0.0,
            pearson_pvalue=1.0,
            spearman=0.0,
            spearman_pvalue=1.0,
            qwk=0.0,
            n_samples=n_samples,
            n_valid=n_valid,
            mean_y_true=float(np.mean(y_true_valid)) if n_valid > 0 else 0.0,
            mean_y_pred=float(np.mean(y_pred_valid)) if n_valid > 0 else 0.0,
            std_y_true=float(np.std(y_true_valid)) if n_valid > 0 else 0.0,
            std_y_pred=float(np.std(y_pred_valid)) if n_valid > 0 else 0.0,
        )

    # Compute metrics
    pearson, pearson_p = compute_pearson(y_true_valid, y_pred_valid)
    spearman, spearman_p = compute_spearman(y_true_valid, y_pred_valid)
    qwk = compute_qwk(y_true_valid, y_pred_valid, y_min, y_max)

    return EvaluationResult(
        pearson=pearson,
        pearson_pvalue=pearson_p,
        spearman=spearman,
        spearman_pvalue=spearman_p,
        qwk=qwk,
        n_samples=n_samples,
        n_valid=n_valid,
        mean_y_true=float(np.mean(y_true_valid)),
        mean_y_pred=float(np.mean(y_pred_valid)),
        std_y_true=float(np.std(y_true_valid)),
        std_y_pred=float(np.std(y_pred_valid)),
    )


def evaluate_scoring_results(
    results: List[Dict],
    prediction_key: str = "y_hat_greedy",
    y_min: Optional[int] = None,
    y_max: Optional[int] = None,
) -> EvaluationResult:
    """
    Evaluate from list of scoring result dictionaries.

    Args:
        results: List of scoring result dicts
        prediction_key: Key for predicted scores
        y_min: Minimum score
        y_max: Maximum score

    Returns:
        EvaluationResult
    """
    y_true = []
    y_pred = []

    for r in results:
        if r.get('y_true') is not None and r.get(prediction_key) is not None:
            y_true.append(r['y_true'])
            y_pred.append(r[prediction_key])

    return evaluate_predictions(y_true, y_pred, y_min, y_max)


class EpochSelector:
    """
    Selects best epoch based on evaluation metrics.
    """

    def __init__(
        self,
        metric: str = "spearman",
        higher_is_better: bool = True,
    ):
        """
        Initialize the selector.

        Args:
            metric: Metric to use for selection ('spearman', 'pearson', 'qwk')
            higher_is_better: Whether higher metric values are better
        """
        self.metric = metric
        self.higher_is_better = higher_is_better
        self.history: Dict[int, EvaluationResult] = {}

    def add_epoch_result(self, epoch: int, result: EvaluationResult):
        """Add evaluation result for an epoch."""
        self.history[epoch] = result

    def get_metric_value(self, result: EvaluationResult) -> float:
        """Get metric value from result."""
        return getattr(result, self.metric)

    def select_best_epoch(self) -> Tuple[int, EvaluationResult]:
        """
        Select best epoch based on metric.

        Returns:
            Tuple of (best_epoch, best_result)
        """
        if not self.history:
            raise ValueError("No epochs recorded")

        if self.higher_is_better:
            best_epoch = max(
                self.history.keys(),
                key=lambda e: self.get_metric_value(self.history[e])
            )
        else:
            best_epoch = min(
                self.history.keys(),
                key=lambda e: self.get_metric_value(self.history[e])
            )

        return best_epoch, self.history[best_epoch]

    def get_epoch_metrics(self) -> Dict[int, Dict]:
        """Get all epoch metrics as dictionary."""
        return {e: r.to_dict() for e, r in self.history.items()}


def select_best_epoch(
    epoch_results: Dict[int, EvaluationResult],
    metric: str = "spearman",
) -> Tuple[int, EvaluationResult]:
    """
    Select best epoch from results.

    Args:
        epoch_results: Dictionary mapping epoch to EvaluationResult
        metric: Metric to use for selection

    Returns:
        Tuple of (best_epoch, best_result)
    """
    selector = EpochSelector(metric=metric)
    for epoch, result in epoch_results.items():
        selector.add_epoch_result(epoch, result)
    return selector.select_best_epoch()


def compute_p_invalid_stats(
    results: List[Dict],
) -> Dict[str, float]:
    """
    Compute statistics about p_invalid (format violation probability).

    Args:
        results: List of scoring result dicts

    Returns:
        Dictionary with p_invalid statistics
    """
    p_invalids = [
        r['p_invalid'] for r in results
        if r.get('p_invalid') is not None
    ]

    if not p_invalids:
        return {
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'n_samples': 0,
        }

    return {
        'mean': float(np.mean(p_invalids)),
        'std': float(np.std(p_invalids)),
        'min': float(np.min(p_invalids)),
        'max': float(np.max(p_invalids)),
        'n_samples': len(p_invalids),
    }
