"""Evaluation module."""
from .metrics import (
    EvaluationResult,
    compute_pearson,
    compute_spearman,
    compute_qwk,
    evaluate_predictions,
    evaluate_scoring_results,
    EpochSelector,
    select_best_epoch,
    compute_p_invalid_stats,
)
from .visualize import (
    plot_true_vs_predicted,
    plot_confusion_matrix,
    plot_score_distribution,
    plot_epoch_metrics,
    plot_all_epoch_metrics,
    create_evaluation_plots,
)

__all__ = [
    "EvaluationResult",
    "compute_pearson",
    "compute_spearman",
    "compute_qwk",
    "evaluate_predictions",
    "evaluate_scoring_results",
    "EpochSelector",
    "select_best_epoch",
    "compute_p_invalid_stats",
    "plot_true_vs_predicted",
    "plot_confusion_matrix",
    "plot_score_distribution",
    "plot_epoch_metrics",
    "plot_all_epoch_metrics",
    "create_evaluation_plots",
]
