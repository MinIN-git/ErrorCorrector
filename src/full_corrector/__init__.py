"""Модульная реализация кластерного корректора ошибок."""

from src.full_corrector.data import CorrectorSplit, load_feature_csv, split_correct_wrong
from src.full_corrector.dispatcher import ClusterDispatcher, ClusterDispatcherConfig
from src.full_corrector.evaluation import (
    BinaryMetrics,
    EvaluationResult,
    evaluate_corrector,
    find_balanced_threshold,
    plot_cluster_score_distributions,
    plot_cluster_sizes,
    plot_confusion_matrix,
    plot_dispatcher_projection,
    plot_roc_comparison,
    plot_roc_curve,
    plot_score_distribution,
)
from src.full_corrector.fisher import FisherSeparator, FisherSeparatorConfig
from src.full_corrector.model import CorrectorConfig, MultiCorrector
from src.full_corrector.projection import FeatureProjector, FeatureProjectorConfig

__all__ = [
    "BinaryMetrics",
    "ClusterDispatcher",
    "ClusterDispatcherConfig",
    "CorrectorConfig",
    "CorrectorSplit",
    "EvaluationResult",
    "FeatureProjector",
    "FeatureProjectorConfig",
    "FisherSeparator",
    "FisherSeparatorConfig",
    "MultiCorrector",
    "evaluate_corrector",
    "find_balanced_threshold",
    "load_feature_csv",
    "plot_cluster_score_distributions",
    "plot_cluster_sizes",
    "plot_confusion_matrix",
    "plot_dispatcher_projection",
    "plot_roc_comparison",
    "plot_roc_curve",
    "plot_score_distribution",
    "split_correct_wrong",
]
