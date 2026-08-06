from dataclasses import asdict, dataclass
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from src.full_corrector.data import CorrectorSplit
from src.full_corrector.model import MultiCorrector


@dataclass(frozen=True)
class BinaryMetrics:
    threshold: float
    sensitivity: float
    specificity: float
    balanced_accuracy: float
    roc_auc: float
    confusion_matrix: list[list[int]]


@dataclass(frozen=True)
class EvaluationResult:
    train: BinaryMetrics
    test: BinaryMetrics

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2))
        return path


def find_balanced_threshold(labels: np.ndarray, error_scores: np.ndarray) -> float:
    """Choose a threshold using training data only."""
    fpr, tpr, thresholds = roc_curve(labels, error_scores)
    index = np.argmax((tpr + (1 - fpr)) / 2)
    return float(thresholds[index])


def evaluate_corrector(
    model: MultiCorrector,
    split: CorrectorSplit,
    clustered: bool = True,
) -> EvaluationResult:
    train_labels, train_scores = _labels_and_scores(
        model, split.correct_train, split.wrong_train, clustered
    )
    threshold = find_balanced_threshold(train_labels, train_scores)
    test_labels, test_scores = _labels_and_scores(
        model, split.correct_test, split.wrong_test, clustered
    )
    return EvaluationResult(
        train=_metrics(train_labels, train_scores, threshold),
        test=_metrics(test_labels, test_scores, threshold),
    )


def plot_score_distribution(
    model: MultiCorrector,
    correct: np.ndarray,
    wrong: np.ndarray,
    threshold: float | None = None,
    clustered: bool = True,
    title: str = "Распределение score корректора",
    path: str | Path | None = None,
):
    correct_scores = -model.score(correct, clustered=clustered)
    wrong_scores = -model.score(wrong, clustered=clustered)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(correct_scores, bins=50, alpha=0.55, density=True, label="Correct")
    axis.hist(wrong_scores, bins=50, alpha=0.55, density=True, label="Wrong")
    if threshold is not None:
        axis.axvline(threshold, color="black", linestyle="--", label="Train threshold")
    axis.set(title=title, xlabel="Error score", ylabel="Density")
    axis.legend()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight")
    return figure


def plot_roc_curve(
    model: MultiCorrector,
    correct: np.ndarray,
    wrong: np.ndarray,
    clustered: bool = True,
    title: str = "ROC multi-corrector",
    path: str | Path | None = None,
):
    labels, scores = _labels_and_scores(model, correct, wrong, clustered)
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    axis.plot([0, 1], [0, 1], "k--", alpha=0.5)
    axis.set(title=title, xlabel="False positive rate", ylabel="True positive rate")
    axis.legend()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight")
    return figure


def plot_roc_comparison(
    model: MultiCorrector,
    correct: np.ndarray,
    wrong: np.ndarray,
    title: str = "Один сепаратор и multi-corrector",
    path: str | Path | None = None,
):
    labels = np.r_[np.zeros(len(correct), dtype=int), np.ones(len(wrong), dtype=int)]
    figure, axis = plt.subplots(figsize=(6, 6))
    for clustered, label in ((False, "One separator"), (True, "Multi-corrector")):
        scores = -model.score(np.vstack((correct, wrong)), clustered=clustered)
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        axis.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    axis.plot([0, 1], [0, 1], "k--", alpha=0.5)
    axis.set(title=title, xlabel="False positive rate", ylabel="True positive rate")
    axis.legend()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight")
    return figure


def plot_confusion_matrix(
    metrics: BinaryMetrics,
    title: str = "Confusion matrix",
    path: str | Path | None = None,
):
    figure, axis = plt.subplots(figsize=(5, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.array(metrics.confusion_matrix),
        display_labels=["Correct", "Wrong"],
    )
    display.plot(ax=axis, values_format="d", colorbar=False)
    axis.set_title(title)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight")
    return figure


def plot_cluster_sizes(
    model: MultiCorrector,
    title: str = "WR train cluster sizes",
    path: str | Path | None = None,
):
    model._check_fitted()
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(np.arange(model.config.n_clusters), model.cluster_sizes_)
    axis.set(title=title, xlabel="Cluster", ylabel="WR train objects")
    axis.set_xticks(np.arange(model.config.n_clusters))
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight")
    return figure


def plot_dispatcher_projection(
    model: MultiCorrector,
    correct: np.ndarray,
    wrong: np.ndarray,
    title: str = "Dispatcher space",
    path: str | Path | None = None,
):
    correct_repr = model.transform(correct)
    wrong_repr = model.transform(wrong)
    labels = model.dispatcher_.predict(wrong_repr)

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(
        correct_repr[:, 0],
        correct_repr[:, 1],
        s=12,
        alpha=0.25,
        color="tab:gray",
        label="Correct",
    )
    scatter = axis.scatter(
        wrong_repr[:, 0],
        wrong_repr[:, 1],
        s=35,
        alpha=0.9,
        c=labels,
        cmap="tab10",
        label="Wrong",
    )
    axis.scatter(
        model.centroids_[:, 0],
        model.centroids_[:, 1],
        s=90,
        marker="x",
        linewidths=2,
        color="black",
        label="WR centroids",
    )
    axis.set(title=title, xlabel="Projection component 1", ylabel="Projection component 2")
    axis.legend(loc="best")
    figure.colorbar(scatter, ax=axis, label="WR cluster")
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight")
    return figure


def plot_cluster_score_distributions(
    model: MultiCorrector,
    correct: np.ndarray,
    wrong: np.ndarray,
    title: str = "Error score by dispatched cluster",
    path: str | Path | None = None,
):
    correct_labels = model.dispatch(correct)
    wrong_labels = model.dispatch(wrong)
    correct_scores = -model.score(correct, clustered=True)
    wrong_scores = -model.score(wrong, clustered=True)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.scatter(
        correct_labels - 0.12,
        correct_scores,
        s=16,
        alpha=0.35,
        color="tab:blue",
        label="Correct",
    )
    axis.scatter(
        wrong_labels + 0.12,
        wrong_scores,
        s=35,
        alpha=0.85,
        color="tab:orange",
        label="Wrong",
    )
    axis.set(title=title, xlabel="Dispatched cluster", ylabel="Error score")
    axis.set_xticks(np.arange(model.config.n_clusters))
    axis.legend()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight")
    return figure


def _labels_and_scores(
    model: MultiCorrector,
    correct: np.ndarray,
    wrong: np.ndarray,
    clustered: bool,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.r_[np.zeros(len(correct), dtype=int), np.ones(len(wrong), dtype=int)]
    scores = -model.score(np.vstack((correct, wrong)), clustered=clustered)
    return labels, scores


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> BinaryMetrics:
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return BinaryMetrics(
        threshold=threshold,
        sensitivity=float(tp / (tp + fn)),
        specificity=float(tn / (tn + fp)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        roc_auc=float(roc_auc_score(labels, scores)),
        confusion_matrix=matrix.tolist(),
    )
