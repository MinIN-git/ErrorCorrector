from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from src.full_corrector.validation import validate_features


@dataclass(frozen=True)
class ClusterDispatcherConfig:
    """Настройки диспетчера, который назначает объект кластеру ошибок."""

    n_clusters: int = 10
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_clusters < 1:
            raise ValueError("n_clusters должен быть положительным")


class ClusterDispatcher:
    """KMeans-диспетчер из статьи: error sample -> ближайший WR-кластер."""

    def __init__(self, config: ClusterDispatcherConfig | None = None):
        self.config = config or ClusterDispatcherConfig()

    def fit(self, wrong: np.ndarray) -> "ClusterDispatcher":
        wrong = validate_features(wrong)
        if self.config.n_clusters > len(wrong):
            raise ValueError("n_clusters не может превышать число wrong-наблюдений")

        self.kmeans_ = KMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_state,
            n_init=10,
        ).fit(wrong)
        self.labels_ = self.kmeans_.labels_
        self.centroids_ = self.kmeans_.cluster_centers_
        self.cluster_sizes_ = np.bincount(
            self.labels_,
            minlength=self.config.n_clusters,
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        self._check_fitted()
        features = validate_features(features)
        distances = ((features[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)

    def _check_fitted(self) -> None:
        if not hasattr(self, "centroids_"):
            raise RuntimeError("Сначала вызовите fit")
