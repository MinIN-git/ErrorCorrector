from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

from src.full_corrector.dispatcher import ClusterDispatcher, ClusterDispatcherConfig
from src.full_corrector.fisher import FisherSeparator, FisherSeparatorConfig
from src.full_corrector.projection import FeatureProjector, FeatureProjectorConfig
from src.full_corrector.validation import validate_correct_wrong, validate_features


@dataclass(frozen=True)
class CorrectorConfig:
    """Hyperparameters of the representation, dispatcher and separators."""

    n_components: int | None = 200
    n_clusters: int = 10
    centre_on: str = "correct"
    whiten: bool = False
    regularization: float = 1e-6
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.centre_on not in {"all", "correct", "wrong"}:
            raise ValueError("centre_on: допустимы 'all', 'correct' и 'wrong'")
        if self.n_components is not None and self.n_components < 1:
            raise ValueError("n_components должен быть положительным или None")
        if self.n_clusters < 1:
            raise ValueError("n_clusters должен быть положительным")
        if self.regularization < 0:
            raise ValueError("regularization не может быть отрицательной")


class MultiCorrector:
    """Фасад полного пайплайна: projection -> dispatcher -> local separators."""

    def __init__(self, config: CorrectorConfig | None = None):
        self.config = config or CorrectorConfig()

    def fit(self, correct: np.ndarray, wrong: np.ndarray) -> "MultiCorrector":
        validate_correct_wrong(correct, wrong)
        projector_config = FeatureProjectorConfig(
            n_components=self.config.n_components,
            centre_on=self.config.centre_on,
            whiten=self.config.whiten,
        )
        self.projector_ = FeatureProjector(projector_config)
        correct_repr, wrong_repr = self.projector_.fit_transform_pair(correct, wrong)

        dispatcher_config = ClusterDispatcherConfig(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_state,
        )
        self.dispatcher_ = ClusterDispatcher(dispatcher_config).fit(wrong_repr)

        separator_config = FisherSeparatorConfig(
            regularization=self.config.regularization,
        )
        self.global_separator_ = FisherSeparator(separator_config).fit(correct_repr, wrong_repr)
        self.local_separators_ = tuple(
            FisherSeparator(separator_config).fit(
                correct_repr,
                wrong_repr[self.dispatcher_.labels_ == cluster],
            )
            for cluster in range(self.config.n_clusters)
        )

        self.centroids_ = self.dispatcher_.centroids_
        self.cluster_sizes_ = self.dispatcher_.cluster_sizes_
        self.global_direction_ = self.global_separator_.direction_
        self.directions_ = np.column_stack(
            [separator.direction_ for separator in self.local_separators_]
        )
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.projector_.transform(features)

    def dispatch(self, features: np.ndarray) -> np.ndarray:
        representation = self.transform(features)
        return self.dispatcher_.predict(representation)

    def score(self, features: np.ndarray, clustered: bool = True) -> np.ndarray:
        """Return a correctness score; smaller values indicate a likely error."""
        representation = self.transform(features)
        if not clustered:
            return self.global_separator_.score(representation)
        labels = self.dispatcher_.predict(representation)
        scores = np.empty(len(representation), dtype=float)
        for cluster, separator in enumerate(self.local_separators_):
            mask = labels == cluster
            if mask.any():
                scores[mask] = separator.score(representation[mask])
        return scores

    def save(self, path: str | Path) -> Path:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MultiCorrector":
        with Path(path).open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls):
            raise TypeError(f"В {path} сохранён объект типа {type(model).__name__}, а не MultiCorrector")
        return model

    @staticmethod
    def _validate_pair(correct: np.ndarray, wrong: np.ndarray) -> None:
        validate_correct_wrong(correct, wrong)

    @staticmethod
    def _validate_features(features: np.ndarray) -> np.ndarray:
        return validate_features(features)

    def _check_fitted(self) -> None:
        if not hasattr(self, "directions_"):
            raise RuntimeError("Сначала вызовите fit")
