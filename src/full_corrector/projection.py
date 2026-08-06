from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from src.full_corrector.validation import validate_features


@dataclass(frozen=True)
class FeatureProjectorConfig:
    """Настройки общего пространства признаков для корректора."""

    n_components: int | None = 200
    centre_on: str = "correct"
    whiten: bool = False

    def __post_init__(self) -> None:
        if self.centre_on not in {"all", "correct", "wrong"}:
            raise ValueError("centre_on: допустимы 'all', 'correct' и 'wrong'")
        if self.n_components is not None and self.n_components < 1:
            raise ValueError("n_components должен быть положительным или None")


class FeatureProjector:
    """Центрирование + PCA/whitening, вынесенные из большого notebook."""

    def __init__(self, config: FeatureProjectorConfig | None = None):
        self.config = config or FeatureProjectorConfig()

    def fit(self, correct: np.ndarray, wrong: np.ndarray) -> "FeatureProjector":
        correct = validate_features(correct)
        wrong = validate_features(wrong)
        reference = self._reference(correct, wrong)
        self.centre_ = reference.mean(axis=0)

        if self.config.n_components is None:
            self.pca_ = None
            self.scale_ = (
                reference.std(axis=0)
                if self.config.whiten
                else np.ones(reference.shape[1])
            )
            self.scale_[self.scale_ == 0] = 1
            return self

        max_components = min(reference.shape[0], reference.shape[1])
        if self.config.n_components > max_components:
            raise ValueError(
                f"n_components={self.config.n_components}, максимум для выбранной базы - {max_components}"
            )
        self.pca_ = PCA(n_components=self.config.n_components, whiten=self.config.whiten)
        self.pca_.fit(reference - self.centre_)
        self.scale_ = None
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        self._check_fitted()
        centered = validate_features(features) - self.centre_
        if self.pca_ is None:
            return centered / self.scale_
        return self.pca_.transform(centered)

    def fit_transform_pair(
        self,
        correct: np.ndarray,
        wrong: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.fit(correct, wrong)
        return self.transform(correct), self.transform(wrong)

    def _reference(self, correct: np.ndarray, wrong: np.ndarray) -> np.ndarray:
        if self.config.centre_on == "all":
            return np.vstack((correct, wrong))
        if self.config.centre_on == "wrong":
            return wrong
        return correct

    def _check_fitted(self) -> None:
        if not hasattr(self, "centre_"):
            raise RuntimeError("Сначала вызовите fit")
