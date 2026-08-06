from dataclasses import dataclass

import numpy as np

from src.full_corrector.validation import validate_features


@dataclass(frozen=True)
class FisherSeparatorConfig:
    """Настройки линейного Fisher-сепаратора."""

    regularization: float = 1e-6

    def __post_init__(self) -> None:
        if self.regularization < 0:
            raise ValueError("regularization не может быть отрицательной")


class FisherSeparator:
    """Линейный separator CR vs WR или CR vs один WR-кластер."""

    def __init__(self, config: FisherSeparatorConfig | None = None):
        self.config = config or FisherSeparatorConfig()

    def fit(self, correct: np.ndarray, wrong: np.ndarray) -> "FisherSeparator":
        correct = validate_features(correct)
        wrong = validate_features(wrong)
        if correct.shape[1] != wrong.shape[1]:
            raise ValueError("correct и wrong должны иметь одинаковое число признаков")
        if len(correct) < 2:
            raise ValueError("Для correct нужны хотя бы два наблюдения")
        covariance = _covariance(correct) + _covariance(wrong)
        trace_scale = np.trace(covariance) / max(covariance.shape[0], 1)
        ridge = self.config.regularization * max(trace_scale, 1.0)
        covariance = covariance + ridge * np.eye(covariance.shape[0])

        direction = np.linalg.pinv(covariance) @ (correct.mean(axis=0) - wrong.mean(axis=0))
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Не удалось построить Fisher-направление: средние классов совпадают")

        self.direction_ = direction / norm
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return validate_features(features) @ self.direction_

    def _check_fitted(self) -> None:
        if not hasattr(self, "direction_"):
            raise RuntimeError("Сначала вызовите fit")


def _covariance(features: np.ndarray) -> np.ndarray:
    if len(features) < 2:
        return np.zeros((features.shape[1], features.shape[1]))
    return np.atleast_2d(np.cov(features, rowvar=False))
