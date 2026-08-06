from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorrectorSplit:
    """Learning/test subsets used by the MATLAB notation in the source notebook."""

    correct_train: np.ndarray
    correct_test: np.ndarray
    wrong_train: np.ndarray
    wrong_test: np.ndarray

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            correct_train=self.correct_train,
            correct_test=self.correct_test,
            wrong_train=self.wrong_train,
            wrong_test=self.wrong_test,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CorrectorSplit":
        with np.load(path) as arrays:
            return cls(**{name: arrays[name] for name in cls.__dataclass_fields__})


def load_feature_csv(path: str | Path, feature_token: str = "feature") -> np.ndarray:
    """Load only numeric feature columns from a detector result CSV."""
    frame = pd.read_csv(path, index_col=0)
    columns = [column for column in frame.columns if feature_token in str(column)]
    if not columns:
        raise ValueError(f"В {path} не найдены столбцы, содержащие {feature_token!r}")

    features = frame.loc[:, columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(features).all():
        raise ValueError(f"В признаках из {path} есть NaN или бесконечные значения")
    return features


def split_correct_wrong(
    correct: np.ndarray,
    wrong: np.ndarray,
    train_fraction: float = 0.5,
    random_state: int = 42,
) -> CorrectorSplit:
    """Independently shuffle and split correct/wrong observations."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction должен находиться строго между 0 и 1")
    if correct.ndim != 2 or wrong.ndim != 2:
        raise ValueError("correct и wrong должны быть двумерными массивами")
    if correct.shape[1] != wrong.shape[1]:
        raise ValueError("correct и wrong должны иметь одинаковое число признаков")
    if min(len(correct), len(wrong)) < 2:
        raise ValueError("Для каждого класса нужны хотя бы два наблюдения")

    rng = np.random.default_rng(random_state)
    correct = correct[rng.permutation(len(correct))]
    wrong = wrong[rng.permutation(len(wrong))]
    n_correct = int(train_fraction * len(correct))
    n_wrong = int(train_fraction * len(wrong))

    return CorrectorSplit(
        correct_train=correct[:n_correct],
        correct_test=correct[n_correct:],
        wrong_train=wrong[:n_wrong],
        wrong_test=wrong[n_wrong:],
    )
