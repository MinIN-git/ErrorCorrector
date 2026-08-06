import numpy as np


def validate_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("Ожидается двумерная матрица объектов и признаков")
    if len(features) == 0:
        raise ValueError("Матрица признаков пуста")
    if not np.isfinite(features).all():
        raise ValueError("Матрица признаков содержит NaN или бесконечные значения")
    return features


def validate_correct_wrong(correct: np.ndarray, wrong: np.ndarray) -> None:
    correct = validate_features(correct)
    wrong = validate_features(wrong)
    if correct.shape[1] != wrong.shape[1]:
        raise ValueError("correct и wrong должны иметь одинаковое число признаков")
    if min(len(correct), len(wrong)) < 2:
        raise ValueError("Для обучения нужны хотя бы два наблюдения каждого класса")
