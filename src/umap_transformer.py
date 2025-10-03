from umap import UMAP
import numpy as np

class UMAPTransformer:
    """
    Обертка над UMAP для унификации с пайплайном PCA.
    
    Args:
        n_components (int): число измерений для проекции (обычно 2 или 3).
        n_neighbors (int): число соседей (регулирует локальность/глобальность структуры).
        min_dist (float): минимальная дистанция между точками (0 -> плотнее, 1 -> равномернее).
        metric (str): метрика расстояния ('euclidean', 'cosine', 'manhattan' и др.)
    """
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine'):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.umap = None

    def fit(self, X):
        self.umap = UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=42
        )
        self.umap.fit(X)
        print(f"UMAP обучен: {self.n_components} компоненты")
        return self

    def transform(self, X):
        return self.umap.transform(X)