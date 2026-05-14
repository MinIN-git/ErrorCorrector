from sklearn.decomposition import PCA
import numpy as np


class PCATransformer:
    """
    Класс для обработки данных методом главных компонент

    На почитать:
        https://practicum.yandex.ru/blog/metod-glavnyh-komponent/
        https://scikit-learn.org/stable/modules/decomposition.html#pca/
    
    Args:
        n_components (int | 'kaiser'): Число главных компонент: фиксированное, автоподбор методом Кайзера
        whiten (bool) : Флаг для отбеливания (http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/)
    """
    def __init__(self, n_components='kaiser', whiten=False):
        self.n_components = n_components
        self.whiten = whiten
        self.pca = None

        self._comp_start = None
        self._comp_end = None

    def fit(self, X):
        """
        Обучение PCA модели
        """
        if self.n_components == "kaiser":
            pca = PCA(n_components=None, whiten=self.whiten)
            pca.fit(X)
            self.n_components = np.sum(pca.explained_variance_ > np.mean(pca.explained_variance_))

        self.pca = PCA(n_components=self.n_components, whiten=self.whiten)
        self.pca.fit(X)

        print("Финальное количество главных компонент: {}".format(
            self.pca.n_components_
        ))
        print("Объяснённая дисперсия: {}".format(
            int(1000*np.sum(self.pca.explained_variance_ratio_))/1000
        ))
        return self

    def transform(self, X):
        """
        Применение обученной PCA модели к новым данным.
        """
        if self.pca is None:
            raise RuntimeError("Сначала вызовите fit()")
            
        X_proj = self.pca.transform(X)
        
        # Если задан диапазон — возвращаем только хвост
        if self._comp_start is not None and self._comp_end is not None:
            return X_proj[:, self._comp_start:self._comp_end]
        
        return X_proj
      
    def get_explained_variance(self):
        return round(np.sum(self.pca.explained_variance_ratio_), 3)
    
    def select_variance_range(self, var_min=0.90, var_max=0.99):
        """
        Настраивает трансформер на работу с компонентами в диапазоне 
        накопленной объясненной дисперсии [var_min, var_max].
        
            
        Returns:
            tuple: (start_idx, end_idx) — индексы выбранных компонент
        """
        if self.pca is None:
            raise RuntimeError("Сначала fit()")
        
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        
        self._comp_start = np.searchsorted(cumsum, var_min, side='left')
        self._comp_end = np.searchsorted(cumsum, var_max, side='right')
        
        variance = round(var_max - var_min, 3)
        print(f"Выбран диапазон компонент [{self._comp_start}:{self._comp_end}] "
              f"(дисперсия: {variance}")
        
        return self._comp_start, self._comp_end, variance
    
    def reset_variance_range(self):
        """
        Сбрасывает настройку диапазона компонент
        """
        self._comp_start = None
        self._comp_end = None
        print("Диапазон компонент сброшен, используются все компоненты")
        return self
