from sklearn.decomposition import PCA
import numpy as np


class PCATransformer:
    """
    Класс для обработки данных методом главных компнент

    На почитать:
        https://practicum.yandex.ru/blog/metod-glavnyh-komponent/
        https://scikit-learn.org/stable/modules/decomposition.html#pca/
    
    Args:
        n_components (int | auto): Число главных компонент: фиксированное, автоподбор методом Кайзера
        whiten (bool) : Флаг для отбеливания (http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/)
    """
    def __init__(self, n_components='auto', whiten=False):
        self.n_components = n_components
        self.whiten = whiten
        self.pca = None
        self.mean_ = None
        self._is_centered = False

    def __determine_components(self, ev):
        """
        Определение оптимального числа компонент
        
        Args:
            ev (np.ndarray): Объясненная дисперсия для всех возможных компонент
            
        Returns:
            int: Число выбранных главных компонент
        """
        if isinstance(self.n_components, int):
            return self.n_components
        elif self.n_components == 'auto':
            return sum(ev > np.mean(ev))
        else:
            return len(ev)
        
        # Тут еще можно метод сломанной кости
        # https://chinapads.ru/c/s/metod_glavnyih_komponent_-_otsenka_chisla_glavnyih_komponent_po_pravilu_slomannoy_trosti
     

    def fit(self, X):
        """
        Обучение PCA модели
        
        Args:
            X (np.ndarray): Матрица признаков формы (n_samples, n_features)
        """
        self._is_centered = np.allclose(np.mean(X, axis=0), 0, atol=1e-6)
        
        if not self._is_centered:
            self.mean_ = np.mean(X, axis=0)
            X_centered = X - self.mean_
        else:
            X_centered = X
        
        self.pca = PCA(n_components=None, whiten=self.whiten)
        self.pca.fit(X_centered)
        
        n_components = self.__determine_components(self.pca.explained_variance_)
        if n_components != self.pca.n_components_:
            self.pca = PCA(n_components=n_components, whiten=self.whiten)
            self.pca.fit(X_centered)

        print("Финальное количество главных компонент: {}".format(
            self.pca.n_components_
        ))

    def transform(self, X):
        """
        Применение обученной PCA модели к новым данным
        
        Данные будут центрированы с использованием тех же средних значений,
        что и при обучении (если исходные данные требовали центрирования)
        
        Args:
            X (np.ndarray): Матрица признаков формы (n_samples, n_features)
            
        Returns:
            np.ndarray: Преобразованные данные формы (n_samples, n_components)
        """
        if not self._is_centered:
            X_centered = X - self.mean_
        else:
            X_centered = X
        return self.pca.transform(X_centered)      

    

if __name__=="__main__":
    arr1 = np.array([5, 5, 5])
    arr2 = np.array([3, 2, 5])

    all_data = np.vstack((arr1, arr2))

    pca = PCATransformer(n_components='auto', whiten=True) 
    pca.fit(all_data)