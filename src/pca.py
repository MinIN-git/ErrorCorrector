from sklearn.decomposition import PCA
import numpy as np


class PCATransformer:
    """
    Класс для обработки данных методом главных компнент

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

    def fit(self, X):
        """
        Обучение PCA модели

        Пайплайн:
            1) Считаем PCA на всех компонентах, чтобы получить их дисперсии (`explained_variance_`).
            2) Анализируем дисперсии и решаем, сколько компонент оставить
        
        Args:
            X (np.ndarray): Матрица признаков формы (n_samples, n_features)
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
        return self.pca.transform(X)      

    

if __name__=="__main__":
    arr1 = np.array([5, 5, 5])
    arr2 = np.array([3, 2, 5])

    all_data = np.vstack((arr1, arr2))

    pca = PCATransformer(n_components='auto', whiten=True) 
    pca.fit(all_data)