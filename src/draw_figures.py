import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px

from sklearn.metrics import roc_auc_score, roc_curve


def plot_projection(
        cr_data: np.ndarray, 
        wr_data: np.ndarray, 
        target_data: np.ndarray | None = None, 
        method: str = "PC",
        path: str | None = None
    ):
    """
    Визуализирует проекции данных CR и WR .
    При наличии target_data также отображает новые данные.
    
    Args:
        cr_data: данные корректных ответов после PCA/UMAP (n_samples, n_components)
        wr_data: данные некорректных ответов после PCA/UMAP
        target_data: новые данные для визуализации (n_samples, n_components)
        method: PCA или UMAP 
        path: Опционально, путь для сохранения
    """
    plt.figure(figsize=(10, 6))
    
    if cr_data.shape[1] < 2 or wr_data.shape[1] < 2:
        raise ValueError("Данные должны содержать как минимум 2 компоненты")
    
    plt.scatter(cr_data[:, 0], cr_data[:, 1], label='CR (correct)', alpha=0.5)
    plt.scatter(wr_data[:, 0], wr_data[:, 1], label='WR (wrong)', alpha=0.5)
    plt.title("Проекция CR/WR + Target")

    if target_data is not None:
        if target_data.shape[1] < 2:
            raise ValueError("target_data должен содержать как минимум 2 компоненты")
        plt.scatter(target_data[:, 0], target_data[:, 1], label='Target (new class)', 
                    alpha=0.8, color='red', marker='x')
        plt.title("Проекция CR/WR + Target")
    
    plt.xlabel(f"{method}-1")
    plt.ylabel(f"{method}-2'")
    plt.legend()
    plt.grid(True)

    if path:
        plt.savefig(path)
    
    plt.show()


def plot_embedding_3d(embedding: np.ndarray, gt_preds: np.ndarray):
    fig = px.scatter_3d(
    x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2],
    color=gt_preds.astype(str),  # цвет по классам
    opacity=0.7
    )
    fig.show()

def plot_loss(
        train_loss: list, 
        val_loss: list | None = None, 
        title: str = "Loss",
        path: str | None = None
    ):
    """
    Строит график ошибки обучения
    
    Args:
        train_loss: список потерь на тренировке
        val_loss: список потерь на валидации (опционально)
        title: заголовок графика
        path: Опционально, путь для сохранения
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Train Loss", marker='o')
    if val_loss is not None:
        plt.plot(val_loss, label="Validation Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if path:
        plt.savefig(path)

    plt.show()


def plot_accuracy(acc: list, title: str = "Accuracy", path: str | None = None):
    """
    Строит график точности модели.
    
    Args:
        acc: список значений точности (например, по эпохам)
        title: заголовок графика
        path: Опционально, путь для сохранения
    """
    plt.figure(figsize=(10, 6))
    plt.plot(acc, label="Accuracy", marker='o', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if path:
        plt.savefig(path)

    plt.show()

def plot_roc_auc(scores: np.ndarray, y: np.ndarray, path: str | None = None):
    """
    Строит roc_auc метрику (https://habr.com/ru/companies/otus/articles/809147/).
    
    Args:
        scores: Уверенность в оценке
        y: y компонента тех данных что загружали в lda (0 или 1)
        path (str | None): Опционально, путь для сохранения
    """

    auc = roc_auc_score(y, scores)

    fpr, tpr, thresholds = roc_curve(y, scores)

    plt.figure(figsize=(10,6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0,1], [0,1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC кривая LDA")
    plt.legend()
    plt.grid()

    if path:
        plt.savefig(path)

    plt.show()

    return auc


def plot_kde(X_lda: np.ndarray,  y: np.ndarray, path: str | None = None):
    """
    Строит ядерную оценку плотности (https://scikit-learn.ru/stable/modules/density.html).
    
    Args:
        X_lda: Координаты объектов вдоль  дискриминантной оси
        y: y компонента тех данных что загружали в lda (0 или 1)
        path (str | None): Опционально, путь для сохранения
    """

    plt.figure(figsize=(8,5))

    sns.kdeplot(X_lda[y==0].ravel(), label="Old classes (CR+WR)", fill=True, alpha=0.5)
    sns.kdeplot(X_lda[y==1].ravel(), label="Unseen class (Target)", fill=True, alpha=0.5)

    plt.title("KDE для LDA ")
    plt.xlabel("X")
    plt.legend()

    if path:
        plt.savefig(path)

    plt.show()