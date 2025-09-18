import matplotlib.pyplot as plt


def plot_pca_projection(cr_data, wr_data, target_data=None, save=False):
    """
    Визуализирует проекции данных CR и WR на первые две главные компоненты.
    При наличии target_data также отображает новые данные.
    
    Args:
        cr_data (np.ndarray): данные корректных ответов после PCA (n_samples, n_components)
        wr_data (np.ndarray): данные некорректных ответов после PCA
        target_data (np.ndarray, optional): новые данные для визуализации (n_samples, n_components)
        save (bool): Флаг для сохранения графика (False по умолчанию)
        
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    
    if cr_data.shape[1] < 2 or wr_data.shape[1] < 2:
        raise ValueError("Данные должны содержать как минимум 2 компоненты")
    
    plt.scatter(cr_data[:, 0], cr_data[:, 1], label='CR (correct)', alpha=0.5)
    plt.scatter(wr_data[:, 0], wr_data[:, 1], label='WR (wrong)', alpha=0.5)

    if target_data is not None:
        if target_data.shape[1] < 2:
            raise ValueError("target_data должен содержать как минимум 2 компоненты")
        plt.scatter(target_data[:, 0], target_data[:, 1], label='Target (new class)', 
                    alpha=0.8, color='red', marker='x')
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("Проекция CR/WR + Target")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig("./figures/projection_first_two_principal_components.png")
    
    plt.show()


def plot_loss(train_loss: list, val_loss: list | None = None, title: str = "Loss") -> None:
    """
    Строит график ошибки обучения.
    
    Args:
        train_loss: список потерь на тренировке
        val_loss: список потерь на валидации (опционально)
        title: заголовок графика
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
    plt.show()


def plot_accuracy(acc: list, title: str = "Accuracy") -> None:
    """
    Строит график точности модели.
    
    Args:
        acc: список значений точности (например, по эпохам)
        title: заголовок графика
    """
    plt.figure(figsize=(10, 6))
    plt.plot(acc, label="Accuracy", marker='o', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    