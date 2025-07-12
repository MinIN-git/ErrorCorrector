import matplotlib.pyplot as plt


def draw_pca_projection(cr_data, wr_data, save=False):
    """
    Визуализирует проекции данных CR и WR на первые две главные компоненты
    
    Args:
        cr_data (np.ndarray): данные корректных ответов после PCA (n_samples, n_components)
        wr_data (np.ndarray): данные некорректных ответов после PCA
        save (bool): Флаг для сохранения графика (False по умолчанию)
        
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    
    if cr_data.shape[1] < 2 or wr_data.shape[1] < 2:
        raise ValueError("Данные должны содержать как минимум 2 компоненты")
    
    plt.scatter(cr_data[:, 0], cr_data[:, 1], label='CR (correct)', alpha=0.5)
    plt.scatter(wr_data[:, 0], wr_data[:, 1], label='WR (wrong)', alpha=0.5)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("Проекция CR/WR")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig("./figures/projection_first_two_principal_components.png")
    
    plt.show()
    