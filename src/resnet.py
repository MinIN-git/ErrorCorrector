import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from typing import Tuple, Dict
from tqdm import tqdm


class ResNet18:
    def __init__(
        self,
        data_dir: str = "../data",
        batch_size: int = 64,
        learning_rate: float = 0.001,
        num_epochs: int = 5,
        pretrained: bool = True,
        device: str | None = None,
    ):
        """
        Инициализация класса для обучения ResNet18.
        
        Args:
            data_dir: Директория с данными
            batch_size: Размер батча
            learning_rate: Скорость обучения
            num_epochs: Количество эпох
            pretrained: Использовать предобученные веса
            device: Устройство для обучения (cuda/cpu)
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.pretrained = pretrained
        
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.loss_func = None
        self.train_loader = None
        self.test_loader = None

        self.__prepare_data()
        self.__prepare_model()
        self.__prepare_optimizer()
        

    
    def __prepare_data(self) -> None:
        """Подготовка данных и загрузчиков"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_data = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        test_data = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        self.train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        self.test_loader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False
        )
    
    def __prepare_model(self) -> None:
        """Инициализация модели"""
        self.model = models.resnet18(weights=self.pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # 10 классов для CIFAR10
        self.model = self.model.to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
    
    def __prepare_optimizer(self) -> None:
        """Инициализация оптимизатора."""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
    
    def train_one_epoch(self) -> float:
        """Обучение модели на одной эпохе
        
        Returns:
            Среднее значение потерь за эпоху
        """
        self.model.train()
        running_loss = 0
        
        for images, labels in tqdm(self.train_loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_func(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        
        return running_loss / len(self.train_loader)
    
    def evaluate(self) -> Tuple[float, float]:
        """Оценка модели на тестовых данных
        
        Returns:
            Точность и средние потери на тестовом наборе
        """
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.loss_func(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = test_loss / len(self.test_loader)
        return accuracy, avg_loss
    
    def train(self) -> Dict[str, list]:
        """обучения модели                
           
        Returns:
            Словарь с историей обучения (потери и точность)
        """
        history = {
            "train_loss": [],
            "test_loss": [],
            "accuracy": []
        }
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}")
            
            train_loss = self.train_one_epoch()
            history["train_loss"].append(train_loss)
            
            accuracy, test_loss = self.evaluate()
            history["test_loss"].append(test_loss)
            history["accuracy"].append(accuracy)
            
            print(
                f"Train Loss: {train_loss:.4f}, "
                f"Test Loss: {test_loss:.4f}, "
                f"Accuracy: {accuracy:.2f}%"
            )
        
        return history
    
    def save_model(self, path: str) -> None:
        """Сохранение весов модели
        
        Args:
            path: Путь для сохранения файла
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


if __name__ == "__main__":
    trainer = ResNet18(
        data_dir="../data",
        batch_size=64,
        learning_rate=0.001,
        num_epochs=5,
        pretrained=True
    )
    
    history = trainer.train()
    trainer.save_model("cifar10_resnet_safe.pth")