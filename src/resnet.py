import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchvision import models
from torch.utils.data import DataLoader
from typing import Tuple, Dict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score



class ResNet18:
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: str,
        weights: str | None = None,
        target_loader: DataLoader | None = None,
        learning_rate: float = 0.001,
        num_epochs: int = 5,
    ):
        """
        Класс для обучения ResNet18.

        Args:
            train_loader: DataLoader для обучающей выборки
            val_loader: DataLoader для валидации
            test_loader: DataLoader для теста
            target_loader: DataLoader для target-класса (опционально)
            learning_rate: скорость обучения
            num_epochs: количество эпох
            weights: использовать предобученные веса
            device: устройство (cuda/cpu)
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.target_loader = target_loader

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = weights
        self.device = device 

        # готовим модель
        self.__prepare_model()
        self.__prepare_optimizer()
    
    def __prepare_model(self) -> None:
        """Инициализация модели"""
        self.model = models.resnet18(weights=None, num_classes=10)
    
        # self.model.fc = nn.Linear(self.model.fc.in_features, 9)
        
        if isinstance(self.weights, str):
            state_dict = torch.load(self.weights, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded weights from {self.weights}")

        self.model = self.model.to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
    
    def __prepare_optimizer(self) -> None:
        """Инициализация оптимизатора"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
 
    def extract_features(self, loader: DataLoader):
        """
        Извлечения признаков из предпоследнего слоя модели
        
        Returns:
            features (np.ndarray): Извлеченные признаки формы (N, 512)
            preds (np.ndarray): Предсказанные метки классов формы (N,)
            gt_preds (np.ndarray): Истинные метки классов формы (N,)
        """
        # Экстрактор признаков без последнего слоя
        feature_extractor = torch.nn.Sequential(
            *list(self.model.children())[:-1]
        ).to(self.device)

        features, preds, gt_preds = [], [], []
        
        self.model.eval()
        feature_extractor.eval()
        
        with torch.no_grad():  
            for images, targets in loader:
                images = images.to(self.device)  
                targets = targets.to(self.device) 

                # извлечение признаков
                outputs = feature_extractor(images)
                outputs = outputs.view(outputs.size(0), -1)
                
                # предсказания модели
                logits = self.model(images)
                pred = torch.argmax(logits, dim=1)
                
                # переводим на CPU и в numpy
                features.append(outputs.cpu().numpy())
                preds.append(pred.cpu().numpy())
                gt_preds.append(targets.cpu().numpy())
        
        features = np.vstack(features) 
        preds = np.concatenate(preds)
        gt_preds = np.concatenate(gt_preds) 
        
        return features, preds, gt_preds
    
    def train_one_epoch(self) -> float:
        """Обучение на одной эпохе"""
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
    
    def evaluate(self, loader: DataLoader) -> dict:
        """Оценка модели на произвольном DataLoader с метриками sklearn"""
        self.model.eval()
        all_preds, all_labels = [], []
        test_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                loss = self.loss_func(outputs, labels)
                test_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # метрики
        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average="macro") 
        f1_micro = f1_score(all_labels, all_preds, average="micro")  

        avg_loss = test_loss / len(loader)

        return {
            "loss": avg_loss,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
        }
    
    def train(self) -> Dict[str, list]:
        """Цикл обучения"""
        history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1_macro": [],
                   "val_f1_micro": []}
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            train_loss = self.train_one_epoch()
            val_info = self.evaluate(self.val_loader)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_info['loss'])
            history["val_accuracy"].append(val_info['accuracy'])
            history["val_f1_macro"].append(val_info['f1_macro'])
            history["val_f1_micro"].append(val_info['f1_micro'])
            
            print(
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_info['loss']:.4f}, "
                f"Val f1_score: {val_info['f1_macro']:.2f}%"
            )
        
        return history
    
    def test(self) -> Tuple[float, float]:
        """Финальная проверка на test"""
        return self.evaluate(self.test_loader)
    
    
    def save_model(self, path: str) -> None:
        """Сохранение весов модели"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
