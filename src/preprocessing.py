import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split


class Preprocessing:
    """
    Класс для предварительной обработки данных и извлечения признаков
    
    Args:
        weigths_path (str): Путь к файлу с весами модели
        data_path (str): Путь к директории с данными (по умолчанию "../data")
    """
    
    def __init__(self, weigths_path, data_path="../data"):
        self.model = self.__load_model(weigths_path) 
        self.test_sample = self.__load_test_sample(data_path) 

    def __load_model(self, weigths_path):
        """
        Загрузка модели 
        
        Args:
            weigths_path (str): Путь к файлу с весами
            
        Returns:
            torch.nn.Module: Загруженная модель
        """ 
        model = resnet18(weights=False, num_classes=10)  
        model.load_state_dict(
            torch.load(
                weigths_path,
                map_location="cuda", 
                weights_only=True  
            )  
        )
        return model
    
    def __load_test_sample(self, data_path):
        """
        Загрузка  CIFAR-10
        
        Args:
            data_path (str): Путь к данным
            
        Returns:
            torch.utils.data.DataLoader: Загрузчик тестовых данных
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        testset = torchvision.datasets.CIFAR10(
            root=data_path, 
            train=False,
            download=True, 
            transform=transform
        )
 
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=100, 
            shuffle=False 
        )

        return testloader
    
    def __extract_features(self):
        """
        Извлечения признаков из предпоследнего слоя модели
        
        Returns:
            features (np.ndarray): Извлеченные признаки формы (N, 512)
            labels (np.ndarray): Метки классов формы (N,)
            correct_preds (np.ndarray): Массив корректности предсказаний формы (N,)
        """
        # Экстрактора признаков без последнего слоя
        feature_extractor = torch.nn.Sequential(
            *list(self.model.children())[:-1] 
        )  

        features, labels, correct_preds = [], [], [] 
        
        with torch.no_grad():  
            for images, targets in self.test_sample:
                outputs = feature_extractor(images)
                outputs = outputs.view(outputs.size(0), -1) 
                
                logits = self.model(images)
                preds = torch.argmax(logits, dim=1)
                
                features.append(outputs.numpy())
                labels.append(targets.numpy())
                correct_preds.append((preds == targets).numpy())
        
        features = np.vstack(features) 
        labels = np.concatenate(labels) 
        correct_preds = np.concatenate(correct_preds)
        
        return features, labels, correct_preds
    
    def get_data(self, fTS=0.5, center="CR", debug=True):
        """
        Получение обработанных данных

        Args:
            fTS (float): Доля данных для обучения (по умолчанию 0.5)
            center (str or None): Как вычислять центр.
                None - без центрирования
                "CR"  - по CR_train
                "WR"  - по WR_train
                "All" - по CR_train + WR_train
            debug (bool): Выводить информацию

        Returns:
            CR_train, CR_test, WR_train, WR_test, centre
        """
        features, _, correct_preds = self.__extract_features()

        CR = features[correct_preds == 1]
        WR = features[correct_preds == 0]

        CR_train, CR_test = train_test_split(
            CR, test_size=1 - fTS, random_state=42
        )
        WR_train, WR_test = train_test_split(
            WR, test_size=1 - fTS, random_state=42
        )

        centre = None
        if center is not None:
            if center == "CR":
                centre = np.mean(CR_train, axis=0)
            elif center == "WR":
                centre = np.mean(WR_train, axis=0)
            elif center == "All":
                centre = np.mean(np.vstack([CR_train, WR_train]), axis=0)
            else:
                raise ValueError(f"Unknown center option: {center}")

            # Центрируем
            CR_train = CR_train - centre
            CR_test  = CR_test - centre
            WR_train = WR_train - centre
            WR_test  = WR_test - centre

        if debug:
            print(f"Правильные предсказания (CR): {len(CR)}")
            print(f"Ошибки (WR): {len(WR)}")
            if centre is not None:
                print(f"Центр посчитан по: {center}")

        return CR_train, CR_test, WR_train, WR_test, centre