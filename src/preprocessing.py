import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from torchvision.models import resnet18
from sklearn.model_selection import train_test_split



class Preprocessing:
    def __init__(self, weigths_path, data_path="../data"):
        self.model = self.__load_model(weigths_path)
        self.test_sample = self.__load_test_sample(data_path)


    def __load_model(self, weigths_path):
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
        
        features = np.vstack(features)  # (N, 512)
        labels = np.concatenate(labels) 
        correct_preds = np.concatenate(correct_preds)
        
        return features, labels, correct_preds
    
    def get_data(self, fTS=0.5, normalize=True, debug=True):
        features, _, correct_preds = self.__extract_features()

        CR = features[correct_preds == 1] 
        WR = features[correct_preds == 0] 

        if normalize:
            CR = (CR - np.mean(CR, axis=0)) / np.std(CR, axis=0)
            WR = (WR - np.mean(WR, axis=0)) / np.std(WR, axis=0)

        CR_train, CR_test = train_test_split(
            CR, test_size=1-fTS, random_state=42
        )

        WR_train, WR_test = train_test_split(
            WR, test_size=1-fTS, random_state=42
        )

        if debug:
            print("Правильные предсказания (CR): {}".format(len(CR)))
            print("Ошибки (WR): {}".format(len(WR)))

        return CR_train, CR_test, WR_train, WR_test