import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image

class CIFARCustom(Dataset):
    def __init__(self, root, exclude_class, train=True,
                 transform=None, download=True):
        """
        exclude_class: класс(ы), который(ые) мы не хотим видеть 
        """
        self.dataset = datasets.CIFAR10(root=root, train=train, download=download)
        self.transform = transform

        data, targets = [], []
        for img, label in zip(self.dataset.data, self.dataset.targets):
            if label not in exclude_class:
                data.append(img)
                targets.append(label)

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label

