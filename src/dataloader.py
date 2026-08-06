from dataclasses import dataclass

from torch.utils.data import Dataset
from torchvision import datasets


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    dataset_cls: type
    num_classes: int
    input_channels: int
    mean: tuple[float, ...]
    std: tuple[float, ...]


DATASET_REGISTRY = {
    "cifar10": DatasetConfig(
        name="cifar10",
        dataset_cls=datasets.CIFAR10,
        num_classes=10,
        input_channels=3,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    try:
        return DATASET_REGISTRY[name]
    except KeyError as error:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Неизвестный датасет {name!r}. Доступны: {available}") from error


def build_excluded_dataset(
    name: str,
    root: str,
    exclude_class,
    train: bool = True,
    transform=None,
    download: bool = True,
) -> "ExcludedClassDataset":
    config = get_dataset_config(name)
    dataset = config.dataset_cls(
        root=root,
        train=train,
        transform=None,
        download=download,
    )
    return ExcludedClassDataset(
        dataset=dataset,
        exclude_class=exclude_class,
        num_classes=config.num_classes,
        transform=transform,
    )


class ExcludedClassDataset(Dataset):
    """Dataset wrapper that removes classes while preserving original labels."""

    def __init__(
        self,
        dataset: Dataset,
        exclude_class,
        num_classes: int,
        transform=None,
    ):
        self.dataset = dataset
        self.exclude_class = set(exclude_class)
        self.num_classes = num_classes
        self.classes = getattr(
            dataset,
            "classes",
            [str(class_id) for class_id in range(num_classes)],
        )
        self.kept_classes = [
            class_id for class_id in range(num_classes)
            if class_id not in self.exclude_class
        ]
        self.transform = transform
        targets = _get_targets(dataset)
        self.indices = [
            index for index, label in enumerate(targets)
            if label not in self.exclude_class
        ]
        self.targets = [
            targets[index]
            for index in self.indices
        ]
        if hasattr(dataset, "data"):
            self.data = [
                dataset.data[index]
                for index in self.indices
            ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label


def _get_targets(dataset: Dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if hasattr(dataset, "labels"):
        return list(dataset.labels)
    raise TypeError("Dataset должен иметь атрибут targets или labels")
