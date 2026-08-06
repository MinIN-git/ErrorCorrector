import numpy as np
import torch
from torch.utils.data import DataLoader


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect true and predicted labels for a fitted torch model."""
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)

            preds.append(pred.cpu().numpy())
            labels.append(targets.cpu().numpy())

    return np.concatenate(labels), np.concatenate(preds)


def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract penultimate-layer features and predictions from a ResNet-like model."""
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device)

    features, preds, gt_preds = [], [], []

    model.eval()
    feature_extractor.eval()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = feature_extractor(images)
            outputs = outputs.view(outputs.size(0), -1)

            logits = model(images)
            pred = torch.argmax(logits, dim=1)

            features.append(outputs.cpu().numpy())
            preds.append(pred.cpu().numpy())
            gt_preds.append(targets.cpu().numpy())

    return np.vstack(features), np.concatenate(preds), np.concatenate(gt_preds)
