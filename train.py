import os
import argparse
import torch
import numpy as np
import mlflow
import mlflow.pytorch

import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    classification_report
)

from src.resnet import ResNet18
from src.dataloader import CIFARCustom
from src.draw_figures import plot_loss, plot_accuracy, plot_confusion_matrix


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    save_path,
    normalize=False
):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype(np.float32)
        cm = cm / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    title = "Confusion Matrix (Normalized)" if normalize else "Confusion Matrix"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def collect_predictions(model, loader, device):
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



def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet18 on CIFAR-10 with excluded target class"
    )

    parser.add_argument(
        "-t", "--target",
        type=int,
        required=True,
        choices=range(10),
        help="Target class to exclude (0-9)"
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_root", type=str, default="./output")
    parser.add_argument("--model_root", type=str, default="./models")

    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(args.seed)

    mlflow.set_experiment("ResNet18_CIFAR10_Exclude_Class")

    with mlflow.start_run(run_name=f"exclude_class_{args.target}_epoch_{args.epochs}"):


        # Log hyperparameters
        mlflow.log_params(vars(args))
        mlflow.log_param("device", device.type)


        # Folders
        out_dir = f"{args.output_root}/{args.target}/train"
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(args.model_root, exist_ok=True)

        # Transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Dataset & splits
        dataset = CIFARCustom(
            root=args.data_root,
            exclude_class=[args.target],
            train=True,
            transform=transform
        )

        train_size = int(0.8 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_set, val_set, test_set = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=generator
        )

        mlflow.log_params({
            "train_size": len(train_set),
            "val_size": len(val_set),
            "test_size": len(test_set)
        })

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            generator=generator
        )

        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )


        # Model
        model = ResNet18(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            learning_rate=args.lr,
            num_epochs=args.epochs
        )

        history = model.train()

        # Log training metrics
        for epoch in range(len(history["train_loss"])):
            mlflow.log_metric("train_loss", history["train_loss"][epoch], step=epoch)
            mlflow.log_metric("val_loss", history["val_loss"][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history["val_accuracy"][epoch], step=epoch)
            mlflow.log_metric("val_f1_macro", history["val_f1_macro"][epoch], step=epoch)

        # Plots
        loss_path = f"{out_dir}/loss.png"
        acc_path = f"{out_dir}/accuracy.png"
        f1_path = f"{out_dir}/f1_macro.png"

        plot_loss(history["train_loss"], history["val_loss"], path=loss_path)
        plot_accuracy(history["val_accuracy"], path=acc_path)
        plot_accuracy(history["val_f1_macro"], "F1 Macro", f1_path)

        mlflow.log_artifact(loss_path, "plots")
        mlflow.log_artifact(acc_path, "plots")
        mlflow.log_artifact(f1_path, "plots")

        # TEST METRICS
        class_names = [str(i) for i in range(10) if i != args.target]

        y_test, y_pred = collect_predictions(model.model, test_loader, device)

        test_acc = accuracy_score(y_test, y_pred)
        test_f1_macro = f1_score(y_test, y_pred, average="macro")
        test_f1_weighted = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_macro", test_f1_macro)
        mlflow.log_metric("test_f1_weighted", test_f1_weighted)

        # per-class f1
        f1_per_class = f1_score(y_test, y_pred, average=None)
        for cls, f1 in zip(class_names, f1_per_class):
            mlflow.log_metric(f"test_f1_class_{cls}", f1)

        # Confusion matrices
        cm_raw_path = f"{out_dir}/confusion_matrix.png"
        cm_norm_path = f"{out_dir}/confusion_matrix_normalized.png"

        plot_confusion_matrix(
            y_test, y_pred, class_names, cm_raw_path, normalize=False
        )
        plot_confusion_matrix(
            y_test, y_pred, class_names, cm_norm_path, normalize=True
        )

        mlflow.log_artifact(cm_raw_path, "confusion_matrix")
        mlflow.log_artifact(cm_norm_path, "confusion_matrix")

        # Classification report
        report = classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            digits=4
        )

        report_path = f"{out_dir}/classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)

        mlflow.log_artifact(report_path, "test_metrics")

        # Save model
        model_path = f"{args.model_root}/resnet18_cifar10_without{args.target}_epoch_{args.epochs}.pth"
        model.save_model(model_path)

        mlflow.log_artifact(model_path, "models")
        mlflow.pytorch.log_model(model.model, f"exlude{args.target}_epoch{args.epochs}")


if __name__ == "__main__":
    main()
