import argparse
from pathlib import Path

import torch
import numpy as np
import mlflow
import mlflow.pytorch

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dotenv import load_dotenv
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report
)

from src.resnet import ResNet18
from src.dataloader import DATASET_REGISTRY, build_excluded_dataset, get_dataset_config
from src.draw_figures import plot_loss, plot_accuracy, plot_confusion_matrix
from src.inference import collect_predictions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet18 on an image dataset with excluded target class"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=sorted(DATASET_REGISTRY),
        help="Dataset registry key"
    )
    parser.add_argument(
        "-t", "--target",
        type=int,
        required=True,
        help="Target class to exclude"
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
    dataset_config = get_dataset_config(args.dataset)
    if args.target < 0 or args.target >= dataset_config.num_classes:
        raise ValueError(
            f"--target должен быть в диапазоне 0..{dataset_config.num_classes - 1} "
            f"для датасета {args.dataset}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(args.seed)

    load_dotenv()
    mlflow.set_experiment(f"ResNet18_{args.dataset}_Exclude_Class")

    with mlflow.start_run(run_name=f"{args.dataset}_exclude_class_{args.target}_epoch_{args.epochs}"):

        # Log hyperparameters
        mlflow.log_params(vars(args))
        mlflow.log_param("device", device.type)

        # Folders
        out_dir = Path(args.output_root) / args.dataset / str(args.target) / "train"
        model_root = Path(args.model_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_root.mkdir(parents=True, exist_ok=True)

        # Transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_config.mean, dataset_config.std)
        ])

        # Dataset & splits
        dataset = build_excluded_dataset(
            name=args.dataset,
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
            "test_size": len(test_set),
            "dataset": args.dataset,
            "num_classes": dataset_config.num_classes,
            "input_channels": dataset_config.input_channels,
            "label_contract": "original_dataset_labels_with_empty_excluded_class",
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
            num_epochs=args.epochs,
            num_classes=dataset_config.num_classes,
            input_channels=dataset_config.input_channels,
        )

        history = model.train()

        # Log training metrics
        for epoch in range(len(history["train_loss"])):
            mlflow.log_metric("train_loss", history["train_loss"][epoch], step=epoch)
            mlflow.log_metric("val_loss", history["val_loss"][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history["val_accuracy"][epoch], step=epoch)
            mlflow.log_metric("val_f1_macro", history["val_f1_macro"][epoch], step=epoch)

        # Plots
        loss_path = out_dir / "loss.png"
        acc_path = out_dir / "accuracy.png"
        f1_path = out_dir / "f1_macro.png"

        plot_loss(history["train_loss"], history["val_loss"], path=loss_path)
        plot_accuracy(history["val_accuracy"], path=acc_path)
        plot_accuracy(history["val_f1_macro"], "F1 Macro", f1_path)

        mlflow.log_artifact(str(loss_path), "plots")
        mlflow.log_artifact(str(acc_path), "plots")
        mlflow.log_artifact(str(f1_path), "plots")

        # TEST METRICS
        all_labels = list(range(dataset_config.num_classes))
        class_names = [
            f"{class_id}: {name}"
            for class_id, name in enumerate(dataset.classes)
        ]

        y_test, y_pred = collect_predictions(model.model, test_loader, device)

        test_acc = accuracy_score(y_test, y_pred)
        test_f1_macro = f1_score(y_test, y_pred, labels=all_labels, average="macro", zero_division=0)
        test_f1_weighted = f1_score(y_test, y_pred, labels=all_labels, average="weighted", zero_division=0)

        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_macro", test_f1_macro)
        mlflow.log_metric("test_f1_weighted", test_f1_weighted)

        # per-class f1
        f1_per_class = f1_score(y_test, y_pred, labels=all_labels, average=None, zero_division=0)
        for cls, f1 in zip(all_labels, f1_per_class):
            mlflow.log_metric(f"test_f1_class_{cls}", f1)

        # Confusion matrices
        cm_raw_path = out_dir / "confusion_matrix.png"
        cm_norm_path = out_dir / "confusion_matrix_normalized.png"

        plot_confusion_matrix(
            y_pred,
            y_test,
            class_names=class_names,
            path=cm_raw_path,
            labels=all_labels,
            normalize=None,
        )
        plot_confusion_matrix(
            y_pred,
            y_test,
            class_names=class_names,
            path=cm_norm_path,
            labels=all_labels,
            normalize="true",
            title="Confusion Matrix (Normalized)",
        )

        mlflow.log_artifact(str(cm_raw_path), "confusion_matrix")
        mlflow.log_artifact(str(cm_norm_path), "confusion_matrix")

        # Classification report
        report = classification_report(
            y_test,
            y_pred,
            labels=all_labels,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )

        report_path = out_dir / "classification_report.txt"
        report_path.write_text(report)

        mlflow.log_artifact(str(report_path), "test_metrics")

        # Save model
        model_path = model_root / f"resnet18_{args.dataset}_without{args.target}_epoch_{args.epochs}.pth"
        model.save_model(model_path)

        mlflow.log_artifact(str(model_path), "models")
        mlflow.pytorch.log_model(model.model, f"exlude{args.target}_epoch{args.epochs}")


if __name__ == "__main__":
    main()
