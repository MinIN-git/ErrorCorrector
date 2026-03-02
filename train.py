import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    classification_report
)
from PIL import Image
from clearml import Task, OutputModel

from src.resnet import ResNet18
from src.dataloader import CIFARCustom
from src.draw_figures import plot_loss, plot_accuracy, plot_confusion_matrix


# api {
#   web_server: http://172.25.249.11:5000
#   api_server: http://172.25.249.11:8008
#   files_server: http://172.25.249.11:8081
#   credentials {
#     "access_key" = "D0D2WJUUD4ZD96NRHIMM9IQGKW9EUA"
#     "secret_key" = "cVNwKiwF9nt3jIlJTfx_IwXMRpdY-1ZJAecPzahigGbc1qILKjIiXbLtwJyP9U-JkyE"
#   }
# }


def fig2img(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


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
    print(device)
    generator = torch.Generator().manual_seed(args.seed)

    # === ClearML Task Initialization ===
    task = Task.init(
        project_name="ResNet18_CIFAR10_Exclude_Class",
        task_name=f"exclude_class_{args.target}_epoch_{args.epochs}",
        output_uri=args.output_root
    )
    task.connect(vars(args))
    logger = task.get_logger()

    # === Folders ===
    out_dir = f"{args.output_root}/{args.target}/train"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.model_root, exist_ok=True)

    # === Transforms ===
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # === Dataset & Splits ===
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

    logger.report_text(f"Dataset sizes: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

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

    # === Model ===
    model = ResNet18(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.lr,
        num_epochs=args.epochs
    )

    history = model.train()

    # === Log training metrics ===
    for epoch in range(len(history["train_loss"])):
        logger.report_scalar("Training Loss", "train_loss", history["train_loss"][epoch], iteration=epoch)
        logger.report_scalar("Validation Loss", "val_loss", history["val_loss"][epoch], iteration=epoch)
        logger.report_scalar("Validation Accuracy", "val_accuracy", history["val_accuracy"][epoch], iteration=epoch)
        logger.report_scalar("Validation F1 Macro", "val_f1_macro", history["val_f1_macro"][epoch], iteration=epoch)

    plot_train_loss = plot_loss(history["train_loss"], history["val_loss"])
    plot_val_acc = plot_accuracy(history["val_accuracy"])
    plot_val_f1 = plot_accuracy(history["val_f1_macro"], "F1 Macro")


    logger.report_image("Loss", "loss_plot", iteration=0, image=fig2img(plot_train_loss))
    logger.report_image("Accuracy", "accuracy_plot", iteration=0, image=fig2img(plot_val_acc))
    logger.report_image("F1 Macro", "f1_macro_plot", iteration=0, image=fig2img(plot_val_f1))


    # === TEST METRICS ===
    class_names = [str(i) for i in range(10) if i != args.target]
    y_gt, y_pred = collect_predictions(model.model, test_loader, device)

    test_acc = accuracy_score(y_gt, y_pred)
    test_f1_macro = f1_score(y_gt, y_pred, average="macro")
    test_f1_weighted = f1_score(y_gt, y_pred, average="weighted")

    last_epoch = len(history["train_loss"]) - 1
    logger.report_scalar("Test Metrics", "accuracy", test_acc, iteration=last_epoch)
    logger.report_scalar("Test Metrics", "f1_macro", test_f1_macro, iteration=last_epoch)
    logger.report_scalar("Test Metrics", "f1_weighted", test_f1_weighted, iteration=last_epoch)

    # per-class f1
    f1_per_class = f1_score(y_gt, y_pred, average=None)
    for cls, f1 in zip(class_names, f1_per_class):
        logger.report_scalar("Test F1 per Class", cls, f1, iteration=last_epoch)

    plot_cfm = plot_confusion_matrix(y_gt, y_pred, class_names)
    plot_cfm_n = plot_confusion_matrix(y_gt, y_pred, class_names, normalize="all")

    logger.report_image("Confusion Matrix", "raw", image=fig2img(plot_cfm))
    logger.report_image("Confusion Matrix", "normalized", image=fig2img(plot_cfm_n))

    # Classification report
    report_dict = classification_report(
        y_gt,
        y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()

    task.upload_artifact(
        name="classification_report",
        artifact_object=report_df
    )

    # Save model
    model_path = f"{args.model_root}/resnet18_cifar10_without{args.target}_epoch_{args.epochs}.pth"
    model.save_model(model_path)
    output_model = OutputModel(
        task=task,
        name=f"resnet18_exclude_{args.target}_epoch_{args.epochs}"
    )

    output_model = OutputModel(
        task=task,
        name=f"resnet18_exclude_{args.target}_epoch_{args.epochs}"
    )

    output_model.update_weights(model_path)

    output_model.set_metadata("epochs", args.epochs)
    output_model.set_metadata("excluded_class", args.target)


if __name__ == "__main__":
    main()
