import argparse
import json
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from config import Config
from dataset import get_dataloaders
from model import SimpleCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for inputs, labels in tqdm(loader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    args = parse_args()
    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.patience is not None:
        cfg.patience = args.patience

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(cfg.data_dir, cfg.batch_size, cfg.num_workers)

    model = SimpleCNN(num_classes=len(cfg.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        print(
            f"Epoch {epoch:2d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        with open(cfg.history_path, "w") as f:
            json.dump(history, f, indent=2)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "val_acc": val_acc, "epoch": epoch},
                cfg.checkpoint_path,
            )
            print(f"  -> saved new best checkpoint (val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                print(f"No improvement in val_acc for {cfg.patience} epochs, stopping early at epoch {epoch}.")
                break

    print(f"Training complete. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
