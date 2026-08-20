import torch

from config import Config
from dataset import get_dataloaders
from model import SimpleCNN


def main():
    cfg = Config()
    device = torch.device(cfg.device)

    _, test_loader = get_dataloaders(cfg.data_dir, cfg.batch_size, cfg.num_workers)

    model = SimpleCNN(num_classes=len(cfg.classes)).to(device)
    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_correct = [0] * len(cfg.classes)
    class_total = [0] * len(cfg.classes)
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for label, pred in zip(labels, preds):
                class_total[label] += 1
                class_correct[label] += int(label == pred)

    print(f"Checkpoint from epoch {checkpoint['epoch']}, val_acc at save time={checkpoint['val_acc']:.4f}")
    print(f"Test accuracy: {correct / total:.4f}\n")
    print("Per-class accuracy:")
    for i, name in enumerate(cfg.classes):
        acc = class_correct[i] / class_total[i] if class_total[i] else 0.0
        print(f"  {name:>6}: {acc:.4f}")


if __name__ == "__main__":
    main()
