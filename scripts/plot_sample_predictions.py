import os
import random
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.join(REPO_ROOT, "ml")
sys.path.insert(0, ML_DIR)

from config import Config
from dataset import CIFAR10_MEAN, CIFAR10_STD
from model import SimpleCNN

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

ROWS, COLS = 4, 6
SEED = 42


def main():
    cfg = Config()
    cfg.data_dir = os.path.join(REPO_ROOT, "data")
    cfg.checkpoint_path = os.path.join(REPO_ROOT, "checkpoints", "best_model.pt")
    device = torch.device(cfg.device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_set = torchvision.datasets.CIFAR10(root=cfg.data_dir, train=False, download=True, transform=transform)
    raw_set = torchvision.datasets.CIFAR10(root=cfg.data_dir, train=False, download=True)

    model = SimpleCNN(num_classes=len(cfg.classes)).to(device)
    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    random.seed(SEED)
    indices = random.sample(range(len(test_set)), ROWS * COLS)

    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 1.8, ROWS * 2.1), facecolor=SURFACE)

    with torch.no_grad():
        for ax, idx in zip(axes.flat, indices):
            img_tensor, true_idx = test_set[idx]
            raw_img, _ = raw_set[idx]

            logits = model(img_tensor.unsqueeze(0).to(device))
            pred_idx = int(logits.argmax(dim=1).item())
            correct = pred_idx == true_idx
            color = STATUS_GOOD if correct else STATUS_CRITICAL

            ax.imshow(raw_img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(2.5)

            mark = "correct" if correct else "wrong"
            title = cfg.classes[pred_idx] if correct else f"{cfg.classes[pred_idx]} ({cfg.classes[true_idx]})"
            ax.set_title(title, fontsize=8.5, color=color, fontweight="bold" if not correct else "normal")

    fig.suptitle(
        "Sample Predictions — green border/label = correct, red = predicted (true)",
        color=TEXT_PRIMARY, fontsize=11, fontweight="bold", y=0.995,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = os.path.join(REPO_ROOT, "docs", "sample_predictions.png")
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
