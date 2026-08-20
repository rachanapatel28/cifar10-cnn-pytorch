import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.join(REPO_ROOT, "ml")
sys.path.insert(0, ML_DIR)

from config import Config
from dataset import get_dataloaders
from model import SimpleCNN

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def main():
    cfg = Config()
    cfg.data_dir = os.path.join(REPO_ROOT, "data")
    cfg.checkpoint_path = os.path.join(REPO_ROOT, "checkpoints", "best_model.pt")
    device = torch.device(cfg.device)

    _, test_loader = get_dataloaders(cfg.data_dir, cfg.batch_size, cfg.num_workers)

    model = SimpleCNN(num_classes=len(cfg.classes)).to(device)
    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    n = len(cfg.classes)
    matrix = np.zeros((n, n), dtype=int)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(dim=1)
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                matrix[t, p] += 1

    pct = matrix / matrix.sum(axis=1, keepdims=True) * 100
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)

    fig, ax = plt.subplots(figsize=(7.5, 7), facecolor=SURFACE)
    im = ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cfg.classes, rotation=45, ha="right", color=TEXT_SECONDARY)
    ax.set_yticklabels(cfg.classes, color=TEXT_SECONDARY)
    ax.set_xlabel("Predicted class", color=TEXT_SECONDARY)
    ax.set_ylabel("True class", color=TEXT_SECONDARY)
    ax.set_title("Confusion Matrix (% of true class)", color=TEXT_PRIMARY, fontsize=13, fontweight="bold")

    for i in range(n):
        for j in range(n):
            value = pct[i, j]
            ink = "#ffffff" if value > 55 else TEXT_PRIMARY
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", color=ink, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% of true class", color=TEXT_SECONDARY)
    cbar.ax.yaxis.set_tick_params(color=TEXT_SECONDARY)
    plt.setp(cbar.ax.get_yticklabels(), color=TEXT_SECONDARY)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(REPO_ROOT, "docs", "confusion_matrix.png")
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
