import json
import os

import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_PATH = os.path.join(REPO_ROOT, "checkpoints", "history.json")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"


def plot_metric(epochs, train_values, val_values, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)

    ax.plot(epochs, train_values, color=TRAIN_COLOR, linewidth=2, label=f"Train {ylabel.lower()}", zorder=3)
    ax.plot(epochs, val_values, color=VAL_COLOR, linewidth=2, label=f"Validation {ylabel.lower()}", zorder=3)

    ax.set_xlabel("Epoch", color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=13, fontweight="bold")
    ax.tick_params(colors=TEXT_SECONDARY)

    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    with open(HISTORY_PATH) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]

    plot_metric(
        epochs,
        [h["train_loss"] for h in history],
        [h["val_loss"] for h in history],
        ylabel="Loss",
        title="Training vs. Validation Loss",
        out_path=os.path.join(DOCS_DIR, "loss_curve.png"),
    )
    plot_metric(
        epochs,
        [h["train_acc"] for h in history],
        [h["val_acc"] for h in history],
        ylabel="Accuracy",
        title="Training vs. Validation Accuracy",
        out_path=os.path.join(DOCS_DIR, "accuracy_curve.png"),
    )


if __name__ == "__main__":
    main()
