import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(REPO_ROOT, "docs", "architecture.png")

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
CONV_BORDER = "#2a78d6"
FC_BORDER = "#eb6834"
NEUTRAL_BORDER = "#898781"
NEUTRAL_FILL = "#f0efec"

STAGES = [
    ("Input", "3 × 32 × 32", "neutral"),
    ("Conv 3×3 → BN → ReLU  (×2, 32 filters)", "32 × 32 × 32", "conv"),
    ("MaxPool 2×2", "32 × 16 × 16", "neutral"),
    ("Conv 3×3 → BN → ReLU  (×2, 64 filters)", "64 × 16 × 16", "conv"),
    ("MaxPool 2×2", "64 × 8 × 8", "neutral"),
    ("Conv 3×3 → BN → ReLU  (128 filters)", "128 × 8 × 8", "conv"),
    ("MaxPool 2×2", "128 × 4 × 4", "neutral"),
    ("Flatten", "2,048", "neutral"),
    ("Dropout(0.4) → Linear → ReLU", "256", "fc"),
    ("Dropout(0.4) → Linear", "10 (logits)", "fc"),
]


def main():
    fig_height = len(STAGES) * 0.95 + 1
    fig, ax = plt.subplots(figsize=(6.5, fig_height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(STAGES))
    ax.axis("off")

    box_w, box_h = 8, 0.62
    x0 = (10 - box_w) / 2

    for i, (label, shape, kind) in enumerate(STAGES):
        y = len(STAGES) - 1 - i
        if kind == "conv":
            face, edge = CONV_BORDER, CONV_BORDER
            alpha = 0.12
        elif kind == "fc":
            face, edge = FC_BORDER, FC_BORDER
            alpha = 0.12
        else:
            face, edge = NEUTRAL_FILL, NEUTRAL_BORDER
            alpha = 1.0

        box = FancyBboxPatch(
            (x0, y + (1 - box_h) / 2), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.5, edgecolor=edge, facecolor=face, alpha=alpha,
        )
        ax.add_patch(box)
        ax.text(5, y + 0.5 + 0.08, label, ha="center", va="center", color=TEXT_PRIMARY, fontsize=9.5)
        ax.text(5, y + 0.5 - 0.16, shape, ha="center", va="center", color=TEXT_SECONDARY, fontsize=8.5)

        if i < len(STAGES) - 1:
            bottom = y + (1 - box_h) / 2
            top_next = (y - 1) + (1 - box_h) / 2 + box_h
            ax.annotate(
                "", xy=(5, top_next), xytext=(5, bottom),
                arrowprops=dict(arrowstyle="-|>", color=NEUTRAL_BORDER, linewidth=1.2),
            )

    legend_items = [
        ("Conv block (learned features)", CONV_BORDER),
        ("Pool / reshape (no params)", NEUTRAL_BORDER),
        ("Classifier head (learned)", FC_BORDER),
    ]
    for i, (label, color) in enumerate(legend_items):
        ly = len(STAGES) + 0.15
        lx = 0.3 + i * 3.3
        ax.add_patch(plt.Rectangle((lx, ly), 0.3, 0.2, facecolor=color, edgecolor=color, alpha=0.8))
        ax.text(lx + 0.45, ly + 0.1, label, ha="left", va="center", color=TEXT_SECONDARY, fontsize=7.5)

    ax.set_ylim(0, len(STAGES) + 0.5)
    fig.suptitle("SimpleCNN Architecture", color=TEXT_PRIMARY, fontsize=13, fontweight="bold", y=0.99)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, facecolor=SURFACE)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
