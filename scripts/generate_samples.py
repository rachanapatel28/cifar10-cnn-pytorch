import os
import sys

import torchvision

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.join(REPO_ROOT, "ml")
sys.path.insert(0, ML_DIR)

from config import Config


def main():
    cfg = Config()
    data_dir = os.path.join(REPO_ROOT, "data")
    out_dir = os.path.join(REPO_ROOT, "app", "frontend", "samples")
    os.makedirs(out_dir, exist_ok=True)

    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)

    remaining = set(range(len(cfg.classes)))
    for img, label_idx in test_set:
        if label_idx not in remaining:
            continue
        class_name = cfg.classes[label_idx]
        img.save(os.path.join(out_dir, f"{class_name}.png"))
        remaining.discard(label_idx)
        if not remaining:
            break

    print(f"Saved {len(cfg.classes)} sample images to {out_dir}")


if __name__ == "__main__":
    main()
