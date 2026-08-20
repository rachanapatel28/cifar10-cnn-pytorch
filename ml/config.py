from dataclasses import dataclass

import torch


@dataclass
class Config:
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    checkpoint_path: str = "./checkpoints/best_model.pt"
    history_path: str = "./checkpoints/history.json"

    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 5e-4
    num_workers: int = 2
    patience: int = 5

    classes: tuple = (
        "plane", "car", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    )

    device: str = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
