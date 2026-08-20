import io
import os
import sys
from contextlib import asynccontextmanager

import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ML_DIR = os.path.join(REPO_ROOT, "ml")
sys.path.insert(0, ML_DIR)

from config import Config
from dataset import CIFAR10_MEAN, CIFAR10_STD
from model import SimpleCNN

CONFIDENCE_THRESHOLD = 0.5

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = Config()
    cfg.checkpoint_path = os.path.join(REPO_ROOT, "checkpoints", "best_model.pt")
    device = torch.device(cfg.device)

    model = SimpleCNN(num_classes=len(cfg.classes)).to(device)
    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    state["cfg"] = cfg
    state["device"] = device
    state["model"] = model
    state["transform"] = transform
    state["checkpoint_epoch"] = checkpoint["epoch"]
    state["val_acc"] = checkpoint["val_acc"]

    yield
    state.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": str(state["device"]),
        "checkpoint_epoch": state["checkpoint_epoch"],
        "val_acc": state["val_acc"],
    }


@app.get("/api/classes")
def classes():
    return {"classes": list(state["cfg"].classes)}


@app.post("/api/predict")
def predict(file: UploadFile = File(...)):
    data = file.file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read image — file may be corrupt or not a supported image format.")

    cfg = state["cfg"]
    device = state["device"]
    model = state["model"]

    tensor = state["transform"](img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze(0).cpu().tolist()

    pred_idx = max(range(len(probs)), key=lambda i: probs[i])
    confidence = probs[pred_idx]

    return {
        "predicted_class": cfg.classes[pred_idx],
        "confidence": confidence,
        "is_uncertain": confidence < CONFIDENCE_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "probabilities": dict(zip(cfg.classes, probs)),
    }
