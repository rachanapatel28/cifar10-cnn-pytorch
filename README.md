# CIFAR-10 Image Classification (PyTorch)

A small CNN trained from scratch on CIFAR-10.

![SimpleCNN architecture](docs/architecture.png)

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python ml/train.py --epochs 40 --batch-size 128 --lr 1e-3 --patience 5
```

CIFAR-10 downloads automatically to `./data` on first run. The best checkpoint
(by validation accuracy) is saved to `./checkpoints/best_model.pt`, and
per-epoch loss/accuracy is logged to `./checkpoints/history.json`. Training
stops early if validation accuracy doesn't improve for `--patience` epochs.
See [docs/TRAINING.md](docs/TRAINING.md) for details.

## Evaluate

```bash
python ml/evaluate.py
```

Prints overall test accuracy and per-class accuracy for the saved checkpoint.

## Results

Current best checkpoint: **88.08%** validation accuracy (epoch 39/40).

**Sample predictions** (green border = correct, red = predicted (true)):

![Sample predictions](docs/sample_predictions.png)

**Loss and accuracy over training:**

![Training vs. validation loss](docs/loss_curve.png)
![Training vs. validation accuracy](docs/accuracy_curve.png)

**Confusion matrix** (row-normalized, % of each true class):

![Confusion matrix](docs/confusion_matrix.png)

Regenerate all of these after any training run:

```bash
python scripts/plot_history.py             # loss_curve.png, accuracy_curve.png
python scripts/plot_confusion_matrix.py     # confusion_matrix.png
python scripts/plot_sample_predictions.py   # sample_predictions.png
```

## Web app

A backend + frontend for interactively classifying images with the saved
checkpoint. Run as two separate local servers.

![Web app screenshot](docs/web_app_screenshot.png)

Generate the 10 sample images (one per class) shown on the page, once:

```bash
python scripts/generate_samples.py
```

Start the backend (port 8000):

```bash
uvicorn app:app --app-dir app/backend --port 8000
```

Start the frontend (port 5500), in a second terminal:

```bash
python3 -m http.server 5500 --directory app/frontend
```

Then open http://localhost:5500 in a browser. Drag/drop or upload an image,
or click one of the 10 sample thumbnails, to see the model's predicted class
probabilities as a bar chart.

## Deploy to Azure

The same two pieces run on Azure: the backend as a Docker container on a
Virtual Machine, and the frontend as a static website on Blob Storage.

![Deployed on Azure](docs/azure_deployment.png)

```
browser -> Storage static website (HTML/JS/CSS)
        -> VM:8000 (FastAPI container, image pulled from Azure Container Registry)
```

All resources live in one resource group, so `az group delete` removes everything.

**Config hooks that make this possible**
- `app/frontend/config.js` sets `window.API_BASE` (the backend URL). `app.js` falls back to `http://localhost:8000` if it isn't set.
- The backend reads allowed CORS origins from the `ALLOWED_ORIGINS` env var (comma-separated), defaulting to the localhost frontend.

**Steps** (portal or `az` CLI; names below are placeholders)

1. Create a resource group: `az group create --name rg-cifar10 --location westeurope`
2. Create a container registry (Basic SKU) with the admin user enabled:
   `az acr create -g rg-cifar10 -n <registry> --sku Basic --admin-enabled true`
3. Build the image for x86 (required on Apple Silicon Macs, since Azure VMs are x86) and push it:
   ```bash
   az acr login --name <registry>
   docker build --platform linux/amd64 -t <registry>.azurecr.io/cifar10-backend:v1 .
   docker push <registry>.azurecr.io/cifar10-backend:v1
   ```
4. Create an Ubuntu 24.04 VM with at least 2 vCPUs / 4 GB RAM (torch needs more than 1 GB) and SSH key auth, then open port 8000:
   `az vm open-port -g rg-cifar10 -n vm-cifar10 --port 8000 --priority 1010`
5. SSH in, install Docker (`sudo apt-get install -y docker.io`), log in to the registry with its admin credentials, and run the container:
   ```bash
   sudo docker login <registry>.azurecr.io
   sudo docker run -d --restart unless-stopped --name cifar10 -p 8000:8000 \
     -e ALLOWED_ORIGINS="http://<storage-account>.z6.web.core.windows.net" \
     <registry>.azurecr.io/cifar10-backend:v1
   ```
   Check it at `http://<vm-public-ip>:8000/api/health`.
6. Create a Storage account (LRS) with "Require secure transfer" turned off, and enable **Static website** with `index.html` as the index document.
7. Set `window.API_BASE = "http://<vm-public-ip>:8000"` in `app/frontend/config.js`, then upload the frontend:
   `az storage blob upload-batch --account-name <storage-account> --source app/frontend --destination '$web' --overwrite`
8. Open `http://<storage-account>.z6.web.core.windows.net`. The banner should say "Backend connected".

## Structure

```
ml/
  config.py    training hyperparameters and paths
  dataset.py   CIFAR-10 loading + augmentation
  model.py     SimpleCNN architecture
  train.py     training loop, checkpointing
  evaluate.py  test-set evaluation
Dockerfile, requirements-backend.txt   backend container image (used for the Azure deploy)
app/
  backend/
    app.py               FastAPI inference API (/api/predict, /api/classes, /api/health)
  frontend/
    index.html, style.css, app.js   drag-drop UI + bar chart
    config.js                       backend URL (window.API_BASE)
    samples/                        generated sample images, one per class
scripts/
  generate_samples.py         regenerates app/frontend/samples/ from the CIFAR-10 test set
  plot_history.py             renders docs/loss_curve.png + accuracy_curve.png
  plot_confusion_matrix.py    renders docs/confusion_matrix.png
  plot_sample_predictions.py  renders docs/sample_predictions.png
  plot_architecture.py        renders docs/architecture.png
notebooks/
  inference_visualization.ipynb   notebook version of the same predict + bar-chart flow
docs/
  TRAINING.md              how the training pipeline works
  web_app_screenshot.png   screenshot of the web app in action
  azure_deployment.png     screenshot of the app running on Azure
  *.png                    charts generated by scripts/plot_*.py, embedded above
```
