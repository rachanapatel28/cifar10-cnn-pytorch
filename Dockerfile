FROM python:3.11-slim

WORKDIR /app

# CPU-only torch: much smaller than the default build, and the VM has no GPU
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

# app.py finds the repo root from its own location, so keep the same layout
COPY ml/ ml/
COPY app/backend/ app/backend/
COPY checkpoints/best_model.pt checkpoints/best_model.pt

EXPOSE 8000
CMD ["uvicorn", "app:app", "--app-dir", "app/backend", "--host", "0.0.0.0", "--port", "8000"]