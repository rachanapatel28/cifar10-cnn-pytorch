# How This Project Works

This document explains the data flow and training process for the CIFAR-10
image classifier in [ml/](../ml/).

## 1. Overview

The pipeline has four stages, each in its own module:

```
dataset.py  ->  model.py  ->  train.py  ->  evaluate.py
(load data)     (define      (fit the       (measure
                 the net)     net)            accuracy)
```

`config.py` holds every tunable value (batch size, learning rate, epochs,
paths, device) so the other modules stay free of hardcoded constants.

## 2. Data flow ([dataset.py](../ml/dataset.py))

CIFAR-10 is 60,000 32x32 RGB images across 10 classes (50k train / 10k test).
`get_dataloaders()` builds two `DataLoader`s:

| Split | Transform | Why |
|---|---|---|
| Train | `RandomCrop(32, padding=4)` → `RandomHorizontalFlip()` → `ToTensor()` → `Normalize()` | Augmentation reduces overfitting by showing a slightly different crop/flip of each image every epoch |
| Test  | `ToTensor()` → `Normalize()` | No augmentation — evaluation must be deterministic |

`Normalize()` uses CIFAR-10's per-channel mean/std so pixel values are
roughly zero-centered, which helps the optimizer converge faster.

Each `DataLoader` yields batches of shape `(batch_size, 3, 32, 32)` for images
and `(batch_size,)` for integer labels, shuffled every epoch on the train
side only.

## 3. Model architecture ([model.py](../ml/model.py))

`SimpleCNN` is a plain convolutional network — no pretrained weights, trained
from scratch. It has two parts:

**`features`** — three convolutional blocks that shrink the spatial size
while growing the channel depth, so the network learns increasingly abstract
patterns (edges → textures → object parts):

```
input            3 x 32 x 32
conv block 1    32 x 32 x 32   (2 conv layers, BatchNorm, ReLU)
maxpool         32 x 16 x 16
conv block 2    64 x 16 x 16   (2 conv layers, BatchNorm, ReLU)
maxpool         64 x  8 x  8
conv block 3   128 x  8 x  8   (1 conv layer, BatchNorm, ReLU)
maxpool        128 x  4 x  4
```

- **Conv2d** layers extract local spatial features with learned filters.
- **BatchNorm2d** normalizes activations between layers, which stabilizes and
  speeds up training.
- **ReLU** is the nonlinearity that lets the network model more than linear
  functions.
- **MaxPool2d** halves height and width each time, discarding position detail
  the network doesn't need while keeping the strongest activations.

**`classifier`** — flattens the final `128 x 4 x 4` feature map (2048 values)
and maps it to 10 class scores:

```
Flatten          2048
Dropout(0.4)
Linear         2048 -> 256
ReLU
Dropout(0.4)
Linear          256 -> 10      (raw class scores / logits)
```

**Dropout** randomly zeroes 40% of activations during training only, forcing
the network to not rely on any single neuron — another anti-overfitting
measure. It's disabled automatically during evaluation (`model.eval()`).

The output is 10 raw scores (logits), one per class — not yet probabilities.

## 4. Training loop ([train.py](../ml/train.py))

For each epoch, `run_epoch()` is called twice: once with `train=True` on the
training set, once with `train=False` on the validation (test) set.

### The training step, per batch

1. **Forward pass** — images go through the model, producing logits.
2. **Loss** — `CrossEntropyLoss` compares logits to true labels. It applies
   softmax internally and penalizes confident wrong predictions more than
   uncertain ones.
3. **Backward pass** — `loss.backward()` computes the gradient of the loss
   with respect to every weight in the network (backpropagation).
4. **Optimizer step** — `Adam` updates every weight a small amount in the
   direction that reduces the loss, scaled by the learning rate.
5. **Zero gradients** — gradients are reset before the next batch so they
   don't accumulate across batches.

### The validation step, per batch

Same forward pass and loss calculation, but wrapped in `torch.no_grad()` (no
gradients computed — cheaper) and no optimizer step (weights aren't updated).
This measures how well the model generalizes to unseen data.

### Per-epoch bookkeeping

- **Loss/accuracy** are accumulated across all batches and averaged.
- **`CosineAnnealingLR`** gradually lowers the learning rate over the course
  of training, following a cosine curve from the initial `lr` down to ~0.
  Large steps early on find a good region of the loss landscape fast; small
  steps later fine-tune without overshooting.
- **Checkpointing** — after every epoch, if validation accuracy improved on
  this epoch, the model's weights are saved to
  `checkpoints/best_model.pt`. This means the saved model is always the best
  one seen so far, even if later epochs overfit and validation accuracy
  drops.
- **Early stopping** — if validation accuracy fails to improve for
  `patience` consecutive epochs (default 5), training stops early rather
  than running the full `epochs` budget. The best checkpoint seen so far is
  unaffected either way; this just avoids spending time on epochs that are
  no longer helping (or that are actively overfitting).
- **History logging** — every epoch's `train_loss`, `train_acc`, `val_loss`,
  and `val_acc` are appended to `checkpoints/history.json` (overwritten after
  each epoch, so a partial/interrupted run still leaves usable data). Load
  it later to plot loss/accuracy curves, e.g.:
  ```python
  import json
  history = json.load(open("checkpoints/history.json"))
  epochs = [h["epoch"] for h in history]
  train_loss = [h["train_loss"] for h in history]
  val_loss = [h["val_loss"] for h in history]
  ```

## 5. Evaluation ([evaluate.py](../ml/evaluate.py))

Loads the saved checkpoint, sets the model to `eval()` mode (disables
Dropout, freezes BatchNorm statistics), and runs a single pass over the test
set with no gradient tracking. Reports:

- Overall test accuracy
- Per-class accuracy (useful for spotting which classes the model confuses —
  e.g. cat vs. dog is a classic CIFAR-10 weak point)

## 6. Key hyperparameters ([config.py](../ml/config.py))

| Parameter | Default | Effect |
|---|---|---|
| `batch_size` | 128 | Images processed per gradient update. Larger = smoother gradients, more memory. |
| `epochs` | 20 | Full passes over the training set. |
| `lr` | 1e-3 | Adam's initial step size. |
| `weight_decay` | 5e-4 | L2 regularization — penalizes large weights to reduce overfitting. |
| `patience` | 5 | Epochs to wait for a validation accuracy improvement before stopping early. |
| `device` | mps/cuda/cpu | Auto-detected; MPS is used on this Mac's Apple GPU. |

## 7. Reproducing a run

```bash
source .venv/bin/activate
python ml/train.py --epochs 20 --batch-size 128 --lr 1e-3 --patience 5
python ml/evaluate.py
```

Expect train accuracy to climb steadily above validation accuracy as
training progresses — a growing gap between the two is the signature of
overfitting, which is exactly what augmentation, dropout, and weight decay
are here to slow down. With early stopping in place, `--epochs` can be set
higher than you expect to need (e.g. 40-50) — training will stop on its own
once validation accuracy plateaus for `--patience` epochs, rather than
running (and potentially overfitting further) for the full budget.
