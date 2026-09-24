#!/usr/bin/env python3
"""
MAIN122 — PHASE 1 / EXPERIMENT 1
RAW TOKENS -> CNN -> LEARNED DOCUMENT VECTOR

Goal
----
Test whether a CNN trained directly on the raw token-ID sequence can learn a
useful document representation for human-vs-machine classification.

Pipeline
--------
token IDs
   -> embedding
   -> multi-kernel 1D CNN
   -> global max/mean pooling
   -> latent vector z
   -> classification head

The learned validation/train embeddings are also evaluated independently with:
  1. Linear SVM
  2. Logistic Regression

This lets us distinguish:
  - a useful learned representation from
  - a useful neural classifier head.

Canonical split
---------------
Exactly the Main64 split:
    train_test_split(all_idx, test_size=0.20, stratify=y, random_state=42)

No Kaggle submission is performed.

Outputs
-------
main122_best.pt
main122_embeddings.npz
main122_report.json

The embedding NPZ contains:
    train_embeddings
    val_embeddings
    y_train
    y_val
"""

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42

DATA_DIR = Path("/home/manu/Desktop/ML4CPMS/project_1")
TRAIN_PATH = DATA_DIR / "train.json"

MAX_LEN = 384
EMBED_DIM = 128
LATENT_DIM = 128

CNN_CHANNELS = 128
KERNELS = (3, 5, 7, 9)

BATCH_SIZE = 64
EPOCHS = 15
LR = 2e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.25
PATIENCE = 4

NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior is preferred for experiment comparison.
    # This can be slightly slower.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)


# ---------------------------------------------------------------------
# JSON / JSONL loader
# ---------------------------------------------------------------------

def load_json_records(path):
    """
    Supports both a JSON array and JSONL.
    Expected fields:
        text: list[int]
        label: A/B or 0/1
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    raw = raw.strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            # Handle a possible wrapper such as {"data": [...]}
            for key in ("data", "train", "records"):
                if key in obj and isinstance(obj[key], list):
                    return obj[key]
            return [obj]
    except json.JSONDecodeError:
        pass

    records = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


records = load_json_records(TRAIN_PATH)

texts = [r["text"] for r in records]

raw_labels = [r.get("label", r.get("target", r.get("y"))) for r in records]

if any(v is None for v in raw_labels):
    raise ValueError("Could not find labels in train.json")

unique_labels = list(dict.fromkeys(raw_labels))

if set(unique_labels).issubset({"A", "B"}):
    label_map = {"A": 0, "B": 1}
elif set(unique_labels).issubset({0, 1}):
    label_map = {0: 0, 1: 1}
elif set(unique_labels).issubset({"0", "1"}):
    label_map = {"0": 0, "1": 1}
else:
    raise ValueError(f"Unexpected labels: {unique_labels}")

y = np.asarray([label_map[v] for v in raw_labels], dtype=np.int64)

# ---------------------------------------------------------------------
# Canonical Main64 split — DO NOT SORT
# ---------------------------------------------------------------------

all_idx = np.arange(len(texts))

train_idx, val_idx = train_test_split(
    all_idx,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

train_idx = np.asarray(train_idx)
val_idx = np.asarray(val_idx)

print("=" * 78)
print("MAIN122 — PHASE 1 / EXPERIMENT 1")
print("RAW TOKENS -> CNN -> LATENT VECTOR")
print("=" * 78)
print(f"Device: {DEVICE}")
print(f"Dataset: {len(texts)}")
print(f"Train: {len(train_idx)}  Val: {len(val_idx)}")
print(
    f"Class counts train={np.bincount(y[train_idx], minlength=2)} "
    f"val={np.bincount(y[val_idx], minlength=2)}"
)

# Verify against the authoritative Main64 validation labels if present.
main64_val_labels_path = DATA_DIR / "main64_val_labels.npy"

if main64_val_labels_path.exists():
    expected_val_y = np.load(main64_val_labels_path)
    if not np.array_equal(y[val_idx], expected_val_y):
        raise RuntimeError(
            "Canonical Main64 validation-label check FAILED. "
            "Do not continue; validation ordering does not match Main64."
        )
    print("Canonical Main64 validation-label check: PASSED")
else:
    print("Main64 validation-label artifact not found; split itself is canonical.")


# ---------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------
#
# Token IDs are already integers. We preserve their identity but reserve:
#   0 = padding
#   1 = unknown/out-of-range
#
# We build vocabulary size from the maximum token ID seen in TRAIN ONLY.
# This avoids learning vocabulary information from validation.
# ---------------------------------------------------------------------

train_token_max = 0
for i in train_idx:
    seq = texts[i]
    if seq:
        train_token_max = max(train_token_max, max(int(t) for t in seq))

VOCAB_SIZE = train_token_max + 2
UNK_ID = 1

print(f"Train-only max token ID: {train_token_max}")
print(f"Vocabulary size: {VOCAB_SIZE}")


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class TokenDataset(Dataset):
    def __init__(self, texts, indices, labels, max_len, vocab_size):
        self.texts = texts
        self.indices = indices
        self.labels = labels
        self.max_len = max_len
        self.vocab_size = vocab_size

    def _prepare(self, seq):
        seq = [int(t) for t in seq]

        # Same truncation convention used in our earlier sequence models:
        # preserve both beginning and ending context for long documents.
        if len(seq) > self.max_len:
            half = self.max_len // 2
            seq = seq[:half] + seq[-half:]

        out = np.zeros(self.max_len, dtype=np.int64)

        for j, token in enumerate(seq):
            if token < 0 or token >= self.vocab_size:
                out[j] = UNK_ID
            else:
                # Token ID 0 is allowed as a real token only if it appears;
                # padding is also zero. The attention/mask is therefore based
                # on sequence length, not token value.
                out[j] = token

        length = len(seq)
        return out, length

    def __getitem__(self, k):
        idx = self.indices[k]
        tokens, length = self._prepare(self.texts[idx])
        return (
            torch.from_numpy(tokens),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

    def __len__(self):
        return len(self.indices)


train_ds = TokenDataset(texts, train_idx, y, MAX_LEN, VOCAB_SIZE)
val_ds = TokenDataset(texts, val_idx, y, MAX_LEN, VOCAB_SIZE)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# ---------------------------------------------------------------------
# CNN encoder
# ---------------------------------------------------------------------

class CNNEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        channels=128,
        kernels=(3, 5, 7, 9),
        latent_dim=128,
        dropout=0.25,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0,
        )

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    embed_dim,
                    channels,
                    kernel_size=k,
                    padding=k // 2,
                )
                for k in kernels
            ]
        )

        # Each kernel contributes mean + max pooled channels.
        pooled_dim = len(kernels) * channels * 2

        self.projection = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim),
        )

    def forward(self, tokens, lengths=None):
        x = self.embedding(tokens)          # B x L x E
        x = x.transpose(1, 2)               # B x E x L

        pooled_parts = []

        # Mask padded positions before pooling.
        if lengths is not None:
            positions = torch.arange(
                x.shape[-1],
                device=x.device
            ).unsqueeze(0)
            mask = positions >= lengths.unsqueeze(1)
        else:
            mask = tokens.eq(0)

        for conv in self.convs:
            h = F.gelu(conv(x))             # B x C x L

            h_mean = h.masked_fill(
                mask.unsqueeze(1), 0.0
            )

            denom = (~mask).sum(dim=1).clamp_min(1).unsqueeze(1)
            h_mean = h_mean.sum(dim=2) / denom

            h_max = h.masked_fill(
                mask.unsqueeze(1), -1e9
            ).max(dim=2).values

            pooled_parts.extend([h_mean, h_max])

        pooled = torch.cat(pooled_parts, dim=1)
        z = self.projection(pooled)

        # Normalize only for the representation output. The classifier uses
        # the normalized vector as well, which makes the latent geometry
        # easier to compare across experiments.
        z = nn.functional.normalize(z, p=2, dim=1)

        return z


class CNNClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, tokens, lengths):
        z = self.encoder(tokens, lengths)
        logits = self.classifier(z).squeeze(1)
        return logits, z


model = CNNClassifier(
    CNNEncoder(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        channels=CNN_CHANNELS,
        kernels=KERNELS,
        latent_dim=LATENT_DIM,
        dropout=DROPOUT,
    )
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model parameters: {n_params:,}")
print(f"Max sequence length: {MAX_LEN}")
print(f"Embedding dimension: {EMBED_DIM}")
print(f"Latent dimension: {LATENT_DIM}")
print(f"CNN kernels: {KERNELS}")


# ---------------------------------------------------------------------
# Loss / optimizer
# ---------------------------------------------------------------------

train_counts = np.bincount(y[train_idx], minlength=2)

# Positive class weight because B is the majority class, but the weighting
# is deliberately mild: the metric is accuracy, and we do not want the
# network to over-correct toward A.
pos_weight_value = float(train_counts[0] / max(train_counts[1], 1))

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(pos_weight_value, device=DEVICE)
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1,
)


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_neural(model, loader):
    model.eval()

    losses = []
    probs = []
    labels = []

    for tokens, lengths, target in loader:
        tokens = tokens.to(DEVICE, non_blocking=True)
        lengths = lengths.to(DEVICE, non_blocking=True)
        target = target.to(DEVICE, non_blocking=True)

        logits, _ = model(tokens, lengths)
        loss = criterion(logits, target)

        losses.append(loss.item())

        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(target.cpu().numpy())

    probs = np.concatenate(probs)
    labels = np.concatenate(labels).astype(np.int64)

    pred = (probs >= 0.5).astype(np.int64)

    acc = accuracy_score(labels, pred)
    auc = roc_auc_score(labels, probs)

    return float(np.mean(losses)), float(acc), float(auc)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

best_val_loss = float("inf")
best_state = None
bad_epochs = 0

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()

    running_loss = 0.0
    seen = 0

    for tokens, lengths, target in train_loader:
        tokens = tokens.to(DEVICE, non_blocking=True)
        lengths = lengths.to(DEVICE, non_blocking=True)
        target = target.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits, _ = model(tokens, lengths)
        loss = criterion(logits, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

        optimizer.step()

        batch_n = target.shape[0]
        running_loss += loss.item() * batch_n
        seen += batch_n

    train_loss = running_loss / max(seen, 1)

    val_loss, val_acc, val_auc = evaluate_neural(model, val_loader)

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.5f} | "
        f"val_loss={val_loss:.5f} | "
        f"val_acc={val_acc:.6f} | "
        f"val_auc={val_auc:.6f} | "
        f"lr={current_lr:.2e}"
    )

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
        bad_epochs = 0
    else:
        bad_epochs += 1

    if bad_epochs >= PATIENCE:
        print("Early stopping.")
        break

elapsed = time.time() - start_time


# ---------------------------------------------------------------------
# Restore best model
# ---------------------------------------------------------------------

if best_state is None:
    raise RuntimeError("No best model state was saved.")

model.load_state_dict(best_state)
model.to(DEVICE)

final_val_loss, neural_val_acc, neural_val_auc = evaluate_neural(
    model,
    val_loader,
)

print()
print("=" * 78)
print("NEURAL CNN RESULT")
print("=" * 78)
print(f"Validation accuracy: {neural_val_acc:.6f}")
print(f"Validation AUC     : {neural_val_auc:.6f}")
print(f"Training time      : {elapsed / 60.0:.2f} min")


# ---------------------------------------------------------------------
# Extract learned embeddings
# ---------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval()

    embeddings = []

    for tokens, lengths, _ in loader:
        tokens = tokens.to(DEVICE, non_blocking=True)
        lengths = lengths.to(DEVICE, non_blocking=True)

        _, z = model(tokens, lengths)
        embeddings.append(z.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


train_eval_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

train_embeddings = extract_embeddings(model, train_eval_loader)
val_embeddings = extract_embeddings(model, val_loader)

print()
print(f"Train embedding shape: {train_embeddings.shape}")
print(f"Val embedding shape  : {val_embeddings.shape}")


# ---------------------------------------------------------------------
# Evaluate learned vector with classical classifiers
# ---------------------------------------------------------------------

# Standardization is fitted ONLY on train embeddings.
scaler = StandardScaler()
z_train = scaler.fit_transform(train_embeddings)
z_val = scaler.transform(val_embeddings)

# ---- Linear SVM ----
svm = LinearSVC(
    C=1.0,
    class_weight="balanced",
    max_iter=5000,
    random_state=SEED,
)

svm.fit(z_train, y[train_idx])

svm_scores = svm.decision_function(z_val)
svm_pred = (svm_scores >= 0).astype(np.int64)

svm_acc = accuracy_score(y[val_idx], svm_pred)
svm_auc = roc_auc_score(y[val_idx], svm_scores)

# ---- Logistic Regression ----
lr = LogisticRegression(
    C=1.0,
    class_weight="balanced",
    max_iter=3000,
    random_state=SEED,
)

lr.fit(z_train, y[train_idx])

lr_scores = lr.decision_function(z_val)
lr_pred = (lr_scores >= 0).astype(np.int64)

lr_acc = accuracy_score(y[val_idx], lr_pred)
lr_auc = roc_auc_score(y[val_idx], lr_scores)

print()
print("=" * 78)
print("CLASSIFIER ON LEARNED CNN VECTOR")
print("=" * 78)
print(
    f"SVM: accuracy={svm_acc:.6f}  AUC={svm_auc:.6f}"
)
print(
    f"LR : accuracy={lr_acc:.6f}  AUC={lr_auc:.6f}"
)


# ---------------------------------------------------------------------
# Latent-space separation diagnostics
# ---------------------------------------------------------------------

def class_centroid_distance(z, labels):
    z0 = z[labels == 0]
    z1 = z[labels == 1]

    c0 = z0.mean(axis=0)
    c1 = z1.mean(axis=0)

    between = float(np.linalg.norm(c0 - c1))

    # Sample pairwise within-class distances to keep computation bounded.
    rng = np.random.default_rng(SEED)

    def sampled_mean_distance(a, max_pairs=20000):
        if len(a) < 2:
            return float("nan")

        n_pairs = min(max_pairs, len(a) * (len(a) - 1) // 2)

        i = rng.integers(0, len(a), size=n_pairs)
        j = rng.integers(0, len(a), size=n_pairs)

        valid = i != j
        i = i[valid]
        j = j[valid]

        if len(i) == 0:
            return float("nan")

        return float(np.linalg.norm(a[i] - a[j], axis=1).mean())

    within0 = sampled_mean_distance(z0)
    within1 = sampled_mean_distance(z1)

    within = float(np.nanmean([within0, within1]))
    ratio = between / max(within, 1e-12)

    return between, within, ratio


between_dist, within_dist, separation_ratio = class_centroid_distance(
    val_embeddings,
    y[val_idx],
)

print()
print("=" * 78)
print("LATENT SPACE DIAGNOSTICS")
print("=" * 78)
print(f"Between-class centroid distance: {between_dist:.6f}")
print(f"Within-class mean distance      : {within_dist:.6f}")
print(f"Between / within ratio          : {separation_ratio:.6f}")


# ---------------------------------------------------------------------
# Compare against Main64/Main115 reference
# ---------------------------------------------------------------------

MAIN64_VAL_ACC = 0.9302656546489564
MAIN115_VAL_ACC = 0.934535104

print()
print("=" * 78)
print("REFERENCE COMPARISON")
print("=" * 78)
print(f"Main64 : {MAIN64_VAL_ACC:.6f}")
print(f"Main115: {MAIN115_VAL_ACC:.6f}")
print(f"CNN NN : {neural_val_acc:.6f}  delta vs Main115 = {neural_val_acc - MAIN115_VAL_ACC:+.6f}")
print(f"CNN SVM: {svm_acc:.6f}  delta vs Main115 = {svm_acc - MAIN115_VAL_ACC:+.6f}")
print(f"CNN LR : {lr_acc:.6f}  delta vs Main115 = {lr_acc - MAIN115_VAL_ACC:+.6f}")


# ---------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": VOCAB_SIZE,
        "max_len": MAX_LEN,
        "embed_dim": EMBED_DIM,
        "latent_dim": LATENT_DIM,
        "cnn_channels": CNN_CHANNELS,
        "kernels": KERNELS,
        "seed": SEED,
    },
    DATA_DIR / "main122_best.pt",
)

np.savez_compressed(
    DATA_DIR / "main122_embeddings.npz",
    train_embeddings=train_embeddings.astype(np.float32),
    val_embeddings=val_embeddings.astype(np.float32),
    y_train=y[train_idx],
    y_val=y[val_idx],
    train_indices=train_idx,
    val_indices=val_idx,
)

report = {
    "experiment": "Phase 1 / Experiment 1",
    "name": "Raw Tokens -> CNN -> Latent Vector",
    "seed": SEED,
    "dataset_size": int(len(texts)),
    "train_size": int(len(train_idx)),
    "val_size": int(len(val_idx)),
    "train_class_counts": np.bincount(
        y[train_idx], minlength=2
    ).tolist(),
    "val_class_counts": np.bincount(
        y[val_idx], minlength=2
    ).tolist(),
    "max_len": MAX_LEN,
    "vocab_size": int(VOCAB_SIZE),
    "embed_dim": EMBED_DIM,
    "latent_dim": LATENT_DIM,
    "cnn_channels": CNN_CHANNELS,
    "kernels": list(KERNELS),
    "parameters": int(n_params),
    "neural_val_loss": float(final_val_loss),
    "neural_val_accuracy": float(neural_val_acc),
    "neural_val_auc": float(neural_val_auc),
    "svm_val_accuracy": float(svm_acc),
    "svm_val_auc": float(svm_auc),
    "lr_val_accuracy": float(lr_acc),
    "lr_val_auc": float(lr_auc),
    "between_class_centroid_distance": float(between_dist),
    "within_class_mean_distance": float(within_dist),
    "between_within_ratio": float(separation_ratio),
    "training_minutes": float(elapsed / 60.0),
    "main64_validation_accuracy": MAIN64_VAL_ACC,
    "main115_validation_accuracy": MAIN115_VAL_ACC,
}

with open(DATA_DIR / "main122_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print()
print("=" * 78)
print("OUTPUTS")
print("=" * 78)
print(f"Wrote: {DATA_DIR / 'main122_best.pt'}")
print(f"Wrote: {DATA_DIR / 'main122_embeddings.npz'}")
print(f"Wrote: {DATA_DIR / 'main122_report.json'}")
print("=" * 78)
