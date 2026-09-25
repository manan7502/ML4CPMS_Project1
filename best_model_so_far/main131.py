#!/usr/bin/env python3
"""
Main131 — Experiment 11
TF-IDF + Structural Features -> Fusion MLP -> 128-D latent representation

Goal:
    Test whether direct word/token TF-IDF information and compact structural
    sequence features provide a complementary learned representation.

Canonical split:
    train/validation = 80/20 stratified split, random_state=42

Branches:
    1. Word/token TF-IDF:
       analyzer="word"
       token_pattern=r"(?u)\S+"
       ngram_range=(1, 3)
       min_df=2
       max_df=.995
       max_features=60000
       sublinear_tf=True

    2. Structural features:
       62 compact sequence/statistical features.
       StandardScaler is fit on training data only.

Architecture:
    TF-IDF -> Linear(60000, 128)
    Structural -> Linear(62, 128)
    concat -> 192 -> 128 latent -> binary classifier

The latent vector is evaluated independently with:
    - LinearSVC
    - LogisticRegression
    - centroid geometry

Only the best validation-loss checkpoint is retained.

Outputs:
    main131_best.pt
    main131_embeddings.npz
    main131_report.json

No Kaggle submission is performed.
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

SEED = 42
MAX_FEATURES = 60000
MAX_EPOCHS = 35
PATIENCE = 6
BATCH_SIZE = 128
LR = 0.002
WEIGHT_DECAY = 1e-4

TFIDF_DIM = 128
STRUCT_DIM = 62
FUSION_HIDDEN = 192
LATENT_DIM = 128

DROPOUT = 0.20

BASE_DIR = Path("/home/manu/Desktop/ML4CPMS/project_1")
TRAIN_PATH = BASE_DIR / "train.json"

OUT_PT = BASE_DIR / "main131_best.pt"
OUT_EMB = BASE_DIR / "main131_embeddings.npz"
OUT_REPORT = BASE_DIR / "main131_report.json"


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 72)
print("MAIN131 — EXPERIMENT 11")
print("TF-IDF + STRUCTURAL -> FUSION MLP -> LATENT")
print("=" * 72)
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# Robust JSON / JSONL loader
# ============================================================

def load_dataset(path):
    """
    Supports:
      - JSON array
      - JSONL, one object per line

    Expected fields:
      text: list[int]
      label: A/B or 0/1
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        first_nonempty = None
        for line in f:
            if line.strip():
                first_nonempty = line.strip()
                break

    if first_nonempty is None:
        raise ValueError("Dataset is empty.")

    if first_nonempty.startswith("["):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON on line {line_no}: {e}"
                    ) from e

    texts = []
    labels = []

    for obj in data:
        seq = obj["text"]
        label = obj.get("label", None)

        texts.append([int(x) for x in seq])

        if label is None:
            raise ValueError("Missing label in training data.")

        if isinstance(label, str):
            if label.upper() == "A":
                labels.append(0)
            elif label.upper() == "B":
                labels.append(1)
            else:
                raise ValueError(f"Unknown label: {label}")
        else:
            labels.append(int(label))

    return texts, np.asarray(labels, dtype=np.int64)


# ============================================================
# Structural features
# ============================================================

def entropy_from_counts(counts):
    if len(counts) == 0:
        return 0.0
    p = counts.astype(np.float64)
    p /= max(p.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def sequence_features(seq):
    """
    62 compact structural/statistical features.

    The implementation deliberately avoids token identity information.
    It describes length, diversity, frequency concentration, transitions,
    positional behavior, chunks and local variability.

    Keep this feature count fixed at 62 for experiment comparability.
    """
    x = np.asarray(seq, dtype=np.int64)

    if x.size == 0:
        return np.zeros(62, dtype=np.float32)

    n = len(x)

    # --------------------------------------------------------
    # Basic length / diversity
    # --------------------------------------------------------
    unique, counts = np.unique(x, return_counts=True)

    length = float(n)
    unique_n = float(len(unique))
    diversity = unique_n / max(length, 1.0)

    count_sorted = np.sort(counts)[::-1].astype(np.float64)
    probs = count_sorted / max(length, 1.0)

    max_freq = float(count_sorted[0] / max(length, 1.0))
    top5_mass = float(count_sorted[:5].sum() / max(length, 1.0))
    top10_mass = float(count_sorted[:10].sum() / max(length, 1.0))

    freq_entropy = entropy_from_counts(counts)
    norm_entropy = freq_entropy / max(np.log2(max(unique_n, 2.0)), 1.0)

    singleton_frac = float(np.sum(counts == 1) / max(unique_n, 1.0))
    repeat_token_frac = float(np.sum(counts[counts > 1]) / max(length, 1.0))

    # --------------------------------------------------------
    # Token ID distribution
    # --------------------------------------------------------
    xf = x.astype(np.float64)

    id_mean = float(np.mean(xf))
    id_std = float(np.std(xf))
    id_min = float(np.min(xf))
    id_max = float(np.max(xf))
    id_range = id_max - id_min

    q10, q25, q50, q75, q90 = np.percentile(xf, [10, 25, 50, 75, 90])

    # --------------------------------------------------------
    # Adjacent transitions
    # --------------------------------------------------------
    if n >= 2:
        d = np.diff(xf)

        diff_mean = float(np.mean(d))
        diff_std = float(np.std(d))
        absdiff_mean = float(np.mean(np.abs(d)))
        absdiff_std = float(np.std(np.abs(d)))

        same_frac = float(np.mean(d == 0))
        positive_frac = float(np.mean(d > 0))
        negative_frac = float(np.mean(d < 0))

        sign_changes = np.diff(np.sign(d))
        sign_change_frac = float(
            np.mean(sign_changes != 0)
        ) if len(sign_changes) else 0.0

        # Return / local reversal behavior
        return_frac = float(
            np.mean(x[1:] == x[:-1])
        )

        # Large jump fractions
        qdiff50, qdiff75, qdiff90 = np.percentile(
            np.abs(d), [50, 75, 90]
        )
        large_jump50 = float(np.mean(np.abs(d) > qdiff50))
        large_jump75 = float(np.mean(np.abs(d) > qdiff75))
        large_jump90 = float(np.mean(np.abs(d) > qdiff90))
    else:
        diff_mean = diff_std = absdiff_mean = absdiff_std = 0.0
        same_frac = positive_frac = negative_frac = 0.0
        sign_change_frac = return_frac = 0.0
        large_jump50 = large_jump75 = large_jump90 = 0.0

    # --------------------------------------------------------
    # Position statistics
    # --------------------------------------------------------
    pos = np.arange(n, dtype=np.float64) / max(n - 1, 1)

    weighted_mean_pos = float(
        np.sum(pos * (x != 0)) / max(np.sum(x != 0), 1)
    )

    first_token = float(x[0])
    last_token = float(x[-1])
    first_last_absdiff = abs(last_token - first_token)

    # --------------------------------------------------------
    # Chunk statistics
    # --------------------------------------------------------
    # Four equal chunks; summarize token statistics in each.
    chunks = np.array_split(xf, 4)

    chunk_means = []
    chunk_stds = []
    chunk_unique = []

    for c in chunks:
        if len(c) == 0:
            chunk_means.append(0.0)
            chunk_stds.append(0.0)
            chunk_unique.append(0.0)
        else:
            chunk_means.append(float(np.mean(c)))
            chunk_stds.append(float(np.std(c)))
            chunk_unique.append(float(len(np.unique(c))) / len(c))

    mean_chunk_mean = float(np.mean(chunk_means))
    std_chunk_mean = float(np.std(chunk_means))

    mean_chunk_std = float(np.mean(chunk_stds))
    std_chunk_std = float(np.std(chunk_stds))

    mean_chunk_unique = float(np.mean(chunk_unique))
    std_chunk_unique = float(np.std(chunk_unique))

    # --------------------------------------------------------
    # Local variability
    # --------------------------------------------------------
    if n >= 4:
        window = max(2, min(16, n // 8))
        local_stds = []

        for i in range(0, n - window + 1, window):
            local_stds.append(float(np.std(xf[i:i + window])))

        if local_stds:
            local_std_mean = float(np.mean(local_stds))
            local_std_std = float(np.std(local_stds))
            local_std_max = float(np.max(local_stds))
            local_std_min = float(np.min(local_stds))
        else:
            local_std_mean = local_std_std = 0.0
            local_std_max = local_std_min = 0.0
    else:
        local_std_mean = local_std_std = 0.0
        local_std_max = local_std_min = 0.0

    # --------------------------------------------------------
    # Run statistics
    # --------------------------------------------------------
    if n >= 2:
        changes = np.concatenate(
            [[True], x[1:] != x[:-1]]
        )
        run_starts = np.where(changes)[0]
        run_lengths = np.diff(
            np.append(run_starts, n)
        ).astype(np.float64)

        run_mean = float(np.mean(run_lengths))
        run_std = float(np.std(run_lengths))
        run_max = float(np.max(run_lengths))
        run_frac_long = float(np.mean(run_lengths >= 3))
    else:
        run_mean = run_std = run_max = 1.0
        run_frac_long = 0.0

    # --------------------------------------------------------
    # Rank / concentration statistics
    # --------------------------------------------------------
    if len(count_sorted) >= 2:
        hhi = float(np.sum(probs ** 2))
        second_freq = float(count_sorted[1] / max(length, 1.0))
    else:
        hhi = max_freq ** 2
        second_freq = 0.0

    vocab_per_100 = 100.0 * unique_n / max(length, 1.0)

    # --------------------------------------------------------
    # Assemble EXACTLY 62 features
    # --------------------------------------------------------
    feat = np.array([
        # 1-12
        length,
        np.log1p(length),
        unique_n,
        diversity,
        max_freq,
        top5_mass,
        top10_mass,
        freq_entropy,
        norm_entropy,
        singleton_frac,
        repeat_token_frac,
        vocab_per_100,

        # 13-21
        id_mean,
        id_std,
        id_min,
        id_max,
        id_range,
        q10,
        q25,
        q50,
        q75,

        # 22-34
        q90,
        diff_mean,
        diff_std,
        absdiff_mean,
        absdiff_std,
        same_frac,
        positive_frac,
        negative_frac,
        sign_change_frac,
        return_frac,
        large_jump50,
        large_jump75,
        large_jump90,

        # 35-38
        weighted_mean_pos,
        first_token,
        last_token,
        first_last_absdiff,

        # 39-44
        mean_chunk_mean,
        std_chunk_mean,
        mean_chunk_std,
        std_chunk_std,
        mean_chunk_unique,
        std_chunk_unique,

        # 45-48
        local_std_mean,
        local_std_std,
        local_std_max,
        local_std_min,

        # 49-52
        run_mean,
        run_std,
        run_max,
        run_frac_long,

        # 53-55
        hhi,
        second_freq,
        np.mean(np.abs(xf - np.median(xf))),

        # 56-62
        np.std(np.diff(xf)) if n >= 2 else 0.0,
        np.mean(np.diff(xf) ** 2) if n >= 2 else 0.0,
        np.percentile(xf, 95),
        np.percentile(xf, 5),
        np.mean(xf[:max(1, n // 4)]),
        np.mean(xf[-max(1, n // 4):]),
        float(np.mean(xf == 0)),
    ], dtype=np.float32)

    if feat.shape[0] != 62:
        raise RuntimeError(
            f"Structural feature count is {feat.shape[0]}, expected 62."
        )

    return feat


# ============================================================
# Build TF-IDF representation
# ============================================================

def token_strings(texts):
    return [" ".join(map(str, seq)) for seq in texts]


# ============================================================
# Dataset
# ============================================================

class PairDataset(Dataset):
    def __init__(self, tfidf_matrix, struct_matrix, labels):
        self.tfidf = tfidf_matrix
        self.struct = struct_matrix
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tfidf_row = self.tfidf[idx].toarray().ravel().astype(np.float32)
        struct_row = self.struct[idx].astype(np.float32)

        return (
            torch.from_numpy(tfidf_row),
            torch.from_numpy(struct_row),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# ============================================================
# Model
# ============================================================

class FusionEncoder(nn.Module):
    def __init__(
        self,
        tfidf_dim,
        struct_dim=62,
        tfidf_hidden=128,
        struct_hidden=128,
        fusion_hidden=192,
        latent_dim=128,
        dropout=0.20,
    ):
        super().__init__()

        self.tfidf_branch = nn.Sequential(
            nn.Linear(tfidf_dim, tfidf_hidden),
            nn.BatchNorm1d(tfidf_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.struct_branch = nn.Sequential(
            nn.Linear(struct_dim, struct_hidden),
            nn.BatchNorm1d(struct_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.fusion = nn.Sequential(
            nn.Linear(tfidf_hidden + struct_hidden, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(fusion_hidden, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(latent_dim, 1)

    def encode(self, tfidf, struct):
        a = self.tfidf_branch(tfidf)
        b = self.struct_branch(struct)
        z = self.fusion(torch.cat([a, b], dim=1))
        return z

    def forward(self, tfidf, struct):
        z = self.encode(tfidf, struct)
        return self.classifier(z).squeeze(1)


# ============================================================
# Evaluation helpers
# ============================================================

@torch.no_grad()
def get_outputs(model, loader):
    model.eval()

    all_z = []
    all_logits = []
    all_y = []

    for tfidf, struct, y in loader:
        tfidf = tfidf.to(device)
        struct = struct.to(device)

        z = model.encode(tfidf, struct)
        logits = model.classifier(z).squeeze(1)

        all_z.append(z.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        all_y.append(y.numpy())

    return (
        np.concatenate(all_z),
        np.concatenate(all_logits),
        np.concatenate(all_y),
    )


def binary_metrics(y, logits):
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    pred = (probs >= 0.5).astype(np.int64)

    acc = accuracy_score(y.astype(np.int64), pred)
    auc = roc_auc_score(y, probs)

    return float(acc), float(auc)


def centroid_geometry(z, y):
    z0 = z[y == 0]
    z1 = z[y == 1]

    c0 = z0.mean(axis=0)
    c1 = z1.mean(axis=0)

    between = float(np.linalg.norm(c1 - c0))

    d0 = np.linalg.norm(z0 - c0, axis=1)
    d1 = np.linalg.norm(z1 - c1, axis=1)

    within = float(
        0.5 * (np.mean(d0) + np.mean(d1))
    )

    ratio = between / max(within, 1e-12)

    return between, within, ratio


# ============================================================
# Main
# ============================================================

start_time = time.time()

texts, y = load_dataset(TRAIN_PATH)

print(f"Dataset: {len(texts)}")
print(f"Class counts: {np.bincount(y)}")

all_idx = np.arange(len(texts))

train_idx, val_idx = train_test_split(
    all_idx,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

train_texts = [texts[i] for i in train_idx]
val_texts = [texts[i] for i in val_idx]

y_train = y[train_idx]
y_val = y[val_idx]

print(f"Train/val: {len(train_idx)}/{len(val_idx)}")

# ------------------------------------------------------------
# TF-IDF
# ------------------------------------------------------------

train_strings = token_strings(train_texts)
val_strings = token_strings(val_texts)

vectorizer = TfidfVectorizer(
    analyzer="word",
    token_pattern=r"(?u)\S+",
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.995,
    max_features=MAX_FEATURES,
    sublinear_tf=True,
    dtype=np.float32,
)

X_train_tfidf = vectorizer.fit_transform(train_strings)
X_val_tfidf = vectorizer.transform(val_strings)

print(f"TF-IDF dimensions: {X_train_tfidf.shape[1]}")
print(f"Train nnz: {X_train_tfidf.nnz}")
print(f"Val nnz: {X_val_tfidf.nnz}")

# ------------------------------------------------------------
# Structural features
# ------------------------------------------------------------

print("Building structural features...")

X_train_struct = np.vstack(
    [sequence_features(s) for s in train_texts]
).astype(np.float32)

X_val_struct = np.vstack(
    [sequence_features(s) for s in val_texts]
).astype(np.float32)

if X_train_struct.shape[1] != STRUCT_DIM:
    raise RuntimeError(
        f"Expected {STRUCT_DIM} structural features, "
        f"got {X_train_struct.shape[1]}"
    )

struct_scaler = StandardScaler()
X_train_struct = struct_scaler.fit_transform(X_train_struct).astype(np.float32)
X_val_struct = struct_scaler.transform(X_val_struct).astype(np.float32)

print(f"Structural dimensions: {X_train_struct.shape[1]}")

# ------------------------------------------------------------
# Data loaders
# ------------------------------------------------------------

train_ds = PairDataset(
    X_train_tfidf,
    X_train_struct,
    y_train,
)

val_ds = PairDataset(
    X_val_tfidf,
    X_val_struct,
    y_val,
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = FusionEncoder(
    tfidf_dim=X_train_tfidf.shape[1],
    struct_dim=STRUCT_DIM,
    tfidf_hidden=128,
    struct_hidden=128,
    fusion_hidden=FUSION_HIDDEN,
    latent_dim=LATENT_DIM,
    dropout=DROPOUT,
).to(device)

params = sum(p.numel() for p in model.parameters())

print(f"Parameters: {params:,}")

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

best_val_loss = float("inf")
best_epoch = -1
patience_counter = 0
history = []

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for tfidf, struct, labels in train_loader:
        tfidf = tfidf.to(device, non_blocking=True)
        struct = struct.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(tfidf, struct)
        loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        train_loss_sum += loss.item() * len(labels)

        preds = (torch.sigmoid(logits) >= 0.5).long()
        train_correct += int(
            (preds == labels.long()).sum().item()
        )
        train_total += len(labels)

    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total

    # Validation
    model.eval()

    val_loss_sum = 0.0
    val_total = 0
    val_logits = []
    val_targets = []

    with torch.no_grad():
        for tfidf, struct, labels in val_loader:
            tfidf = tfidf.to(device, non_blocking=True)
            struct = struct.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)

            logits = model(tfidf, struct)
            loss = criterion(logits, labels_device)

            val_loss_sum += loss.item() * len(labels)
            val_total += len(labels)

            val_logits.append(logits.cpu().numpy())
            val_targets.append(labels.numpy())

    val_loss = val_loss_sum / val_total
    val_logits_np = np.concatenate(val_logits)
    val_targets_np = np.concatenate(val_targets)

    val_acc, val_auc = binary_metrics(
        val_targets_np,
        val_logits_np,
    )

    record = {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "val_auc": float(val_auc),
    }
    history.append(record)

    print(
        f"Epoch {epoch:02d} | "
        f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
        f"val loss {val_loss:.4f} acc {val_acc:.4f} "
        f"AUC {val_auc:.4f}"
    )

    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "tfidf_vocabulary": vectorizer.vocabulary_,
                "tfidf_idf": vectorizer.idf_,
                "struct_scaler_mean": struct_scaler.mean_,
                "struct_scaler_scale": struct_scaler.scale_,
                "config": {
                    "tfidf_dim": X_train_tfidf.shape[1],
                    "struct_dim": STRUCT_DIM,
                    "tfidf_hidden": TFIDF_DIM,
                    "struct_hidden": 128,
                    "fusion_hidden": FUSION_HIDDEN,
                    "latent_dim": LATENT_DIM,
                    "dropout": DROPOUT,
                },
                "seed": SEED,
                "best_epoch": best_epoch,
            },
            OUT_PT,
        )

        print(f"  -> saved best checkpoint (epoch {epoch})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break


# ============================================================
# Restore best checkpoint
# ============================================================

checkpoint = torch.load(
    OUT_PT,
    map_location=device,
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])

print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.6f}")

# ============================================================
# Extract latent embeddings
# ============================================================

train_eval_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

z_train, train_logits, train_y = get_outputs(
    model,
    train_eval_loader,
)

z_val, val_logits, val_y = get_outputs(
    model,
    val_loader,
)

neural_val_acc, neural_val_auc = binary_metrics(
    val_y,
    val_logits,
)

# ============================================================
# Classical classifiers on frozen latent z
# ============================================================

print("\nFrozen latent evaluation:")

svm = LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=SEED,
)

svm.fit(z_train, train_y)

svm_pred = svm.predict(z_val)
svm_acc = accuracy_score(val_y.astype(np.int64), svm_pred)

svm_decision = svm.decision_function(z_val)
svm_auc = roc_auc_score(val_y, svm_decision)

print(
    f"SVM latent: accuracy={svm_acc:.6f}, "
    f"AUC={svm_auc:.6f}"
)

lr = LogisticRegression(
    C=1.0,
    class_weight="balanced",
    max_iter=2000,
    random_state=SEED,
)

lr.fit(z_train, train_y)

lr_pred = lr.predict(z_val)
lr_prob = lr.predict_proba(z_val)[:, 1]

lr_acc = accuracy_score(
    val_y.astype(np.int64),
    lr_pred,
)
lr_auc = roc_auc_score(val_y, lr_prob)

print(
    f"LR latent:  accuracy={lr_acc:.6f}, "
    f"AUC={lr_auc:.6f}"
)

between, within, ratio = centroid_geometry(
    z_val,
    val_y,
)

print(
    f"Geometry: between={between:.6f}, "
    f"within={within:.6f}, "
    f"ratio={ratio:.6f}"
)

runtime = time.time() - start_time

# ============================================================
# Save embeddings
# ============================================================

np.savez_compressed(
    OUT_EMB,
    z_train=z_train.astype(np.float32),
    z_val=z_val.astype(np.float32),
    y_train=train_y.astype(np.int64),
    y_val=val_y.astype(np.int64),
    train_indices=train_idx.astype(np.int64),
    val_indices=val_idx.astype(np.int64),
)

# ============================================================
# Save report
# ============================================================

report = {
    "experiment": "Exp11",
    "main": "Main131",
    "description": (
        "TF-IDF + structural features -> fusion MLP -> "
        "128-D latent representation"
    ),

    "dataset": {
        "n_samples": int(len(texts)),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "class_counts": np.bincount(y).tolist(),
        "split": "train_test_split(test_size=0.20, stratify=y, random_state=42)",
    },

    "tfidf": {
        "analyzer": "word",
        "token_pattern": r"(?u)\S+",
        "ngram_range": [1, 3],
        "min_df": 2,
        "max_df": 0.995,
        "max_features": MAX_FEATURES,
        "sublinear_tf": True,
        "dimensions": int(X_train_tfidf.shape[1]),
        "train_nnz": int(X_train_tfidf.nnz),
        "val_nnz": int(X_val_tfidf.nnz),
    },

    "structural": {
        "dimensions": STRUCT_DIM,
        "scaler": "StandardScaler fit on train only",
    },

    "model": {
        "tfidf_hidden": 128,
        "struct_hidden": 128,
        "fusion_hidden": FUSION_HIDDEN,
        "latent_dim": LATENT_DIM,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "gradient_clip": 5.0,
        "parameters": int(params),
    },

    "best_epoch": int(best_epoch),
    "best_val_loss": float(best_val_loss),

    "results": {
        "neural_val_accuracy": float(neural_val_acc),
        "neural_val_auc": float(neural_val_auc),
        "svm_latent_accuracy": float(svm_acc),
        "svm_latent_auc": float(svm_auc),
        "lr_latent_accuracy": float(lr_acc),
        "lr_latent_auc": float(lr_auc),
        "between_centroid_distance": float(between),
        "within_centroid_distance": float(within),
        "between_within_ratio": float(ratio),
    },

    "training_history": history,

    "runtime_seconds": float(runtime),

    "outputs": [
        str(OUT_PT),
        str(OUT_EMB),
        str(OUT_REPORT),
    ],
}

with OUT_REPORT.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 72)
print("FINAL RESULTS — MAIN131")
print("=" * 72)
print(f"Best epoch:              {best_epoch}")
print(f"Neural validation acc:   {neural_val_acc:.6f}")
print(f"Neural validation AUC:   {neural_val_auc:.6f}")
print(f"SVM latent acc:          {svm_acc:.6f}")
print(f"SVM latent AUC:          {svm_auc:.6f}")
print(f"LR latent acc:           {lr_acc:.6f}")
print(f"LR latent AUC:           {lr_auc:.6f}")
print(f"Between centroid:        {between:.6f}")
print(f"Within centroid:         {within:.6f}")
print(f"Between/within ratio:    {ratio:.6f}")
print(f"Runtime:                 {runtime / 60:.2f} min")
print("=" * 72)
print(f"Saved: {OUT_PT}")
print(f"Saved: {OUT_EMB}")
print(f"Saved: {OUT_REPORT}")
