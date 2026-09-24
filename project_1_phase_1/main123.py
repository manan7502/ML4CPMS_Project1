#!/usr/bin/env python3
"""
Experiment 2 — RAW TOKENS → BiGRU → LATENT VECTOR

ML4CPMS Project 1
Human vs Machine-Generated Text

Purpose:
    Test whether a BiGRU learns a useful document representation from raw
    token-ID sequences.

Canonical protocol:
    - 80/20 stratified split, random_state=42
    - Same dataset and split as Experiment 1
    - No Kaggle submission
    - Evaluate:
        1. Neural BiGRU classifier
        2. SVM on learned latent vector z
        3. Logistic Regression on learned latent vector z
        4. AUC for the above
        5. Between/within-class latent distance ratio

Outputs:
    exp2_bigru_best.pt
    exp2_bigru_embeddings.npz
    exp2_bigru_report.json

The script intentionally does NOT use Main64/Main115 features during training.
This experiment is supposed to test the raw-token representation independently.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

SEED = 42
DATA_DIR = Path(".")
TRAIN_JSON = DATA_DIR / "train.json"

MAX_LEN = 384

# Keep this deliberately comparable to Experiment 1.
EMBED_DIM = 128
HIDDEN_DIM = 128
LATENT_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.20

BATCH_SIZE = 64
EPOCHS = 12
LR = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Special padding ID. We use vocab_size as the padding index so it is
# guaranteed not to collide with an observed token ID.
PAD_EXTRA = 1


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility is preferred for this experiment.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Data loading
# ============================================================

def load_train(path):
    """
    Load the project dataset.

    The CP219 project train.json is JSONL/NDJSON: one JSON object per line,
    not one large JSON array. This loader reads it line-by-line and also
    accepts a JSON array as a fallback.
    """
    texts = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        first_nonspace = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_nonspace = ch
                break
        f.seek(0)

        if first_nonspace == "[":
            data = json.load(f)
            rows = data
        else:
            rows = []
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_no}: {exc}"
                    ) from exc

        for i, row in enumerate(rows, 1):
            if "text" not in row:
                raise ValueError(f"Dataset row {i} is missing 'text'.")
            label = row.get("label", row.get("labels"))
            if label is None:
                raise ValueError(
                    f"Dataset row {i} is missing 'label'/'labels'."
                )
            texts.append(row["text"])
            labels.append(label)

    return texts, labels


def encode_labels(labels):
    """
    Convert A/B or 0/1 labels to binary integers.

    The experiment only assumes there are exactly two classes.
    """
    unique = list(dict.fromkeys(labels))
    if len(unique) != 2:
        raise ValueError(f"Expected exactly 2 classes, found {unique}")

    # Preserve A/B convention when present.
    if set(unique) == {"A", "B"}:
        mapping = {"A": 0, "B": 1}
    elif set(unique) == {0, 1}:
        mapping = {0: 0, 1: 1}
    elif set(unique) == {"0", "1"}:
        mapping = {"0": 0, "1": 1}
    else:
        mapping = {unique[0]: 0, unique[1]: 1}

    return np.asarray([mapping[x] for x in labels], dtype=np.int64), mapping


# ============================================================
# Sequence preparation
# ============================================================

def prepare_sequences(texts, max_len):
    """
    Pad/truncate token-ID sequences.

    We keep the most recent max_len tokens when a document is longer than
    max_len. This is deterministic and keeps the same maximum length used
    in Experiment 1.
    """
    cleaned = []
    max_token = 0

    for seq in texts:
        arr = np.asarray(seq, dtype=np.int64).reshape(-1)

        if arr.size == 0:
            arr = np.asarray([0], dtype=np.int64)

        max_token = max(max_token, int(arr.max()))
        cleaned.append(arr)

    vocab_size = max_token + 1
    pad_id = vocab_size

    X = np.full((len(cleaned), max_len), pad_id, dtype=np.int64)
    lengths = np.zeros(len(cleaned), dtype=np.int64)

    for i, arr in enumerate(cleaned):
        arr = arr[-max_len:]
        n = len(arr)
        X[i, :n] = arr
        lengths[i] = n

    return X, lengths, vocab_size + PAD_EXTRA, pad_id


# ============================================================
# Dataset
# ============================================================

class SequenceDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X = torch.from_numpy(X).long()
        self.y = torch.from_numpy(y).float()
        self.lengths = torch.from_numpy(lengths).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]


# ============================================================
# BiGRU model
# ============================================================

class BiGRUEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_id,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_id,
        )

        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout,
        )

        self.latent = nn.Sequential(
            nn.Linear(hidden_dim * 4, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, x, lengths):
        emb = self.embedding(x)

        # pack_padded_sequence needs CPU lengths and strictly positive lengths.
        safe_lengths = lengths.clamp_min(1).cpu()

        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            safe_lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, h = self.gru(packed)

        # h:
        # [num_layers * 2, batch, hidden_dim]
        # Last forward and backward hidden states.
        h_forward = h[-2]
        h_backward = h[-1]
        h_last = torch.cat([h_forward, h_backward], dim=1)

        # Also unpack for masked mean pooling. This gives the representation
        # both final sequential state and a global average of the sequence.
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=x.shape[1],
        )

        mask = (
            torch.arange(x.shape[1], device=x.device)[None, :]
            < lengths[:, None].to(x.device)
        )

        mask = mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp_min(1.0)

        mean_pool = (out * mask).sum(dim=1) / denom

        combined = torch.cat([h_last, mean_pool], dim=1)

        z = self.latent(combined)
        logits = self.classifier(z).squeeze(1)

        return logits, z


# ============================================================
# Training / evaluation
# ============================================================

def evaluate_model(model, loader):
    model.eval()

    losses = []
    all_y = []
    all_p = []
    all_z = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for x, y, lengths in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            lengths = lengths.to(DEVICE, non_blocking=True)

            logits, z = model(x, lengths)
            loss = criterion(logits, y)

            losses.append(float(loss.item()))
            all_y.append(y.cpu().numpy())
            all_p.append(torch.sigmoid(logits).cpu().numpy())
            all_z.append(z.cpu().numpy())

    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    z = np.concatenate(all_z)

    acc = accuracy_score(y, p >= 0.5)
    auc = roc_auc_score(y, p)

    return {
        "loss": float(np.mean(losses)),
        "acc": float(acc),
        "auc": float(auc),
        "y": y,
        "p": p,
        "z": z,
    }


def train_model(model, train_loader, val_loader):
    # Balanced BCE because class frequencies are not exactly equal.
    train_y = train_loader.dataset.y.numpy()
    n0 = np.sum(train_y == 0)
    n1 = np.sum(train_y == 1)

    pos_weight = torch.tensor(
        n0 / max(n1, 1),
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    patience_left = PATIENCE

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running = []

        for x, y, lengths in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            lengths = lengths.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(x, lengths)
            loss = criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running.append(float(loss.item()))

        val = evaluate_model(model, val_loader)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(running)),
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_auc": val["auc"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={row['train_loss']:.5f} | "
            f"val_loss={row['val_loss']:.5f} | "
            f"val_acc={100*row['val_acc']:.4f}% | "
            f"val_auc={row['val_auc']:.5f}"
        )

        if val["loss"] < best_val_loss - 1e-5:
            best_val_loss = val["loss"]
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError("No best model state was recorded.")

    model.load_state_dict(best_state)
    return model, history, best_epoch


# ============================================================
# Simple classifiers on frozen z
# ============================================================

def evaluate_frozen_embeddings(z_train, y_train, z_val, y_val):
    scaler = StandardScaler()
    ztr = scaler.fit_transform(z_train)
    zv = scaler.transform(z_val)

    results = {}

    svm = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=10000,
        random_state=SEED,
    )
    svm.fit(ztr, y_train)
    svm_score = svm.decision_function(zv)

    results["svm_acc"] = float(accuracy_score(y_val, svm_score >= 0))
    results["svm_auc"] = float(roc_auc_score(y_val, svm_score))

    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
        random_state=SEED,
    )
    lr.fit(ztr, y_train)
    lr_score = lr.predict_proba(zv)[:, 1]

    results["lr_acc"] = float(accuracy_score(y_val, lr_score >= 0.5))
    results["lr_auc"] = float(roc_auc_score(y_val, lr_score))

    return results


# ============================================================
# Latent-space diagnostics
# ============================================================

def distance_diagnostics(z, y):
    z0 = z[y == 0]
    z1 = z[y == 1]

    c0 = z0.mean(axis=0)
    c1 = z1.mean(axis=0)

    between = float(np.linalg.norm(c0 - c1))

    # Sample within-class pairwise distances so this remains cheap.
    rng = np.random.default_rng(SEED)

    def sample_mean_pair_distance(arr, n_pairs=10000):
        if len(arr) < 2:
            return 0.0

        i = rng.integers(0, len(arr), size=n_pairs)
        j = rng.integers(0, len(arr), size=n_pairs)

        keep = i != j
        i = i[keep]
        j = j[keep]

        if len(i) == 0:
            return 0.0

        return float(np.linalg.norm(arr[i] - arr[j], axis=1).mean())

    within0 = sample_mean_pair_distance(z0)
    within1 = sample_mean_pair_distance(z1)
    within = float((within0 + within1) / 2.0)

    ratio = float(between / max(within, 1e-12))

    return {
        "between_class_centroid_distance": between,
        "within_class_mean_distance": within,
        "between_within_ratio": ratio,
    }


# ============================================================
# Main
# ============================================================

def main():
    seed_everything()

    print("=" * 72)
    print("MAIN123 — RAW TOKENS → BiGRU → LATENT VECTOR")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")

    start_time = time.time()

    texts, labels_raw = load_train(TRAIN_JSON)
    y, label_mapping = encode_labels(labels_raw)

    X, lengths, vocab_size, pad_id = prepare_sequences(
        texts,
        MAX_LEN,
    )

    n = len(y)
    all_idx = np.arange(n)

    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    # Explicit canonical split check.
    if len(train_idx) != 8428 or len(val_idx) != 2108:
        raise RuntimeError(
            f"Unexpected split sizes: {len(train_idx)} / {len(val_idx)}"
        )

    print(f"Dataset size: {n}")
    print(f"Train/validation: {len(train_idx)} / {len(val_idx)}")
    print(f"Max sequence length: {MAX_LEN}")
    print(f"Vocabulary size incl. padding: {vocab_size}")
    print(f"Embedding: {EMBED_DIM}-D")
    print(f"BiGRU hidden: {HIDDEN_DIM}-D per direction")
    print(f"Latent vector: {LATENT_DIM}-D")

    train_ds = SequenceDataset(
        X[train_idx],
        y[train_idx],
        lengths[train_idx],
    )
    val_ds = SequenceDataset(
        X[val_idx],
        y[val_idx],
        lengths[val_idx],
    )

    pin = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )

    model = BiGRUEncoder(
        vocab_size=vocab_size,
        pad_id=pad_id,
    ).to(DEVICE)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,}")

    model, history, best_epoch = train_model(
        model,
        train_loader,
        val_loader,
    )

    # Evaluate final selected checkpoint.
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )

    train_eval = evaluate_model(model, train_eval_loader)
    val_eval = evaluate_model(model, val_loader)

    frozen = evaluate_frozen_embeddings(
        train_eval["z"],
        train_eval["y"].astype(np.int64),
        val_eval["z"],
        val_eval["y"].astype(np.int64),
    )

    diagnostics = distance_diagnostics(
        val_eval["z"],
        val_eval["y"].astype(np.int64),
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 72)
    print("EXPERIMENT 2 RESULTS")
    print("=" * 72)
    print(f"Best epoch: {best_epoch}")
    print(f"Neural BiGRU validation accuracy: {100*val_eval['acc']:.4f}%")
    print(f"Neural BiGRU validation AUC:      {val_eval['auc']:.6f}")
    print(f"SVM on learned z accuracy:        {100*frozen['svm_acc']:.4f}%")
    print(f"SVM on learned z AUC:             {frozen['svm_auc']:.6f}")
    print(f"LR on learned z accuracy:         {100*frozen['lr_acc']:.4f}%")
    print(f"LR on learned z AUC:              {frozen['lr_auc']:.6f}")
    print(
        "Between/within ratio:             "
        f"{diagnostics['between_within_ratio']:.6f}"
    )
    print(f"Training + evaluation time:       {elapsed/60:.2f} min")

    # Save model checkpoint.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "seed": SEED,
                "max_len": MAX_LEN,
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "latent_dim": LATENT_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "vocab_size": vocab_size,
                "pad_id": pad_id,
                "best_epoch": best_epoch,
            },
            "label_mapping": label_mapping,
        },
        "exp2_bigru_best.pt",
    )

    # Save embeddings in the exact train/validation row order.
    np.savez_compressed(
        "exp2_bigru_embeddings.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        y_train=y[train_idx],
        y_val=y[val_idx],
        z_train=train_eval["z"],
        z_val=val_eval["z"],
        p_train=train_eval["p"],
        p_val=val_eval["p"],
    )

    report = {
        "experiment": 2,
        "name": "RAW TOKENS -> BiGRU -> LATENT VECTOR",
        "status": "completed",
        "seed": SEED,
        "dataset_size": int(n),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "max_len": MAX_LEN,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "best_epoch": int(best_epoch),
        "trainable_parameters": int(params),
        "neural_val_accuracy": float(val_eval["acc"]),
        "neural_val_auc": float(val_eval["auc"]),
        "svm_z_val_accuracy": float(frozen["svm_acc"]),
        "svm_z_val_auc": float(frozen["svm_auc"]),
        "lr_z_val_accuracy": float(frozen["lr_acc"]),
        "lr_z_val_auc": float(frozen["lr_auc"]),
        **diagnostics,
        "runtime_minutes": float(elapsed / 60.0),
        "main64_reference_accuracy": 0.930265655,
        "main115_reference_accuracy": 0.934535104,
        "outputs": [
            "exp2_bigru_best.pt",
            "exp2_bigru_embeddings.npz",
            "exp2_bigru_report.json",
        ],
    }

    with open("exp2_bigru_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nSaved:")
    print("  exp2_bigru_best.pt")
    print("  exp2_bigru_embeddings.npz")
    print("  exp2_bigru_report.json")

    print("\nReference:")
    print("  Main64  = 93.0266%")
    print("  Main115 = 93.4535%")
    print("\nDo NOT submit this experiment to Kaggle.")
    print("Send me the terminal results; then we will decide whether")
    print("Experiment 2 is retired or promoted to later fusion experiments.")


if __name__ == "__main__":
    main()
