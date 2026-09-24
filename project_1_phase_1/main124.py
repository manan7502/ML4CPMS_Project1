#!/usr/bin/env python3
"""
MAIN124 — RAW TOKENS -> TRANSFORMER -> LATENT VECTOR

Phase 1, Experiment 3 from the neural representation experiment series.

Protocol:
- Canonical 80/20 stratified split, random_state=42
- Raw token-ID sequences
- Embedding -> Transformer encoder -> masked mean pooling -> latent z
- Evaluate:
    1) neural classifier
    2) SVM on z
    3) Logistic Regression on z
    4) AUC
    5) between/within distance ratio
- Save best checkpoint, train/validation embeddings, and JSON report.

No Kaggle submission.
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


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

SEED = 42
MAX_LEN = 384

EMBED_DIM = 128
NHEAD = 4
NUM_LAYERS = 2
FF_DIM = 256
DROPOUT = 0.20
LATENT_DIM = 128

BATCH_SIZE = 32
EPOCHS = 12
LR = 2e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 3
GRAD_CLIP = 1.0

TRAIN_JSON = Path("train.json")

OUT_MODEL = Path("main124_best.pt")
OUT_EMBED = Path("main124_embeddings.npz")
OUT_REPORT = Path("main124_report.json")


# ---------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior is useful for experiment comparison.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

def load_train(path):
    """
    The project train.json is JSONL/NDJSON: one JSON object per line.
    A JSON-array fallback is also supported.
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
            rows = json.load(f)
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
    unique = sorted(set(labels))
    if len(unique) != 2:
        raise ValueError(f"Expected exactly two labels, got {unique}")

    mapping = {unique[0]: 0, unique[1]: 1}
    y = np.asarray([mapping[x] for x in labels], dtype=np.int64)
    return y, mapping


def prepare_sequences(texts, max_len=MAX_LEN):
    """
    Token IDs are preserved exactly. Long documents are truncated from the
    left so that the most recent/max-position portion is retained, matching
    the existing raw-token experiment convention.
    """
    max_token = 0
    for seq in texts:
        if len(seq):
            max_token = max(max_token, int(max(seq)))

    # Reserve one ID for padding.
    pad_id = max_token + 1
    vocab_size = pad_id + 1

    X = np.full((len(texts), max_len), pad_id, dtype=np.int64)
    lengths = np.zeros(len(texts), dtype=np.int64)

    for i, seq in enumerate(texts):
        seq = [int(t) for t in seq]
        if len(seq) > max_len:
            seq = seq[-max_len:]

        n = len(seq)
        if n:
            X[i, :n] = np.asarray(seq, dtype=np.int64)
        lengths[i] = n

    return X, lengths, vocab_size, pad_id


# ---------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------

class TokenDataset(Dataset):
    def __init__(self, X, lengths, y):
        self.X = torch.from_numpy(X)
        self.lengths = torch.from_numpy(lengths)
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.lengths[idx], self.y[idx]


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------

class TransformerEncoderClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_id,
        embed_dim=EMBED_DIM,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        latent_dim=LATENT_DIM,
    ):
        super().__init__()

        self.pad_id = pad_id

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_id,
        )

        # Learned positional embedding keeps the model aware of token order
        # without requiring a separate positional feature representation.
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, MAX_LEN, embed_dim)
        )
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(embed_dim)

        if latent_dim == embed_dim:
            self.latent = nn.Identity()
        else:
            self.latent = nn.Sequential(
                nn.Linear(embed_dim, latent_dim),
                nn.GELU(),
                nn.LayerNorm(latent_dim),
            )

        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, tokens, lengths):
        pad_mask = tokens.eq(self.pad_id)

        x = self.embedding(tokens)
        x = x + self.pos_embedding[:, :x.size(1)]

        x = self.transformer(
            x,
            src_key_padding_mask=pad_mask,
        )

        # Masked mean pooling.
        valid = (~pad_mask).unsqueeze(-1).float()
        denom = valid.sum(dim=1).clamp_min(1.0)
        pooled = (x * valid).sum(dim=1) / denom

        z = self.latent(self.norm(pooled))
        logits = self.classifier(z).squeeze(-1)

        return logits, z


# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_neural(model, loader, device):
    model.eval()

    losses = []
    all_y = []
    all_p = []

    # Loss is recreated outside this function only for clean evaluation.
    criterion = nn.BCEWithLogitsLoss()

    for tokens, lengths, y in loader:
        tokens = tokens.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits, _ = model(tokens, lengths)
        loss = criterion(logits, y)

        losses.append(loss.item() * len(y))
        all_y.append(y.cpu().numpy())
        all_p.append(torch.sigmoid(logits).cpu().numpy())

    y_true = np.concatenate(all_y)
    probs = np.concatenate(all_p)
    preds = (probs >= 0.5).astype(np.int64)

    return (
        float(np.sum(losses) / len(y_true)),
        float(accuracy_score(y_true, preds)),
        float(roc_auc_score(y_true, probs)),
    )


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()

    all_z = []
    all_y = []

    for tokens, lengths, y in loader:
        tokens = tokens.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        _, z = model(tokens, lengths)

        all_z.append(z.cpu().numpy())
        all_y.append(y.numpy())

    return np.concatenate(all_z), np.concatenate(all_y)


def embedding_distance_ratio(z, y):
    z0 = z[y == 0]
    z1 = z[y == 1]

    c0 = z0.mean(axis=0)
    c1 = z1.mean(axis=0)

    between = float(np.linalg.norm(c0 - c1))

    # To keep the metric consistent with the previous experiments,
    # calculate mean distance to each class centroid.
    d0 = np.linalg.norm(z0 - c0, axis=1)
    d1 = np.linalg.norm(z1 - c1, axis=1)
    within = float((d0.mean() + d1.mean()) / 2.0)

    ratio = between / max(within, 1e-12)
    return between, within, ratio


def evaluate_classical_on_z(z_train, y_train, z_val, y_val):
    scaler = StandardScaler()
    ztr = scaler.fit_transform(z_train)
    zv = scaler.transform(z_val)

    svm = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=SEED,
        max_iter=10000,
    )
    svm.fit(ztr, y_train)

    svm_dec = svm.decision_function(zv)
    svm_pred = (svm_dec >= 0).astype(np.int64)

    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
    )
    lr.fit(ztr, y_train)

    lr_prob = lr.predict_proba(zv)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(np.int64)

    return {
        "svm_z_val_accuracy": float(accuracy_score(y_val, svm_pred)),
        "svm_z_val_auc": float(roc_auc_score(y_val, svm_dec)),
        "lr_z_val_accuracy": float(accuracy_score(y_val, lr_pred)),
        "lr_z_val_auc": float(roc_auc_score(y_val, lr_prob)),
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("=" * 72)
    print("MAIN124 — RAW TOKENS → TRANSFORMER → LATENT VECTOR")
    print("=" * 72)

    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {SEED}")

    texts, labels_raw = load_train(TRAIN_JSON)
    y, label_mapping = encode_labels(labels_raw)

    print(f"Dataset size: {len(texts)}")

    all_idx = np.arange(len(texts))

    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )

    print(f"Train/validation: {len(train_idx)} / {len(val_idx)}")

    X, lengths, vocab_size, pad_id = prepare_sequences(texts, MAX_LEN)

    print(f"Max sequence length: {MAX_LEN}")
    print(f"Vocabulary size incl. padding: {vocab_size}")
    print(f"Embedding: {EMBED_DIM}-D")
    print(f"Transformer heads: {NHEAD}")
    print(f"Transformer layers: {NUM_LAYERS}")
    print(f"Feed-forward dimension: {FF_DIM}")
    print(f"Latent vector: {LATENT_DIM}-D")

    train_ds = TokenDataset(
        X[train_idx],
        lengths[train_idx],
        y[train_idx],
    )
    val_ds = TokenDataset(
        X[val_idx],
        lengths[val_idx],
        y[val_idx],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = TransformerEncoderClassifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Balanced BCE, matching the intent of previous raw-token experiments.
    n0 = int(np.sum(y[train_idx] == 0))
    n1 = int(np.sum(y[train_idx] == 1))
    pos_weight = torch.tensor(
        [n0 / max(n1, 1)],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

    best_val_loss = float("inf")
    best_epoch = 0
    patience_count = 0

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        seen = 0

        for tokens, lengths_batch, y_batch in train_loader:
            tokens = tokens.to(device, non_blocking=True)
            lengths_batch = lengths_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(tokens, lengths_batch)
            loss = criterion(logits, y_batch)

            loss.backward()

            if GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

            optimizer.step()

            running_loss += loss.item() * len(y_batch)
            seen += len(y_batch)

        train_loss = running_loss / max(seen, 1)

        val_loss, val_acc, val_auc = evaluate_neural(
            model,
            val_loader,
            device,
        )

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | "
            f"val_acc={val_acc*100:.4f}% | "
            f"val_auc={val_auc:.5f}"
        )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_count = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "seed": SEED,
                        "max_len": MAX_LEN,
                        "embed_dim": EMBED_DIM,
                        "nhead": NHEAD,
                        "num_layers": NUM_LAYERS,
                        "ff_dim": FF_DIM,
                        "dropout": DROPOUT,
                        "latent_dim": LATENT_DIM,
                        "vocab_size": vocab_size,
                        "pad_id": pad_id,
                        "label_mapping": label_mapping,
                    },
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                },
                OUT_MODEL,
            )
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print("Early stopping.")
                break

    runtime = (time.time() - start_time) / 60.0

    # Restore the best checkpoint selected strictly by validation loss.
    checkpoint = torch.load(
        OUT_MODEL,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    best_val_loss, neural_acc, neural_auc = evaluate_neural(
        model,
        val_loader,
        device,
    )

    z_train, y_train = extract_embeddings(
        model,
        train_eval_loader,
        device,
    )
    z_val, y_val = extract_embeddings(
        model,
        val_loader,
        device,
    )

    classical = evaluate_classical_on_z(
        z_train,
        y_train,
        z_val,
        y_val,
    )

    between, within, ratio = embedding_distance_ratio(
        z_val,
        y_val,
    )

    # Save embeddings with their exact canonical split indices so that they
    # can be audited or used for later fusion without ambiguity.
    np.savez_compressed(
        OUT_EMBED,
        z_train=z_train.astype(np.float32),
        z_val=z_val.astype(np.float32),
        y_train=y_train.astype(np.int64),
        y_val=y_val.astype(np.int64),
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
    )

    report = {
        "experiment": 3,
        "name": "RAW TOKENS -> TRANSFORMER -> LATENT VECTOR",
        "status": "completed",
        "seed": SEED,
        "dataset_size": len(texts),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "max_len": MAX_LEN,
        "embed_dim": EMBED_DIM,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "ff_dim": FF_DIM,
        "dropout": DROPOUT,
        "latent_dim": LATENT_DIM,
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "best_epoch": best_epoch,
        "trainable_parameters": n_params,
        "neural_val_accuracy": neural_acc,
        "neural_val_auc": neural_auc,
        **classical,
        "between_class_centroid_distance": between,
        "within_class_mean_distance": within,
        "between_within_ratio": ratio,
        "runtime_minutes": runtime,
        "main64_reference_accuracy": 0.930265655,
        "main115_reference_accuracy": 0.934535104,
        "outputs": [
            str(OUT_MODEL),
            str(OUT_EMBED),
            str(OUT_REPORT),
        ],
    }

    OUT_REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("EXPERIMENT 3 RESULTS")
    print("=" * 72)
    print(f"Best epoch: {best_epoch}")
    print(f"Neural Transformer validation accuracy: {neural_acc*100:.4f}%")
    print(f"Neural Transformer validation AUC:      {neural_auc:.6f}")
    print(f"SVM on learned z accuracy:             {classical['svm_z_val_accuracy']*100:.4f}%")
    print(f"SVM on learned z AUC:                  {classical['svm_z_val_auc']:.6f}")
    print(f"LR on learned z accuracy:              {classical['lr_z_val_accuracy']*100:.4f}%")
    print(f"LR on learned z AUC:                   {classical['lr_z_val_auc']:.6f}")
    print(f"Between/within ratio:                  {ratio:.6f}")
    print(f"Training + evaluation time:            {runtime:.2f} min")
    print()
    print("Saved:")
    print(f"  {OUT_MODEL}")
    print(f"  {OUT_EMBED}")
    print(f"  {OUT_REPORT}")
    print()
    print("Reference:")
    print("  Main64  = 93.0266%")
    print("  Main115 = 93.4535%")
    print()
    print("Do NOT submit this experiment to Kaggle.")
    print("Send me the terminal results; then we will decide whether")
    print("Experiment 3 is retired or promoted.")
    print("=" * 72)


if __name__ == "__main__":
    main()
