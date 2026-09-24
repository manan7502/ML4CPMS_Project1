#!/usr/bin/env python3
"""
Main125 — Phase 1, Experiment 4
TF-IDF → SVD → MLP ENCODER

Purpose
-------
Test whether a compressed classical lexical representation can be turned into
a useful learned document embedding.

Pipeline
--------
JSONL train data
    ↓
token/word TF-IDF
    ↓
TruncatedSVD
    ↓
MLP encoder
    ↓
128-D latent vector z
    ├── neural classifier
    ├── SVM(z)
    └── LogisticRegression(z)

Evaluation uses the canonical 80/20 stratified split:
train=8428, validation=2108, random_state=42.

Important:
- SVD and TF-IDF vocabulary are fitted on TRAIN ONLY.
- MLP is trained only on the training split.
- Validation is used for model selection/early stopping only.
- No Kaggle submission is made.
- Learned train/validation embeddings are saved for possible later fusion.
"""

import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Configuration
# ============================================================

SEED = 42

DATA_PATH = Path("train.json")
OUT_DIR = Path(".")

MAX_TFIDF_FEATURES = 60000
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.995
TFIDF_NGRAM_RANGE = (1, 3)
TFIDF_SUBLINEAR = True

SVD_DIM = 256
LATENT_DIM = 128

HIDDEN1 = 256
HIDDEN2 = 192
DROPOUT = 0.20

BATCH_SIZE = 128
MAX_EPOCHS = 35
PATIENCE = 6
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data loading
# ============================================================

def load_jsonl(path):
    texts = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            if "text" not in obj:
                raise KeyError(f"Missing 'text' at line {line_no}")

            if "label" not in obj:
                raise KeyError(f"Missing 'label' at line {line_no}")

            token_ids = obj["text"]

            # The project data stores documents as integer token IDs.
            # Convert each token ID to a stable whitespace-separated token.
            text = " ".join(map(str, token_ids))
            texts.append(text)

            labels.append(obj["label"])

    return texts, labels


def encode_labels(labels):
    """
    Map A/B (or any two labels) deterministically to 0/1.
    The mapping is printed so the experiment is auditable.
    """
    unique = sorted(set(labels))
    if len(unique) != 2:
        raise ValueError(f"Expected exactly 2 classes, found: {unique}")

    mapping = {unique[0]: 0, unique[1]: 1}
    y = np.asarray([mapping[x] for x in labels], dtype=np.int64)

    print("Label mapping:", mapping)
    return y, mapping


# ============================================================
# Model
# ============================================================

class TfidfSVDMLP(nn.Module):
    def __init__(self, input_dim, hidden1=HIDDEN1, hidden2=HIDDEN2,
                 latent_dim=LATENT_DIM, dropout=DROPOUT):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(dropout),

            nn.Linear(hidden2, latent_dim),
        )

        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z).squeeze(1)
        return logits, z


# ============================================================
# Helpers
# ============================================================

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


@torch.no_grad()
def predict_model(model, X, batch_size=512):
    model.eval()

    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    logits_all = []
    z_all = []

    for (xb,) in loader:
        xb = xb.to(DEVICE)
        logits, z = model(xb)
        logits_all.append(logits.cpu().numpy())
        z_all.append(z.cpu().numpy())

    logits = np.concatenate(logits_all)
    z = np.concatenate(z_all)

    return logits, z


def centroid_stats(z, y):
    z0 = z[y == 0]
    z1 = z[y == 1]

    c0 = z0.mean(axis=0)
    c1 = z1.mean(axis=0)

    between = float(np.linalg.norm(c0 - c1))

    # Sample a bounded number of points for the within-class distance.
    rng = np.random.default_rng(SEED)

    def mean_distance_to_centroid(a, c):
        if len(a) > 3000:
            idx = rng.choice(len(a), 3000, replace=False)
            a = a[idx]
        return float(np.mean(np.linalg.norm(a - c, axis=1)))

    within0 = mean_distance_to_centroid(z0, c0)
    within1 = mean_distance_to_centroid(z1, c1)
    within = (within0 + within1) / 2.0

    ratio = between / (within + 1e-12)

    return between, within, ratio


# ============================================================
# Main experiment
# ============================================================

def main():
    seed_everything()
    start_time = time.time()

    print("=" * 72)
    print("MAIN125 — PHASE 1 / EXPERIMENT 4")
    print("TF-IDF → SVD → MLP ENCODER")
    print("=" * 72)
    print("Device:", DEVICE)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    texts, labels = load_jsonl(DATA_PATH)
    y, label_mapping = encode_labels(labels)

    n = len(y)
    all_idx = np.arange(n)

    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )

    print(f"Dataset: {n}")
    print(f"Train/validation: {len(train_idx)} / {len(val_idx)}")

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------
    print("\n[1/5] Fitting TF-IDF on TRAIN ONLY...")

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\S+",
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        max_features=MAX_TFIDF_FEATURES,
        sublinear_tf=TFIDF_SUBLINEAR,
        dtype=np.float32,
    )

    X_train_sparse = vectorizer.fit_transform(train_texts)
    X_val_sparse = vectorizer.transform(val_texts)

    print("TF-IDF train shape:", X_train_sparse.shape)
    print("TF-IDF val shape  :", X_val_sparse.shape)

    # --------------------------------------------------------
    # SVD
    # --------------------------------------------------------
    print(f"\n[2/5] Fitting TruncatedSVD ({SVD_DIM}-D) on TRAIN ONLY...")

    svd = TruncatedSVD(
        n_components=SVD_DIM,
        random_state=SEED,
        n_iter=5,
    )

    X_train = svd.fit_transform(X_train_sparse).astype(np.float32)
    X_val = svd.transform(X_val_sparse).astype(np.float32)

    explained = float(np.sum(svd.explained_variance_ratio_))

    print("SVD train shape:", X_train.shape)
    print("SVD val shape  :", X_val.shape)
    print(f"SVD explained variance ratio sum: {explained:.6f}")

    # Release sparse matrices before neural training.
    del X_train_sparse, X_val_sparse

    # --------------------------------------------------------
    # Standardize SVD coordinates using TRAIN ONLY
    # --------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------
    print("\n[3/5] Training MLP encoder...")

    model = TfidfSVDMLP(input_dim=SVD_DIM).to(DEVICE)

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train.astype(np.float32)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation loss
        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(X_val).to(DEVICE)
            yv = torch.from_numpy(y_val.astype(np.float32)).to(DEVICE)

            val_logits, _ = model(xv)
            val_loss = criterion(val_logits, yv).item()

        train_loss = float(np.mean(train_losses))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f}"
        )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping after epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("No best model state was recorded.")

    model.load_state_dict(best_state)
    model.to(DEVICE)

    # --------------------------------------------------------
    # Neural validation result + embeddings
    # --------------------------------------------------------
    print("\n[4/5] Evaluating learned representation...")

    train_logits, z_train = predict_model(model, X_train)
    val_logits, z_val = predict_model(model, X_val)

    train_prob = sigmoid_np(train_logits)
    val_prob = sigmoid_np(val_logits)

    train_pred = (train_prob >= 0.5).astype(np.int64)
    val_pred = (val_prob >= 0.5).astype(np.int64)

    neural_train_acc = accuracy_score(y_train, train_pred)
    neural_val_acc = accuracy_score(y_val, val_pred)
    neural_val_auc = roc_auc_score(y_val, val_prob)

    between, within, ratio = centroid_stats(z_val, y_val)

    print(f"Best epoch: {best_epoch}")
    print(f"Neural train accuracy: {neural_train_acc:.6f}")
    print(f"Neural validation accuracy: {neural_val_acc:.6f}")
    print(f"Neural validation AUC: {neural_val_auc:.6f}")
    print(f"Between-class centroid distance: {between:.6f}")
    print(f"Within-class mean distance: {within:.6f}")
    print(f"Between/within ratio: {ratio:.6f}")

    # --------------------------------------------------------
    # Simple classifiers on learned z
    # --------------------------------------------------------
    print("\n[5/5] Training SVM and Logistic Regression on learned z...")

    z_scaler = StandardScaler()
    z_train_s = z_scaler.fit_transform(z_train)
    z_val_s = z_scaler.transform(z_val)

    svm = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=SEED,
        max_iter=5000,
    )
    svm.fit(z_train_s, y_train)

    svm_decision = svm.decision_function(z_val_s)
    svm_pred = (svm_decision >= 0).astype(np.int64)

    svm_acc = accuracy_score(y_val, svm_pred)
    svm_auc = roc_auc_score(y_val, svm_decision)

    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
        random_state=SEED,
    )
    lr.fit(z_train_s, y_train)

    lr_prob = lr.predict_proba(z_val_s)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(np.int64)

    lr_acc = accuracy_score(y_val, lr_pred)
    lr_auc = roc_auc_score(y_val, lr_prob)

    print(f"SVM on z accuracy: {svm_acc:.6f}")
    print(f"SVM on z AUC     : {svm_auc:.6f}")
    print(f"LR on z accuracy : {lr_acc:.6f}")
    print(f"LR on z AUC      : {lr_auc:.6f}")

    # --------------------------------------------------------
    # Save compact outputs
    # --------------------------------------------------------
    elapsed = time.time() - start_time

    results = {
        "experiment": "Main125",
        "phase": 1,
        "experiment_number": 4,
        "description": "TF-IDF -> SVD -> MLP encoder",

        "seed": SEED,
        "dataset_size": n,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "random_state": SEED,

        "tfidf": {
            "ngram_range": list(TFIDF_NGRAM_RANGE),
            "min_df": TFIDF_MIN_DF,
            "max_df": TFIDF_MAX_DF,
            "max_features": MAX_TFIDF_FEATURES,
            "sublinear_tf": TFIDF_SUBLINEAR,
            "train_features": int(X_train.shape[1]),
        },

        "svd": {
            "components": SVD_DIM,
            "explained_variance_ratio_sum": explained,
        },

        "mlp": {
            "hidden1": HIDDEN1,
            "hidden2": HIDDEN2,
            "latent_dim": LATENT_DIM,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "best_epoch": best_epoch,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "parameters": int(sum(p.numel() for p in model.parameters())),
        },

        "neural": {
            "train_accuracy": float(neural_train_acc),
            "val_accuracy": float(neural_val_acc),
            "val_auc": float(neural_val_auc),
        },

        "svm_on_z": {
            "accuracy": float(svm_acc),
            "auc": float(svm_auc),
        },

        "lr_on_z": {
            "accuracy": float(lr_acc),
            "auc": float(lr_auc),
        },

        "geometry": {
            "between_centroid_distance": float(between),
            "within_class_mean_distance": float(within),
            "between_within_ratio": float(ratio),
        },

        "runtime_minutes": float(elapsed / 60.0),
        "label_mapping": label_mapping,
        "history": history,
    }

    report_path = OUT_DIR / "main125_report.json"
    report_path.write_text(json.dumps(results, indent=2))

    embedding_path = OUT_DIR / "main125_embeddings.npz"
    np.savez_compressed(
        embedding_path,
        train_idx=train_idx,
        val_idx=val_idx,
        y_train=y_train,
        y_val=y_val,
        z_train=z_train.astype(np.float32),
        z_val=z_val.astype(np.float32),
    )

    model_path = OUT_DIR / "main125_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "svd_dim": SVD_DIM,
                "latent_dim": LATENT_DIM,
                "hidden1": HIDDEN1,
                "hidden2": HIDDEN2,
                "dropout": DROPOUT,
                "best_epoch": best_epoch,
            },
        },
        model_path,
    )

    print("\n" + "=" * 72)
    print("MAIN125 COMPLETE")
    print("=" * 72)
    print(f"Neural validation accuracy : {neural_val_acc * 100:.4f}%")
    print(f"Neural validation AUC      : {neural_val_auc:.6f}")
    print(f"SVM on z accuracy          : {svm_acc * 100:.4f}%")
    print(f"SVM on z AUC               : {svm_auc:.6f}")
    print(f"LR on z accuracy           : {lr_acc * 100:.4f}%")
    print(f"LR on z AUC                : {lr_auc:.6f}")
    print(f"Best epoch                 : {best_epoch}")
    print(f"Parameters                 : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Runtime                    : {elapsed / 60.0:.2f} min")
    print("\nSaved:")
    print("  main125_report.json")
    print("  main125_embeddings.npz")
    print("  main125_best.pt")


if __name__ == "__main__":
    main()
