#!/usr/bin/env python3
"""
MAIN126 — PHASE 1 / EXPERIMENT 6
STRUCTURAL FEATURES → MLP → LATENT VECTOR

Goal:
    Test whether compact structural/statistical properties of the token
    sequences can themselves form a useful learned document representation.

Canonical split:
    80/20 stratified split, random_state=42.

Pipeline:
    integer token ID sequence
        -> compact structural/statistical feature vector
        -> StandardScaler
        -> MLP encoder
        -> latent vector z
        -> binary classifier

Also evaluates:
    - neural classifier
    - LinearSVC on z
    - LogisticRegression on z
    - latent-space centroid geometry

No Kaggle submission.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42
LATENT_DIM = 128

BATCH_SIZE = 128
MAX_EPOCHS = 40
PATIENCE = 7
LR = 0.002
WEIGHT_DECAY = 1e-4
DROPOUT = 0.20

DATA_PATH = Path("train.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_dataset(path):
    texts = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(row["label"])

    label_values = sorted(set(labels))
    label_map = {lab: i for i, lab in enumerate(label_values)}

    y = np.asarray([label_map[x] for x in labels], dtype=np.int64)
    return texts, y, label_map


# ---------------------------------------------------------------------
# Structural feature extraction
# ---------------------------------------------------------------------

def extract_structural_features(seq):
    """
    Build a compact sequence-level feature vector.

    Features are deliberately representation-level statistics rather
    than token identity features. They describe length, repetition,
    transition behavior, diversity, and local variability.

    The feature set is designed to be robust to documents having
    different lengths and to avoid using the class label.
    """

    x = np.asarray(seq, dtype=np.float64)

    n = len(x)

    if n == 0:
        return np.zeros(62, dtype=np.float32)

    # Basic sequence statistics.
    unique = np.unique(x)
    unique_n = len(unique)

    # Frequencies.
    counts = np.bincount(
        np.searchsorted(unique, x),
        minlength=unique_n,
    ).astype(np.float64)

    probs = counts / n

    # Shannon entropy.
    entropy = -np.sum(
        probs * np.log(probs + 1e-12)
    )

    # Normalized entropy.
    entropy_norm = entropy / max(np.log(n), 1.0)

    # Frequency concentration.
    sorted_counts = np.sort(counts)[::-1]

    top1 = sorted_counts[0] / n
    top5 = sorted_counts[:5].sum() / n
    top10 = sorted_counts[:10].sum() / n

    # Singleton / repeated token statistics.
    singleton_fraction = np.mean(counts == 1)
    repeated_types = np.sum(counts >= 2) / max(unique_n, 1)
    max_frequency = sorted_counts[0] / n

    # Transition statistics.
    if n >= 2:
        d = np.diff(x)

        abs_d = np.abs(d)

        mean_diff = np.mean(d)
        std_diff = np.std(d)
        mean_abs_diff = np.mean(abs_d)
        median_abs_diff = np.median(abs_d)

        zero_transition_rate = np.mean(d == 0)
        positive_transition_rate = np.mean(d > 0)
        negative_transition_rate = np.mean(d < 0)

        sign_changes = np.sum(
            (d[:-1] * d[1:]) < 0
        ) / max(len(d) - 1, 1)

        # Distinct adjacent pairs.
        pairs = np.column_stack((x[:-1], x[1:]))
        unique_pairs = np.unique(pairs, axis=0).shape[0]

        pair_diversity = unique_pairs / max(n - 1, 1)

        # Transition repetition.
        _, pair_counts = np.unique(
            pairs,
            axis=0,
            return_counts=True,
        )

        pair_counts = np.sort(pair_counts)[::-1]

        pair_top1 = pair_counts[0] / max(n - 1, 1)
        pair_top5 = pair_counts[:5].sum() / max(n - 1, 1)
        pair_top10 = pair_counts[:10].sum() / max(n - 1, 1)

        # Runs.
        run_breaks = np.sum(d != 0)
        run_count = run_breaks + 1
        mean_run_length = n / max(run_count, 1)

    else:
        mean_diff = 0.0
        std_diff = 0.0
        mean_abs_diff = 0.0
        median_abs_diff = 0.0
        zero_transition_rate = 0.0
        positive_transition_rate = 0.0
        negative_transition_rate = 0.0
        sign_changes = 0.0
        unique_pairs = 0
        pair_diversity = 0.0
        pair_top1 = 0.0
        pair_top5 = 0.0
        pair_top10 = 0.0
        run_count = 1
        mean_run_length = float(n)

    # Token-position statistics.
    positions = np.arange(n, dtype=np.float64)

    if n >= 2:
        # Correlation between token ID and position.
        pos_std = np.std(positions)
        token_std = np.std(x)

        if pos_std > 0 and token_std > 0:
            position_corr = np.corrcoef(
                positions,
                x,
            )[0, 1]
        else:
            position_corr = 0.0

        # Statistics over chunks.
        q1 = max(n // 4, 1)
        q2 = max(n // 2, 1)
        q3 = max(3 * n // 4, 1)

        chunks = [
            x[:q1],
            x[q1:q2],
            x[q2:q3],
            x[q3:],
        ]

        chunk_means = [
            float(np.mean(c)) if len(c) else 0.0
            for c in chunks
        ]

        chunk_stds = [
            float(np.std(c)) if len(c) else 0.0
            for c in chunks
        ]

        chunk_ranges = [
            float(np.ptp(c)) if len(c) else 0.0
            for c in chunks
        ]

        mean_shift_12 = chunk_means[1] - chunk_means[0]
        mean_shift_23 = chunk_means[2] - chunk_means[1]
        mean_shift_34 = chunk_means[3] - chunk_means[2]

        first_last_mean_diff = chunk_means[3] - chunk_means[0]

        # Diversity in each quarter.
        chunk_unique_rates = [
            len(np.unique(c)) / max(len(c), 1)
            for c in chunks
        ]

    else:
        position_corr = 0.0
        chunk_means = [float(x[0]), 0.0, 0.0, 0.0]
        chunk_stds = [0.0, 0.0, 0.0, 0.0]
        chunk_ranges = [0.0, 0.0, 0.0, 0.0]
        mean_shift_12 = 0.0
        mean_shift_23 = 0.0
        mean_shift_34 = 0.0
        first_last_mean_diff = 0.0
        chunk_unique_rates = [1.0, 0.0, 0.0, 0.0]

    # Token ID distribution statistics.
    token_mean = np.mean(x)
    token_std = np.std(x)
    token_min = np.min(x)
    token_max = np.max(x)
    token_range = token_max - token_min

    token_median = np.median(x)
    token_q25 = np.percentile(x, 25)
    token_q75 = np.percentile(x, 75)
    token_iqr = token_q75 - token_q25

    # First-order local variability.
    if n >= 3:
        second_diff = np.diff(x, n=2)

        second_diff_mean = np.mean(second_diff)
        second_diff_std = np.std(second_diff)
        second_diff_abs_mean = np.mean(np.abs(second_diff))

        # Local direction consistency.
        d1 = np.diff(x)
        same_direction = np.mean(
            np.sign(d1[:-1]) == np.sign(d1[1:])
        )
    else:
        second_diff_mean = 0.0
        second_diff_std = 0.0
        second_diff_abs_mean = 0.0
        same_direction = 0.0

    # Assemble exactly 62 features.
    features = np.array([
        # 1-10: length/diversity/frequency
        n,
        np.log1p(n),
        unique_n,
        unique_n / n,
        entropy,
        entropy_norm,
        top1,
        top5,
        top10,
        singleton_fraction,

        # 11-20: repetition/transitions
        repeated_types,
        max_frequency,
        mean_diff,
        std_diff,
        mean_abs_diff,
        median_abs_diff,
        zero_transition_rate,
        positive_transition_rate,
        negative_transition_rate,
        sign_changes,

        # 21-30: transition structure
        unique_pairs,
        pair_diversity,
        pair_top1,
        pair_top5,
        pair_top10,
        run_count,
        mean_run_length,
        position_corr,
        mean_shift_12,
        mean_shift_23,

        # 31-40: positional/chunk statistics
        mean_shift_34,
        first_last_mean_diff,
        chunk_means[0],
        chunk_means[1],
        chunk_means[2],
        chunk_means[3],
        chunk_stds[0],
        chunk_stds[1],
        chunk_stds[2],
        chunk_stds[3],

        # 41-50: chunk ranges/diversity
        chunk_ranges[0],
        chunk_ranges[1],
        chunk_ranges[2],
        chunk_ranges[3],
        chunk_unique_rates[0],
        chunk_unique_rates[1],
        chunk_unique_rates[2],
        chunk_unique_rates[3],
        token_mean,
        token_std,

        # 51-62: distribution/local variability
        token_min,
        token_max,
        token_range,
        token_median,
        token_q25,
        token_q75,
        token_iqr,
        second_diff_mean,
        second_diff_std,
        second_diff_abs_mean,
        same_direction,
        repeated_types * entropy_norm,
    ], dtype=np.float32)

    assert len(features) == 62, len(features)

    return features


def build_features(texts):
    return np.stack(
        [extract_structural_features(seq) for seq in texts]
    ).astype(np.float32)


# ---------------------------------------------------------------------
# MLP encoder
# ---------------------------------------------------------------------

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(128, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(96, latent_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, x, return_z=False):
        z = self.encoder(x)
        logits = self.classifier(z).squeeze(1)

        if return_z:
            return logits, z

        return logits


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def safe_auc(y, scores):
    try:
        return roc_auc_score(y, scores)
    except Exception:
        return float("nan")


def latent_geometry(z, y):
    z0 = z[y == 0]
    z1 = z[y == 1]

    c0 = z0.mean(axis=0)
    c1 = z1.mean(axis=0)

    between = float(np.linalg.norm(c0 - c1))

    within0 = np.linalg.norm(z0 - c0, axis=1).mean()
    within1 = np.linalg.norm(z1 - c1, axis=1).mean()
    within = float((within0 + within1) / 2.0)

    ratio = between / max(within, 1e-12)

    return between, within, ratio


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    seed_everything()

    start_time = time.time()

    print("=" * 72)
    print("MAIN126 — PHASE 1 / EXPERIMENT 6")
    print("STRUCTURAL FEATURES → MLP → LATENT VECTOR")
    print("=" * 72)
    print(f"Device: {DEVICE}")

    texts, y, label_map = load_dataset(DATA_PATH)

    n = len(y)
    all_idx = np.arange(n)

    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )

    print(f"Label mapping: {label_map}")
    print(f"Dataset: {n}")
    print(
        f"Train/validation: {len(train_idx)} / {len(val_idx)}"
    )

    # -------------------------------------------------------------
    # Structural features
    # -------------------------------------------------------------

    print("\n[1/5] Extracting compact structural features...")

    all_features = build_features(texts)

    X_train = all_features[train_idx]
    X_val = all_features[val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]

    print(f"Structural train shape: {X_train.shape}")
    print(f"Structural val shape  : {X_val.shape}")

    # Replace possible numerical non-finite values safely.
    X_train = np.nan_to_num(
        X_train,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    X_val = np.nan_to_num(
        X_val,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    # -------------------------------------------------------------
    # Standardization
    # -------------------------------------------------------------

    print("\n[2/5] Standardizing structural features on TRAIN ONLY...")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    Xtr = torch.from_numpy(X_train)
    ytr = torch.from_numpy(y_train.astype(np.float32))
    Xva = torch.from_numpy(X_val)
    yva = torch.from_numpy(y_val.astype(np.float32))

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, ytr),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    # -------------------------------------------------------------
    # MLP
    # -------------------------------------------------------------

    print("\n[3/5] Training MLP encoder...")

    model = MLPEncoder(
        input_dim=X_train.shape[1],
        latent_dim=LATENT_DIM,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total_count = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_count += bs

        train_loss = total_loss / total_count

        model.eval()

        with torch.no_grad():
            val_logits = model(Xva.to(DEVICE)).detach().cpu()
            val_loss = criterion(val_logits, yva).item()

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f}"
        )

        if val_loss < best_val_loss:
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

    model.load_state_dict(best_state)
    model.eval()

    # -------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------

    print("\n[4/5] Evaluating learned representation...")

    with torch.no_grad():
        train_logits, z_train_t = model(
            Xtr.to(DEVICE),
            return_z=True,
        )
        val_logits, z_val_t = model(
            Xva.to(DEVICE),
            return_z=True,
        )

    train_scores = torch.sigmoid(train_logits).cpu().numpy()
    val_scores = torch.sigmoid(val_logits).cpu().numpy()

    z_train = z_train_t.cpu().numpy()
    z_val = z_val_t.cpu().numpy()

    neural_train_acc = accuracy_score(
        y_train,
        train_scores >= 0.5,
    )
    neural_val_acc = accuracy_score(
        y_val,
        val_scores >= 0.5,
    )
    neural_val_auc = safe_auc(y_val, val_scores)

    between, within, ratio = latent_geometry(z_val, y_val)

    print(f"Best epoch: {best_epoch}")
    print(f"Neural train accuracy: {neural_train_acc:.6f}")
    print(f"Neural validation accuracy: {neural_val_acc:.6f}")
    print(f"Neural validation AUC: {neural_val_auc:.6f}")
    print(f"Between-class centroid distance: {between:.6f}")
    print(f"Within-class mean distance: {within:.6f}")
    print(f"Between/within ratio: {ratio:.6f}")

    # -------------------------------------------------------------
    # Classical classifiers on z
    # -------------------------------------------------------------

    print("\n[5/5] Training SVM and Logistic Regression on learned z...")

    svm = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
    )
    svm.fit(z_train, y_train)

    svm_pred = svm.predict(z_val)
    svm_scores = svm.decision_function(z_val)

    svm_acc = accuracy_score(y_val, svm_pred)
    svm_auc = safe_auc(y_val, svm_scores)

    print(f"SVM on z accuracy: {svm_acc:.6f}")
    print(f"SVM on z AUC     : {svm_auc:.6f}")

    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
    )
    lr.fit(z_train, y_train)

    lr_pred = lr.predict(z_val)
    lr_scores = lr.decision_function(z_val)

    lr_acc = accuracy_score(y_val, lr_pred)
    lr_auc = safe_auc(y_val, lr_scores)

    print(f"LR on z accuracy : {lr_acc:.6f}")
    print(f"LR on z AUC      : {lr_auc:.6f}")

    # -------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------

    runtime = (time.time() - start_time) / 60.0

    report = {
        "experiment": "Main126",
        "phase": 1,
        "experiment_number": 6,
        "description": "Structural features -> MLP -> latent vector",
        "seed": SEED,
        "dataset_size": int(n),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "random_state": SEED,
        "structural_features": {
            "dimension": int(X_train.shape[1]),
            "description": (
                "length, token diversity, entropy, frequency "
                "concentration, transition statistics, positional "
                "statistics, chunk statistics, and local variability"
            ),
        },
        "mlp": {
            "input_dim": int(X_train.shape[1]),
            "hidden1": 128,
            "hidden2": 96,
            "latent_dim": LATENT_DIM,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "best_epoch": int(best_epoch),
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "parameters": int(
                sum(p.numel() for p in model.parameters())
            ),
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
        "runtime_minutes": float(runtime),
        "label_mapping": label_map,
        "history": history,
    }

    with open("main126_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    np.savez_compressed(
        "main126_embeddings.npz",
        z_train=z_train.astype(np.float32),
        z_val=z_val.astype(np.float32),
        raw_features_train=X_train.astype(np.float32),
        raw_features_val=X_val.astype(np.float32),
        y_train=y_train.astype(np.int64),
        y_val=y_val.astype(np.int64),
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "config": {
                "seed": SEED,
                "input_dim": int(X_train.shape[1]),
                "latent_dim": LATENT_DIM,
            },
        },
        "main126_best.pt",
    )

    print("\n" + "=" * 72)
    print("MAIN126 COMPLETE")
    print("=" * 72)
    print(f"Neural validation accuracy : {neural_val_acc:.4%}")
    print(f"Neural validation AUC      : {neural_val_auc:.6f}")
    print(f"SVM on z accuracy          : {svm_acc:.4%}")
    print(f"SVM on z AUC               : {svm_auc:.6f}")
    print(f"LR on z accuracy           : {lr_acc:.4%}")
    print(f"LR on z AUC                : {lr_auc:.6f}")
    print(f"Best epoch                 : {best_epoch}")
    print(
        "Parameters                 : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )
    print(f"Runtime                    : {runtime:.2f} min")
    print("\nSaved:")
    print("  main126_report.json")
    print("  main126_embeddings.npz")
    print("  main126_best.pt")


if __name__ == "__main__":
    main()
