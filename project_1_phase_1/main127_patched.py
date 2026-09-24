#!/usr/bin/env python3
"""
Main127 — Experiment 7
TOKENS + TF-IDF -> fusion MLP -> latent vector

This is Phase 2, Experiment 7 from the experiment tracker.

Architecture:
    raw token sequence -> token encoder ─┐
                                        ├-> fusion MLP -> z -> classifier
    word TF-IDF --------> TF-IDF encoder ┘

IMPORTANT:
    This experiment intentionally does NOT use SVD.
    Experiment 4 already tested TF-IDF -> SVD -> MLP as a standalone branch.
    Here we test whether the raw TF-IDF representation provides complementary
    information when learned jointly with the token representation.

Canonical split:
    train/val = 80/20 stratified, random_state=42

Outputs:
    main127_best.pt
    main127_embeddings.npz
    main127_report.json
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


DATA_PATH = Path("/home/manu/Desktop/ML4CPMS/project_1/train.json")

SEED = 42
MAXLEN = 384

# Token branch
EMBED_DIM = 96
TOKEN_HIDDEN = 96
TOKEN_OUT = 128

# TF-IDF branch
TFIDF_MAX_FEATURES = 60000
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.995
TFIDF_NGRAM = (1, 3)
TFIDF_HIDDEN = 192
TFIDF_OUT = 96

# Fusion
LATENT_DIM = 128
FUSION_HIDDEN = 192

BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 6
LR = 2e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data():
    # The project train.json is JSONL (one JSON object per line), despite
    # the .json extension. Support both JSONL and ordinary JSON so the
    # experiment is robust to either format.
    with open(DATA_PATH, "r") as f:
        raw = f.read()

    try:
        data = json.loads(raw)
        items = (
            data["train"]
            if isinstance(data, dict) and "train" in data
            else data
        )
    except json.JSONDecodeError:
        items = []
        for line_no, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Could not parse train.json at line {line_no}: {e}"
                ) from e

    if isinstance(items, dict):
        items = list(items.values())

    token_lists = []
    labels = []

    for item in items:
        token_lists.append([int(x) for x in item["text"]])
        lab = item.get("label")
        if isinstance(lab, str):
            lab = 0 if lab == "A" else 1
        labels.append(int(lab))

    return token_lists, np.asarray(labels, dtype=np.int64)


def make_token_matrix(token_lists, maxlen):
    x = np.zeros((len(token_lists), maxlen), dtype=np.int64)

    for i, seq in enumerate(token_lists):
        n = min(len(seq), maxlen)
        if n:
            x[i, :n] = np.asarray(seq[:n], dtype=np.int64)

    return x


def make_word_strings(token_lists):
    return [" ".join(map(str, seq)) for seq in token_lists]


class TokenEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            EMBED_DIM,
            padding_idx=0,
        )

        self.gru = nn.GRU(
            EMBED_DIM,
            TOKEN_HIDDEN,
            batch_first=True,
            bidirectional=True,
        )

        self.proj = nn.Sequential(
            nn.Linear(2 * TOKEN_HIDDEN, TOKEN_OUT),
            nn.BatchNorm1d(TOKEN_OUT),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        mask = (x != 0)
        emb = self.embedding(x)
        out, _ = self.gru(emb)

        maskf = mask.unsqueeze(-1).float()
        denom = maskf.sum(dim=1).clamp_min(1.0)
        pooled = (out * maskf).sum(dim=1) / denom

        return self.proj(pooled)


class FusionModel(nn.Module):
    def __init__(self, vocab_size, tfidf_dim):
        super().__init__()

        self.token_encoder = TokenEncoder(vocab_size)

        # Direct TF-IDF encoder.
        # No SVD or other pre-compression is used in Experiment 7.
        self.tfidf_encoder = nn.Sequential(
            nn.Linear(tfidf_dim, TFIDF_HIDDEN),
            nn.BatchNorm1d(TFIDF_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(TFIDF_HIDDEN, TFIDF_OUT),
            nn.BatchNorm1d(TFIDF_OUT),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        self.fusion = nn.Sequential(
            nn.Linear(TOKEN_OUT + TFIDF_OUT, FUSION_HIDDEN),
            nn.BatchNorm1d(FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FUSION_HIDDEN, LATENT_DIM),
        )

        self.classifier = nn.Linear(LATENT_DIM, 1)

    def forward(self, token_x, tfidf_x):
        z_token = self.token_encoder(token_x)
        z_tfidf = self.tfidf_encoder(tfidf_x)
        z = self.fusion(torch.cat([z_token, z_tfidf], dim=1))
        logits = self.classifier(z).squeeze(1)
        return z, logits


def sparse_batch_to_dense(matrix, indices):
    return torch.from_numpy(
        matrix[indices].toarray().astype(np.float32, copy=False)
    )


@torch.no_grad()
def extract_embeddings(model, token_x, tfidf_matrix):
    model.eval()
    zs = []

    for start in range(0, len(token_x), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(token_x))

        tok = torch.from_numpy(token_x[start:end]).to(DEVICE)
        tf = sparse_batch_to_dense(
            tfidf_matrix,
            np.arange(start, end),
        ).to(DEVICE)

        z, _ = model(tok, tf)
        zs.append(z.cpu().numpy())

    return np.concatenate(zs, axis=0)


def evaluate_latent(z_train, y_train, z_val, y_val):
    scaler = StandardScaler()
    ztr = scaler.fit_transform(z_train)
    zva = scaler.transform(z_val)

    svm = LinearSVC(C=1.0)
    svm.fit(ztr, y_train)
    svm_score = svm.decision_function(zva)

    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
    )
    lr.fit(ztr, y_train)
    lr_score = lr.decision_function(zva)

    return {
        "svm_accuracy": float(
            accuracy_score(y_val, svm_score >= 0)
        ),
        "svm_auc": float(
            roc_auc_score(y_val, svm_score)
        ),
        "lr_accuracy": float(
            accuracy_score(y_val, lr_score >= 0)
        ),
        "lr_auc": float(
            roc_auc_score(y_val, lr_score)
        ),
    }


def main():
    start_time = time.time()
    seed_everything(SEED)

    print("=" * 78)
    print("MAIN127 — EXPERIMENT 7: TOKENS + TF-IDF")
    print("=" * 78)
    print("Tracker architecture:")
    print("  Token encoder + TF-IDF encoder -> fusion MLP -> z")
    print("SVD: NONE")
    print(f"Device: {DEVICE}")

    token_lists, y = load_data()
    n = len(y)

    all_idx = np.arange(n)

    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    print(f"Dataset: {n}")
    print(f"Train/Val: {len(train_idx)}/{len(val_idx)}")

    # ---------------------------------------------------------
    # Raw token branch
    # ---------------------------------------------------------
    token_all = make_token_matrix(token_lists, MAXLEN)

    vocab_size = (
        max((max(seq) if seq else 0) for seq in token_lists) + 1
    )

    print(f"Vocab size: {vocab_size}")
    print(f"Token maxlen: {MAXLEN}")

    # ---------------------------------------------------------
    # Word TF-IDF branch — no SVD
    # ---------------------------------------------------------
    print("\nBuilding word TF-IDF (TRAIN ONLY fit)...")

    strings = make_word_strings(token_lists)

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\S+",
        ngram_range=TFIDF_NGRAM,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )

    tf_train = vectorizer.fit_transform(
        [strings[i] for i in train_idx]
    )
    tf_val = vectorizer.transform(
        [strings[i] for i in val_idx]
    )

    print(f"TF-IDF dimensions: {tf_train.shape[1]}")
    print(f"Train nnz: {tf_train.nnz}")
    print(f"Val nnz: {tf_val.nnz}")

    # Train/val token arrays
    token_train = token_all[train_idx]
    token_val = token_all[val_idx]

    y_train_np = y[train_idx]
    y_val_np = y[val_idx]

    model = FusionModel(
        vocab_size=vocab_size,
        tfidf_dim=tf_train.shape[1],
    ).to(DEVICE)

    nparams = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {nparams:,}")

    pos = float(y_train_np.sum())
    neg = float(len(y_train_np) - pos)

    pos_weight = torch.tensor(
        [neg / max(pos, 1.0)],
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = PATIENCE

    history = []

    print("\nTraining...")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        order = np.random.permutation(len(train_idx))

        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for start in range(0, len(order), BATCH_SIZE):
            batch_ids = order[start:start + BATCH_SIZE]

            tok = torch.from_numpy(
                token_train[batch_ids]
            ).to(DEVICE)

            tf = sparse_batch_to_dense(
                tf_train,
                batch_ids,
            ).to(DEVICE)

            yb = torch.from_numpy(
                y_train_np[batch_ids].astype(np.float32)
            ).to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            _, logits = model(tok, tf)

            loss = criterion(logits, yb)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            train_loss_sum += loss.item() * len(batch_ids)

            pred = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            train_correct += int(
                (pred == yb).sum().item()
            )
            train_count += len(batch_ids)

        train_loss = train_loss_sum / train_count
        train_acc = train_correct / train_count

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------
        model.eval()

        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0
        val_scores = []

        with torch.no_grad():
            for start in range(0, len(val_idx), BATCH_SIZE):
                end = min(
                    start + BATCH_SIZE,
                    len(val_idx),
                )

                tok = torch.from_numpy(
                    token_val[start:end]
                ).to(DEVICE)

                tf = sparse_batch_to_dense(
                    tf_val,
                    np.arange(start, end),
                ).to(DEVICE)

                yb = torch.from_numpy(
                    y_val_np[start:end].astype(np.float32)
                ).to(DEVICE)

                _, logits = model(tok, tf)

                loss = criterion(logits, yb)

                val_loss_sum += (
                    loss.item() * (end - start)
                )

                pred = (
                    torch.sigmoid(logits) >= 0.5
                ).float()

                val_correct += int(
                    (pred == yb).sum().item()
                )

                val_count += end - start
                val_scores.append(
                    logits.cpu().numpy()
                )

        val_loss = val_loss_sum / val_count
        val_acc = val_correct / val_count

        val_scores_np = np.concatenate(val_scores)

        val_auc = roc_auc_score(
            y_val_np,
            val_scores_np,
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "val_auc": float(val_auc),
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} "
            f"acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} "
            f"acc {val_acc:.4f} "
            f"AUC {val_auc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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

    # Restore best model
    model.load_state_dict(best_state)
    model.to(DEVICE)

    # ---------------------------------------------------------
    # Final neural validation metrics
    # ---------------------------------------------------------
    model.eval()

    val_logits = []

    with torch.no_grad():
        for start in range(0, len(val_idx), BATCH_SIZE):
            end = min(
                start + BATCH_SIZE,
                len(val_idx),
            )

            tok = torch.from_numpy(
                token_val[start:end]
            ).to(DEVICE)

            tf = sparse_batch_to_dense(
                tf_val,
                np.arange(start, end),
            ).to(DEVICE)

            _, logits = model(tok, tf)

            val_logits.append(
                logits.cpu().numpy()
            )

    val_logits = np.concatenate(val_logits)

    neural_prob = 1.0 / (
        1.0 + np.exp(
            -np.clip(val_logits, -50, 50)
        )
    )

    neural_acc = accuracy_score(
        y_val_np,
        neural_prob >= 0.5,
    )

    neural_auc = roc_auc_score(
        y_val_np,
        val_logits,
    )

    # ---------------------------------------------------------
    # Latent vectors
    # ---------------------------------------------------------
    print("\nExtracting latent vectors...")

    z_train = extract_embeddings(
        model,
        token_train,
        tf_train,
    )

    z_val = extract_embeddings(
        model,
        token_val,
        tf_val,
    )

    latent_metrics = evaluate_latent(
        z_train,
        y_train_np,
        z_val,
        y_val_np,
    )

    # ---------------------------------------------------------
    # Geometry
    # ---------------------------------------------------------
    z0 = z_val[y_val_np == 0]
    z1 = z_val[y_val_np == 1]

    c0 = z0.mean(axis=0)
    c1 = z1.mean(axis=0)

    between = float(
        np.linalg.norm(c0 - c1)
    )

    within = float(
        0.5 * (
            np.mean(
                np.linalg.norm(z0 - c0, axis=1)
            )
            +
            np.mean(
                np.linalg.norm(z1 - c1, axis=1)
            )
        )
    )

    ratio = between / max(within, 1e-12)

    elapsed = time.time() - start_time

    report = {
        "experiment": 7,
        "name": "TOKENS + TF-IDF -> fusion MLP -> latent",
        "description": (
            "Token encoder + direct word TF-IDF encoder + fusion MLP. "
            "No SVD."
        ),
        "seed": SEED,
        "device": str(DEVICE),
        "dataset_size": n,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "maxlen": MAXLEN,
        "vocab_size": vocab_size,
        "tfidf": {
            "ngram_range": list(TFIDF_NGRAM),
            "min_df": TFIDF_MIN_DF,
            "max_df": TFIDF_MAX_DF,
            "max_features": TFIDF_MAX_FEATURES,
            "actual_dimensions": int(tf_train.shape[1]),
            "svd_used": False,
        },
        "model": {
            "embed_dim": EMBED_DIM,
            "token_hidden": TOKEN_HIDDEN,
            "token_out": TOKEN_OUT,
            "tfidf_hidden": TFIDF_HIDDEN,
            "tfidf_out": TFIDF_OUT,
            "fusion_hidden": FUSION_HIDDEN,
            "latent_dim": LATENT_DIM,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "epochs_max": EPOCHS,
            "best_epoch": best_epoch,
            "parameters": nparams,
        },
        "neural": {
            "accuracy": float(neural_acc),
            "auc": float(neural_auc),
        },
        "latent_classifier": latent_metrics,
        "geometry": {
            "between_centroid_distance": between,
            "within_class_mean_radius": within,
            "between_within_ratio": ratio,
        },
        "runtime_seconds": elapsed,
        "history": history,
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "tfidf_dim": int(tf_train.shape[1]),
                "latent_dim": LATENT_DIM,
                "svd_used": False,
            },
            "best_epoch": best_epoch,
        },
        "main127_best.pt",
    )

    np.savez_compressed(
        "main127_embeddings.npz",
        z_train=z_train.astype(np.float32),
        z_val=z_val.astype(np.float32),
        y_train=y_train_np,
        y_val=y_val_np,
        train_idx=train_idx,
        val_idx=val_idx,
    )

    with open("main127_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 78)
    print("FINAL RESULT")
    print("=" * 78)
    print(f"Best epoch:       {best_epoch}")
    print(f"Neural accuracy:  {neural_acc:.6f}")
    print(f"Neural AUC:       {neural_auc:.6f}")
    print(
        f"SVM latent:       "
        f"{latent_metrics['svm_accuracy']:.6f} "
        f"(AUC {latent_metrics['svm_auc']:.6f})"
    )
    print(
        f"LR latent:        "
        f"{latent_metrics['lr_accuracy']:.6f} "
        f"(AUC {latent_metrics['lr_auc']:.6f})"
    )
    print(f"Between/within:   {ratio:.6f}")
    print(f"Runtime:          {elapsed / 60:.2f} min")

    print("\nSaved:")
    print("  main127_best.pt")
    print("  main127_embeddings.npz")
    print("  main127_report.json")


if __name__ == "__main__":
    main()
