#!/usr/bin/env python3
"""
Main133.py
Phase 3 — Experiment 13:
RAW TOKENS + WORD TF-IDF + STRUCTURAL FEATURES -> FUSED LATENT REPRESENTATION

Goal:
Test whether three complementary representation families can jointly learn
a useful latent space:
  1) raw token sequence -> BiGRU
  2) word/token TF-IDF -> SVD -> MLP
  3) 62-D structural features -> MLP

Canonical split:
  train_test_split(all_idx, test_size=0.20, stratify=y, random_state=42)

Outputs only:
  main133_best.pt
  main133_embeddings.npz
  main133_report.json

No Kaggle submission.
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path("/home/manu/Desktop/ML4CPMS/project_1")
TRAIN_PATH = ROOT / "train.json"

SEED = 133
MAXLEN = 384
TOKEN_EMBED = 96
TOKEN_HIDDEN = 96
TFIDF_DIM = 256
TFIDF_HIDDEN = 128
STRUCT_HIDDEN = 96
LATENT = 128
FUSION_HIDDEN = 256
BATCH = 64
EPOCHS = 35
PATIENCE = 6
LR = 2e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json_or_jsonl(path):
    raw = path.read_text(encoding="utf-8").strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ("train", "data", "records", "documents"):
                if key in obj and isinstance(obj[key], list):
                    return obj[key]
    except json.JSONDecodeError:
        pass

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_text_and_label(row):
    tokens = row["text"]
    label = row.get("label", row.get("target", row.get("y")))
    return tokens, label


def label_to_int(v):
    if isinstance(v, (int, np.integer)):
        return int(v)
    s = str(v).strip().upper()
    if s == "A":
        return 0
    if s == "B":
        return 1
    try:
        return int(v)
    except Exception:
        raise ValueError(f"Unknown label: {v}")


def tokens_to_string(tokens):
    return " ".join(map(str, tokens))


def structural_features(tokens):
    x = np.asarray(tokens, dtype=np.int64)
    n = len(x)

    if n == 0:
        return np.zeros(62, dtype=np.float32)

    u, counts = np.unique(x, return_counts=True)
    counts = counts.astype(np.float64)
    p = counts / n

    # Frequency statistics
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    concentration = float((p * p).sum())
    top1 = float(counts.max() / n)
    sorted_counts = np.sort(counts)[::-1]
    top5 = float(sorted_counts[:5].sum() / n)
    top10 = float(sorted_counts[:10].sum() / n)

    # Transition statistics
    if n > 1:
        a, b = x[:-1], x[1:]
        changes = (a != b)
        tr_unique = len(np.unique(np.stack([a, b], axis=1), axis=0))
        tr_change = float(changes.mean())
        tr_repeat = 1.0 - tr_change
        dif = np.diff(x).astype(np.float64)
        absdif = np.abs(dif)
        tr_mean_abs = float(absdif.mean())
        tr_std_abs = float(absdif.std())
        tr_zero = float((dif == 0).mean())
        tr_sign_changes = float(
            (np.sign(dif[1:]) != np.sign(dif[:-1])).mean()
        ) if len(dif) > 1 else 0.0
    else:
        tr_unique = 0
        tr_change = tr_repeat = tr_mean_abs = tr_std_abs = tr_zero = tr_sign_changes = 0.0

    # Position statistics
    pos = np.arange(n, dtype=np.float64) / max(n - 1, 1)
    weighted_mean_pos = float((pos * np.repeat(p, counts.astype(int))).mean()) if n else 0.0
    q = np.quantile(x.astype(np.float64), [0.1, .25, .5, .75, .9])
    diffs_sorted = np.diff(np.sort(x.astype(np.float64)))
    med_gap = float(np.median(diffs_sorted)) if len(diffs_sorted) else 0.0

    # Chunk statistics
    chunks = np.array_split(x, 4)
    chunk_len = [len(c) for c in chunks]
    chunk_unique = [len(np.unique(c)) if len(c) else 0 for c in chunks]
    chunk_mean = [float(c.mean()) if len(c) else 0.0 for c in chunks]
    chunk_std = [float(c.std()) if len(c) else 0.0 for c in chunks]

    # Local variability
    win = max(2, min(16, n // 8 if n >= 16 else 2))
    local_means = []
    local_stds = []
    for i in range(0, n, win):
        c = x[i:i + win]
        if len(c):
            local_means.append(float(c.mean()))
            local_stds.append(float(c.std()))
    local_mean_std = float(np.std(local_means)) if local_means else 0.0
    local_std_mean = float(np.mean(local_stds)) if local_stds else 0.0

    # Summary statistics
    vals = x.astype(np.float64)
    feats = [
        float(n),
        float(np.log1p(n)),
        float(len(u)),
        float(len(u) / n),
        entropy,
        float(entropy / np.log(max(len(u), 2))),
        concentration,
        top1,
        top5,
        top10,
        float(np.mean(vals)),
        float(np.std(vals)),
        float(np.median(vals)),
        float(np.min(vals)),
        float(np.max(vals)),
        float(np.percentile(vals, 10)),
        float(np.percentile(vals, 25)),
        float(np.percentile(vals, 75)),
        float(np.percentile(vals, 90)),
        float(np.ptp(vals)),
        float(np.mean(np.abs(vals - np.median(vals)))),
        float(np.mean(np.diff(vals))) if n > 1 else 0.0,
        float(np.std(np.diff(vals))) if n > 1 else 0.0,
        float(np.mean(np.abs(np.diff(vals)))) if n > 1 else 0.0,
        tr_change,
        tr_repeat,
        float(tr_unique / max(n - 1, 1)),
        tr_mean_abs,
        tr_std_abs,
        tr_zero,
        tr_sign_changes,
        float(med_gap),
        weighted_mean_pos,
        local_mean_std,
        local_std_mean,
    ]

    feats += [float(v) for v in chunk_len]
    feats += [float(v) for v in chunk_unique]
    feats += [float(v / max(l, 1)) for v, l in zip(chunk_unique, chunk_len)]
    feats += chunk_mean
    feats += chunk_std

    # First/last token and frequency of boundary tokens
    feats += [
        float(x[0]),
        float(x[-1]),
        float(np.mean(x[:max(1, n // 10)])),
        float(np.mean(x[-max(1, n // 10):])),
        float(np.std(x[:max(1, n // 10)])),
        float(np.std(x[-max(1, n // 10):])),
    ]

    # Pad exactly to 62 features if implementation changes slightly.
    if len(feats) < 62:
        feats += [0.0] * (62 - len(feats))
    return np.asarray(feats[:62], dtype=np.float32)


class MultiInputModel(nn.Module):
    def __init__(self, vocab_size, pad_id):
        super().__init__()

        # Raw-token branch
        self.embedding = nn.Embedding(
            vocab_size, TOKEN_EMBED, padding_idx=pad_id
        )
        self.gru = nn.GRU(
            TOKEN_EMBED,
            TOKEN_HIDDEN,
            batch_first=True,
            bidirectional=True,
        )
        self.token_proj = nn.Sequential(
            nn.Linear(TOKEN_HIDDEN * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

        # TF-IDF branch
        self.tfidf_branch = nn.Sequential(
            nn.Linear(TFIDF_DIM, TFIDF_HIDDEN),
            nn.BatchNorm1d(TFIDF_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(TFIDF_HIDDEN, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

        # Structural branch
        self.struct_branch = nn.Sequential(
            nn.Linear(62, STRUCT_HIDDEN),
            nn.BatchNorm1d(STRUCT_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(STRUCT_HIDDEN, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 128, FUSION_HIDDEN),
            nn.BatchNorm1d(FUSION_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(FUSION_HIDDEN, LATENT),
        )
        self.classifier = nn.Linear(LATENT, 1)

    def encode_tokens(self, x):
        e = self.embedding(x)
        out, _ = self.gru(e)

        mask = (x != self.embedding.padding_idx).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = (out * mask).sum(dim=1) / denom
        return self.token_proj(pooled)

    def forward(self, tokens, tfidf, struct):
        a = self.encode_tokens(tokens)
        b = self.tfidf_branch(tfidf)
        c = self.struct_branch(struct)
        z = self.fusion(torch.cat([a, b, c], dim=1))
        logit = self.classifier(z).squeeze(1)
        return logit, z


def make_vocab(all_token_lists):
    # Reserve 0 for PAD and 1 for UNK.
    vocab = {}
    for seq in all_token_lists:
        for t in seq:
            if t not in vocab:
                vocab[t] = len(vocab) + 2
    return vocab


def encode_sequences(seqs, vocab, maxlen):
    arr = np.zeros((len(seqs), maxlen), dtype=np.int64)
    for i, seq in enumerate(seqs):
        ids = [vocab.get(t, 1) for t in seq[:maxlen]]
        if ids:
            arr[i, :len(ids)] = ids
    return arr


def batched_indices(n, batch, shuffle=True):
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for s in range(0, n, batch):
        yield idx[s:s + batch]


def main():
    seed_all(SEED)
    t0 = time.time()

    rows = load_json_or_jsonl(TRAIN_PATH)
    token_lists = []
    labels = []
    for r in rows:
        toks, lab = get_text_and_label(r)
        token_lists.append(list(toks))
        labels.append(label_to_int(lab))

    y = np.asarray(labels, dtype=np.int64)
    n = len(y)
    print("Dataset:", n)
    print("Class counts:", np.bincount(y))

    all_idx = np.arange(n)
    tr_idx, va_idx = train_test_split(
        all_idx,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )
    print("Split:", len(tr_idx), len(va_idx))

    # ---- representations fit only on train ----
    print("Building vocabulary...")
    vocab = make_vocab(token_lists)
    vocab_size = max(vocab.values(), default=1) + 1
    print("Vocab size:", vocab_size)

    print("Encoding raw tokens...")
    X_tokens = encode_sequences(token_lists, vocab, MAXLEN)

    print("Building word TF-IDF...")
    text_strings = [tokens_to_string(s) for s in token_lists]
    tfidf = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\S+",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.995,
        max_features=60000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    X_tfidf_tr = tfidf.fit_transform([text_strings[i] for i in tr_idx])
    X_tfidf_va = tfidf.transform([text_strings[i] for i in va_idx])

    print("TF-IDF shape:", X_tfidf_tr.shape)

    print("SVD...")
    svd = TruncatedSVD(n_components=TFIDF_DIM, random_state=SEED)
    X_svd_tr = svd.fit_transform(X_tfidf_tr).astype(np.float32)
    X_svd_va = svd.transform(X_tfidf_va).astype(np.float32)
    print("SVD explained variance:", float(svd.explained_variance_ratio_.sum()))

    print("Building structural features...")
    X_struct = np.stack([structural_features(s) for s in token_lists])
    scaler = StandardScaler()
    X_struct_tr = scaler.fit_transform(X_struct[tr_idx]).astype(np.float32)
    X_struct_va = scaler.transform(X_struct[va_idx]).astype(np.float32)

    # ---- model ----
    model = MultiInputModel(vocab_size, pad_id=0).to(DEVICE)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    pos = float((y[tr_idx] == 0).sum())
    neg = float((y[tr_idx] == 1).sum())
    # BCEWithLogitsLoss weight for class 0, matching previous class balancing convention.
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    Xt_tr = torch.from_numpy(X_tokens[tr_idx])
    Xt_va = torch.from_numpy(X_tokens[va_idx])
    Xf_tr = torch.from_numpy(X_svd_tr)
    Xf_va = torch.from_numpy(X_svd_va)
    Xs_tr = torch.from_numpy(X_struct_tr)
    Xs_va = torch.from_numpy(X_struct_va)
    yt_tr = torch.from_numpy(y[tr_idx].astype(np.float32))
    yt_va = torch.from_numpy(y[va_idx].astype(np.float32))

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for bi in batched_indices(len(tr_idx), BATCH, shuffle=True):
            tb = Xt_tr[bi].to(DEVICE)
            fb = Xf_tr[bi].to(DEVICE)
            sb = Xs_tr[bi].to(DEVICE)
            yb = yt_tr[bi].to(DEVICE)

            opt.zero_grad(set_to_none=True)
            logits, _ = model(tb, fb, sb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            train_loss += float(loss.item()) * len(bi)
            train_correct += int(((torch.sigmoid(logits) >= 0.5) == yb.bool()).sum())
            train_total += len(bi)

        train_loss /= len(tr_idx)
        train_acc = train_correct / train_total

        model.eval()
        with torch.no_grad():
            val_logits = []
            val_z = []
            for s in range(0, len(va_idx), BATCH):
                tb = Xt_va[s:s+BATCH].to(DEVICE)
                fb = Xf_va[s:s+BATCH].to(DEVICE)
                sb = Xs_va[s:s+BATCH].to(DEVICE)
                lg, zz = model(tb, fb, sb)
                val_logits.append(lg.cpu().numpy())
                val_z.append(zz.cpu().numpy())
            val_logits = np.concatenate(val_logits)
            val_z_epoch = np.concatenate(val_z)
            val_prob = 1.0 / (1.0 + np.exp(-val_logits))
            val_loss = float(
                criterion(
                    torch.from_numpy(val_logits).to(DEVICE),
                    yt_va.to(DEVICE)
                ).item()
            )
            val_acc = accuracy_score(y[va_idx], val_prob >= 0.5)
            val_auc = roc_auc_score(y[va_idx], val_prob)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": float(train_acc),
            "val_loss": val_loss,
            "val_accuracy": float(val_acc),
            "val_auc": float(val_auc),
        })
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} auc {val_auc:.6f}"
        )

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping.")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    # Final validation latent / neural scores
    with torch.no_grad():
        val_logits = []
        val_z = []
        for s in range(0, len(va_idx), BATCH):
            lg, zz = model(
                Xt_va[s:s+BATCH].to(DEVICE),
                Xf_va[s:s+BATCH].to(DEVICE),
                Xs_va[s:s+BATCH].to(DEVICE),
            )
            val_logits.append(lg.cpu().numpy())
            val_z.append(zz.cpu().numpy())
        val_logits = np.concatenate(val_logits)
        z_va = np.concatenate(val_z)

    neural_prob = 1.0 / (1.0 + np.exp(-val_logits))
    neural_acc = accuracy_score(y[va_idx], neural_prob >= 0.5)
    neural_auc = roc_auc_score(y[va_idx], neural_prob)

    print("\nEvaluating classical classifiers on frozen latent z...")

    svm = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        random_state=SEED,
    )
    svm.fit(z_va, y[va_idx])
    svm_prob = svm.predict_proba(z_va)[:, 1]
    svm_acc = accuracy_score(y[va_idx], svm_prob >= 0.5)
    svm_auc = roc_auc_score(y[va_idx], svm_prob)

    lr = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
    )
    lr.fit(z_va, y[va_idx])
    lr_prob = lr.predict_proba(z_va)[:, 1]
    lr_acc = accuracy_score(y[va_idx], lr_prob >= 0.5)
    lr_auc = roc_auc_score(y[va_idx], lr_prob)

    # Geometry diagnostics
    z0 = z_va[y[va_idx] == 0]
    z1 = z_va[y[va_idx] == 1]
    c0 = z0.mean(axis=0)
    c1 = z1.mean(axis=0)
    between = float(np.linalg.norm(c0 - c1))
    within = float(
        0.5 * (
            np.linalg.norm(z0 - c0, axis=1).mean()
            + np.linalg.norm(z1 - c1, axis=1).mean()
        )
    )
    ratio = between / max(within, 1e-12)

    report = {
        "experiment": "Main133 / Phase3 Exp13",
        "description": "Raw tokens + word TF-IDF + structural features fused into a learned 128-D latent representation",
        "seed": SEED,
        "dataset_size": n,
        "train_size": len(tr_idx),
        "val_size": len(va_idx),
        "class_counts": np.bincount(y).tolist(),
        "split": {
            "method": "train_test_split",
            "test_size": 0.20,
            "stratify": True,
            "random_state": 42,
            "order_preserved": True,
        },
        "representations": {
            "raw_tokens": {
                "maxlen": MAXLEN,
                "embedding": TOKEN_EMBED,
                "gru_hidden": TOKEN_HIDDEN,
                "bidirectional": True,
                "projection": 128,
            },
            "word_tfidf": {
                "analyzer": "word",
                "token_pattern": r"(?u)\\S+",
                "ngram_range": [1, 3],
                "min_df": 2,
                "max_df": 0.995,
                "max_features": 60000,
                "sublinear_tf": True,
                "svd_dim": TFIDF_DIM,
                "explained_variance": float(svd.explained_variance_ratio_.sum()),
            },
            "structural": {
                "dimensions": 62,
                "standard_scaler_fit_on_train": True,
            },
        },
        "model": {
            "token_branch_output": 128,
            "tfidf_branch_output": 128,
            "structural_branch_output": 128,
            "fusion_hidden": FUSION_HIDDEN,
            "latent_dim": LATENT,
            "dropout": 0.20,
            "fusion_dropout": 0.25,
            "batch_size": BATCH,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": EPOCHS,
            "patience": PATIENCE,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "results": {
            "neural_accuracy": float(neural_acc),
            "neural_auc": float(neural_auc),
            "svm_on_latent_accuracy": float(svm_acc),
            "svm_on_latent_auc": float(svm_auc),
            "lr_on_latent_accuracy": float(lr_acc),
            "lr_on_latent_auc": float(lr_auc),
            "between_centroid_distance": between,
            "within_class_mean_radius": within,
            "between_within_ratio": ratio,
        },
        "history": history,
        "runtime_seconds": time.time() - t0,
    }

    # Save exactly three useful artifacts.
    torch.save(
        {
            "state_dict": best_state,
            "vocab": vocab,
            "tfidf_vocabulary": tfidf.vocabulary_,
            "tfidf_idf": tfidf.idf_.astype(np.float32),
            "svd_components": svd.components_.astype(np.float32),
            "struct_scaler_mean": scaler.mean_.astype(np.float32),
            "struct_scaler_scale": scaler.scale_.astype(np.float32),
            "config": report["model"],
            "best_epoch": best_epoch,
        },
        ROOT / "main133_best.pt",
    )

    np.savez_compressed(
        ROOT / "main133_embeddings.npz",
        val_indices=va_idx.astype(np.int64),
        val_labels=y[va_idx].astype(np.int64),
        z_val=z_va.astype(np.float32),
        neural_prob=neural_prob.astype(np.float32),
        svm_prob=svm_prob.astype(np.float32),
        lr_prob=lr_prob.astype(np.float32),
    )

    with (ROOT / "main133_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== MAIN133 RESULT ===")
    print("Neural accuracy:", neural_acc)
    print("Neural AUC:", neural_auc)
    print("SVM-on-z accuracy:", svm_acc)
    print("SVM-on-z AUC:", svm_auc)
    print("LR-on-z accuracy:", lr_acc)
    print("LR-on-z AUC:", lr_auc)
    print("Between:", between)
    print("Within:", within)
    print("Ratio:", ratio)
    print("Best epoch:", best_epoch)
    print("Runtime:", time.time() - t0, "sec")
    print("\nSaved:")
    print(ROOT / "main133_best.pt")
    print(ROOT / "main133_embeddings.npz")
    print(ROOT / "main133_report.json")


if __name__ == "__main__":
    main()
