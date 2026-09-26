#!/usr/bin/env python3
"""
MAIN156 / EXPERIMENT 36
STRICT OOF MAIN131 + MAIN133 FUSION

Goal
----
Test whether the two strongest learned representations can exploit each
other's complementary information without using Main115.

Protocol
--------
- Canonical split: train_test_split(..., test_size=.20, stratify=y,
  random_state=42), order preserved.
- Main131 validation latent is taken from its saved canonical artifact.
- Main133 validation latent is taken from its saved canonical artifact.
- For the 8428 canonical training rows, BOTH representations are generated
  strictly out-of-fold using the same 5 folds.
- Main131 fold models use the selected Main131 training regime: 1 epoch.
- Main133 fold models use the selected Main133 training regime: 3 epochs.
- TF-IDF vocabularies, IDF/SVD, structural scalers, token vocabularies and
  model weights are fit only on each fold's fit partition.
- Fusion LR/SVM are trained only on the strict OOF 256-D representation.
- The 2108-row canonical validation set is never used to fit the fusion.
- No validation tuning.
- No Kaggle submission.
"""

import json
import random
import time
import importlib.util
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path("/home/manu/Desktop/ML4CPMS/project_1")
TRAIN_PATH = ROOT / "train.json"
MAIN131_EMBEDDINGS = ROOT / "main131_embeddings.npz"
MAIN133_EMBEDDINGS = ROOT / "main133_embeddings.npz"
MAIN115_RESULTS = ROOT / "main115_results.npz"

SEED = 156
N_SPLITS = 5

# Match the selected representation-training regimes.
MAIN131_EPOCHS = 1
MAIN133_EPOCHS = 3

# Main131 exact settings.
MAIN131_MAX_FEATURES = 60000
MAIN131_STRUCT_DIM = 62
MAIN131_TFIDF_HIDDEN = 128
MAIN131_STRUCT_HIDDEN = 128
MAIN131_FUSION_HIDDEN = 192
MAIN131_LATENT = 128
MAIN131_DROPOUT = 0.20
MAIN131_LR = 0.002
MAIN131_WEIGHT_DECAY = 1e-4
MAIN131_BATCH = 128

# Main133 exact settings.
MAIN133_MAXLEN = 384
MAIN133_TOKEN_EMBED = 96
MAIN133_TOKEN_HIDDEN = 96
MAIN133_TFIDF_DIM = 256
MAIN133_TFIDF_HIDDEN = 128
MAIN133_STRUCT_HIDDEN = 96
MAIN133_LATENT = 128
MAIN133_FUSION_HIDDEN = 256
MAIN133_DROPOUT = 0.20
MAIN133_FUSION_DROPOUT = 0.25
MAIN133_LR = 2e-3
MAIN133_WEIGHT_DECAY = 1e-4
MAIN133_BATCH = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def label_to_int(v):
    if isinstance(v, (int, np.integer)):
        return int(v)
    s = str(v).strip().upper()
    if s == "A":
        return 0
    if s == "B":
        return 1
    return int(v)


def tokens_to_string(tokens):
    return " ".join(map(str, tokens))


def load_reference_module(filename, module_name):
    path = ROOT / filename
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def encode_sequences(seqs, vocab, maxlen):
    arr = np.zeros((len(seqs), maxlen), dtype=np.int64)
    for i, seq in enumerate(seqs):
        ids = [vocab.get(t, 1) for t in seq[:maxlen]]
        if ids:
            arr[i, :len(ids)] = ids
    return arr


def batched_indices(n, batch, rng):
    idx = np.arange(n)
    rng.shuffle(idx)
    for s in range(0, n, batch):
        yield idx[s:s + batch]


# ============================================================
# Main131 strict fold representation
# ============================================================

def main131_fold_representation(token_lists, y, fit_idx, hold_idx,
                                main131, fold_no):
    fit_tokens = [token_lists[i] for i in fit_idx]
    hold_tokens = [token_lists[i] for i in hold_idx]

    fit_text = [tokens_to_string(s) for s in fit_tokens]
    hold_text = [tokens_to_string(s) for s in hold_tokens]

    tfidf = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\S+",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.995,
        max_features=MAIN131_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )
    X_fit = tfidf.fit_transform(fit_text)
    X_hold = tfidf.transform(hold_text)

    struct_fit = np.vstack(
        [main131.sequence_features(s) for s in fit_tokens]
    ).astype(np.float32)
    struct_hold = np.vstack(
        [main131.sequence_features(s) for s in hold_tokens]
    ).astype(np.float32)

    scaler = StandardScaler()
    struct_fit = scaler.fit_transform(struct_fit).astype(np.float32)
    struct_hold = scaler.transform(struct_hold).astype(np.float32)

    model = main131.FusionEncoder(
        tfidf_dim=X_fit.shape[1],
        struct_dim=MAIN131_STRUCT_DIM,
        tfidf_hidden=MAIN131_TFIDF_HIDDEN,
        struct_hidden=MAIN131_STRUCT_HIDDEN,
        fusion_hidden=MAIN131_FUSION_HIDDEN,
        latent_dim=MAIN131_LATENT,
        dropout=MAIN131_DROPOUT,
    ).to(DEVICE)

    y_fit = y[fit_idx].astype(np.float32)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MAIN131_LR,
        weight_decay=MAIN131_WEIGHT_DECAY,
    )

    # IMPORTANT: X_fit is a sparse 60k-dimensional matrix. Never call
    # X_fit.toarray() for the complete fold. Dense conversion of a
    # 6742 x 60000 matrix is ~1.6 GB by itself and was the cause of the
    # previous process being killed.
    xs = torch.from_numpy(struct_fit)
    yt = torch.from_numpy(y_fit)

    rng = np.random.default_rng(SEED + fold_no)
    model.train()

    for epoch in range(1, MAIN131_EPOCHS + 1):
        total_loss = 0.0

        for bi in batched_indices(len(fit_idx), MAIN131_BATCH, rng):
            # Convert ONLY this mini-batch from sparse -> dense.
            batch_tfidf = X_fit[bi].toarray().astype(np.float32, copy=False)

            tb = torch.from_numpy(batch_tfidf).to(
                DEVICE, non_blocking=True
            )
            sb = xs[bi].to(DEVICE, non_blocking=True)
            yb = yt[bi].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(tb, sb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(bi)

            del batch_tfidf, tb, sb, yb, logits, loss

        print(
            f"  Main131 | epoch {epoch}/{MAIN131_EPOCHS} | "
            f"loss {total_loss / len(fit_idx):.5f}"
        )

    model.eval()

    z_parts = []
    with torch.no_grad():
        for s in range(0, len(hold_idx), MAIN131_BATCH):
            e = min(s + MAIN131_BATCH, len(hold_idx))

            # Again, densify ONLY one inference batch.
            batch_tfidf = X_hold[s:e].toarray().astype(
                np.float32, copy=False
            )
            tb = torch.from_numpy(batch_tfidf).to(
                DEVICE, non_blocking=True
            )
            sb = torch.from_numpy(struct_hold[s:e]).to(
                DEVICE, non_blocking=True
            )

            z = model.encode(tb, sb)
            z_parts.append(z.cpu().numpy())

            del batch_tfidf, tb, sb, z

    result = np.concatenate(z_parts).astype(np.float32)

    # Release the fold model and large preprocessing objects before the
    # Main133 branch starts.
    del model, X_fit, X_hold, struct_fit, struct_hold
    del xs, yt, optimizer, criterion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# Main133 strict fold representation
# ============================================================

def main133_fold_representation(token_lists, y, fit_idx, hold_idx,
                                main133, fold_no):
    fit_tokens = [token_lists[i] for i in fit_idx]
    hold_tokens = [token_lists[i] for i in hold_idx]

    # Fold-only token vocabulary.
    vocab = main133.make_vocab(fit_tokens)
    vocab_size = max(vocab.values(), default=1) + 1

    tokens_fit = encode_sequences(
        fit_tokens, vocab, MAIN133_MAXLEN
    )
    tokens_hold = encode_sequences(
        hold_tokens, vocab, MAIN133_MAXLEN
    )

    fit_text = [tokens_to_string(s) for s in fit_tokens]
    hold_text = [tokens_to_string(s) for s in hold_tokens]

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
    X_tf_fit = tfidf.fit_transform(fit_text)
    X_tf_hold = tfidf.transform(hold_text)

    svd = TruncatedSVD(
        n_components=MAIN133_TFIDF_DIM,
        random_state=SEED + fold_no,
    )
    X_svd_fit = svd.fit_transform(X_tf_fit).astype(np.float32)
    X_svd_hold = svd.transform(X_tf_hold).astype(np.float32)

    struct_fit = np.vstack(
        [main133.structural_features(s) for s in fit_tokens]
    ).astype(np.float32)
    struct_hold = np.vstack(
        [main133.structural_features(s) for s in hold_tokens]
    ).astype(np.float32)

    scaler = StandardScaler()
    struct_fit = scaler.fit_transform(struct_fit).astype(np.float32)
    struct_hold = scaler.transform(struct_hold).astype(np.float32)

    model = main133.MultiInputModel(
        vocab_size,
        pad_id=0,
    ).to(DEVICE)

    y_fit = y[fit_idx].astype(np.float32)
    n0 = float((y_fit == 0).sum())
    n1 = float((y_fit == 1).sum())
    pos_weight = torch.tensor(
        [n1 / max(n0, 1.0)],
        device=DEVICE,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MAIN133_LR,
        weight_decay=MAIN133_WEIGHT_DECAY,
    )

    xt = torch.from_numpy(tokens_fit)
    xf = torch.from_numpy(X_svd_fit)
    xs = torch.from_numpy(struct_fit)
    yt = torch.from_numpy(y_fit)

    rng = np.random.default_rng(SEED + 100 + fold_no)
    model.train()

    for epoch in range(1, MAIN133_EPOCHS + 1):
        total_loss = 0.0
        for bi in batched_indices(len(fit_idx), MAIN133_BATCH, rng):
            tb = xt[bi].to(DEVICE)
            fb = xf[bi].to(DEVICE)
            sb = xs[bi].to(DEVICE)
            yb = yt[bi].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(tb, fb, sb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(bi)

        print(
            f"  Main133 | epoch {epoch}/{MAIN133_EPOCHS} | "
            f"loss {total_loss / len(fit_idx):.5f}"
        )

    model.eval()
    xt_h = torch.from_numpy(tokens_hold)
    xf_h = torch.from_numpy(X_svd_hold)
    xs_h = torch.from_numpy(struct_hold)

    z_parts = []
    with torch.no_grad():
        for s in range(0, len(hold_idx), MAIN133_BATCH):
            _, z = model(
                xt_h[s:s + MAIN133_BATCH].to(DEVICE),
                xf_h[s:s + MAIN133_BATCH].to(DEVICE),
                xs_h[s:s + MAIN133_BATCH].to(DEVICE),
            )
            z_parts.append(z.cpu().numpy())

    result = np.concatenate(z_parts).astype(np.float32)

    # Release all large fold-local objects before the next fold.
    del model, X_tf_fit, X_tf_hold, X_svd_fit, X_svd_hold
    del tokens_fit, tokens_hold, struct_fit, struct_hold
    del optimizer, criterion, xt, xf, xs, yt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return result


def main():
    start = time.time()
    seed_all(SEED)

    print("=" * 78)
    print("MAIN156 — STRICT OOF MAIN131 + MAIN133 FUSION")
    print("=" * 78)
    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    for p in (
        TRAIN_PATH,
        MAIN131_EMBEDDINGS,
        MAIN133_EMBEDDINGS,
        MAIN115_RESULTS,
    ):
        if not p.exists():
            raise FileNotFoundError(p)

    rows = load_json_or_jsonl(TRAIN_PATH)
    token_lists = [list(map(int, r["text"])) for r in rows]
    y = np.asarray(
        [label_to_int(r["label"]) for r in rows],
        dtype=np.int64,
    )

    print("Dataset:", len(y))

    main131 = load_reference_module("main131.py", "main131_ref_156")
    main133 = load_reference_module("main133.py", "main133_ref_156")
    print("Loaded Main131 and Main133 reference implementations.")

    all_idx = np.arange(len(y))
    tr_idx, va_idx = train_test_split(
        all_idx,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )
    tr_idx = np.asarray(tr_idx, dtype=np.int64)
    va_idx = np.asarray(va_idx, dtype=np.int64)

    print(
        f"Canonical train/validation: {len(tr_idx)} {len(va_idx)}"
    )
    print("Canonical validation order preserved: YES")

    # ------------------------------------------------------------
    # Load and validate saved validation embeddings.
    # ------------------------------------------------------------
    m131 = np.load(MAIN131_EMBEDDINGS, allow_pickle=True)
    m133 = np.load(MAIN133_EMBEDDINGS, allow_pickle=True)

    m131_va_idx = np.asarray(m131["val_indices"], dtype=np.int64)
    m133_va_idx = np.asarray(m133["val_indices"], dtype=np.int64)

    if not np.array_equal(m131_va_idx, va_idx):
        raise RuntimeError("Main131 saved validation indices mismatch.")
    if not np.array_equal(m133_va_idx, va_idx):
        raise RuntimeError("Main133 saved validation indices mismatch.")

    z131_val = np.asarray(m131["z_val"], dtype=np.float32)
    z133_val = np.asarray(m133["z_val"], dtype=np.float32)

    if z131_val.shape != (len(va_idx), 128):
        raise RuntimeError(f"Unexpected Main131 val shape: {z131_val.shape}")
    if z133_val.shape != (len(va_idx), 128):
        raise RuntimeError(f"Unexpected Main133 val shape: {z133_val.shape}")

    print("Main131 validation embedding alignment: PASS")
    print("Main133 validation embedding alignment: PASS")
    print("Main131 z_val shape:", z131_val.shape)
    print("Main133 z_val shape:", z133_val.shape)

    # ------------------------------------------------------------
    # Strict 5-fold OOF for BOTH representations.
    # ------------------------------------------------------------
    print("\nStarting strict 5-fold OOF generation for BOTH representations...")

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    z131_oof = np.zeros((len(tr_idx), 128), dtype=np.float32)
    z133_oof = np.zeros((len(tr_idx), 128), dtype=np.float32)

    y_train = y[tr_idx]

    for fold_no, (fit_pos, hold_pos) in enumerate(
        skf.split(tr_idx, y_train),
        start=1,
    ):
        fit_idx = tr_idx[fit_pos]
        hold_idx = tr_idx[hold_pos]

        print("\n" + "-" * 70)
        print(
            f"FOLD {fold_no}/{N_SPLITS} | "
            f"fit={len(fit_idx)} holdout={len(hold_idx)}"
        )

        # Explicitly seed each representation separately so that the two
        # models are deterministic but do not accidentally share RNG state.
        seed_all(SEED + fold_no)

        z131_hold = main131_fold_representation(
            token_lists,
            y,
            fit_idx,
            hold_idx,
            main131,
            fold_no,
        )

        # Main131 is fully released before Main133 starts.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        seed_all(SEED + 100 + fold_no)

        z133_hold = main133_fold_representation(
            token_lists,
            y,
            fit_idx,
            hold_idx,
            main133,
            fold_no,
        )

        if z131_hold.shape != (len(hold_idx), 128):
            raise RuntimeError(
                f"Fold {fold_no}: Main131 holdout shape mismatch."
            )
        if z133_hold.shape != (len(hold_idx), 128):
            raise RuntimeError(
                f"Fold {fold_no}: Main133 holdout shape mismatch."
            )

        z131_oof[hold_pos] = z131_hold
        z133_oof[hold_pos] = z133_hold

        del z131_hold, z133_hold
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not np.all(np.isfinite(z131_oof)):
        raise RuntimeError("Main131 OOF contains non-finite values.")
    if not np.all(np.isfinite(z133_oof)):
        raise RuntimeError("Main133 OOF contains non-finite values.")

    print("\nStrict Main131 OOF generation: PASS")
    print("Strict Main133 OOF generation: PASS")
    print("Main131 z_oof shape:", z131_oof.shape)
    print("Main133 z_oof shape:", z133_oof.shape)

    # ------------------------------------------------------------
    # Fusion.
    # ------------------------------------------------------------
    X_oof = np.hstack([z131_oof, z133_oof]).astype(np.float32)
    X_val = np.hstack([z131_val, z133_val]).astype(np.float32)

    print("Fusion train shape:", X_oof.shape)
    print("Fusion validation shape:", X_val.shape)

    lr = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=3000,
        random_state=SEED,
    )
    lr.fit(X_oof, y_train)

    lr_oof_prob = lr.predict_proba(X_oof)[:, 1]
    lr_val_prob = lr.predict_proba(X_val)[:, 1]

    lr_oof_pred = (lr_oof_prob >= 0.5).astype(np.int64)
    lr_val_pred = (lr_val_prob >= 0.5).astype(np.int64)

    lr_oof_acc = accuracy_score(y_train, lr_oof_pred)
    lr_val_acc = accuracy_score(y[va_idx], lr_val_pred)
    lr_oof_auc = roc_auc_score(y_train, lr_oof_prob)
    lr_val_auc = roc_auc_score(y[va_idx], lr_val_prob)

    print("\n=== STRICT OOF LOGISTIC REGRESSION FUSION ===")
    print(f"OOF accuracy: {lr_oof_acc}")
    print(f"OOF AUC     : {lr_oof_auc}")
    print(f"VAL accuracy: {lr_val_acc}")
    print(f"VAL AUC     : {lr_val_auc}")

    svm = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=SEED,
    )
    svm.fit(X_oof, y_train)

    svm_oof_score = svm.decision_function(X_oof)
    svm_val_score = svm.decision_function(X_val)

    svm_oof_pred = (svm_oof_score >= 0.0).astype(np.int64)
    svm_val_pred = (svm_val_score >= 0.0).astype(np.int64)

    svm_oof_acc = accuracy_score(y_train, svm_oof_pred)
    svm_val_acc = accuracy_score(y[va_idx], svm_val_pred)
    svm_oof_auc = roc_auc_score(y_train, svm_oof_score)
    svm_val_auc = roc_auc_score(y[va_idx], svm_val_score)

    print("\n=== STRICT OOF LINEAR SVM FUSION ===")
    print(f"OOF accuracy: {svm_oof_acc}")
    print(f"OOF AUC     : {svm_oof_auc}")
    print(f"VAL accuracy: {svm_val_acc}")
    print(f"VAL AUC     : {svm_val_auc}")

    # ------------------------------------------------------------
    # Reference models for context.
    # ------------------------------------------------------------
    m115 = np.load(MAIN115_RESULTS, allow_pickle=True)
    y115_val = np.asarray(m115["y_val"], dtype=np.int64)
    specialist_val = np.asarray(
        m115["specialist_p_val"], dtype=np.float32
    )

    if not np.array_equal(y115_val, y[va_idx]):
        raise RuntimeError("Main115 validation alignment failed.")

    mean_p = specialist_val.mean(axis=1)
    mean_acc = accuracy_score(y[va_idx], mean_p >= 0.5)
    mean_auc = roc_auc_score(y[va_idx], mean_p)

    frozen = np.asarray(
        m115["corrected_val"], dtype=np.int64
    )
    frozen_acc = accuracy_score(y[va_idx], frozen)

    print("\n=== REFERENCES ===")
    print(f"Main115 specialist mean VAL accuracy: {mean_acc}")
    print(f"Main115 specialist mean VAL AUC     : {mean_auc}")
    print(f"Main115 frozen consensus VAL accuracy: {frozen_acc}")

    # Individual learned representations using train-fitted probes where
    # available from the saved Main131/Main133 artifacts are deliberately
    # NOT used as fusion training inputs. The strict OOF fusion above is the
    # primary experiment.

    report = {
        "experiment": "Main156 / Phase9 Exp36",
        "description": (
            "Strict 5-fold OOF fusion of Main131 and Main133 learned "
            "representations."
        ),
        "seed": SEED,
        "device": str(DEVICE),
        "dataset_size": int(len(y)),
        "canonical_train_size": int(len(tr_idx)),
        "canonical_val_size": int(len(va_idx)),
        "canonical_split": {
            "method": "train_test_split",
            "test_size": 0.2,
            "stratify": True,
            "random_state": 42,
            "order_preserved": True,
        },
        "strict_ooo": {
            "folds": N_SPLITS,
            "fold_seed": SEED,
            "main131_fold_epochs": MAIN131_EPOCHS,
            "main133_fold_epochs": MAIN133_EPOCHS,
            "main131_fit_only_on_fold_train": True,
            "main133_fit_only_on_fold_train": True,
            "no_validation_tuning": True,
        },
        "alignment": {
            "main131_saved_val_indices_alignment": True,
            "main133_saved_val_indices_alignment": True,
            "main115_val_alignment": True,
        },
        "fusion": {
            "main131_dim": 128,
            "main133_dim": 128,
            "total_dim": 256,
            "lr_C": 0.1,
            "svm_C": 1.0,
        },
        "results": {
            "lr_oof_accuracy": float(lr_oof_acc),
            "lr_oof_auc": float(lr_oof_auc),
            "lr_val_accuracy": float(lr_val_acc),
            "lr_val_auc": float(lr_val_auc),
            "svm_oof_accuracy": float(svm_oof_acc),
            "svm_oof_auc": float(svm_oof_auc),
            "svm_val_accuracy": float(svm_val_acc),
            "svm_val_auc": float(svm_val_auc),
            "main115_mean_val_accuracy": float(mean_acc),
            "main115_mean_val_auc": float(mean_auc),
            "main115_frozen_rule_val_accuracy": float(frozen_acc),
        },
        "runtime_seconds": float(time.time() - start),
    }

    np.savez_compressed(
        ROOT / "main156_results.npz",
        z131_oof=z131_oof,
        z133_oof=z133_oof,
        z131_val=z131_val,
        z133_val=z133_val,
        y_oof=y_train,
        y_val=y[va_idx],
        lr_oof_prob=lr_oof_prob,
        lr_val_prob=lr_val_prob,
        svm_oof_score=svm_oof_score,
        svm_val_score=svm_val_score,
        lr_val_pred=lr_val_pred,
        svm_val_pred=svm_val_pred,
    )

    with open(ROOT / "main156_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== MAIN156 SUMMARY ===")
    print(f"LR  strict OOF fusion VAL accuracy : {lr_val_acc:.6f}")
    print(f"LR  strict OOF fusion VAL AUC      : {lr_val_auc:.6f}")
    print(f"SVM strict OOF fusion VAL accuracy : {svm_val_acc:.6f}")
    print(f"SVM strict OOF fusion VAL AUC      : {svm_val_auc:.6f}")
    print(f"Main115 frozen rule VAL accuracy   : {frozen_acc:.6f}")

    print("\nSaved:")
    print(ROOT / "main156_results.npz")
    print(ROOT / "main156_report.json")
    print(f"Runtime: {time.time() - start:.3f} sec")


if __name__ == "__main__":
    main()
