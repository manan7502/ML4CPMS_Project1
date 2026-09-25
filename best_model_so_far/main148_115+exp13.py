#!/usr/bin/env python3
"""
Main148.py
Phase 7 — Experiment 28

MAIN133 latent z + MAIN115 specialist consensus.

Purpose:
    Test whether the strongest learned representation from Main133
    contributes complementary information beyond the strict Main115
    specialist consensus.

Safety:
    - Exact canonical split/order.
    - Reuse Main133's SAVED vocabulary, TF-IDF vocabulary/IDF, SVD,
      structural scaler, and checkpoint.
    - Reconstruct Main133 z_val independently.
    - Abort if reconstructed z_val differs from saved Main133 z_val
      by more than 1e-5.
    - Verify Main115 OOF/VAL labels align with canonical train/val labels.
    - No Kaggle submission.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/manu/Desktop/ML4CPMS/project_1")
TRAIN_PATH = ROOT / "train.json"
MAIN133_PT = ROOT / "main133_best.pt"
MAIN133_NPZ = ROOT / "main133_embeddings.npz"
MAIN115_NPZ = ROOT / "main115_results.npz"

SEED = 148
MAXLEN = 384
TOKEN_EMBED = 96
TOKEN_HIDDEN = 96
TFIDF_DIM = 256
TFIDF_HIDDEN = 128
STRUCT_HIDDEN = 96
LATENT = 128
FUSION_HIDDEN = 256
BATCH = 64

VAL_RECON_TOL = 1e-5

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


# EXACT Main133 structural_features implementation.
def structural_features(tokens):
    x = np.asarray(tokens, dtype=np.int64)
    n = len(x)

    if n == 0:
        return np.zeros(62, dtype=np.float32)

    u, counts = np.unique(x, return_counts=True)
    counts = counts.astype(np.float64)
    p = counts / n

    entropy = float(-(p * np.log(p + 1e-12)).sum())
    concentration = float((p * p).sum())
    top1 = float(counts.max() / n)
    sorted_counts = np.sort(counts)[::-1]
    top5 = float(sorted_counts[:5].sum() / n)
    top10 = float(sorted_counts[:10].sum() / n)

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

    pos = np.arange(n, dtype=np.float64) / max(n - 1, 1)
    weighted_mean_pos = float((pos * np.repeat(p, counts.astype(int))).mean()) if n else 0.0
    q = np.quantile(x.astype(np.float64), [0.1, .25, .5, .75, .9])
    diffs_sorted = np.diff(np.sort(x.astype(np.float64)))
    med_gap = float(np.median(diffs_sorted)) if len(diffs_sorted) else 0.0

    chunks = np.array_split(x, 4)
    chunk_len = [len(c) for c in chunks]
    chunk_unique = [len(np.unique(c)) if len(c) else 0 for c in chunks]
    chunk_mean = [float(c.mean()) if len(c) else 0.0 for c in chunks]
    chunk_std = [float(c.std()) if len(c) else 0.0 for c in chunks]

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

    feats += [
        float(x[0]),
        float(x[-1]),
        float(np.mean(x[:max(1, n // 10)])),
        float(np.mean(x[-max(1, n // 10):])),
        float(np.std(x[:max(1, n // 10)])),
        float(np.std(x[-max(1, n // 10):])),
    ]

    if len(feats) < 62:
        feats += [0.0] * (62 - len(feats))
    return np.asarray(feats[:62], dtype=np.float32)


class MultiInputModel(nn.Module):
    def __init__(self, vocab_size, pad_id):
        super().__init__()

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

        self.tfidf_branch = nn.Sequential(
            nn.Linear(TFIDF_DIM, TFIDF_HIDDEN),
            nn.BatchNorm1d(TFIDF_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(TFIDF_HIDDEN, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

        self.struct_branch = nn.Sequential(
            nn.Linear(62, STRUCT_HIDDEN),
            nn.BatchNorm1d(STRUCT_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(STRUCT_HIDDEN, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

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


def encode_sequences(seqs, vocab, maxlen):
    arr = np.zeros((len(seqs), maxlen), dtype=np.int64)
    for i, seq in enumerate(seqs):
        ids = [vocab.get(t, 1) for t in seq[:maxlen]]
        if len(ids) > maxlen:
            ids = ids[:maxlen]
        if ids:
            arr[i, :len(ids)] = ids
    return arr


def main():
    seed_all(SEED)

    print("=" * 72)
    print("MAIN148 / EXPERIMENT 28")
    print("MAIN133 z + MAIN115 SPECIALIST CONSENSUS")
    print("=" * 72)

    rows = load_json_or_jsonl(TRAIN_PATH)
    token_lists = [list(r["text"]) for r in rows]
    y = np.asarray([label_to_int(r.get("label", r.get("target", r.get("y"))))
                    for r in rows], dtype=np.int64)

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
    print("Canonical split:", len(tr_idx), len(va_idx))
    print("Device:", DEVICE)

    if not MAIN133_PT.exists():
        raise FileNotFoundError(MAIN133_PT)
    if not MAIN133_NPZ.exists():
        raise FileNotFoundError(MAIN133_NPZ)
    if not MAIN115_NPZ.exists():
        raise FileNotFoundError(MAIN115_NPZ)

    # ---------------------------------------------------------------
    # Load Main133 checkpoint. Crucially, Main133 saved the EXACT vocab,
    # so we use that mapping directly instead of reconstructing it.
    # ---------------------------------------------------------------
    ckpt = torch.load(MAIN133_PT, map_location="cpu", weights_only=False)

    state = ckpt["state_dict"]
    vocab = ckpt["vocab"]

    checkpoint_vocab_size = int(state["embedding.weight"].shape[0])
    saved_vocab_size = int(max(vocab.values(), default=1) + 1)

    print("Checkpoint embedding rows:", checkpoint_vocab_size)
    print("Saved Main133 vocab entries:", len(vocab))
    print("Saved Main133 vocab-derived embedding size:", saved_vocab_size)

    if checkpoint_vocab_size != saved_vocab_size:
        raise RuntimeError(
            f"Main133 checkpoint inconsistency: embedding rows "
            f"{checkpoint_vocab_size}, vocab-derived size {saved_vocab_size}."
        )

    # This is expected from Main133's source:
    # vocab = make_vocab(token_lists), i.e. ALL documents, with 0=PAD, 1=UNK.
    expected_vocab_size = max(vocab.values(), default=1) + 1
    print("Using exact saved Main133 vocabulary. No vocabulary guessing.")

    X_tokens = encode_sequences(token_lists, vocab, MAXLEN)
    print("Token matrix:", X_tokens.shape)

    # ---------------------------------------------------------------
    # Reproduce Main133 TF-IDF/SVD EXACTLY using saved fitted artifacts.
    # This avoids refitting and avoids any floating-point/preprocessing drift.
    # ---------------------------------------------------------------
    tfidf_vocab = ckpt["tfidf_vocabulary"]
    tfidf_idf = np.asarray(ckpt["tfidf_idf"], dtype=np.float32)
    svd_components = np.asarray(ckpt["svd_components"], dtype=np.float32)

    # TfidfVectorizer's vocabulary_ maps token/ngram -> column.
    # Recreate only the transform stage using the saved vocabulary + IDF.
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\S+",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.995,
        max_features=60000,
        sublinear_tf=True,
        dtype=np.float32,
        vocabulary=tfidf_vocab,
    )
    vectorizer.idf_ = tfidf_idf
    vectorizer._tfidf.idf_ = tfidf_idf
    vectorizer._tfidf._idf_diag = None

    text_strings = [tokens_to_string(s) for s in token_lists]

    # Safer route: reconstruct the fitted vectorizer state via a fresh
    # CountVectorizer + TfidfTransformer, because sklearn versions may
    # differ in private TfidfVectorizer internals.
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    cv = CountVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\S+",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.995,
        max_features=60000,
        vocabulary=tfidf_vocab,
    )
    counts_tr = cv.transform([text_strings[i] for i in tr_idx])
    counts_va = cv.transform([text_strings[i] for i in va_idx])

    transformer = TfidfTransformer(
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
    )
    transformer.idf_ = tfidf_idf
    transformer._idf_diag = None

    X_tfidf_tr = transformer.transform(counts_tr).astype(np.float32)
    X_tfidf_va = transformer.transform(counts_va).astype(np.float32)

    # Main133 fitted SVD on training TF-IDF. Use saved components exactly.
    svd = TruncatedSVD(n_components=TFIDF_DIM, random_state=133)
    svd.components_ = svd_components
    X_svd_tr = X_tfidf_tr @ svd_components.T
    X_svd_va = X_tfidf_va @ svd_components.T
    X_svd_tr = np.asarray(X_svd_tr, dtype=np.float32)
    X_svd_va = np.asarray(X_svd_va, dtype=np.float32)

    print("Main133 word TF-IDF/SVD:", X_svd_tr.shape, X_svd_va.shape)
    print("Main133 word vocab:", len(tfidf_vocab))
    print("Main133 SVD variance:", ckpt.get("config", {}).get("explained_variance", "saved in report"))

    # ---------------------------------------------------------------
    # Structural features + exact saved Main133 StandardScaler.
    # ---------------------------------------------------------------
    X_struct = np.stack([structural_features(s) for s in token_lists])

    scaler_mean = np.asarray(ckpt["struct_scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(ckpt["struct_scaler_scale"], dtype=np.float32)

    if scaler_mean.shape != (62,) or scaler_scale.shape != (62,):
        raise RuntimeError("Main133 structural scaler shape is not 62-D.")

    X_struct_tr = ((X_struct[tr_idx] - scaler_mean) / scaler_scale).astype(np.float32)
    X_struct_va = ((X_struct[va_idx] - scaler_mean) / scaler_scale).astype(np.float32)

    print("Structural:", X_struct.shape)

    # ---------------------------------------------------------------
    # Recreate model and load exact best checkpoint.
    # ---------------------------------------------------------------
    model = MultiInputModel(checkpoint_vocab_size, pad_id=0).to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    Xt_va = torch.from_numpy(X_tokens[va_idx])
    Xf_va = torch.from_numpy(X_svd_va)
    Xs_va = torch.from_numpy(X_struct_va)

    with torch.no_grad():
        val_z_parts = []
        val_logits_parts = []
        for s in range(0, len(va_idx), BATCH):
            lg, zz = model(
                Xt_va[s:s+BATCH].to(DEVICE),
                Xf_va[s:s+BATCH].to(DEVICE),
                Xs_va[s:s+BATCH].to(DEVICE),
            )
            val_logits_parts.append(lg.cpu().numpy())
            val_z_parts.append(zz.cpu().numpy())

    z_val_reconstructed = np.concatenate(val_z_parts).astype(np.float32)
    neural_prob = 1.0 / (1.0 + np.exp(-np.concatenate(val_logits_parts)))

    # ---------------------------------------------------------------
    # STRICT Main133 artifact alignment check.
    # ---------------------------------------------------------------
    main133_emb = np.load(MAIN133_NPZ, allow_pickle=False)
    saved_indices = main133_emb["val_indices"]
    saved_labels = main133_emb["val_labels"]
    z_val_saved = main133_emb["z_val"]

    if not np.array_equal(saved_indices, va_idx):
        raise RuntimeError("Main133 val_indices do not exactly match canonical validation order.")

    if not np.array_equal(saved_labels, y[va_idx]):
        raise RuntimeError("Main133 saved val_labels do not exactly match canonical validation labels.")

    max_abs_diff = float(np.max(np.abs(z_val_reconstructed - z_val_saved)))
    mean_abs_diff = float(np.mean(np.abs(z_val_reconstructed - z_val_saved)))

    print("Main133 z_val max abs diff:", max_abs_diff)
    print("Main133 z_val mean abs diff:", mean_abs_diff)

    if max_abs_diff > VAL_RECON_TOL:
        raise RuntimeError(
            "STRICT SAFETY FAILURE: reconstructed Main133 z_val differs "
            f"from saved z_val by max {max_abs_diff:.9g} > {VAL_RECON_TOL}. "
            "Do not continue to fusion."
        )

    print("PASS: Main133 z_val reconstruction is within strict tolerance.")

    # ---------------------------------------------------------------
    # Load and verify Main115 strict OOF/VAL specialist outputs.
    # ---------------------------------------------------------------
    m115 = np.load(MAIN115_NPZ, allow_pickle=True)

    y_oof = np.asarray(m115["y_oof"]).astype(np.int64)
    y_val_115 = np.asarray(m115["y_val"]).astype(np.int64)

    specialist_p_oof = np.asarray(m115["specialist_p_oof"], dtype=np.float64)
    specialist_p_val = np.asarray(m115["specialist_p_val"], dtype=np.float64)
    specialist_names = [str(x) for x in m115["specialist_names"]]

    if not np.array_equal(y_oof, y[tr_idx]):
        raise RuntimeError("Main115 y_oof does not exactly match canonical train split order.")
    if not np.array_equal(y_val_115, y[va_idx]):
        raise RuntimeError("Main115 y_val does not exactly match canonical validation order.")

    if specialist_p_oof.shape[0] != len(tr_idx) or specialist_p_val.shape[0] != len(va_idx):
        raise RuntimeError("Main115 specialist prediction row counts do not match canonical split.")

    print("Main115 specialists:", specialist_names)
    print("Main115 OOF:", specialist_p_oof.shape)
    print("Main115 VAL:", specialist_p_val.shape)

    # Use specialist consensus as the probability mean for a clean,
    # cross-fitted representation. The frozen Main115 rule is also reported
    # separately if present.
    m115_mean_oof = specialist_p_oof.mean(axis=1)
    m115_mean_val = specialist_p_val.mean(axis=1)

    # ---------------------------------------------------------------
    # Strict OOF fusion:
    # z_train is NOT saved by Main133. Therefore we cannot legitimately
    # train a fusion model on Main133 z_train reconstructed from a single
    # model trained on all canonical training data without introducing
    # optimistic leakage. Instead, this experiment uses Main133 z_val
    # for the frozen validation complementarity diagnostic and trains
    # probes only on validation as a diagnostic, explicitly labeled.
    #
    # The complementarity diagnostic itself is leakage-safe because it
    # compares fixed predictions on the same held-out validation set.
    # ---------------------------------------------------------------
    yv = y[va_idx]

    # Main133 latent probes: diagnostic only, matching the original
    # Main133 evaluation style, NOT a submit-safe estimate.
    lr_z = LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED
    )
    lr_z.fit(z_val_reconstructed, yv)
    p_z_lr = lr_z.predict_proba(z_val_reconstructed)[:, 1]

    svm_z = SVC(
        C=1.0, kernel="rbf", gamma="scale",
        probability=True, random_state=SEED
    )
    svm_z.fit(z_val_reconstructed, yv)
    p_z_svm = svm_z.predict_proba(z_val_reconstructed)[:, 1]

    # Main115 specialist mean probe: directly evaluated on fixed specialist
    # outputs. Also diagnostic.
    lr_m = LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED
    )
    lr_m.fit(specialist_p_val, yv)
    p_m_lr = lr_m.predict_proba(specialist_p_val)[:, 1]

    svm_m = SVC(
        C=1.0, kernel="rbf", gamma="scale",
        probability=True, random_state=SEED
    )
    svm_m.fit(specialist_p_val, yv)
    p_m_svm = svm_m.predict_proba(specialist_p_val)[:, 1]

    # Fused diagnostics.
    fused_val = np.column_stack([z_val_reconstructed, specialist_p_val])

    lr_f = LogisticRegression(
        C=1.0, max_iter=3000, class_weight="balanced", random_state=SEED
    )
    lr_f.fit(fused_val, yv)
    p_f_lr = lr_f.predict_proba(fused_val)[:, 1]

    svm_f = SVC(
        C=1.0, kernel="rbf", gamma="scale",
        probability=True, random_state=SEED
    )
    svm_f.fit(fused_val, yv)
    p_f_svm = svm_f.predict_proba(fused_val)[:, 1]

    def metrics(name, p):
        acc = float(accuracy_score(yv, p >= 0.5))
        auc = float(roc_auc_score(yv, p))
        print(f"{name:24s} acc={acc:.6f} auc={auc:.6f}")
        return acc, auc

    print("\n=== MAIN148 RESULTS ===")
    z_lr_acc, z_lr_auc = metrics("LR z only", p_z_lr)
    m_lr_acc, m_lr_auc = metrics("LR Main115 only", p_m_lr)
    f_lr_acc, f_lr_auc = metrics("LR fused", p_f_lr)

    z_svm_acc, z_svm_auc = metrics("SVM z only", p_z_svm)
    m_svm_acc, m_svm_auc = metrics("SVM Main115 only", p_m_svm)
    f_svm_acc, f_svm_auc = metrics("SVM fused", p_f_svm)

    # Raw specialist mean diagnostics.
    mean_acc, mean_auc = metrics("Main115 mean", m115_mean_val)

    # Complementarity: among cases where Main115 mean is wrong, how often
    # is Main133 latent probe correct?
    m_pred = (m115_mean_val >= 0.5).astype(np.int64)
    z_lr_pred = (p_z_lr >= 0.5).astype(np.int64)
    z_svm_pred = (p_z_svm >= 0.5).astype(np.int64)

    wrong_m = m_pred != yv

    lr_comp_total = int(wrong_m.sum())
    lr_comp_correct = int((z_lr_pred[wrong_m] == yv[wrong_m]).sum())
    svm_comp_correct = int((z_svm_pred[wrong_m] == yv[wrong_m]).sum())

    print("\nComplementarity:")
    print("Main115-mean wrong:", lr_comp_total)
    print("Main133 LR correct on those:", lr_comp_correct)
    print("Main133 SVM correct on those:", svm_comp_correct)
    if lr_comp_total:
        print("LR complementarity precision:", lr_comp_correct / lr_comp_total)
        print("SVM complementarity precision:", svm_comp_correct / lr_comp_total)

    # If Main115 frozen rule outputs exist, report them exactly.
    frozen_val = None
    if "corrected_val" in m115.files:
        frozen_val = np.asarray(m115["corrected_val"]).astype(np.int64)
        if len(frozen_val) == len(yv):
            frozen_acc = float(accuracy_score(yv, frozen_val))
            print("Main115 frozen rule:", f"{frozen_acc:.6f}")
        else:
            frozen_acc = None
    else:
        frozen_acc = None

    report = {
        "experiment": "Main148 / Phase7 Exp28",
        "description": "Main133 exact latent representation + Main115 specialist consensus",
        "seed": SEED,
        "canonical_split": {
            "train_size": len(tr_idx),
            "val_size": len(va_idx),
            "random_state": 42,
            "order_preserved": True,
        },
        "main133_reconstruction": {
            "checkpoint_vocab_rows": checkpoint_vocab_size,
            "saved_vocab_entries": len(vocab),
            "z_val_max_abs_diff": max_abs_diff,
            "z_val_mean_abs_diff": mean_abs_diff,
            "tolerance": VAL_RECON_TOL,
            "passed": True,
        },
        "main115_specialists": specialist_names,
        "results": {
            "lr_z_accuracy": z_lr_acc,
            "lr_z_auc": z_lr_auc,
            "lr_main115_accuracy": m_lr_acc,
            "lr_main115_auc": m_lr_auc,
            "lr_fused_accuracy": f_lr_acc,
            "lr_fused_auc": f_lr_auc,
            "svm_z_accuracy": z_svm_acc,
            "svm_z_auc": z_svm_auc,
            "svm_main115_accuracy": m_svm_acc,
            "svm_main115_auc": m_svm_auc,
            "svm_fused_accuracy": f_svm_acc,
            "svm_fused_auc": f_svm_auc,
            "main115_mean_accuracy": mean_acc,
            "main115_mean_auc": mean_auc,
            "main115_frozen_rule_accuracy": frozen_acc,
            "main115_wrong_cases": lr_comp_total,
            "main133_lr_correct_on_main115_wrong": lr_comp_correct,
            "main133_svm_correct_on_main115_wrong": svm_comp_correct,
        },
    }

    out_report = ROOT / "main148_report.json"
    with out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSaved:", out_report)
    print("No Kaggle submission performed.")


if __name__ == "__main__":
    main()
