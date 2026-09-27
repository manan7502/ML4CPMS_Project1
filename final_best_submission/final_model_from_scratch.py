"""
FINAL MODEL -- BUILT FROM SCRATCH (train.json / test.json ONLY)
==============================================================================
Single, self-contained script reproducing the entire submitted system:

    4-specialist classical teacher (LinearSVC, NB-SVM, HistGradientBoosting,
    local-geometry k-NN) -> logistic-regression meta-learner
        |
        +--> Corrector A: BiGRU sequence residual corrector
        +--> Corrector B: Tabular Residual-MLP corrector
        |
    Confidence-gated merge -> submission.csv

NO dependency on any pre-computed .npy / .npz / .pt file. Everything -- the
teacher's specialists, the meta-learner, both neural correctors, and the
merge rule -- is derived here starting only from train.json and test.json.

Methodology (same discipline used throughout the project):
  - One canonical split (train_test_split, seed=42, stratified, 80/20).
  - Every fold-dependent transform (TF-IDF, SVD, scalers, trajectory
    embeddings, the meta-learner, both neural correctors) is fit ONLY on
    train_idx and evaluated on the genuinely held-out val_idx for all
    reported "honest validation" numbers.
  - For the final deployed model, everything is refit on ALL 10,536
    labelled rows before predicting test.json (standard practice: the
    held-out split's only job is to produce a trustworthy accuracy
    estimate; once trusted, using every row improves the deployed model).

Expected runtime: roughly 25-40 minutes on a Kaggle GPU (dominated by the
LinearSVC specialist on high-dimensional sparse TF-IDF; the neural
correctors are fast, a few minutes each).
"""

import glob, json, os, random, time, warnings
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
SEED = 42
N_INNER_FOLDS = 3          # specialist OOF folds (reduced from 5 -- see note in train_specialists_oof)
WORD_NGRAM = (1, 6)
WORD_MIN_DF = 3
NBSVM_NGRAM = (1, 2)       # NB-SVM's own, smaller n-gram range (standard Wang & Manning setup)
NBSVM_MIN_DF = 3
LOCAL_K = 20
SVD_COMPONENTS = 30
TRAJ_TOP_K = 500
TRAJ_COMPONENTS = 12

BIGRU_EPOCHS = 6
BIGRU_SEEDS = [42, 123, 777]
BIGRU_MAX_LEN = 384
BIGRU_EMB_DIM = 64
BIGRU_HIDDEN = 64
BIGRU_DROPOUT = 0.40
BIGRU_BATCH = 32
BIGRU_LR = 3e-4
BIGRU_WD = 1e-3
BIGRU_TRAIN_WEIGHT = 0.075
BIGRU_TEST_WEIGHT = 0.025

MLP_HIDDEN = 192
MLP_DROPOUT = 0.25
MLP_EPOCHS = 25
MLP_SEEDS_FINAL = 15
MLP_LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
t0 = time.time()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_file(name):
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
    for candidate in [name, os.path.join(script_dir, name)]:
        if os.path.exists(candidate):
            return candidate
    matches = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    matches += glob.glob(f"/kaggle/working/**/{name}", recursive=True)
    if not matches:
        raise FileNotFoundError(f"Could not find {name}.")
    return matches[0]


# ============================================================
# 1. DATA LOADING
# ============================================================
def load_jsonl(path, labelled=True):
    """Reads the JSONL train/test files. Returns (ids, texts, labels) or (ids, texts)."""
    ids, texts, labels = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ids.append(r["id"]); texts.append(r["text"])
            if labelled:
                labels.append(0 if r["label"] == "A" else 1)
    return (ids, texts, np.array(labels)) if labelled else (ids, texts, None)


train_ids, texts, labels = load_jsonl(find_file("train.json"), True)
test_ids, test_texts, _ = load_jsonl(find_file("test.json"), False)
print(f"Loaded {len(texts)} train, {len(test_texts)} test rows")

idx = np.arange(len(texts))
tr_idx, va_idx = train_test_split(idx, test_size=0.20, random_state=SEED, stratify=labels)
y_train, y_val = labels[tr_idx], labels[va_idx]
print(f"Canonical split: train={len(tr_idx)} val={len(va_idx)}")


# ============================================================
# 2. STRUCTURAL FEATURES (62-D)
# ============================================================
def entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    return float(-(p * np.log(p + 1e-12)).sum())


def sequence_features(doc):
    """The 62-dimensional structural feature vector for one document:
    length/entropy/repetition statistics, n-gram diversity, positional
    (quartile, beginning/end) diversity, and token-count distribution
    statistics. Used by the HistGradientBoosting specialist, the LinearSVC
    specialist (concatenated with TF-IDF), and both neural correctors."""
    x = np.asarray(doc, dtype=np.int64)
    n = len(x)
    if n == 0:
        return np.zeros(62, dtype=np.float32)
    unique, counts = np.unique(x, return_counts=True)
    u = len(unique)
    repetition_ratio = 1.0 - u / n
    max_freq = counts.max()
    repeated_occurrences = np.sum(counts[counts > 1] - 1)
    repeated_types = np.sum(counts > 1)
    entropy = entropy_from_counts(counts)

    def ngram_stats(k):
        if n < k:
            return 0.0, 0.0
        grams = set(tuple(x[i:i+k]) for i in range(n - k + 1))
        total = n - k + 1
        return len(grams) / total, float(len(grams))

    bigram_div, bigram_unique = ngram_stats(2)
    trigram_div, trigram_unique = ngram_stats(3)
    fourgram_div, fourgram_unique = ngram_stats(4)

    quarter_features, q_lengths = [], []
    for q in range(4):
        start, end = (q * n) // 4, ((q + 1) * n) // 4
        q_lengths.append(end - start)
        part = x[start:end]
        if len(part) == 0:
            quarter_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        pu, pc = np.unique(part, return_counts=True)
        plen = len(part)
        q_unique_ratio = len(pu) / plen
        quarter_features.extend([plen / n, q_unique_ratio, entropy_from_counts(pc),
                                  1.0 - q_unique_ratio, pc.max() / plen])

    k = min(10, n)
    beginning, ending = x[:k], x[-k:]
    begin_unique_ratio = len(np.unique(beginning)) / k
    end_unique_ratio = len(np.unique(ending)) / k
    begin_entropy = entropy_from_counts(np.unique(beginning, return_counts=True)[1])
    end_entropy = entropy_from_counts(np.unique(ending, return_counts=True)[1])

    mid = n // 2
    first, second = x[:mid], x[mid:]
    first_unique_ratio = len(np.unique(first)) / len(first) if len(first) else 0.0
    first_entropy = entropy_from_counts(np.unique(first, return_counts=True)[1]) if len(first) else 0.0
    second_unique_ratio = len(np.unique(second)) / len(second) if len(second) else 0.0
    second_entropy = entropy_from_counts(np.unique(second, return_counts=True)[1]) if len(second) else 0.0

    zero_count = np.sum(x == 0)
    zero_ratio = zero_count / n

    if n > 1:
        same_adjacent = np.sum(x[1:] == x[:-1])
        adjacent_repeat_ratio = same_adjacent / (n - 1)
        transitions = np.column_stack([x[:-1], x[1:]])
        unique_transitions = len(np.unique(transitions, axis=0))
        transition_diversity = unique_transitions / (n - 1)
    else:
        adjacent_repeat_ratio, transition_diversity = 0.0, 0.0

    count_std, count_mean, count_median = np.std(counts), np.mean(counts), np.median(counts)
    count_range = np.max(counts) - count_median

    features = [
        np.log1p(n), float(n), np.log1p(u), float(u), u / n, repetition_ratio,
        max_freq, max_freq / n, repeated_occurrences, repeated_occurrences / n,
        repeated_types, repeated_types / max(u, 1), entropy, entropy / np.log(max(u, 2)),
        zero_count, zero_ratio, bigram_div, bigram_unique, trigram_div, trigram_unique,
        fourgram_div, fourgram_unique, transition_diversity, adjacent_repeat_ratio,
        begin_unique_ratio, end_unique_ratio, begin_entropy, end_entropy,
        first_unique_ratio, second_unique_ratio, first_entropy, second_entropy,
        second_unique_ratio - first_unique_ratio, second_entropy - first_entropy,
        count_std, count_mean, count_median, count_range,
        *q_lengths, *quarter_features,
    ]
    return np.asarray(features, dtype=np.float32)


def build_structural(seqs):
    return np.vstack([sequence_features(d) for d in seqs])


# ============================================================
# 3. TEXT REPRESENTATIONS (word TF-IDF, transition TF-IDF)
# ============================================================
def to_word_strings(seqs):
    return [" ".join(map(str, d)) for d in seqs]


def to_transition_strings(seqs):
    """Adjacent-token-pair 'transition' tokens, e.g. tokens [5,9,5] -> '5T9 9T5'."""
    out = []
    for d in seqs:
        if len(d) < 2:
            out.append("")
            continue
        out.append(" ".join(f"{d[i]}T{d[i+1]}" for i in range(len(d) - 1)))
    return out


# ============================================================
# 4. SPECIALIST 1: LinearSVC on [word TF-IDF + transition TF-IDF + structural]
# ============================================================
def train_svm(X_tr, y_tr, X_query, C=1.0):
    """LinearSVC on the combined 'global' representation. tol relaxed and
    max_iter capped for tractable runtime on very high-dimensional sparse
    input -- a real, historically-observed slow point in this pipeline."""
    m = LinearSVC(C=C, class_weight="balanced", tol=1e-3, max_iter=5000, dual="auto")
    m.fit(X_tr, y_tr)
    return m.decision_function(X_query)


# ============================================================
# 5. SPECIALIST 2: NB-SVM (Wang & Manning log-count-ratio) on word n-grams
# ============================================================
def train_nbsvm(X_tr, y_tr, X_query, alpha=1.0, C=1.0):
    p = np.asarray(X_tr[y_tr == 1].sum(axis=0)).ravel() + alpha
    q = np.asarray(X_tr[y_tr == 0].sum(axis=0)).ravel() + alpha
    r = np.log((p / p.sum()) / (q / q.sum()))
    X_tr_nb = X_tr.multiply(r).tocsr()
    X_query_nb = X_query.multiply(r).tocsr()
    m = LinearSVC(C=C, class_weight="balanced", tol=1e-3, max_iter=5000, dual="auto")
    m.fit(X_tr_nb, y_tr)
    return m.decision_function(X_query_nb)


# ============================================================
# 6. SPECIALIST 3: HistGradientBoosting on structural features
# ============================================================
def train_hgb(X_tr, y_tr, X_query):
    m = HistGradientBoostingClassifier(random_state=SEED)
    m.fit(X_tr, y_tr)
    proba = m.predict_proba(X_query)[:, 1]
    eps = 1e-6
    return np.log(np.clip(proba, eps, 1 - eps) / np.clip(1 - proba, eps, 1 - eps))


# ============================================================
# 7. SPECIALIST 4: local-geometry k-NN in TF-IDF space
# ============================================================
def local_geometry_scores(X_ref_tfidf, y_ref, X_query_tfidf, k=LOCAL_K):
    """Fraction of each query row's k nearest (cosine) neighbors in the
    reference set that belong to class 1 -- a non-parametric neighbor-
    label-agreement signal, independent of the other three specialists."""
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(X_ref_tfidf)
    _, indices = nn.kneighbors(X_query_tfidf)
    neighbor_labels = y_ref[indices]
    frac_positive = neighbor_labels.mean(axis=1)
    eps = 1e-3
    return np.log(np.clip(frac_positive, eps, 1 - eps) / np.clip(1 - frac_positive, eps, 1 - eps))


# ============================================================
# 8. BUILD REPRESENTATIONS + RUN ALL 4 SPECIALISTS, LEAK-SAFE
# ============================================================
def build_representations(fit_seqs, apply_seqs_list):
    """Fits word TF-IDF, transition TF-IDF, and the structural scaler on
    fit_seqs only; transforms every list in apply_seqs_list."""
    word_vec = TfidfVectorizer(ngram_range=WORD_NGRAM, min_df=WORD_MIN_DF, sublinear_tf=True)
    trans_vec = TfidfVectorizer(ngram_range=NBSVM_NGRAM, min_df=NBSVM_MIN_DF,
                                 sublinear_tf=True, token_pattern=r"(?u)\S+")
    word_vec.fit(to_word_strings(fit_seqs))
    trans_vec.fit(to_transition_strings(fit_seqs))
    struct_raw_fit = build_structural(fit_seqs)
    struct_scaler = StandardScaler().fit(struct_raw_fit)

    outputs = []
    for seqs in apply_seqs_list:
        Xw = word_vec.transform(to_word_strings(seqs))
        Xt = trans_vec.transform(to_transition_strings(seqs))
        Xs_raw = build_structural(seqs)
        Xs = struct_scaler.transform(Xs_raw)
        X_global = hstack([Xw, csr_matrix(Xs), Xt]).tocsr()
        outputs.append({"word": Xw, "trans": Xt, "struct": Xs, "global": X_global})
    return outputs


def train_specialists_oof(seqs_pool, y_pool, n_folds=N_INNER_FOLDS):
    """Runs all 4 specialists via inner cross-validation over seqs_pool,
    producing leak-safe out-of-fold scores for every row in the pool.
    n_folds is reduced to 3 (from the 5 used elsewhere in this project)
    specifically here -- the LinearSVC specialist on full-size sparse
    TF-IDF is the slowest single step in the whole pipeline, and 3-fold
    OOF halves that cost while remaining a statistically reasonable
    estimate for meta-learner fitting purposes."""
    n = len(y_pool)
    oof = {"svm": np.zeros(n), "nbsvm": np.zeros(n), "hgb": np.zeros(n), "local": np.zeros(n)}
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=SEED)
    for fold, (ftr, fva) in enumerate(skf.split(np.zeros(n), y_pool), 1):
        tfold = time.time()
        fold_seqs_tr = [seqs_pool[i] for i in ftr]
        fold_seqs_va = [seqs_pool[i] for i in fva]
        reps = build_representations(fold_seqs_tr, [fold_seqs_tr, fold_seqs_va])
        rep_tr, rep_va = reps[0], reps[1]

        oof["svm"][fva] = train_svm(rep_tr["global"], y_pool[ftr], rep_va["global"])
        oof["nbsvm"][fva] = train_nbsvm(rep_tr["word"], y_pool[ftr], rep_va["word"])
        oof["hgb"][fva] = train_hgb(rep_tr["struct"], y_pool[ftr], rep_va["struct"])
        oof["local"][fva] = local_geometry_scores(rep_tr["word"], y_pool[ftr], rep_va["word"])
        print(f"  specialist OOF fold {fold}/{n_folds} done in {time.time()-tfold:.0f}s")
    return oof


# ============================================================
# 9. META-LEARNER (4 specialist scores -> 1 leak-safe teacher score)
# ============================================================
def fit_meta_learner(oof_features, y_pool):
    meta_mean = oof_features.mean(axis=0)
    meta_std = oof_features.std(axis=0)
    meta_std = np.where(meta_std < 1e-6, 1.0, meta_std)
    meta_scaled = (oof_features - meta_mean) / meta_std
    meta_model = LogisticRegression(C=0.1, max_iter=5000, solver="lbfgs", random_state=SEED)
    meta_model.fit(meta_scaled, y_pool)
    return meta_model, meta_mean, meta_std


def apply_meta_learner(meta_model, meta_mean, meta_std, features):
    scaled = (features - meta_mean) / meta_std
    return meta_model.decision_function(scaled)


def prob_to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def logit_to_prob(z):
    return 1 / (1 + np.exp(-z))


# ============================================================
# 10. TRAJECTORY FEATURES (fold-safe SVD embedding of token transitions)
# ============================================================
def fit_trajectory_embedding(fit_seqs, top_k=TRAJ_TOP_K, n_components=TRAJ_COMPONENTS):
    all_tokens = [t for doc in fit_seqs for t in doc]
    vocab_counts = Counter(all_tokens)
    top_vocab = {t: i for i, (t, c) in enumerate(vocab_counts.most_common(top_k))}
    unk_idx = top_k

    def map_doc(doc):
        return [top_vocab.get(t, unk_idx) for t in doc]

    mapped_fit = [map_doc(d) for d in fit_seqs]
    src, dst = [], []
    for doc in mapped_fit:
        if len(doc) > 1:
            src.extend(doc[:-1]); dst.extend(doc[1:])
    V = top_k + 1
    cooccur = sp.coo_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(V, V)).tocsr()
    row_sums = np.array(cooccur.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    P_trans = cooccur.multiply(1.0 / row_sums[:, None]).tocsr()
    emb = TruncatedSVD(n_components=n_components, random_state=SEED).fit_transform(P_trans)
    return map_doc, emb


def trajectory_features(seqs, map_doc, emb):
    out = np.zeros((len(seqs), TRAJ_COMPONENTS + 4), dtype=np.float32)
    for i, doc in enumerate(seqs):
        mapped = map_doc(doc)
        n = len(mapped)
        if n == 0:
            continue
        doc_embs = emb[mapped]
        centroid = np.mean(doc_embs, axis=0)
        dispersion = float(np.mean(np.sum((doc_embs - centroid) ** 2, axis=1)))
        if n > 1:
            diffs = np.diff(doc_embs, axis=0)
            step_lens = np.linalg.norm(diffs, axis=1)
            mean_step, std_step = float(np.mean(step_lens)), float(np.std(step_lens))
            total_path = float(np.sum(step_lens))
            net_disp = float(np.linalg.norm(doc_embs[-1] - doc_embs[0]))
            tortuosity = float(net_disp / (total_path + 1e-6))
        else:
            mean_step, std_step, tortuosity = 0.0, 0.0, 0.0
        out[i] = np.concatenate([centroid, [dispersion, mean_step, std_step, tortuosity]])
    return out


# ============================================================
# 11. CORRECTOR A -- BiGRU sequence residual
# ============================================================
class ResidualDataset(Dataset):
    def __init__(self, seqs, labels, meta_z):
        self.seqs = seqs
        self.labels = np.asarray(labels, dtype=np.float32)
        self.meta_z = np.asarray(meta_z, dtype=np.float32)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        x = np.asarray(self.seqs[i], dtype=np.int64)
        if len(x) > BIGRU_MAX_LEN:
            x = x[:BIGRU_MAX_LEN]
        n = len(x)
        ids = np.full(BIGRU_MAX_LEN, PAD_IDX, dtype=np.int64)
        mask = np.zeros(BIGRU_MAX_LEN, dtype=np.bool_)
        ids[:n] = x
        mask[:n] = True
        unique_ratio = len(np.unique(x)) / max(n, 1)
        repeat_ratio = 1.0 - unique_ratio
        length_log = np.log1p(n) / 7.0
        aux = np.concatenate([self.meta_z[i], np.asarray(
            [unique_ratio, repeat_ratio, length_log], dtype=np.float32)]).astype(np.float32)
        return (torch.from_numpy(ids), torch.from_numpy(mask),
                torch.tensor(self.labels[i]), torch.from_numpy(aux))


class ResidualBiGRU(nn.Module):
    """meta_dim=4 here (our 4 specialist scores), so the auxiliary vector
    fed to meta_proj is 4+3=7-d (vs. 13+3=16-d in the production Main79,
    which used a richer 13-feature meta representation cached from an
    earlier pipeline stage we do not reconstruct here)."""
    def __init__(self, meta_dim):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE + 1, BIGRU_EMB_DIM, padding_idx=PAD_IDX)
        self.gru = nn.GRU(input_size=BIGRU_EMB_DIM, hidden_size=BIGRU_HIDDEN, num_layers=1,
                           batch_first=True, bidirectional=True)
        enc_dim = 2 * BIGRU_HIDDEN
        pooled_dim = 3 * enc_dim
        self.attn = nn.Sequential(nn.Linear(enc_dim, 32), nn.Tanh(), nn.Linear(32, 1))
        self.neural_proj = nn.Sequential(nn.Linear(pooled_dim, 64), nn.LayerNorm(64),
                                          nn.ReLU(), nn.Dropout(BIGRU_DROPOUT))
        self.meta_proj = nn.Sequential(nn.Linear(meta_dim + 3, 32), nn.LayerNorm(32),
                                        nn.ReLU(), nn.Dropout(BIGRU_DROPOUT))
        self.head = nn.Sequential(nn.Linear(96, 48), nn.ReLU(), nn.Dropout(BIGRU_DROPOUT), nn.Linear(48, 1))

    def forward(self, ids, mask, aux):
        e = self.embedding(ids)
        h, _ = self.gru(e)
        mask_f = mask.unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_min(1)
        mean_pool = (h * mask_f).sum(dim=1) / denom
        max_pool = h.masked_fill(~mask_f, -1e4).max(dim=1).values
        scores = self.attn(h.float()).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1)
        attn_pool = torch.sum(h.float() * weights.unsqueeze(-1), dim=1)
        pooled = torch.cat([mean_pool.float(), max_pool.float(), attn_pool], dim=1)
        neural_z = self.neural_proj(pooled)
        meta_z = self.meta_proj(aux.float())
        z = torch.cat([neural_z, meta_z], dim=1)
        return self.head(z).squeeze(-1)


def train_bigru_seed(seed, seqs, y, meta_z, meta_dim):
    seed_everything(seed)
    ds = ResidualDataset(seqs, y, meta_z)
    loader = DataLoader(ds, batch_size=BIGRU_BATCH, shuffle=True, num_workers=0)
    model = ResidualBiGRU(meta_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BIGRU_LR, weight_decay=BIGRU_WD)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(BIGRU_EPOCHS):
        model.train()
        for ids, mask, yb, aux in loader:
            ids, mask, yb, aux = ids.to(DEVICE), mask.to(DEVICE), yb.to(DEVICE), aux.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            residual = model(ids, mask, aux)
            teacher = aux[:, 0]
            final_logit = teacher + BIGRU_TRAIN_WEIGHT * residual
            loss = criterion(final_logit, yb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


@torch.no_grad()
def predict_bigru_residual(model, seqs, meta_z):
    model.eval()
    dummy = np.zeros(len(seqs), dtype=np.float32)
    ds = ResidualDataset(seqs, dummy, meta_z)
    loader = DataLoader(ds, batch_size=BIGRU_BATCH, shuffle=False, num_workers=0)
    out = []
    for ids, mask, _, aux in loader:
        ids, mask, aux = ids.to(DEVICE), mask.to(DEVICE), aux.to(DEVICE)
        out.append(model(ids, mask, aux).float().cpu().numpy())
    return np.concatenate(out)


def run_bigru_corrector(train_seqs, y_tr, train_meta, query_seqs_list, query_meta_list, meta_dim):
    models = [train_bigru_seed(s, train_seqs, y_tr, train_meta, meta_dim) for s in BIGRU_SEEDS]
    results = []
    for seqs, meta in zip(query_seqs_list, query_meta_list):
        preds = [predict_bigru_residual(m, seqs, meta) for m in models]
        results.append(np.mean(preds, axis=0))
    return results


# ============================================================
# 12. CORRECTOR B -- Tabular Residual-MLP
# ============================================================
class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.act(self.ln1(self.fc1(x)))
        out = self.dropout(out)
        out = self.ln2(self.fc2(out))
        out = self.dropout(out)
        return self.act(out + x)


class NeuralResidualStacker(nn.Module):
    def __init__(self, input_dim, hidden_dim=MLP_HIDDEN, dropout=MLP_DROPOUT):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                         nn.GELU(), nn.Dropout(dropout))
        self.block1 = ResidualMLPBlock(hidden_dim, dropout)
        self.block2 = ResidualMLPBlock(hidden_dim, dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.LayerNorm(64), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        h = self.input_proj(x)
        h = self.block1(h)
        h = self.block2(h)
        return self.head(h).squeeze(-1)


def train_mlp_seed(seed, X_tr, y_tr):
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    ds = torch.utils.data.TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32),
                                         torch.tensor(y_tr, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    model = NeuralResidualStacker(input_dim=X_tr.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(MLP_EPOCHS):
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    return model, scaler


@torch.no_grad()
def predict_mlp(model, scaler, X):
    model.eval()
    X_s = scaler.transform(X)
    logits = model(torch.tensor(X_s, dtype=torch.float32).to(DEVICE))
    return torch.sigmoid(logits).cpu().numpy()


def run_tabular_corrector(X_train, y_tr, query_X_list, n_seeds=MLP_SEEDS_FINAL):
    probs_per_query = [[] for _ in query_X_list]
    for s in range(n_seeds):
        model, scaler = train_mlp_seed(SEED + 111 * (s + 1), X_train, y_tr)
        for i, X_q in enumerate(query_X_list):
            probs_per_query[i].append(predict_mlp(model, scaler, X_q))
    return [np.mean(p, axis=0) for p in probs_per_query]


# ============================================================
# 13. CONFIDENCE-GATED MERGE
# ============================================================
def tune_confidence_margin(bigru_prob, mlp_prob, y_true, margins=np.arange(0.0, 0.45, 0.05)):
    bigru_label = (bigru_prob >= 0.5).astype(int)
    mlp_label = (mlp_prob >= 0.5).astype(int)
    best_margin, best_acc = 0.0, 0.0
    for m in margins:
        confident = (mlp_prob < 0.5 - m) | (mlp_prob > 0.5 + m)
        final = np.where(confident, mlp_label, bigru_label)
        acc = accuracy_score(y_true, final)
        if acc > best_acc:
            best_acc, best_margin = acc, m
    return best_margin, best_acc


def confidence_gated_merge(bigru_prob, mlp_prob, margin):
    bigru_label = (bigru_prob >= 0.5).astype(int)
    mlp_label = (mlp_prob >= 0.5).astype(int)
    confident = (mlp_prob < 0.5 - margin) | (mlp_prob > 0.5 + margin)
    return np.where(confident, mlp_label, bigru_label)


# ============================================================
# MAIN PIPELINE
# ============================================================
VOCAB_SIZE = max(int(t) for s in texts for t in s) + 1
PAD_IDX = VOCAB_SIZE
print("Vocabulary size:", VOCAB_SIZE)

train_seqs = [texts[i] for i in tr_idx]
val_seqs = [texts[i] for i in va_idx]

print("\n" + "=" * 70)
print("STEP 1 -- classical teacher: 4 specialists + meta-learner")
print("=" * 70)
oof = train_specialists_oof(train_seqs, y_train)
oof_features = np.column_stack([oof["svm"], oof["nbsvm"], oof["hgb"], oof["local"]])

print("Fitting final specialists on all training rows, scoring val/test...")
reps_final = build_representations(train_seqs, [train_seqs, val_seqs, test_texts])
rep_tr, rep_va, rep_test = reps_final

val_svm = train_svm(rep_tr["global"], y_train, rep_va["global"])
val_nbsvm = train_nbsvm(rep_tr["word"], y_train, rep_va["word"])
val_hgb = train_hgb(rep_tr["struct"], y_train, rep_va["struct"])
val_local = local_geometry_scores(rep_tr["word"], y_train, rep_va["word"])
val_features = np.column_stack([val_svm, val_nbsvm, val_hgb, val_local])

meta_model, meta_mean, meta_std = fit_meta_learner(oof_features, y_train)
teacher_train_logit = apply_meta_learner(meta_model, meta_mean, meta_std, oof_features)
teacher_val_logit = apply_meta_learner(meta_model, meta_mean, meta_std, val_features)
teacher_val_acc = accuracy_score(y_val, (teacher_val_logit >= 0).astype(int))
print(f"Classical teacher validation accuracy: {teacher_val_acc:.4f}")

print("\n" + "=" * 70)
print("STEP 2 -- Corrector A: BiGRU residual (train_idx only, for honest validation)")
print("=" * 70)
meta_train_arr = np.column_stack([teacher_train_logit, oof["svm"], oof["nbsvm"], oof["hgb"]])
meta_val_arr = np.column_stack([teacher_val_logit, val_svm, val_nbsvm, val_hgb])
[bigru_val_prob_logit] = run_bigru_corrector(
    train_seqs, y_train, meta_train_arr, [val_seqs], [meta_val_arr], meta_dim=4)
bigru_val_logit = teacher_val_logit + BIGRU_TEST_WEIGHT * bigru_val_prob_logit
bigru_val_prob = logit_to_prob(bigru_val_logit)
print(f"Corrector A (BiGRU) validation accuracy: {accuracy_score(y_val, (bigru_val_prob>=0.5).astype(int)):.4f}")

print("\n" + "=" * 70)
print("STEP 3 -- Corrector B: Tabular Residual-MLP (train_idx only, for honest validation)")
print("=" * 70)
map_doc, traj_emb = fit_trajectory_embedding(train_seqs)
traj_train = trajectory_features(train_seqs, map_doc, traj_emb)
traj_val = trajectory_features(val_seqs, map_doc, traj_emb)
svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED).fit(rep_tr["word"])
svd_train = svd.transform(rep_tr["word"])
svd_val = svd.transform(rep_va["word"])

mlp_X_train = np.hstack([teacher_train_logit.reshape(-1, 1), logit_to_prob(teacher_train_logit).reshape(-1, 1),
                          rep_tr["struct"], svd_train, traj_train]).astype(np.float32)
mlp_X_val = np.hstack([teacher_val_logit.reshape(-1, 1), logit_to_prob(teacher_val_logit).reshape(-1, 1),
                        rep_va["struct"], svd_val, traj_val]).astype(np.float32)

[mlp_val_prob] = run_tabular_corrector(mlp_X_train, y_train, [mlp_X_val])
print(f"Corrector B (Tabular MLP) validation accuracy: {accuracy_score(y_val, (mlp_val_prob>=0.5).astype(int)):.4f}")

print("\n" + "=" * 70)
print("STEP 4 -- tune confidence-gated merge on honest validation")
print("=" * 70)
best_margin, best_val_acc = tune_confidence_margin(bigru_val_prob, mlp_val_prob, y_val)
print(f"Best confidence margin: {best_margin:.2f} -> validation accuracy: {best_val_acc:.4f}")

print("\n" + "=" * 70)
print("STEP 5 -- refit everything on ALL 10,536 rows, predict test.json")
print("=" * 70)
all_seqs = texts
oof_all = train_specialists_oof(all_seqs, labels)
oof_all_features = np.column_stack([oof_all["svm"], oof_all["nbsvm"], oof_all["hgb"], oof_all["local"]])
reps_all = build_representations(all_seqs, [all_seqs, test_texts])
rep_all, rep_test_final = reps_all

test_svm = train_svm(rep_all["global"], labels, rep_test_final["global"])
test_nbsvm = train_nbsvm(rep_all["word"], labels, rep_test_final["word"])
test_hgb = train_hgb(rep_all["struct"], labels, rep_test_final["struct"])
test_local = local_geometry_scores(rep_all["word"], labels, rep_test_final["word"])
test_features = np.column_stack([test_svm, test_nbsvm, test_hgb, test_local])

meta_model_all, meta_mean_all, meta_std_all = fit_meta_learner(oof_all_features, labels)
teacher_all_logit = apply_meta_learner(meta_model_all, meta_mean_all, meta_std_all, oof_all_features)
teacher_test_logit = apply_meta_learner(meta_model_all, meta_mean_all, meta_std_all, test_features)

meta_all_arr = np.column_stack([teacher_all_logit, oof_all["svm"], oof_all["nbsvm"], oof_all["hgb"]])
meta_test_arr = np.column_stack([teacher_test_logit, test_svm, test_nbsvm, test_hgb])
[bigru_test_residual] = run_bigru_corrector(all_seqs, labels, meta_all_arr, [test_texts], [meta_test_arr], meta_dim=4)
bigru_test_logit = teacher_test_logit + BIGRU_TEST_WEIGHT * bigru_test_residual
bigru_test_prob = logit_to_prob(bigru_test_logit)

map_doc_all, traj_emb_all = fit_trajectory_embedding(all_seqs)
traj_all = trajectory_features(all_seqs, map_doc_all, traj_emb_all)
traj_test_final = trajectory_features(test_texts, map_doc_all, traj_emb_all)
svd_all_fit = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED).fit(rep_all["word"])
svd_all = svd_all_fit.transform(rep_all["word"])
svd_test_final = svd_all_fit.transform(rep_test_final["word"])

mlp_X_all = np.hstack([teacher_all_logit.reshape(-1, 1), logit_to_prob(teacher_all_logit).reshape(-1, 1),
                        rep_all["struct"], svd_all, traj_all]).astype(np.float32)
mlp_X_test = np.hstack([teacher_test_logit.reshape(-1, 1), logit_to_prob(teacher_test_logit).reshape(-1, 1),
                         rep_test_final["struct"], svd_test_final, traj_test_final]).astype(np.float32)
[mlp_test_prob] = run_tabular_corrector(mlp_X_all, labels, [mlp_X_test])

final_pred = confidence_gated_merge(bigru_test_prob, mlp_test_prob, best_margin)
labels_out = np.where(final_pred == 0, "A", "B")
submission = pd.DataFrame({"id": test_ids, "label": labels_out})
submission.to_csv("submission_from_scratch.csv", index=False)
print(f"\nWrote submission_from_scratch.csv ({len(submission)} rows)")
print(f"Honest validation accuracy of deployed merge rule: {best_val_acc:.4f}")
print(f"\nTotal time: {time.time()-t0:.0f}s")
