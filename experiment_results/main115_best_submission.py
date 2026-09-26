#!/usr/bin/env python3
"""
MAIN115 BEST-ACCURACY SUBMISSION GENERATOR
==========================================

Project 1: Human-written vs Machine-generated text.

REFERENCE USED FOR THIS SUBMISSION
-----------------------------------
Main64 validation accuracy : 93.0265655%
Main115 validation accuracy: 93.4535104%   <-- CURRENT BEST LOCAL REFERENCE

Main115 frozen correction rule:
    >= 2 / 5 specialists disagree with the Main64 teacher
    mean disagreement margin >= 0.15
    weakest disagreeing specialist confidence >= 0.55

Main115 validation result:
    93.4535104%
    +0.426945 percentage points over Main64

IMPORTANT:
    This script does NOT tune on the Kaggle test set.
    The Main115 rule is frozen from the OOF experiment.
    The script stops if the reconstructed 62-D Main115 feature space
    does not exactly match the saved OOF representation.

FINAL TEST PIPELINE
-------------------
train.json
    -> Main64 full-data SVM / NB-SVM / HGB / local geometry
    -> Main64 13-D meta representation
    -> Main64 teacher probability
    -> exact Main114 39-D feature construction
    -> Main111 23-D structural features
    -> exact 62-D Main115 feature vector
    -> five Main115 specialists
    -> frozen consensus correction
    -> submission.csv

Expected input files in the project directory:
    train.json
    test.json
    main64_oof_svm.npy
    main64_oof_nbsvm.npy
    main64_oof_hgb.npy
    main64_oof_local.npy
    main64_oof_meta_features.npy
    main64_val_labels.npy
    main64_val_scores.npy
    main111_features.npz
    corrected_main114_results.npz

Optional:
    sample_submission.csv

Outputs:
    main115_best_submission.csv
    main115_best_submission_metadata.json

No Kaggle submission is performed automatically.
"""

import json
import math
import os
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# ============================================================
# LOCKED REFERENCE
# ============================================================

SEED = 42

MAIN64_VALIDATION_ACCURACY = 0.9302656546489564
MAIN115_VALIDATION_ACCURACY = 0.9345351043643264

MAIN64_TEACHER_THRESHOLD = 0.3745

# Frozen Main115 rule.
MAIN115_MIN_VOTES = 2
MAIN115_MEAN_MARGIN = 0.15
MAIN115_MIN_CONFIDENCE = 0.55

# Main64 representation.
WORD_NGRAM = (1, 6)
WORD_MIN_DF = 3
TRANS_NGRAM = (1, 2)
TRANS_MIN_DF = 2
SVM_C = 10.0
NBSVM_C = 10.0
LOCAL_K = 20
META_C = 0.1

TRAIN_FILE = "train.json"
TEST_FILE = "test.json"

OOF_META_FILE = "main64_oof_meta_features.npy"
OOF_SVM_FILE = "main64_oof_svm.npy"
OOF_NBSVM_FILE = "main64_oof_nbsvm.npy"
OOF_HGB_FILE = "main64_oof_hgb.npy"
OOF_LOCAL_FILE = "main64_oof_local.npy"

VAL_LABELS_FILE = "main64_val_labels.npy"
VAL_SCORES_FILE = "main64_val_scores.npy"

MAIN111_FILE = "main111_features.npz"
MAIN114_RESULTS = "corrected_main114_results.npz"

SUBMISSION_FILE = "main115_best_submission.csv"
METADATA_FILE = "main115_best_submission_metadata.json"


# ============================================================
# DATA
# ============================================================

def load_train(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    ids = [r["id"] for r in records]
    texts = [r["text"] for r in records]
    labels = np.asarray(
        [0 if r["label"] == "A" else 1 for r in records],
        dtype=np.int64,
    )

    return ids, texts, labels


def load_test(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    ids = [r["id"] for r in records]
    texts = [r["text"] for r in records]

    return ids, texts


def to_strings(texts):
    return [" ".join(map(str, doc)) for doc in texts]


def make_transition_strings(texts):
    out = []

    for doc in texts:
        if len(doc) < 2:
            out.append("")
            continue

        out.append(
            " ".join(
                f"{doc[i]}T{doc[i + 1]}"
                for i in range(len(doc) - 1)
            )
        )

    return out


# ============================================================
# MAIN64 COMPACT FEATURES
# Exact implementation from Main64.
# ============================================================

def entropy_from_counts(counts):
    if len(counts) == 0:
        return 0.0

    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()

    if total <= 0:
        return 0.0

    p = counts / total
    return float(-(p * np.log(p + 1e-12)).sum())


def sequence_features(doc):
    x = np.asarray(doc, dtype=np.int64)
    n = len(x)

    if n == 0:
        return np.zeros(56, dtype=np.float32)

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

        grams = set()
        for i in range(n - k + 1):
            grams.add(tuple(x[i:i + k]))

        total = n - k + 1
        return len(grams) / total, float(len(grams))

    bigram_div, bigram_unique = ngram_stats(2)
    trigram_div, trigram_unique = ngram_stats(3)
    fourgram_div, fourgram_unique = ngram_stats(4)

    quarter_features = []

    for q in range(4):
        start = (q * n) // 4
        end = ((q + 1) * n) // 4
        part = x[start:end]

        if len(part) == 0:
            quarter_features.extend([0.0] * 5)
            continue

        pu, pc = np.unique(part, return_counts=True)
        plen = len(part)

        q_unique_ratio = len(pu) / plen
        q_entropy = entropy_from_counts(pc)
        q_repetition = 1.0 - q_unique_ratio
        q_max_ratio = pc.max() / plen

        quarter_features.extend([
            plen / n,
            q_unique_ratio,
            q_entropy,
            q_repetition,
            q_max_ratio,
        ])

    k = min(10, n)

    beginning = x[:k]
    ending = x[-k:]

    begin_unique_ratio = len(np.unique(beginning)) / k
    end_unique_ratio = len(np.unique(ending)) / k

    begin_entropy = entropy_from_counts(
        np.unique(beginning, return_counts=True)[1]
    )
    end_entropy = entropy_from_counts(
        np.unique(ending, return_counts=True)[1]
    )

    mid = n // 2
    first = x[:mid]
    second = x[mid:]

    if len(first) > 0:
        first_unique_ratio = len(np.unique(first)) / len(first)
        first_entropy = entropy_from_counts(
            np.unique(first, return_counts=True)[1]
        )
    else:
        first_unique_ratio = 0.0
        first_entropy = 0.0

    if len(second) > 0:
        second_unique_ratio = len(np.unique(second)) / len(second)
        second_entropy = entropy_from_counts(
            np.unique(second, return_counts=True)[1]
        )
    else:
        second_unique_ratio = 0.0
        second_entropy = 0.0

    zero_count = np.sum(x == 0)
    zero_ratio = zero_count / n

    if n > 1:
        same_adjacent = np.sum(x[1:] == x[:-1])
        adjacent_repeat_ratio = same_adjacent / (n - 1)

        transitions = np.column_stack([x[:-1], x[1:]])
        unique_transitions = len(
            np.unique(transitions, axis=0)
        )

        transition_diversity = unique_transitions / (n - 1)
    else:
        adjacent_repeat_ratio = 0.0
        transition_diversity = 0.0

    q_lengths = []
    for q in range(4):
        start = (q * n) // 4
        end = ((q + 1) * n) // 4
        q_lengths.append(end - start)

    count_mean = np.mean(counts)
    count_std = np.std(counts)
    count_median = np.median(counts)
    count_range = np.max(counts) - count_median

    features = [
        np.log1p(n),
        float(n),
        np.log1p(u),
        float(u),
        u / n,
        repetition_ratio,

        max_freq,
        max_freq / n,

        repeated_occurrences,
        repeated_occurrences / n,

        repeated_types,
        repeated_types / max(u, 1),

        entropy,
        entropy / np.log(max(u, 2)),

        zero_count,
        zero_ratio,

        bigram_div,
        bigram_unique,

        trigram_div,
        trigram_unique,

        fourgram_div,
        fourgram_unique,

        transition_diversity,
        adjacent_repeat_ratio,

        begin_unique_ratio,
        end_unique_ratio,

        begin_entropy,
        end_entropy,

        first_unique_ratio,
        second_unique_ratio,

        first_entropy,
        second_entropy,

        second_unique_ratio - first_unique_ratio,
        second_entropy - first_entropy,

        count_std,
        count_mean,
        count_median,
        count_range,

        *q_lengths,
        *quarter_features,
    ]

    return np.asarray(features, dtype=np.float32)


def build_compact_features(texts):
    return np.vstack([
        sequence_features(doc)
        for doc in texts
    ])


# ============================================================
# MAIN64 BASE MODELS
# ============================================================

def train_svm(X_train, y_train, X_query):
    model = LinearSVC(
        C=SVM_C,
        class_weight="balanced",
        tol=1e-2,
        max_iter=500000,
    )
    model.fit(X_train, y_train)
    return model.decision_function(X_query)


def train_nbsvm(train_strings, y_train, query_strings):
    vectorizer = CountVectorizer(
        ngram_range=(1, 3),
        min_df=2,
        binary=True,
    )

    X_train = vectorizer.fit_transform(train_strings)
    X_query = vectorizer.transform(query_strings)

    A = X_train[y_train == 0]
    B = X_train[y_train == 1]

    alpha = 1.0

    pA = np.asarray(A.sum(axis=0)).ravel() + alpha
    pB = np.asarray(B.sum(axis=0)).ravel() + alpha

    pA /= pA.sum()
    pB /= pB.sum()

    ratio = np.log(pA / pB)

    X_train_nb = X_train.multiply(ratio)
    X_query_nb = X_query.multiply(ratio)

    model = LinearSVC(
        C=NBSVM_C,
        class_weight="balanced",
        tol=1e-2,
        max_iter=500000,
    )

    model.fit(X_train_nb, y_train)
    return model.decision_function(X_query_nb)


def train_hgb(X_train, y_train, X_query):
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        l2_regularization=1.0,
        random_state=SEED,
    )

    weight_B = (
        np.sum(y_train == 0) /
        np.sum(y_train == 1)
    )

    weights = np.where(
        y_train == 0,
        1.0,
        weight_B,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=weights,
    )

    return model.predict_proba(X_query)[:, 1] - 0.5


def local_geometry_features(
    X_reference,
    y_reference,
    X_query,
    k=20,
):
    A_mask = y_reference == 0
    B_mask = y_reference == 1

    XA = X_reference[A_mask]
    XB = X_reference[B_mask]

    kA = min(k, XA.shape[0])
    kB = min(k, XB.shape[0])

    nnA = NearestNeighbors(
        n_neighbors=kA,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    nnB = NearestNeighbors(
        n_neighbors=kB,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    nnA.fit(XA)
    nnB.fit(XB)

    distA, _ = nnA.kneighbors(X_query)
    distB, _ = nnB.kneighbors(X_query)

    simA = 1.0 - distA
    simB = 1.0 - distB

    output = []

    for i in range(X_query.shape[0]):
        gap1 = distB[i, 0] - distA[i, 0]
        similarity_gap1 = simA[i, 0] - simB[i, 0]

        mean_similarity_gap = (
            np.mean(simA[i]) -
            np.mean(simB[i])
        )

        weights_A = np.exp(5.0 * simA[i])
        weights_B = np.exp(5.0 * simB[i])

        vote_A = np.sum(weights_A)
        vote_B = np.sum(weights_B)

        weighted_vote = (
            (vote_A - vote_B) /
            (vote_A + vote_B + 1e-12)
        )

        density_gap = (
            np.mean(simA[i]) -
            np.mean(simB[i])
        )

        multi = []

        for kk in [1, 2, 3, 5, 10, 20]:
            kkA = min(kk, kA)
            kkB = min(kk, kB)

            mean_A = np.mean(simA[i, :kkA])
            mean_B = np.mean(simB[i, :kkB])

            multi.append(mean_A - mean_B)

        output.append([
            gap1,
            similarity_gap1,
            mean_similarity_gap,
            weighted_vote,
            density_gap,
            *multi,
        ])

    return np.asarray(output, dtype=np.float64)


# ============================================================
# MAIN64 GLOBAL REPRESENTATION
# ============================================================

def fit_global_representation(texts):
    strings = to_strings(texts)
    transitions = make_transition_strings(texts)

    tfidf = TfidfVectorizer(
        ngram_range=WORD_NGRAM,
        min_df=WORD_MIN_DF,
        sublinear_tf=True,
    )

    X_tfidf = tfidf.fit_transform(strings)

    trans_vectorizer = TfidfVectorizer(
        ngram_range=TRANS_NGRAM,
        min_df=TRANS_MIN_DF,
        sublinear_tf=True,
        token_pattern=r"(?u)\S+",
    )

    X_transition = trans_vectorizer.fit_transform(
        transitions
    )

    compact = build_compact_features(texts)

    compact_scaler = StandardScaler()
    compact_s = compact_scaler.fit_transform(compact)

    X_global = hstack([
        X_tfidf,
        csr_matrix(compact_s),
        X_transition,
    ]).tocsr()

    return (
        strings,
        tfidf,
        trans_vectorizer,
        compact_scaler,
        X_tfidf,
        compact,
        X_global,
    )


def transform_global_representation(
    texts,
    tfidf,
    trans_vectorizer,
    compact_scaler,
):
    strings = to_strings(texts)
    transitions = make_transition_strings(texts)

    X_tfidf = tfidf.transform(strings)
    X_transition = trans_vectorizer.transform(transitions)

    compact = build_compact_features(texts)
    compact_s = compact_scaler.transform(compact)

    X_global = hstack([
        X_tfidf,
        csr_matrix(compact_s),
        X_transition,
    ]).tocsr()

    return (
        strings,
        X_tfidf,
        compact,
        X_global,
    )


# ============================================================
# MAIN64 META FEATURES
# ============================================================

def make_meta_features(
    svm,
    nbsvm,
    hgb,
    local,
    scaler,
):
    local_b = -local

    raw = np.column_stack([
        svm,
        nbsvm,
        hgb,
        local_b[:, 0],
        local_b[:, 1],
        local_b[:, 2],
        local_b[:, 3],
        local_b[:, 6],
        local_b[:, 7],
        local_b[:, 8],
        local_b[:, 9],
        local_b[:, 10],
    ])

    transformed = scaler.transform(raw)

    uncertainty = np.exp(-np.abs(svm))
    geometry_signal = local_b[:, 3]

    interaction = (
        geometry_signal * uncertainty
    ).reshape(-1, 1)

    return np.hstack([
        transformed,
        interaction,
    ])


# ============================================================
# EXACT MAIN114 39-D FEATURE BLOCK
#
# This is the feature definition used to create the 62-D
# Main115 representation:
#     39 Main64-derived features + 23 Main111 features.
# ============================================================

def empirical_percentile(train, other):
    train = np.asarray(train, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)

    order = np.argsort(train)

    ranks = np.empty(
        len(train),
        dtype=np.float64,
    )

    ranks[order] = np.arange(len(train))

    if len(train) > 1:
        train_pct = ranks / (len(train) - 1)
    else:
        train_pct = np.zeros(len(train))

    sorted_train = np.sort(train)

    other_rank = np.searchsorted(
        sorted_train,
        other,
        side="right",
    )

    other_pct = (
        other_rank /
        max(len(train), 1)
    )

    return train_pct, other_pct


def build_main114_39_features(
    oof_meta,
    oof_svm,
    oof_nbsvm,
    oof_hgb,
    test_meta,
    test_svm,
    test_nbsvm,
    test_hgb,
):
    oof_components = np.column_stack([
        oof_svm,
        oof_nbsvm,
        oof_hgb,
    ])

    test_components = np.column_stack([
        test_svm,
        test_nbsvm,
        test_hgb,
    ])

    parts_o = [oof_meta]
    parts_t = [test_meta]

    # Explicit component scores.
    parts_o.append(oof_components)
    parts_t.append(test_components)

    # Pairwise disagreement.
    for i in range(3):
        for j in range(i + 1, 3):
            parts_o.append(
                np.abs(
                    oof_components[:, i] -
                    oof_components[:, j]
                )[:, None]
            )
            parts_t.append(
                np.abs(
                    test_components[:, i] -
                    test_components[:, j]
                )[:, None]
            )

    # Mean / std / range.
    for fn in (
        lambda z: np.mean(z, axis=1),
        lambda z: np.std(z, axis=1),
        lambda z: np.ptp(z, axis=1),
    ):
        parts_o.append(
            fn(oof_components)[:, None]
        )
        parts_t.append(
            fn(test_components)[:, None]
        )

    # Hard component votes.
    oof_hard = (
        oof_components >= 0.5
    ).astype(np.float64)

    test_hard = (
        test_components >= 0.5
    ).astype(np.float64)

    parts_o.append(oof_hard)
    parts_t.append(test_hard)

    parts_o.append(
        np.sum(oof_hard, axis=1)[:, None]
    )
    parts_t.append(
        np.sum(test_hard, axis=1)[:, None]
    )

    # Percentile of every saved Main64 meta feature.
    for j in range(oof_meta.shape[1]):
        po, pt = empirical_percentile(
            oof_meta[:, j],
            test_meta[:, j],
        )

        parts_o.append(po[:, None])
        parts_t.append(pt[:, None])

    Xo = np.hstack(parts_o)
    Xt = np.hstack(parts_t)

    Xo = np.nan_to_num(
        Xo,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    Xt = np.nan_to_num(
        Xt,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if Xo.shape[1] != 39 or Xt.shape[1] != 39:
        raise RuntimeError(
            f"Main114 feature construction expected 39 columns; "
            f"got OOF={Xo.shape}, TEST={Xt.shape}"
        )

    return (
        Xo.astype(np.float32),
        Xt.astype(np.float32),
    )


# ============================================================
# MAIN111 23-D STRUCTURAL FEATURES
# Exact implementation from main111.py.
# ============================================================

def make_main111_struct_features(texts):
    out = []

    for x in texts:
        n = len(x)

        a = np.asarray(
            x,
            dtype=np.float64,
        )

        c = Counter(x)

        vals = np.asarray(
            list(c.values()),
            dtype=np.float64,
        )

        p = (
            vals /
            max(vals.sum(), 1.0)
        )

        entropy = (
            float(
                -(p * np.log(p + 1e-12)).sum()
            )
            if len(p)
            else 0.0
        )

        d = (
            np.diff(a)
            if n > 1
            else np.empty(0)
        )

        ad = np.abs(d)

        runs = (
            1 + int(np.sum(d == 0))
            if n > 1
            else 1
        )

        if (
            n > 1
            and np.std(a) > 0
        ):
            corr = float(
                np.corrcoef(
                    np.arange(n),
                    a,
                )[0, 1]
            )
        else:
            corr = 0.0

        top = np.sort(vals)[::-1]

        out.append([
            n,
            len(c),
            len(c) / max(n, 1),

            float(a.mean()) if n else 0,
            float(a.std()) if n else 0,
            float(a.min()) if n else 0,
            float(a.max()) if n else 0,
            float(np.median(a)) if n else 0,

            float(ad.mean()) if len(ad) else 0,
            float(ad.std()) if len(ad) else 0,

            float(np.mean(d == 0)) if len(d) else 0,
            float(np.mean(np.abs(d) <= 2)) if len(d) else 0,

            entropy,
            entropy / max(math.log(len(c) + 1), 1),

            float(np.mean(vals == 1)) if len(vals) else 0,
            float(np.mean(vals >= 2)) if len(vals) else 0,
            float(np.mean(vals >= 3)) if len(vals) else 0,

            float(top[0] / n) if len(top) else 0,
            float(top[1] / n) if len(top) > 1 else 0,
            float(top[2] / n) if len(top) > 2 else 0,

            corr,

            float(
                np.mean(d > 0) -
                np.mean(d < 0)
            ) if len(d) else 0,

            float(
                runs /
                max(n, 1)
            ),
        ])

    return np.asarray(
        out,
        dtype=np.float32,
    )


# ============================================================
# MAIN115 SPECIALISTS
# Exact frozen model family from Main115.
# ============================================================

def build_specialists():
    return {
        "LR_C1": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
        ),

        "HGB_small": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.5,
            random_state=SEED,
        ),

        "HGB_medium": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=260,
            max_leaf_nodes=31,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=SEED + 1,
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED + 2,
        ),

        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED + 3,
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    start = time.time()

    print("=" * 80)
    print("MAIN115 — BEST LOCAL VALIDATION SUBMISSION")
    print("=" * 80)

    print(
        f"REFERENCE VALIDATION ACCURACY: "
        f"Main64={MAIN64_VALIDATION_ACCURACY * 100:.4f}% | "
        f"Main115={MAIN115_VALIDATION_ACCURACY * 100:.4f}%"
    )

    print(
        "FROZEN RULE: "
        f"votes>={MAIN115_MIN_VOTES}, "
        f"mean_margin>={MAIN115_MEAN_MARGIN:.2f}, "
        f"weak_conf>={MAIN115_MIN_CONFIDENCE:.2f}"
    )

    required = [
        TRAIN_FILE,
        TEST_FILE,
        OOF_META_FILE,
        OOF_SVM_FILE,
        OOF_NBSVM_FILE,
        OOF_HGB_FILE,
        OOF_LOCAL_FILE,
        VAL_LABELS_FILE,
        VAL_SCORES_FILE,
        MAIN111_FILE,
        MAIN114_RESULTS,
    ]

    missing = [
        p for p in required
        if not os.path.exists(p)
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" +
            "\n".join(missing)
        )

    # --------------------------------------------------------
    # Load train/test.
    # --------------------------------------------------------

    train_ids, texts, labels = load_train(TRAIN_FILE)
    test_ids, test_texts = load_test(TEST_FILE)

    print(
        f"Train documents: {len(texts)}"
    )
    print(
        f"Test documents:  {len(test_texts)}"
    )

    if len(test_texts) != 3000:
        raise RuntimeError(
            f"Expected exactly 3000 test documents; "
            f"found {len(test_texts)}."
        )

    if len(set(test_ids)) != len(test_ids):
        raise RuntimeError(
            "Test IDs are not unique."
        )

    # --------------------------------------------------------
    # Canonical split.
    # --------------------------------------------------------

    all_idx = np.arange(len(labels))

    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=0.20,
        stratify=labels,
        random_state=SEED,
    )

    # Do NOT sort.
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)

    # --------------------------------------------------------
    # Load authoritative OOF Main64 artifacts.
    # --------------------------------------------------------

    oof_meta = np.load(
        OOF_META_FILE
    ).astype(np.float64)

    oof_svm = np.load(
        OOF_SVM_FILE
    ).astype(np.float64)

    oof_nbsvm = np.load(
        OOF_NBSVM_FILE
    ).astype(np.float64)

    oof_hgb = np.load(
        OOF_HGB_FILE
    ).astype(np.float64)

    oof_local = np.load(
        OOF_LOCAL_FILE
    ).astype(np.float64)

    if oof_meta.shape != (len(train_idx), 13):
        raise RuntimeError(
            f"Unexpected Main64 OOF meta shape: "
            f"{oof_meta.shape}"
        )

    # --------------------------------------------------------
    # Reconstruct exact Main64 meta scaler.
    # --------------------------------------------------------

    oof_local_b = -oof_local

    oof_raw_meta = np.column_stack([
        oof_svm,
        oof_nbsvm,
        oof_hgb,
        oof_local_b[:, 0],
        oof_local_b[:, 1],
        oof_local_b[:, 2],
        oof_local_b[:, 3],
        oof_local_b[:, 6],
        oof_local_b[:, 7],
        oof_local_b[:, 8],
        oof_local_b[:, 9],
        oof_local_b[:, 10],
    ])

    meta_scaler = StandardScaler()
    meta_scaler.fit(oof_raw_meta)

    reconstructed_oof_meta = (
        meta_scaler.transform(oof_raw_meta)
    )

    max_meta_diff = np.max(
        np.abs(
            reconstructed_oof_meta -
            oof_meta[:, :12]
        )
    )

    print(
        f"Main64 meta reconstruction max error: "
        f"{max_meta_diff:.3e}"
    )

    if max_meta_diff > 1e-5:
        raise RuntimeError(
            "STOP: Main64 13-D meta representation could not "
            "be reconstructed exactly."
        )

    # --------------------------------------------------------
    # Main64 meta-model.
    #
    # This is the validated Main64 meta-model trained on
    # leakage-safe OOF meta-features.
    # --------------------------------------------------------

    main64_meta_model = LogisticRegression(
        C=META_C,
        class_weight="balanced",
        max_iter=5000,
        random_state=SEED,
    )

    main64_meta_model.fit(
        oof_meta,
        labels[train_idx],
    )

    # --------------------------------------------------------
    # Fit Main64 representation on ALL labelled data.
    # --------------------------------------------------------

    print("\nFitting Main64 representation on ALL labelled data...")

    (
        all_strings,
        full_tfidf,
        full_trans_vectorizer,
        full_compact_scaler,
        X_all_tfidf,
        X_all_compact,
        X_all_global,
    ) = fit_global_representation(texts)

    (
        test_strings,
        X_test_tfidf,
        X_test_compact,
        X_test_global,
    ) = transform_global_representation(
        test_texts,
        full_tfidf,
        full_trans_vectorizer,
        full_compact_scaler,
    )

    # --------------------------------------------------------
    # Full-data Main64 base models -> test.
    # --------------------------------------------------------

    print("Training Main64 SVM...")
    test_svm = train_svm(
        X_all_global,
        labels,
        X_test_global,
    )

    print("Training Main64 NB-SVM...")
    test_nbsvm = train_nbsvm(
        all_strings,
        labels,
        test_strings,
    )

    print("Training Main64 HGB...")
    test_hgb = train_hgb(
        X_all_compact,
        labels,
        X_test_compact,
    )

    print("Computing Main64 local geometry...")
    test_local = local_geometry_features(
        X_all_tfidf,
        labels,
        X_test_tfidf,
        k=LOCAL_K,
    )

    # --------------------------------------------------------
    # Main64 13-D test meta representation.
    # --------------------------------------------------------

    test_meta = make_meta_features(
        test_svm,
        test_nbsvm,
        test_hgb,
        test_local,
        meta_scaler,
    )

    if test_meta.shape != (
        len(test_texts),
        13,
    ):
        raise RuntimeError(
            f"Unexpected test meta shape: "
            f"{test_meta.shape}"
        )

    main64_test_probability = (
        main64_meta_model
        .predict_proba(test_meta)[:, 1]
    )

    main64_teacher_test = (
        main64_test_probability >=
        MAIN64_TEACHER_THRESHOLD
    ).astype(np.int64)

    print(
        "\nMain64 test teacher:"
    )
    print(
        f"  A: {np.sum(main64_teacher_test == 0)}"
    )
    print(
        f"  B: {np.sum(main64_teacher_test == 1)}"
    )

    # --------------------------------------------------------
    # Build exact 39-D Main114 features.
    # --------------------------------------------------------

    print(
        "\nBuilding exact Main114 39-D feature representation..."
    )

    raw_oof_39, raw_test_39 = (
        build_main114_39_features(
            oof_meta,
            oof_svm,
            oof_nbsvm,
            oof_hgb,
            test_meta,
            test_svm,
            test_nbsvm,
            test_hgb,
        )
    )

    # --------------------------------------------------------
    # Main111 structural features.
    # --------------------------------------------------------

    main111 = np.load(
        MAIN111_FILE,
    )

    saved_train_idx = main111["train_idx"]

    if not np.array_equal(
        saved_train_idx,
        train_idx,
    ):
        raise RuntimeError(
            "STOP: main111_features.npz train_idx does not "
            "match the exact canonical Main64 train order."
        )

    saved_val_idx = main111["val_idx"]

    if not np.array_equal(
        saved_val_idx,
        val_idx,
    ):
        raise RuntimeError(
            "STOP: main111_features.npz val_idx does not "
            "match the exact canonical validation order."
        )

    structural_all = (
        main111["structural_features"]
        .astype(np.float32)
    )

    if structural_all.shape[1] != 23:
        raise RuntimeError(
            f"Expected 23 Main111 structural features; "
            f"got {structural_all.shape}"
        )

    structural_oof = (
        structural_all[train_idx]
        .astype(np.float64)
    )

    print(
        "Computing Main111 structural features for test..."
    )

    structural_test = (
        make_main111_struct_features(
            test_texts
        ).astype(np.float64)
    )

    if structural_test.shape != (
        len(test_texts),
        23,
    ):
        raise RuntimeError(
            f"Unexpected test structural shape: "
            f"{structural_test.shape}"
        )

    # --------------------------------------------------------
    # Exact 62-D Main115 representation.
    # --------------------------------------------------------

    raw_oof_62 = np.hstack([
        raw_oof_39.astype(np.float64),
        structural_oof,
    ])

    raw_test_62 = np.hstack([
        raw_test_39.astype(np.float64),
        structural_test,
    ])

    if raw_oof_62.shape != (
        len(train_idx),
        62,
    ):
        raise RuntimeError(
            f"Unexpected OOF Main115 raw feature shape: "
            f"{raw_oof_62.shape}"
        )

    if raw_test_62.shape != (
        len(test_texts),
        62,
    ):
        raise RuntimeError(
            f"Unexpected test Main115 raw feature shape: "
            f"{raw_test_62.shape}"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Main114 standardized the complete 62-D matrix using OOF
    # rows. Reproduce exactly.
    # --------------------------------------------------------

    feature_mu = np.mean(
        raw_oof_62,
        axis=0,
    )

    feature_sd = np.std(
        raw_oof_62,
        axis=0,
    )

    feature_sd = np.where(
        feature_sd < 1e-8,
        1.0,
        feature_sd,
    )

    X_oof_62 = (
        raw_oof_62 -
        feature_mu
    ) / feature_sd

    X_test_62 = (
        raw_test_62 -
        feature_mu
    ) / feature_sd

    X_oof_62 = np.nan_to_num(
        X_oof_62,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    X_test_62 = np.nan_to_num(
        X_test_62,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    # --------------------------------------------------------
    # Critical integrity check against saved Main115 training
    # representation.
    # --------------------------------------------------------

    saved_results = np.load(
        MAIN114_RESULTS,
        allow_pickle=True,
    )

    saved_feature_oof = (
        saved_results["feature_oof"]
        .astype(np.float32)
    )

    if saved_feature_oof.shape != X_oof_62.shape:
        raise RuntimeError(
            "STOP: saved Main115 OOF feature shape differs "
            "from reconstructed feature shape."
        )

    max_feature_diff = np.max(
        np.abs(
            X_oof_62 -
            saved_feature_oof
        )
    )

    print(
        f"Main115 62-D OOF reconstruction max error: "
        f"{max_feature_diff:.3e}"
    )

    if max_feature_diff > 1e-4:
        raise RuntimeError(
            "STOP: reconstructed Main115 62-D representation "
            "does not match the representation used to obtain "
            "93.4535% validation accuracy."
        )

    print(
        "Main115 62-D representation: EXACT MATCH"
    )

    # --------------------------------------------------------
    # Train exact five Main115 specialists on the saved OOF
    # representation.
    # --------------------------------------------------------

    print(
        "\nTraining frozen Main115 specialists..."
    )

    specialists = build_specialists()
    specialist_probabilities = []

    for name, model in specialists.items():
        print(
            f"  {name}..."
        )

        model.fit(
            X_oof_62,
            labels[train_idx],
        )

        probability = (
            model.predict_proba(
                X_test_62
            )[:, 1]
        )

        specialist_probabilities.append(
            probability
        )

    p = np.column_stack(
        specialist_probabilities
    )

    # --------------------------------------------------------
    # Frozen consensus correction.
    # --------------------------------------------------------

    specialist_hard = (
        p >= 0.5
    ).astype(np.int64)

    disagreement = (
        specialist_hard !=
        main64_teacher_test[:, None]
    )

    confidence = np.maximum(
        p,
        1.0 - p,
    )

    votes = np.sum(
        disagreement,
        axis=1,
    )

    disagreement_strength = np.sum(
        np.where(
            disagreement,
            confidence - 0.5,
            0.0,
        ),
        axis=1,
    )

    mean_margin = np.divide(
        disagreement_strength,
        votes,
        out=np.zeros_like(
            disagreement_strength
        ),
        where=votes > 0,
    )

    weakest_confidence = np.where(
        disagreement,
        confidence,
        1.0,
    ).min(axis=1)

    change = (
        (votes >= MAIN115_MIN_VOTES)
        &
        (mean_margin >= MAIN115_MEAN_MARGIN)
        &
        (weakest_confidence >= MAIN115_MIN_CONFIDENCE)
    )

    final_prediction = (
        main64_teacher_test.copy()
    )

    # The five specialists can only agree on one opposite class when
    # they are voting against the teacher. Flip the teacher in that case.
    final_prediction[change] = (
        1 -
        final_prediction[change]
    )

    final_labels = np.where(
        final_prediction == 0,
        "A",
        "B",
    )

    # --------------------------------------------------------
    # Sanity checks.
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )
    print(
        "FINAL SUBMISSION SANITY CHECK"
    )
    print(
        "=" * 80
    )

    if len(final_labels) != 3000:
        raise RuntimeError(
            "Submission does not contain exactly 3000 predictions."
        )

    if len(set(test_ids)) != 3000:
        raise RuntimeError(
            "Test IDs are not unique."
        )

    if not np.all(
        np.isfinite(
            main64_test_probability
        )
    ):
        raise RuntimeError(
            "Main64 test probabilities contain NaN/Inf."
        )

    if not np.all(
        np.isfinite(p)
    ):
        raise RuntimeError(
            "Main115 specialist probabilities contain NaN/Inf."
        )

    if not np.all(
        np.isin(
            final_labels,
            ["A", "B"],
        )
    ):
        raise RuntimeError(
            "Invalid labels generated."
        )

    print(
        f"Main64 reference validation accuracy : "
        f"{MAIN64_VALIDATION_ACCURACY * 100:.4f}%"
    )

    print(
        f"Main115 reference validation accuracy: "
        f"{MAIN115_VALIDATION_ACCURACY * 100:.4f}%"
    )

    print(
        "Main115 frozen rule:"
    )
    print(
        f"  votes >= {MAIN115_MIN_VOTES}"
    )
    print(
        f"  mean disagreement margin >= "
        f"{MAIN115_MEAN_MARGIN:.2f}"
    )
    print(
        f"  weakest confidence >= "
        f"{MAIN115_MIN_CONFIDENCE:.2f}"
    )

    print(
        f"\nTest teacher predictions: "
        f"A={np.sum(main64_teacher_test == 0)} "
        f"B={np.sum(main64_teacher_test == 1)}"
    )

    print(
        f"Main115 corrections applied: "
        f"{int(change.sum())}"
    )

    print(
        f"Final predictions: "
        f"A={np.sum(final_labels == 'A')} "
        f"B={np.sum(final_labels == 'B')}"
    )

    # --------------------------------------------------------
    # Write submission.
    # --------------------------------------------------------

    submission = pd.DataFrame({
        "id": test_ids,
        "label": final_labels,
    })

    # If sample_submission.csv exists, preserve its ID order/schema
    # exactly after verifying it matches test.json.
    if os.path.exists(
        "sample_submission.csv"
    ):
        sample = pd.read_csv(
            "sample_submission.csv"
        )

        if list(sample.columns) != [
            "id",
            "label",
        ]:
            raise RuntimeError(
                "sample_submission.csv does not have "
                "the expected [id, label] columns."
            )

        if not np.array_equal(
            sample["id"].to_numpy(),
            np.asarray(test_ids),
        ):
            raise RuntimeError(
                "sample_submission.csv IDs/order do not "
                "match test.json."
            )

        submission = sample.copy()
        submission["label"] = final_labels

    submission.to_csv(
        SUBMISSION_FILE,
        index=False,
    )

    # Re-read the exact file that will be uploaded.
    check = pd.read_csv(
        SUBMISSION_FILE
    )

    if len(check) != 3000:
        raise RuntimeError(
            "Written submission does not have 3000 rows."
        )

    if list(check.columns) != [
        "id",
        "label",
    ]:
        raise RuntimeError(
            "Written submission has incorrect columns."
        )

    if not check["id"].is_unique:
        raise RuntimeError(
            "Written submission contains duplicate IDs."
        )

    if not set(
        check["label"].unique()
    ).issubset({"A", "B"}):
        raise RuntimeError(
            "Written submission contains invalid labels."
        )

    # --------------------------------------------------------
    # Metadata explicitly records which validated version this is.
    # --------------------------------------------------------

    metadata = {
        "submission_file": SUBMISSION_FILE,
        "model_version": "Main115",
        "main64_validation_accuracy": MAIN64_VALIDATION_ACCURACY,
        "main115_validation_accuracy": MAIN115_VALIDATION_ACCURACY,
        "main115_validation_delta": (
            MAIN115_VALIDATION_ACCURACY -
            MAIN64_VALIDATION_ACCURACY
        ),
        "frozen_rule": {
            "min_votes": MAIN115_MIN_VOTES,
            "mean_margin": MAIN115_MEAN_MARGIN,
            "min_confidence": MAIN115_MIN_CONFIDENCE,
        },
        "test_documents": len(test_texts),
        "main115_corrections_applied": int(
            change.sum()
        ),
        "final_A": int(
            np.sum(final_labels == "A")
        ),
        "final_B": int(
            np.sum(final_labels == "B")
        ),
        "note": (
            "This is the Main115 frozen OOF-selected submission. "
            "No test labels or Kaggle scores were used to tune the rule."
        ),
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        "\n" + "=" * 80
    )
    print(
        "SUBMISSION READY"
    )
    print(
        "=" * 80
    )

    print(
        f"Written: {SUBMISSION_FILE}"
    )

    print(
        f"Validation reference: "
        f"Main115 = {MAIN115_VALIDATION_ACCURACY * 100:.4f}%"
    )

    print(
        f"Metadata: {METADATA_FILE}"
    )

    print(
        f"Runtime: {time.time() - start:.1f} s"
    )


if __name__ == "__main__":
    main()
