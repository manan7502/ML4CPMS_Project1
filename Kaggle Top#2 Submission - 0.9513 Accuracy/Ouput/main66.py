
import json
import joblib
import os
import time
import warnings

import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")

# ============================================================
# MAIN66
# Final submission pipeline
#
# Strategy:
#   1. Reproduce Main64 validation pipeline.
#   2. Apply calibrated probability threshold = 0.3745.
#   3. If validation sanity check passes, refit base models
#      on ALL 10,536 labelled documents.
#   4. Predict test.json.
#   5. Write submission.csv.
#
# NOTE:
# The meta-model is intentionally the validated Main64
# meta-model, rather than another expensive 5-fold OOF run.
# ============================================================

SEED = 42

WORD_NGRAM = (1, 6)
WORD_MIN_DF = 3
TRANS_NGRAM = (1, 2)
TRANS_MIN_DF = 2

SVM_C = 10.0
NBSVM_C = 10.0
LOCAL_K = 20
META_C = 0.1

CALIBRATED_THRESHOLD = 0.3745

TRAIN_FILE = "train.json"
TEST_FILE = "test.json"
SUBMISSION_FILE = "submission.csv"

EXPECTED_VALIDATION = 0.930266
VALIDATION_TOLERANCE = 0.015

# ============================================================
# DATA
# ============================================================

def load_jsonl(path, labelled=True):
    records = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    texts = [r["text"] for r in records]
    ids = [r["id"] for r in records]

    if labelled:
        labels = np.array(
            [0 if r["label"] == "A" else 1 for r in records],
            dtype=np.int64
        )
        return ids, texts, labels

    return ids, texts


def to_strings(texts):
    return [" ".join(map(str, doc)) for doc in texts]


def make_transition_strings(texts):
    output = []

    for doc in texts:
        if len(doc) < 2:
            output.append("")
            continue

        transitions = [
            f"{doc[i]}T{doc[i + 1]}"
            for i in range(len(doc) - 1)
        ]

        output.append(" ".join(transitions))

    return output


# ============================================================
# COMPACT FEATURES — EXACTLY MAIN64
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
        return np.zeros(62, dtype=np.float32)

    unique, counts = np.unique(x, return_counts=True)
    u = len(unique)

    repetition_ratio = 1.0 - (u / n)
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
        return len(grams) / total, len(grams)

    bdiv, buniq = ngram_stats(2)
    tdiv, tuniq = ngram_stats(3)

    quarter_values = []

    for q in range(4):
        start = (q * n) // 4
        end = ((q + 1) * n) // 4
        part = x[start:end]

        if len(part) == 0:
            quarter_values.extend([0.0] * 5)
            continue

        pu, pc = np.unique(part, return_counts=True)
        plen = len(part)

        quarter_values.extend([
            plen / n,
            len(pu) / plen,
            entropy_from_counts(pc),
            1.0 - len(pu) / plen,
            pc.max() / plen
        ])

    k = min(10, n)

    beginning = x[:k]
    ending = x[-k:]

    begin_unique = len(np.unique(beginning)) / k
    end_unique = len(np.unique(ending)) / k

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

        transitions = np.stack([x[:-1], x[1:]], axis=1)
        unique_transitions = len(
            np.unique(transitions, axis=0)
        )
        transition_diversity = unique_transitions / (n - 1)
    else:
        adjacent_repeat_ratio = 0.0
        transition_diversity = 0.0

    max_freq_ratio = max_freq / n
    repeated_type_ratio = repeated_types / max(u, 1)

    features = [
        np.log1p(n), n,
        np.log1p(u), u,
        u / n, repetition_ratio,
        max_freq, max_freq_ratio,
        repeated_occurrences, repeated_occurrences / n,
        repeated_types, repeated_type_ratio,
        entropy, entropy / np.log(max(u, 2)),
        zero_count, zero_ratio,

        bdiv, buniq,
        tdiv, tuniq,
        transition_diversity,
        adjacent_repeat_ratio,

        begin_unique, end_unique,
        begin_entropy, end_entropy,

        first_unique_ratio, second_unique_ratio,
        first_entropy, second_entropy,
        second_unique_ratio - first_unique_ratio,
        second_entropy - first_entropy,

        np.std(counts),
        np.mean(counts),
        np.median(counts),
        np.max(counts) - np.median(counts),

        *quarter_values
    ]

    return np.asarray(features, dtype=np.float32)


def build_compact_features(texts):
    return np.vstack([sequence_features(doc) for doc in texts])


# ============================================================
# BASE MODELS — EXACTLY MAIN64 CONVENTIONS
# ============================================================

def train_svm(X_train, y_train, X_query):
    model = LinearSVC(
        C=SVM_C,
        class_weight="balanced",
        tol=1e-2,
        max_iter=500000
    )
    model.fit(X_train, y_train)

    # Positive = B
    return model.decision_function(X_query)


def train_nbsvm(train_strings, y_train, query_strings):
    vectorizer = CountVectorizer(
        ngram_range=(1, 3),
        min_df=2,
        binary=True
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
        max_iter=500000
    )

    model.fit(X_train_nb, y_train)

    # Positive = B
    return model.decision_function(X_query_nb)


def train_hgb(X_train, y_train, X_query):
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        l2_regularization=1.0,
        random_state=SEED
    )

    weights = np.where(
        y_train == 0,
        1.0,
        np.sum(y_train == 0) / np.sum(y_train == 1)
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=weights
    )

    probability_B = model.predict_proba(X_query)[:, 1]

    # Positive = B
    return probability_B - 0.5


def local_geometry_features(
    X_reference,
    y_reference,
    X_query,
    k=20
):
    """
    A-oriented geometry.
    Positive = A, negative = B.
    """

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
        n_jobs=-1
    )

    nnB = NearestNeighbors(
        n_neighbors=kB,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1
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
            np.mean(simA[i]) - np.mean(simB[i])
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
            np.mean(simA[i]) - np.mean(simB[i])
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
            *multi
        ])

    return np.asarray(output, dtype=np.float64)


# ============================================================
# BUILD REPRESENTATIONS
# ============================================================

def build_representation(texts, fit_data=None):
    """
    fit_data is unused here; kept only to make the workflow clear.
    Vectorizers are always fitted explicitly where needed.
    """
    strings = to_strings(texts)
    transitions = make_transition_strings(texts)

    return strings, transitions


def fit_global_representation(train_texts):
    train_strings = to_strings(train_texts)
    train_transitions = make_transition_strings(train_texts)

    tfidf = TfidfVectorizer(
        ngram_range=WORD_NGRAM,
        min_df=WORD_MIN_DF,
        sublinear_tf=True
    )

    X_tfidf = tfidf.fit_transform(train_strings)

    trans_vectorizer = TfidfVectorizer(
        ngram_range=TRANS_NGRAM,
        min_df=TRANS_MIN_DF,
        sublinear_tf=True,
        token_pattern=r"(?u)\S+"
    )

    X_transition = trans_vectorizer.fit_transform(train_transitions)

    compact = build_compact_features(train_texts)

    scaler = StandardScaler()
    compact_s = scaler.fit_transform(compact)

    X_global = hstack([
        X_tfidf,
        csr_matrix(compact_s),
        X_transition
    ]).tocsr()

    return (
        train_strings,
        train_transitions,
        tfidf,
        trans_vectorizer,
        scaler,
        X_tfidf,
        compact,
        X_global
    )


def transform_global_representation(
    texts,
    tfidf,
    trans_vectorizer,
    compact_scaler
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
        X_transition
    ]).tocsr()

    return (
        strings,
        transitions,
        X_tfidf,
        compact,
        X_global
    )


# ============================================================
# META FEATURE CONSTRUCTION
# ============================================================

def make_meta_features(
    svm,
    nbsvm,
    hgb,
    local,
    scaler,
    fit_scaler=False
):
    # local is A-positive, so convert to B-positive
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
        local_b[:, 10]
    ])

    if fit_scaler:
        transformed = scaler.fit_transform(raw)
    else:
        transformed = scaler.transform(raw)

    uncertainty = np.exp(-np.abs(svm))
    geometry_signal = local_b[:, 3]

    interaction = (
        geometry_signal * uncertainty
    ).reshape(-1, 1)

    return np.hstack([transformed, interaction])


# ============================================================
# MAIN
# ============================================================

def main():
    start = time.time()

    print("=" * 70)
    print("MAIN66")
    print("Final calibrated submission pipeline")
    print("=" * 70)

    # --------------------------------------------------------
    # Load labelled data
    # --------------------------------------------------------

    train_ids, texts, labels = load_jsonl(
        TRAIN_FILE,
        labelled=True
    )

    print("\nDocuments:", len(texts))
    print("Class A:", np.sum(labels == 0))
    print("Class B:", np.sum(labels == 1))

    # --------------------------------------------------------
    # Recreate exact Main64 external validation split
    # --------------------------------------------------------

    indices = np.arange(len(texts))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=SEED,
        stratify=labels
    )

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]

    y_train = labels[train_idx]
    y_val = labels[val_idx]

    print("\nValidation sanity stage")
    print("Training:", len(train_idx))
    print("Validation:", len(val_idx))

    # --------------------------------------------------------
    # Fit representation on 8428 training documents
    # --------------------------------------------------------

    print("\nFitting validation-stage representation...")

    (
        train_strings,
        train_transitions,
        tfidf,
        trans_vectorizer,
        compact_scaler,
        X_train_tfidf,
        X_train_compact,
        X_train_global
    ) = fit_global_representation(train_texts)

    (
        val_strings,
        val_transitions,
        X_val_tfidf,
        X_val_compact,
        X_val_global
    ) = transform_global_representation(
        val_texts,
        tfidf,
        trans_vectorizer,
        compact_scaler
    )

    print("Global TF-IDF:", X_train_tfidf.shape)
    print("Transition TF-IDF:", trans_vectorizer.transform(train_transitions).shape)
    print("Compact:", X_train_compact.shape)
    print("Combined:", X_train_global.shape)

    # --------------------------------------------------------
    # Fit validation base models
    # --------------------------------------------------------

    print("\nTraining validation SVM...")
    val_svm = train_svm(
        X_train_global,
        y_train,
        X_val_global
    )

    print("Training validation NBSVM...")
    val_nbsvm = train_nbsvm(
        train_strings,
        y_train,
        val_strings
    )

    print("Training validation HGB...")
    val_hgb = train_hgb(
        X_train_compact,
        y_train,
        X_val_compact
    )

    print("Computing validation local geometry...")
    val_local = local_geometry_features(
        X_train_tfidf,
        y_train,
        X_val_tfidf,
        k=LOCAL_K
    )

    # --------------------------------------------------------
    # Load Main64 OOF features to recreate its meta-model
    # --------------------------------------------------------

    print("\nLoading Main64 OOF meta-features...")

    oof_meta_path = "main64_oof_meta_features.npy"

    if not os.path.exists(oof_meta_path):
        raise FileNotFoundError(
            f"Missing {oof_meta_path}. "
            "Keep the Main64 result files in the project folder."
        )

    oof_meta = np.load(oof_meta_path)

    if oof_meta.shape[1] != 13:
        raise ValueError(
            f"Expected 13 Main64 meta features, got {oof_meta.shape}"
        )

    print("OOF meta shape:", oof_meta.shape)

    # Recreate Main64 meta model.
    # Main64 meta features were already standardized and include
    # the final interaction column.
    meta_model = LogisticRegression(
        C=META_C,
        class_weight="balanced",
        max_iter=5000,
        random_state=SEED
    )

    meta_model.fit(oof_meta, y_train)

    # Main64's scaler is not saved separately, so reconstruct it
    # from OOF meta features after removing the interaction column.
    #
    # The OOF file contains:
    #   12 standardized base features + interaction.
    #
    # We therefore standardize the new validation base features
    # using the distribution implied by Main64's OOF feature file.
    #
    # The first 12 OOF columns are zero mean/unit variance, so this
    # is equivalent to applying their OOF means/stds (0 and 1).
    #
    # For exact transfer, use the saved Main64 validation meta
    # distribution as a diagnostic and keep the same score units.
    #
    # In practice Main64's saved OOF meta is already standardized,
    # so validation raw features are standardized independently
    # below using the OOF base score arrays.

    oof_svm = np.load("main64_oof_svm.npy")
    oof_nbsvm = np.load("main64_oof_nbsvm.npy")
    oof_hgb = np.load("main64_oof_hgb.npy")
    oof_local = np.load("main64_oof_local.npy")

    # Reconstruct the exact Main64 meta scaler from raw OOF scores.
    oof_local_b = -oof_local

    oof_raw = np.column_stack([
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
        oof_local_b[:, 10]
    ])

    exact_meta_scaler = StandardScaler()
    exact_meta_scaler.fit(oof_raw)

    # Sanity: this should recreate Main64's first 12 columns.
    reconstructed_oof_base = exact_meta_scaler.transform(oof_raw)

    max_diff = np.max(
        np.abs(reconstructed_oof_base - oof_meta[:, :12])
    )

    print("Meta-feature reconstruction max difference:", max_diff)

    if max_diff > 1e-5:
        raise RuntimeError(
            "Main64 meta-feature reconstruction failed. "
            "Stopping before submission."
        )

    # Build validation meta features
    val_meta = make_meta_features(
        val_svm,
        val_nbsvm,
        val_hgb,
        val_local,
        exact_meta_scaler
    )

    val_probability_B = meta_model.predict_proba(
        val_meta
    )[:, 1]

    # --------------------------------------------------------
    # VALIDATION CHECK
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VALIDATION SANITY CHECK")
    print("=" * 70)

    default_pred = (
        val_probability_B >= 0.5
    ).astype(int)

    calibrated_pred = (
        val_probability_B >= CALIBRATED_THRESHOLD
    ).astype(int)

    default_acc = accuracy_score(
        y_val,
        default_pred
    )

    calibrated_acc = accuracy_score(
        y_val,
        calibrated_pred
    )

    print(
        f"Main64 threshold 0.5000 : "
        f"{default_acc * 100:.4f}%"
    )

    print(
        f"Calibrated threshold "
        f"{CALIBRATED_THRESHOLD:.4f} : "
        f"{calibrated_acc * 100:.4f}%"
    )

    print("\nCalibrated confusion:")
    print(confusion_matrix(y_val, calibrated_pred))

    print("\nErrors:", np.sum(calibrated_pred != y_val))

    # Orientation sanity
    print("\nPrediction counts:")
    print("A:", np.sum(calibrated_pred == 0))
    print("B:", np.sum(calibrated_pred == 1))

    if calibrated_acc < EXPECTED_VALIDATION - VALIDATION_TOLERANCE:
        raise RuntimeError(
            "\nVALIDATION SANITY CHECK FAILED.\n"
            f"Expected approximately {EXPECTED_VALIDATION * 100:.2f}% "
            f"but obtained {calibrated_acc * 100:.2f}%.\n"
            "No submission was generated."
        )

    print("\nVALIDATION SANITY CHECK: PASS")

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)

    test_ids, test_texts = load_jsonl(
        TEST_FILE,
        labelled=False
    )

    print("Test documents:", len(test_texts))

    if len(test_texts) != 3000:
        print(
            "WARNING: expected 3000 test documents, "
            f"found {len(test_texts)}"
        )

    # --------------------------------------------------------
    # FIT REPRESENTATION ON ALL LABELLED DATA
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RETRAINING BASE MODELS ON ALL LABELLED DATA")
    print("=" * 70)

    (
        all_strings,
        all_transitions,
        full_tfidf,
        full_trans_vectorizer,
        full_compact_scaler,
        X_all_tfidf,
        X_all_compact,
        X_all_global
    ) = fit_global_representation(texts)

    print("Full TF-IDF:", X_all_tfidf.shape)
    print("Full compact:", X_all_compact.shape)
    print("Full combined:", X_all_global.shape)

    (
        test_strings,
        test_transitions,
        X_test_tfidf,
        X_test_compact,
        X_test_global
    ) = transform_global_representation(
        test_texts,
        full_tfidf,
        full_trans_vectorizer,
        full_compact_scaler
    )

    # --------------------------------------------------------
    # FINAL BASE MODELS
    # --------------------------------------------------------

    print("\nFinal SVM...")
    test_svm = train_svm(
        X_all_global,
        labels,
        X_test_global
    )

    print("Final NBSVM...")
    test_nbsvm = train_nbsvm(
        all_strings,
        labels,
        test_strings
    )

    print("Final HGB...")
    test_hgb = train_hgb(
        X_all_compact,
        labels,
        X_test_compact
    )

    print("Final local geometry...")
    test_local = local_geometry_features(
        X_all_tfidf,
        labels,
        X_test_tfidf,
        k=LOCAL_K
    )

    # --------------------------------------------------------
    # FINAL META PREDICTION
    # --------------------------------------------------------

    print("\nBuilding final meta features...")

    test_meta = make_meta_features(
        test_svm,
        test_nbsvm,
        test_hgb,
        test_local,
        exact_meta_scaler
    )

    test_probability_B = meta_model.predict_proba(
        test_meta
    )[:, 1]

    test_prediction = (
        test_probability_B >= CALIBRATED_THRESHOLD
    ).astype(int)

    test_labels = np.where(
        test_prediction == 0,
        "A",
        "B"
    )

    # --------------------------------------------------------
    # SANITY CHECKS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST SANITY CHECKS")
    print("=" * 70)

    assert len(test_ids) == len(test_prediction)
    assert len(test_ids) == len(test_labels)
    assert len(test_ids) == len(test_probability_B)

    assert len(set(test_ids)) == len(test_ids)

    assert np.all(
        np.isin(test_labels, ["A", "B"])
    )

    assert np.all(
        np.isfinite(test_probability_B)
    )

    print("Prediction count:", len(test_prediction), "PASS")
    print("Unique IDs:", len(set(test_ids)), "PASS")
    print("Labels only A/B: PASS")
    print("No NaN/Inf scores: PASS")

    print("\nTest predictions:")
    print("A:", np.sum(test_labels == "A"))
    print("B:", np.sum(test_labels == "B"))

    # --------------------------------------------------------
    # SUBMISSION CSV
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CREATING SUBMISSION")
    print("=" * 70)

    # Competition-specific submission schemas vary.
    # The assignment's expected schema is ID + label.
    submission = pd.DataFrame({
        "id": test_ids,
        "label": test_labels
    })

    submission.to_csv(
        SUBMISSION_FILE,
        index=False
    )

    # Re-read to verify what was actually written.
    check = pd.read_csv(SUBMISSION_FILE)

    assert len(check) == len(test_ids)
    assert list(check.columns) == ["id", "label"]
    assert check["id"].is_unique
    assert set(check["label"].unique()).issubset({"A", "B"})

    print("\nSUBMISSION CREATED SUCCESSFULLY")
    print("File:", os.path.abspath(SUBMISSION_FILE))
    print("Rows:", len(check))
    print("Columns:", list(check.columns))


if __name__ == "__main__":
    main()
    print("A:", np.sum(check["label"] == "A"))
    print("B:", np.sum(check["label"] == "B"))

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"\nTotal time: {time.time() - start:.2f} seconds"
    )


if __name__ == "__main__":
    main()
