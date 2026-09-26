"""
================================================================================
Canonical Usage Guide & Exact Replication of Exp 4 SOTA (91.74% 5-Fold CV)
================================================================================
Directory: Core_Analysis/Feature_Engineering_Consolidation/
Script: replicate_exp4_sota.py

This script serves two primary purposes:
  1. CANONICAL CLASS USAGE REFERENCE:
     Demonstrates the standard developer workflow for using `FeatureExtractor`:
     - Initializing the extractor with domain configurations.
     - Extracting individual modular feature families independently.
     - End-to-end `fit_transform` and `transform` following Scikit-Learn semantics.
     - Serializing and reloading the fitted pipeline via `.save()` and `.load()`.

  2. NUMERICAL REPLICATION OF EXP 4 SOTA:
     Executes the 5-fold Stratified Cross-Validation benchmark on the full
     dataset (N = 10,536) to reproduce the exact breakthrough SOTA accuracy:
       - Baseline (Stage 4):  91.31% (+/- 0.40%)
       - Replicated Exp 4:    91.74% (+/- 0.35%)  [+0.44% Gain]
================================================================================
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Dict, Any

from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import MaxAbsScaler

# Ensure local Feature_Engineering_Consolidation module is accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import FeatureExtractor


def resolve_dataset_paths() -> Tuple[str, str]:
    """Finds the dataset and log-DF files across standard repository paths."""
    candidate_data = [
        "../../../../train.json",
        "/home/justin/PhD/ML4CPS/Kaggle Project/Analysis_II/train.json",
        "train.json",
    ]
    candidate_log_df = [
        "../data/unigram_log_df_ratios.json",
        "data/unigram_log_df_ratios.json",
        "/home/justin/PhD/ML4CPS/Kaggle Project/Analysis_II/ML4CPMS_Project1/Main79 - Classical + Residual/Core_Analysis/data/unigram_log_df_ratios.json",
    ]

    data_path = next((p for p in candidate_data if os.path.exists(p)), None)
    log_df_path = next((p for p in candidate_log_df if os.path.exists(p)), None)

    if not data_path:
        raise FileNotFoundError(f"Could not locate 'train.json'. Looked in: {candidate_data}")

    return data_path, log_df_path


def load_dataset(data_path: str) -> Tuple[List[List[int]], np.ndarray]:
    """Loads raw token sequences and integer labels from JSONL format."""
    print(f"[DataLoader] Ingesting tokenized dataset from: {os.path.abspath(data_path)}")
    t0 = time.time()
    with open(data_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    token_sequences = [rec["text"] for rec in records]
    labels = np.array([1 if rec["label"] == "B" else 0 for rec in records], dtype=np.int32)
    elapsed = time.time() - t0

    n_machine = int(np.sum(labels == 1))
    n_human = int(np.sum(labels == 0))
    print(
        f"[DataLoader] Successfully loaded {len(token_sequences):,} documents in {elapsed:.2f}s "
        f"(Machine: {n_machine:,} | Human: {n_human:,} | Majority Baseline: {100 * n_machine / len(labels):.2f}%)"
    )
    return token_sequences, labels


# ==============================================================================
# PART 1: CANONICAL CLASS USAGE DEMONSTRATION
# ==============================================================================
def demonstrate_class_usage(sample_tokens: List[List[int]], sample_labels: np.ndarray) -> None:
    """
    Demonstrates the clean developer API of the FeatureExtractor class:
      - Modular extraction functions
      - Scikit-Learn transformer API
      - Persistence helpers
    """
    print("\n" + "=" * 80)
    print("PART 1: CANONICAL FeatureExtractor API & USAGE WORKFLOW")
    print("=" * 80)

    print("\n[Step 1] Initializing FeatureExtractor instance...")
    extractor = FeatureExtractor(
        ngram_range=(1, 5),
        min_df=3,
        sublinear_tf=True,
        trajectory_top_k=500,
        trajectory_n_components=12,
        random_state=42,
    )
    print(f"  Initialized: {extractor}")

    print("\n[Step 2] Independent Extraction via Modular Functions (First 5 docs):")
    mini_docs = sample_tokens[:5]

    # Length (1 dim)
    len_feats = extractor.extract_length_features(mini_docs)
    print(f"  • extract_length_features()                -> shape: {len_feats.shape}")

    # Stage 1: Lexical Diversity & Entropy (4 dims)
    div_feats = extractor.extract_lexical_diversity_features(mini_docs)
    print(f"  • extract_lexical_diversity_features()     -> shape: {div_feats.shape}")

    # Stage 2: Repetition & Token ID Profiling (6 dims)
    rep_feats = extractor.extract_repetition_profiling_features(mini_docs)
    print(f"  • extract_repetition_profiling_features()  -> shape: {rep_feats.shape}")

    # Stage 3: Log-DF Machine Likelihood Moments (5 dims)
    log_feats = extractor.extract_log_df_likelihood_features(mini_docs)
    print(f"  • extract_log_df_likelihood_features()     -> shape: {log_feats.shape}")

    # Stage 4: Sequential Dynamics & Burstiness (2 dims)
    seq_feats = extractor.extract_sequential_dynamics_features(mini_docs)
    print(f"  • extract_sequential_dynamics_features()   -> shape: {seq_feats.shape}")

    print("\n[Step 3] Fitting Extractor Pipeline on Train Split (80/20 train/test)...")
    train_docs, test_docs, y_train, y_test = train_test_split(
        sample_tokens, sample_labels, test_size=0.20, random_state=42, stratify=sample_labels
    )

    t0 = time.time()
    # fit_transform() builds SVD syntactic manifold, fits TF-IDF, and fits MaxAbsScaler
    X_train_comp = extractor.fit_transform(train_docs)
    train_time = time.time() - t0
    print(f"  Fitted on {len(train_docs):,} documents in {train_time:.2f}s!")
    print(f"  Train Composite Feature Matrix shape: {X_train_comp.shape}")

    print("\n[Step 4] Transforming Unseen Validation Split...")
    t0 = time.time()
    X_test_comp = extractor.transform(test_docs)
    test_time = time.time() - t0
    print(f"  Transformed {len(test_docs):,} validation docs in {test_time:.2f}s!")
    print(f"  Test Composite Feature Matrix shape:  {X_test_comp.shape}")

    print("\n[Step 5] Training Linear Classifier (LinearSVC)...")
    clf = LinearSVC(C=1.0, dual="auto", random_state=42)
    clf.fit(X_train_comp, y_train)
    val_preds = clf.predict(X_test_comp)
    holdout_acc = accuracy_score(y_test, val_preds) * 100
    holdout_f1 = f1_score(y_test, val_preds, average="macro") * 100

    print(f"  Holdout 20% Accuracy: {holdout_acc:.2f}% | Macro F1: {holdout_f1:.2f}%")

    print("\n[Step 6] Pipeline Persistence (save & load):")
    save_path = "fitted_feature_extractor_demo.pkl"
    extractor.save(save_path)
    restored = FeatureExtractor.load(save_path)
    X_reloaded_test = restored.transform(test_docs[:10])
    assert (X_test_comp[:10] != X_reloaded_test).nnz == 0
    print("  Bitwise verification passed: Reloaded model matches fitted output exactly!")
    if os.path.exists(save_path):
        os.remove(save_path)


# ==============================================================================
# PART 2: EXACT NUMERICAL REPLICATION OF EXP 4 SOTA (5-FOLD STRATIFIED CV)
# ==============================================================================
def replicate_exp4_sota(token_sequences: List[List[int]], labels: np.ndarray) -> Dict[str, Any]:
    """
    Executes the exact 5-fold Stratified Cross-Validation protocol from Exp 4
    to reproduce the 91.74% SOTA benchmark.
    """
    print("\n" + "=" * 80)
    print("PART 2: EXACT 5-FOLD STRATIFIED CV REPLICATION OF EXP 4 SOTA")
    print("=" * 80)

    n_samples = len(token_sequences)
    print(f"[Replication] Total corpus size: {n_samples:,} documents.")

    # 1. Initialize FeatureExtractor
    print("[Replication] Initializing FeatureExtractor for Exp 4 SOTA pipeline...")
    extractor = FeatureExtractor(
        ngram_range=(1, 5),
        min_df=3,
        sublinear_tf=True,
        trajectory_top_k=500,
        trajectory_n_components=12,
        random_state=42,
    )

    # 2. Extract Spectral Syntactic Manifold & Dense Features across corpus
    print("[Replication] Step 1/3: Fitting spectral syntactic transition manifold...")
    t0 = time.time()
    extractor._fit_spectral_syntactic_manifold(token_sequences)
    print(f"  Extracted 12-D Syntactic Manifold in {time.time()-t0:.2f}s.")

    print("[Replication] Step 2/3: Extracting all 34 dense feature descriptors...")
    t0 = time.time()
    dense_matrix = extractor.extract_all_dense_features(token_sequences)
    print(f"  Dense Feature Matrix extracted: {dense_matrix.shape} in {time.time()-t0:.2f}s.")

    print("[Replication] Step 3/3: Vectorizing cumulative 1-5 gram TF-IDF matrix...")
    t0 = time.time()
    X_tfidf = extractor.generate_tfidf_features(token_sequences, is_training=True)
    print(f"  Sparse TF-IDF Matrix generated: {X_tfidf.shape} in {time.time()-t0:.2f}s.")

    # 3. 5-Fold Stratified Cross-Validation
    print("\n[Replication] Running 5-Fold Stratified Cross-Validation (Seed = 42)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_f1_scores = []
    fold_durations = []

    print("-" * 80)
    print(f"{'Fold':^8} | {'Train / Val':^18} | {'Accuracy (%)':^15} | {'Macro F1 (%)':^15} | {'Duration':^10}")
    print("-" * 80)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_tfidf, labels), 1):
        t_fold = time.time()

        # In-fold MaxAbsScaler fitting strictly on training split
        scaler = MaxAbsScaler()
        tr_dense_scaled = scaler.fit_transform(dense_matrix[train_idx])
        val_dense_scaled = scaler.transform(dense_matrix[val_idx])

        # Composite feature assembly: [Sparse TF-IDF || Scaled Dense]
        X_train_fold = sp.hstack([X_tfidf[train_idx], sp.csr_matrix(tr_dense_scaled)]).tocsr()
        X_val_fold = sp.hstack([X_tfidf[val_idx], sp.csr_matrix(val_dense_scaled)]).tocsr()

        # Fit LibLinear classifier
        clf = LinearSVC(C=1.0, dual="auto", random_state=42)
        clf.fit(X_train_fold, labels[train_idx])
        preds = clf.predict(X_val_fold)

        acc = accuracy_score(labels[val_idx], preds) * 100.0
        f1 = f1_score(labels[val_idx], preds, average="macro") * 100.0
        fold_time = time.time() - t_fold

        fold_accuracies.append(acc)
        fold_f1_scores.append(f1)
        fold_durations.append(fold_time)

        print(
            f"Fold {fold_idx:^3} | "
            f"{len(train_idx):>6} / {len(val_idx):<6} | "
            f"{acc:>14.4f}% | "
            f"{f1:>14.4f}% | "
            f"{fold_time:>8.2f}s"
        )

    print("-" * 80)
    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    mean_f1 = float(np.mean(fold_f1_scores))
    total_time = sum(fold_durations)

    print(
        f"{'MEAN':^8} | {'5 Folds':^18} | "
        f"{mean_acc:>14.4f}% | "
        f"{mean_f1:>14.4f}% | "
        f"{total_time:>8.2f}s"
    )
    print(f"{'STD':^8}  | {'':^18} | {std_acc:>14.4f}% | {'':^15} |")
    print("-" * 80)

    # Comparison against historical stages
    stage4_acc = 91.3060
    delta = mean_acc - stage4_acc

    print("\n" + "=" * 80)
    print("REPLICATION BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"  • Stage 4 Baseline Accuracy:          {stage4_acc:.4f}%")
    print(f"  • Replicated Exp 4 Accuracy:          {mean_acc:.4f}% (+/- {std_acc:.4f}%)")
    print(f"  • Delta vs. Stage 4 Baseline:         {delta:+.4f}%")
    print(f"  • Target Exp 4 SOTA Accuracy:         91.7426%")
    print(f"  • Exact Numerical Match:              {abs(mean_acc - 91.7426) < 1e-4}")
    print("=" * 80)

    results = {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_f1": mean_f1,
        "fold_accuracies": fold_accuracies,
        "stage4_baseline": stage4_acc,
        "delta": delta,
    }
    return results


# ==============================================================================
# MAIN EXECUTION ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Canonical Usage & Exact Replication of Exp 4 SOTA (FeatureExtractor)"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "demo", "replicate"],
        default="all",
        help="Execution mode: 'demo' (API usage), 'replicate' (5-Fold SOTA CV), or 'all' (both).",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("FEATURE EXTRACTOR SOTA REPLICATION & CANONICAL USAGE SCRIPT")
    print("=" * 80)

    data_path, log_df_path = resolve_dataset_paths()
    token_sequences, labels = load_dataset(data_path)

    if args.mode in ["all", "demo"]:
        demonstrate_class_usage(token_sequences, labels)

    if args.mode in ["all", "replicate"]:
        replicate_exp4_sota(token_sequences, labels)


if __name__ == "__main__":
    main()
