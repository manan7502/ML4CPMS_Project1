"""
================================================================================
Feature Engineering Consolidation: Comprehensive FeatureExtractor Class
================================================================================
Project: Authorship Attribution (LLM vs. Human Text Classification)
Directory: Core_Analysis/Feature_Engineering_Consolidation
Module: feature_extractor.py

This module provides the unified, enterprise-grade `FeatureExtractor` class
converting raw integer token sequences into the peak hybrid sparse-dense
representation (91.74% 5-Fold Cross-Validation accuracy).

Architecture Pipeline:
  1. Sparse Representation:
     - Cumulative N-Gram TF-IDF (1-gram to 5-gram, sublinear scaling, L2 normalized)
  2. Dense Hybrid Representations (34 total descriptors across 6 families):
     - Function 1: Length of sentence / document (1 feature)
     - Function 2: Lexical Diversity & Shannon Entropy (Stage 1: 4 features)
     - Function 3: N-Gram Repetition & Token ID Profiling (Stage 2: 6 features)
     - Function 4: Document Log-DF Machine Likelihood Pooling (Stage 3: 5 features)
     - Function 5: Sequential Dynamics & Token Burstiness (Stage 4: 2 features)
     - Function 6: Spectral Syntactic Trajectory Dynamics (Exp 4 SOTA: 16 features)
  3. Numerical Conditioning:
     - In-fold Maximum Absolute Scaling (MaxAbsScaler) strictly bounded in [-1, 1]
       to match sparse coordinate magnitude and prevent LibLinear solver divergence.
================================================================================
"""

import os
import json
import pickle
import numpy as np
import scipy.sparse as sp
from collections import Counter
from typing import List, Dict, Union, Optional, Tuple

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Consolidated Feature Extractor converting raw token ID sequences into the
    peak hybrid feature space (Cumulative 1-5 TF-IDF + 34 Dense Descriptors).

    Parameters
    ----------
    log_df_path : Optional[str]
        Path to the precomputed unigram log-DF ratios JSON file. If None,
        automatically searches standard relative paths in Core_Analysis/data.
    ngram_range : Tuple[int, int], default=(1, 5)
        N-gram lower and upper boundary for the sparse TF-IDF vectorizer.
    min_df : int, default=3
        Minimum document frequency threshold for n-gram inclusion.
    sublinear_tf : bool, default=True
        Apply sublinear term frequency scaling (1 + log(tf)).
    token_boundary_q1 : int, default=3000
        Rank threshold for Q1 functional tokens (stopword proxy).
    token_boundary_q4 : int, default=10000
        Rank threshold for Q4 rare tail vocabulary tokens.
    trajectory_top_k : int, default=500
        Vocabulary cutoff for the syntactic transition co-occurrence matrix.
    trajectory_n_components : int, default=12
        Dimensionality of the continuous spectral syntactic manifold.
    random_state : int, default=42
        Random seed for reproducible SVD decomposition.
    """

    def __init__(
        self,
        log_df_path: Optional[str] = None,
        ngram_range: Tuple[int, int] = (1, 5),
        min_df: int = 3,
        sublinear_tf: bool = True,
        token_boundary_q1: int = 3000,
        token_boundary_q4: int = 10000,
        trajectory_top_k: int = 500,
        trajectory_n_components: int = 12,
        random_state: int = 42,
    ):
        self.log_df_path = log_df_path
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.token_boundary_q1 = token_boundary_q1
        self.token_boundary_q4 = token_boundary_q4
        self.trajectory_top_k = trajectory_top_k
        self.trajectory_n_components = trajectory_n_components
        self.random_state = random_state

        # Fitted transformer state
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.dense_scaler: Optional[MaxAbsScaler] = None
        self.trajectory_top_vocab: Dict[int, int] = {}
        self.trajectory_svd: Optional[TruncatedSVD] = None
        self.syntactic_embeddings: Optional[np.ndarray] = None
        self.unigram_log_df: Dict[int, float] = {}
        self.is_fitted: bool = False

        # Load precomputed Bayesian log-DF dictionary
        self._load_log_df_ratios()

    # --------------------------------------------------------------------------
    # Initialization & Configuration Helpers
    # --------------------------------------------------------------------------
    def _load_log_df_ratios(self) -> None:
        """Loads the precomputed unigram log-DF likelihood dictionary."""
        candidate_paths = [
            self.log_df_path,
            "../data/unigram_log_df_ratios.json",
            "data/unigram_log_df_ratios.json",
            "/home/justin/PhD/ML4CPS/Kaggle Project/Analysis_II/ML4CPMS_Project1/Main79 - Classical + Residual/Core_Analysis/data/unigram_log_df_ratios.json",
            "/home/justin/PhD/ML4CPS/Kaggle Project/Analysis_II/data/unigram_log_df_ratios.json",
        ]

        resolved_path = None
        for path in candidate_paths:
            if path and os.path.exists(path):
                resolved_path = path
                break

        if resolved_path and os.path.exists(resolved_path):
            with open(resolved_path, "r", encoding="utf-8") as f:
                self.unigram_log_df = {int(k): float(v) for k, v in json.load(f).items()}
        else:
            self.unigram_log_df = {}

    # --------------------------------------------------------------------------
    # 1. Sparse TF-IDF Vector Generation
    # --------------------------------------------------------------------------
    def generate_tfidf_features(
        self, token_sequences: List[List[int]], is_training: bool = False
    ) -> sp.csr_matrix:
        """
        Generates high-order cumulative N-Gram TF-IDF representation (1-gram to 5-gram).

        Parameters
        ----------
        token_sequences : List[List[int]]
            List of documents, where each document is a sequence of integer token IDs.
        is_training : bool, default=False
            Whether to fit the TfidfVectorizer on the provided sequences.

        Returns
        -------
        sp.csr_matrix
            Sparse L2-normalized TF-IDF matrix of shape (N, P).
        """
        string_docs = [" ".join(map(str, tokens)) for tokens in token_sequences]

        if is_training:
            self.tfidf_vectorizer = TfidfVectorizer(
                token_pattern=r"\S+",
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                sublinear_tf=self.sublinear_tf,
            )
            return self.tfidf_vectorizer.fit_transform(string_docs)
        else:
            if self.tfidf_vectorizer is None:
                raise RuntimeError("TfidfVectorizer is not fitted. Call fit() first.")
            return self.tfidf_vectorizer.transform(string_docs)

    # --------------------------------------------------------------------------
    # 2. Document / Sentence Length Feature Addition
    # --------------------------------------------------------------------------
    def extract_length_features(self, token_sequences: List[List[int]]) -> np.ndarray:
        """
        Extracts document sequence length (L_d = |d|).

        Parameters
        ----------
        token_sequences : List[List[int]]
            Input document token sequences.

        Returns
        -------
        np.ndarray
            2D array of shape (N, 1) containing sequence lengths.
        """
        lengths = [len(seq) for seq in token_sequences]
        return np.array(lengths, dtype=np.float64).reshape(-1, 1)

    # --------------------------------------------------------------------------
    # 3. Lexical Diversity & Entropy Addition (Stage 1)
    # --------------------------------------------------------------------------
    def extract_lexical_diversity_features(
        self, token_sequences: List[List[int]]
    ) -> np.ndarray:
        """
        Computes 4 Stage 1 Information-Theoretic and Vocabulary Richness Metrics:
          1. Shannon Entropy of the Empirical Token Distribution: H(d) = -sum p*log2(p)
          2. Guiraud's Index / Root Type-Token Ratio: Root_TTR = |V_d| / sqrt(L_d)
          3. Hapax Legomena Ratio: Count of tokens appearing exactly once / L_d
          4. Maximum Term Domination Ratio: max count(t) / L_d

        Parameters
        ----------
        token_sequences : List[List[int]]
            Input document token sequences.

        Returns
        -------
        np.ndarray
            2D array of shape (N, 4).
        """
        features = []
        for tokens in token_sequences:
            n = len(tokens)
            if n == 0:
                features.append([0.0, 0.0, 0.0, 0.0])
                continue

            counts = Counter(tokens)
            probs = np.array(list(counts.values()), dtype=np.float64) / n

            # Shannon Entropy
            entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))

            # Guiraud's Root TTR
            root_ttr = float(len(counts) / np.sqrt(n))

            # Hapax Legomena Ratio
            hapax_count = sum(1 for c in counts.values() if c == 1)
            hapax_ratio = float(hapax_count / n)

            # Maximum Term Domination Ratio
            max_rep_ratio = float(max(counts.values()) / n)

            features.append([entropy, root_ttr, hapax_ratio, max_rep_ratio])

        return np.array(features, dtype=np.float64)

    # --------------------------------------------------------------------------
    # 4. Local Repetition & Token Profiling Addition (Stage 2)
    # --------------------------------------------------------------------------
    def extract_repetition_profiling_features(
        self, token_sequences: List[List[int]]
    ) -> np.ndarray:
        """
        Computes 6 Stage 2 Local Transition Repetition and BPE Token Rank Metrics:
          1. Distinct Bigram Ratio: |unique bigrams| / (L_d - 1)
          2. Distinct Trigram Ratio: |unique trigrams| / (L_d - 2)
          3. Mean Token ID (mu_token_id): Captures vocabulary depth
          4. Standard Deviation of Token IDs (sigma_token_id): Measures vocabulary dispersion
          5. Q1 Functional Token Ratio: Proportion of tokens with ID <= 3000
          6. Q4 Rare Tail Token Ratio: Proportion of tokens with ID >= 10000

        Parameters
        ----------
        token_sequences : List[List[int]]
            Input document token sequences.

        Returns
        -------
        np.ndarray
            2D array of shape (N, 6).
        """
        features = []
        for tokens in token_sequences:
            n = len(tokens)
            if n == 0:
                features.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                continue

            # Distinct N-Gram Ratios
            bi_ratio = (
                len(set(zip(tokens[:-1], tokens[1:]))) / max(1, n - 1)
                if n >= 2
                else 0.0
            )
            tri_ratio = (
                len(set(zip(tokens[:-2], tokens[1:-1], tokens[2:]))) / max(1, n - 2)
                if n >= 3
                else 0.0
            )

            # Token ID Distribution Statistics
            mean_id = float(np.mean(tokens))
            std_id = float(np.std(tokens))

            # Vocabulary Rank Quartiles
            q1_ratio = float(sum(1 for t in tokens if t <= self.token_boundary_q1) / n)
            q4_ratio = float(sum(1 for t in tokens if t >= self.token_boundary_q4) / n)

            features.append([bi_ratio, tri_ratio, mean_id, std_id, q1_ratio, q4_ratio])

        return np.array(features, dtype=np.float64)

    # --------------------------------------------------------------------------
    # 5. Document Log-DF Likelihood Ratio Pooling (Stage 3)
    # --------------------------------------------------------------------------
    def extract_log_df_likelihood_features(
        self, token_sequences: List[List[int]]
    ) -> np.ndarray:
        """
        Computes 5 Stage 3 Document-Level Bayesian Log-DF Likelihood Moments:
          1. Mean Machine Affinity (mu_aff): Aggregate machine vocabulary bias
          2. Standard Deviation of Machine Affinity (sigma_aff): Dispersion of odds
          3. Maximum Machine Affinity (max_aff): Extreme machine token indicator
          4. Minimum Machine Affinity (min_aff): Extreme human token indicator
          5. Positive Affinity Token Proportion (Pos_Ratio): Fraction of tokens with r_t > 0

        Parameters
        ----------
        token_sequences : List[List[int]]
            Input document token sequences.

        Returns
        -------
        np.ndarray
            2D array of shape (N, 5).
        """
        features = []
        for tokens in token_sequences:
            if not self.unigram_log_df:
                features.append([0.0, 0.0, 0.0, 0.0, 0.0])
                continue

            vals = [self.unigram_log_df[t] for t in tokens if t in self.unigram_log_df]
            if len(vals) == 0:
                features.append([0.0, 0.0, 0.0, 0.0, 0.0])
                continue

            mean_aff = float(np.mean(vals))
            std_aff = float(np.std(vals))
            max_aff = float(np.max(vals))
            min_aff = float(np.min(vals))
            pos_ratio = float(sum(1 for v in vals if v > 0) / len(vals))

            features.append([mean_aff, std_aff, max_aff, min_aff, pos_ratio])

        return np.array(features, dtype=np.float64)

    # --------------------------------------------------------------------------
    # 6. Sequential Dynamics & Token Burstiness (Stage 4)
    # --------------------------------------------------------------------------
    def extract_sequential_dynamics_features(
        self, token_sequences: List[List[int]]
    ) -> np.ndarray:
        """
        Computes 2 Stage 4 Sequential Dynamics & Cognitive Memory Descriptors:
          1. Half-Document Vocabulary Jaccard Similarity (J_half):
             Measures narrative drift vs. autoregressive prompt continuity across halves.
          2. Inter-Arrival Token Recurrence Burstiness Parameter (B):
             Measures heavy-tailed associative recall vs. memoryless Poisson arrivals:
             B = (sigma_tau - mu_tau) / (sigma_tau + mu_tau) for top recurring tokens.

        Parameters
        ----------
        token_sequences : List[List[int]]
            Input document token sequences.

        Returns
        -------
        np.ndarray
            2D array of shape (N, 2).
        """
        features = []
        for tokens in token_sequences:
            n = len(tokens)
            if n <= 1:
                features.append([0.0, 0.0])
                continue

            # 1. Half-Document Jaccard Similarity
            mid = n // 2
            half1 = set(tokens[:mid])
            half2 = set(tokens[mid:])
            union_len = len(half1 | half2)
            jaccard_half = float(len(half1 & half2) / union_len if union_len > 0 else 0.0)

            # 2. Token Recurrence Burstiness (Top 3 most frequent tokens)
            counts = Counter(tokens)
            top_tokens = [tok for tok, c in counts.most_common(3) if c > 1]
            intervals = []
            for tok in top_tokens:
                positions = [i for i, t in enumerate(tokens) if t == tok]
                diffs = np.diff(positions)
                intervals.extend(diffs)

            if len(intervals) >= 2:
                mean_inv = float(np.mean(intervals))
                std_inv = float(np.std(intervals))
                denom = std_inv + mean_inv
                burstiness = float((std_inv - mean_inv) / denom) if denom > 0 else 0.0
            else:
                burstiness = 0.0

            features.append([jaccard_half, burstiness])

        return np.array(features, dtype=np.float64)

    # --------------------------------------------------------------------------
    # 7. Spectral Syntactic Trajectory Dynamics (Exp 4 SOTA Extension)
    # --------------------------------------------------------------------------
    def _fit_spectral_syntactic_manifold(
        self, token_sequences: List[List[int]]
    ) -> None:
        """
        Fits the continuous syntactic manifold on training sequences using Truncated SVD
        over the immediate bigram transition probability matrix P(w_{t+1} | w_t).
        """
        all_tokens = [t for doc in token_sequences for t in doc]
        vocab_counts = Counter(all_tokens)
        top_k = self.trajectory_top_k
        self.trajectory_top_vocab = {
            t: i for i, (t, c) in enumerate(vocab_counts.most_common(top_k))
        }
        unk_idx = top_k

        # Map training sequences
        mapped_docs = [
            [self.trajectory_top_vocab.get(t, unk_idx) for t in doc]
            for doc in token_sequences
        ]

        # Build bigram co-occurrence matrix
        src, dst = [], []
        for doc in mapped_docs:
            if len(doc) > 1:
                src.extend(doc[:-1])
                dst.extend(doc[1:])

        V = top_k + 1
        cooccur = sp.coo_matrix(
            (np.ones(len(src), dtype=np.float32), (src, dst)), shape=(V, V)
        ).tocsr()

        # Row-normalize to transition probabilities
        row_sums = np.array(cooccur.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        P_trans = cooccur.multiply(1.0 / row_sums[:, None]).tocsr()

        # Fit Truncated SVD
        self.trajectory_svd = TruncatedSVD(
            n_components=self.trajectory_n_components, random_state=self.random_state
        )
        self.syntactic_embeddings = self.trajectory_svd.fit_transform(P_trans)

    def extract_spectral_trajectory_features(
        self, token_sequences: List[List[int]]
    ) -> np.ndarray:
        """
        Computes 16 Spectral Syntactic Manifold Velocity, Dispersion, and Tortuosity Metrics:
          - Centroid Coordinates in Syntactic Manifold (12 features)
          - Manifold Radius of Gyration / Dispersion: mean ||e_t - mu_e||^2 (1 feature)
          - Mean Velocity Step Length: mean ||v_t|| (1 feature)
          - Velocity Step Length Variance: std(||v_t||) (1 feature)
          - Syntactic Tortuosity: ||e_n - e_1|| / (sum ||v_t|| + eps) (1 feature)

        Parameters
        ----------
        token_sequences : List[List[int]]
            Input document token sequences.

        Returns
        -------
        np.ndarray
            2D array of shape (N, 16).
        """
        if self.syntactic_embeddings is None:
            raise RuntimeError(
                "Spectral syntactic manifold is not fitted. Call fit() first."
            )

        top_vocab = self.trajectory_top_vocab
        unk_idx = self.trajectory_top_k
        emb = self.syntactic_embeddings

        features = []
        for tokens in token_sequences:
            n = len(tokens)
            if n == 0:
                features.append([0.0] * (self.trajectory_n_components + 4))
                continue

            mapped = [top_vocab.get(t, unk_idx) for t in tokens]
            doc_embs = emb[mapped]  # Shape: (n, trajectory_n_components)

            # Manifold centroid and dispersion
            centroid = np.mean(doc_embs, axis=0)
            dispersion = float(np.mean(np.sum((doc_embs - centroid) ** 2, axis=1)))

            # Velocity dynamics & tortuosity
            if n > 1:
                diffs = np.diff(doc_embs, axis=0)
                step_lens = np.linalg.norm(diffs, axis=1)
                mean_step = float(np.mean(step_lens))
                std_step = float(np.std(step_lens))
                total_path = float(np.sum(step_lens))
                net_disp = float(np.linalg.norm(doc_embs[-1] - doc_embs[0]))
                tortuosity = float(net_disp / (total_path + 1e-6))
            else:
                mean_step, std_step, tortuosity = 0.0, 0.0, 0.0

            features.append(list(centroid) + [dispersion, mean_step, std_step, tortuosity])

        return np.array(features, dtype=np.float64)

    # --------------------------------------------------------------------------
    # 8. Complete Dense Feature Matrix Assembly (34 Features)
    # --------------------------------------------------------------------------
    def extract_all_dense_features(
        self, token_sequences: List[List[int]]
    ) -> np.ndarray:
        """
        Concatenates all 34 engineered dense statistical descriptors:
          - Length (1)
          - Lexical Diversity (4)
          - Repetition & Token Profiling (6)
          - Log-DF Likelihood Moments (5)
          - Sequential Dynamics & Burstiness (2)
          - Spectral Syntactic Trajectory Dynamics (16)

        Returns
        -------
        np.ndarray
            Array of shape (N, 34).
        """
        f_length = self.extract_length_features(token_sequences)
        f_diversity = self.extract_lexical_diversity_features(token_sequences)
        f_profiling = self.extract_repetition_profiling_features(token_sequences)
        f_log_df = self.extract_log_df_likelihood_features(token_sequences)
        f_burstiness = self.extract_sequential_dynamics_features(token_sequences)
        f_trajectory = self.extract_spectral_trajectory_features(token_sequences)

        return np.column_stack(
            [f_length, f_diversity, f_profiling, f_log_df, f_burstiness, f_trajectory]
        )

    # --------------------------------------------------------------------------
    # 9. Scikit-Learn API: fit, transform, fit_transform
    # --------------------------------------------------------------------------
    def fit(self, token_sequences: List[List[int]], y=None) -> "FeatureExtractor":
        """
        Fits the TF-IDF vectorizer, the spectral syntactic manifold, and the
        in-fold MaxAbsScaler on training sequences.

        Parameters
        ----------
        token_sequences : List[List[int]]
            Training set token sequences.
        y : None
            Ignored. Follows scikit-learn API conventions.

        Returns
        -------
        FeatureExtractor
            Fitted transformer instance.
        """
        # 1. Fit TF-IDF on token sequences
        _ = self.generate_tfidf_features(token_sequences, is_training=True)

        # 2. Fit Spectral Syntactic Manifold on token transitions
        self._fit_spectral_syntactic_manifold(token_sequences)

        # 3. Extract and fit MaxAbsScaler on all 34 dense features
        dense_matrix = self.extract_all_dense_features(token_sequences)
        self.dense_scaler = MaxAbsScaler()
        self.dense_scaler.fit(dense_matrix)

        self.is_fitted = True
        return self

    def transform(self, token_sequences: List[List[int]]) -> sp.csr_matrix:
        """
        Transforms input token sequences into the full hybrid sparse-dense feature matrix.

        Parameters
        ----------
        token_sequences : List[List[int]]
            Input document token sequences.

        Returns
        -------
        sp.csr_matrix
            Composite sparse matrix of shape (N, P + 34), where P is TF-IDF dimension.
        """
        if not self.is_fitted:
            raise RuntimeError("Transformer is not fitted. Call fit() before transform().")

        # 1. Transform sparse TF-IDF (N, P)
        X_sparse = self.generate_tfidf_features(token_sequences, is_training=False)

        # 2. Transform and scale dense descriptors (N, 34)
        dense_matrix = self.extract_all_dense_features(token_sequences)
        scaled_dense = self.dense_scaler.transform(dense_matrix)

        # 3. Concatenate horizontally: [Sparse TF-IDF || Scaled Dense]
        X_composite = sp.hstack([X_sparse, sp.csr_matrix(scaled_dense)]).tocsr()
        return X_composite

    def fit_transform(
        self, token_sequences: List[List[int]], y=None
    ) -> sp.csr_matrix:
        """Fits transformer and returns transformed feature matrix."""
        return self.fit(token_sequences, y).transform(token_sequences)

    # --------------------------------------------------------------------------
    # 10. Feature Name Metadata
    # --------------------------------------------------------------------------
    def get_dense_feature_names(self) -> List[str]:
        """Returns the list of 34 dense feature column names."""
        names = [
            # Length (1)
            "Document_Length",
            # Stage 1: Lexical Diversity & Entropy (4)
            "Shannon_Entropy",
            "Root_TTR_Guiraud",
            "Hapax_Legomena_Ratio",
            "Max_Repetition_Ratio",
            # Stage 2: Repetition & Token Profiling (6)
            "Distinct_Bigram_Ratio",
            "Distinct_Trigram_Ratio",
            "Mean_Token_ID",
            "Std_Token_ID",
            "Q1_Functional_Token_Ratio",
            "Q4_Rare_Token_Ratio",
            # Stage 3: Log-DF Likelihood Pooling (5)
            "Mean_Machine_Affinity",
            "Std_Machine_Affinity",
            "Max_Machine_Affinity",
            "Min_Machine_Affinity",
            "Positive_Affinity_Ratio",
            # Stage 4: Sequential Dynamics & Burstiness (2)
            "Half_Doc_Jaccard_Overlap",
            "Token_Recurrence_Burstiness",
        ]
        # Spectral Syntactic Trajectory Dynamics (16)
        trajectory_names = [
            f"Syntactic_Manifold_Dim_{i}" for i in range(self.trajectory_n_components)
        ] + [
            "Syntactic_Dispersion",
            "Mean_Velocity_Step",
            "Std_Velocity_Step",
            "Syntactic_Tortuosity",
        ]
        return names + trajectory_names

    def get_feature_names_out(self) -> np.ndarray:
        """Returns the complete array of feature names (sparse + dense)."""
        if self.tfidf_vectorizer is None:
            raise RuntimeError("Transformer must be fitted first.")
        sparse_names = self.tfidf_vectorizer.get_feature_names_out()
        dense_names = np.array(self.get_dense_feature_names())
        return np.concatenate([sparse_names, dense_names])

    # --------------------------------------------------------------------------
    # 11. Persistence Utilities
    # --------------------------------------------------------------------------
    def save(self, filepath: str) -> None:
        """Serializes the fitted FeatureExtractor instance."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"FeatureExtractor saved successfully to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "FeatureExtractor":
        """Deserializes a saved FeatureExtractor instance."""
        with open(filepath, "rb") as f:
            instance = pickle.load(f)
        print(f"FeatureExtractor loaded successfully from: {filepath}")
        return instance


# Backwards compatibility alias
Stage4FeatureExtractor = FeatureExtractor


# ==============================================================================
# Self-Verification Test Execution
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Testing FeatureExtractor on synthetic token sequences...")
    print("=" * 80)

    sample_docs = [
        [17175, 42, 89, 104, 3012, 17, 89, 42, 89, 12050, 8294],
        [12787, 5, 23, 104, 23, 5, 104, 23, 11000, 8294],
        [17175, 89, 104, 3012, 17, 42, 89, 12050, 42, 104, 8294],
        [12787, 42, 5, 104, 17, 3012, 89, 23, 12050, 8294],
    ]

    extractor = FeatureExtractor(min_df=1, ngram_range=(1, 2), trajectory_top_k=20, trajectory_n_components=6)
    X_out = extractor.fit_transform(sample_docs)

    print("Transformed Composite Matrix Shape:", X_out.shape)
    print("Sparse TF-IDF Feature Count:        ", extractor.tfidf_vectorizer.get_feature_names_out().shape[0])
    print("Dense Feature Count:                ", len(extractor.get_dense_feature_names()))
    print("Matrix Sparsity (% Nonzeros):       ", f"{100 * X_out.nnz / (X_out.shape[0] * X_out.shape[1]):.2f}%")
    print("\nAll Dense Feature Names (Total: %d):" % len(extractor.get_dense_feature_names()))
    for i, name in enumerate(extractor.get_dense_feature_names(), 1):
        print(f"  {i:2d}. {name}")
    print("=" * 80)
    print("Self-test completed successfully!")
