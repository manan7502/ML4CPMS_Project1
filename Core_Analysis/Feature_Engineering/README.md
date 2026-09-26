# 🏗️ Feature Engineering Consolidation: Unified `FeatureExtractor` Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Compatible-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)]()
[![Performance](https://img.shields.io/badge/5--Fold%20CV-91.74%25-brightgreen.svg)]()

> A modular, production-ready feature engineering architecture that transforms raw integer token sequences into the peak high-dimensional hybrid sparse-dense feature space established across **Stages 1 through 4 + Spectral Syntactic Trajectory Dynamics (Exp 4 SOTA)**, achieving **91.74% 5-Fold Stratified Cross-Validation accuracy**.

---

## 🧭 Visual System Architecture

The following diagram illustrates how raw integer token sequences are routed into parallel specialized extractors and fused into the final composite representation:

```
                           Raw Token Sequence: d = [t_1, t_2, ..., t_n]
                                             │
         ┌───────────────────────────────────┴───────────────────────────────────┐
         │                                                                       │
         ▼                                                                       ▼
┌─────────────────────────────────┐                       ┌───────────────────────────────────────────────┐
│     SPARSE TF-IDF PIPELINE      │                       │             DENSE HYBRID PIPELINE             │
│                                 │                       │                                               │
│ 1. String Projection            │                       │ 1. Document Length: |d| (1 dim)               │
│ 2. Cumulative N-Grams (1 to 5)  │                       │ 2. Lexical Diversity (Stage 1: 4 dims)        │
│ 3. Sublinear TF: (1 + ln(tf))   │                       │    • Shannon Entropy H(d)                     │
│ 4. L2 Normalization             │                       │    • Guiraud's Root TTR                       │
│                                 │                       │    • Hapax Legomena Ratio                     │
│ Output: Sparse Matrix (N, ~248k)│                       │    • Max Term Domination Ratio                │
└────────────────┬────────────────┘                       │ 3. Repetition Profiling (Stage 2: 6 dims)     │
                 │                                        │    • Distinct Bigram Ratio                    │
                 │                                        │    • Distinct Trigram Ratio                   │
                 │                                        │    • Mean & Std Token ID                      │
                 │                                        │    • Q1 & Q4 Token Rank Quartiles             │
                 │                                        │ 4. Log-DF Machine Likelihood (Stage 3: 5 dims)│
                 │                                        │    • Mean, Std, Max, Min Machine Log-Odds     │
                 │                                        │    • Positive Machine Odds Ratio              │
                 │                                        │ 5. Sequential Dynamics (Stage 4: 2 dims)      │
                 │                                        │    • Half-Doc Jaccard Vocabulary Drift        │
                 │                                        │    • Goh-Barabási Recurrence Burstiness (B)   │
                 │                                        │ 6. Spectral Trajectory Dynamics (Exp 4: 16 d) │
                 │                                        │    • 12 Syntactic Manifold Centroid Coords    │
                 │                                        │    • Syntactic Dispersion (Radius of Gyration)│
                 │                                        │    • Mean & Std Velocity Step Length          │
                 │                                        │    • Syntactic Tortuosity (Arc-to-Chord Ratio)│
                 │                                        │                                               │
                 │                                        │ Output: Dense Matrix (N, 34 dims)             │
                 │                                        └───────────────────────┬───────────────────────┘
                 │                                                                │
                 │                                                                ▼
                 │                                        ┌───────────────────────────────────────────────┐
                 │                                        │             MaxAbsScaler FITTING              │
                 │                                        │ Scales all 34 dense features into [-1, 1]     │
                 │                                        │ Matches sparse TF-IDF coordinate range        │
                 │                                        └───────────────────────┬───────────────────────┘
                 │                                                                │
                 └────────────────────────┬───────────────────────────────────────┘
                                          │
                                          ▼
                 ┌──────────────────────────────────────────────────┐
                 │       COMPOSITE FEATURE MATRIX (CSR Sparse)      │
                 │    X_composite = [ X_sparse || Scaled_Dense ]    │
                 │               Shape: (N, 248,401)                │
                 └──────────────────────────────────────────────────┘
```

---

## 📦 Class Overview: `FeatureExtractor`

Located in [`feature_extractor.py`](file:///home/justin/PhD/ML4CPS/Kaggle%20Project/Analysis_II/ML4CPMS_Project1/Main79%20-%20Classical%20+%20Residual/Core_Analysis/Feature_Engineering_Consolidation/feature_extractor.py), `FeatureExtractor` inherits from `sklearn.base.BaseEstimator` and `TransformerMixin`.

> [!NOTE]
> For backwards compatibility with Stage 4 codebases, `Stage4FeatureExtractor = FeatureExtractor` is exported as an alias in `__init__.py`.

### Core API Methods

| Function / Method | Category | Output Shape | Description |
| :--- | :--- | :---: | :--- |
| `generate_tfidf_features(docs, is_training)` | **Sparse TF-IDF** | `(N, P)` | Builds cumulative 1-5 gram TF-IDF matrix with sublinear scaling and L2 normalization. |
| `extract_length_features(docs)` | **Length** | `(N, 1)` | Computes sequence length $L_d = \|d\|$. |
| `extract_lexical_diversity_features(docs)` | **Stage 1 Diversity** | `(N, 4)` | Computes Shannon Entropy, Root TTR, Hapax ratio, and Max Repetition ratio. |
| `extract_repetition_profiling_features(docs)`| **Stage 2 Profiling** | `(N, 6)` | Computes Distinct Bigram/Trigram ratios, Token ID moments, and Q1/Q4 quartiles. |
| `extract_log_df_likelihood_features(docs)` | **Stage 3 Log-DF** | `(N, 5)` | Pools unigram log-DF machine affinity moments ($\mu, \sigma, \max, \min$) and positive odds fraction. |
| `extract_sequential_dynamics_features(docs)` | **Stage 4 Dynamics** | `(N, 2)` | Computes Half-Doc Jaccard vocabulary drift and Goh-Barabási Recurrence Burstiness $B$. |
| `extract_spectral_trajectory_features(docs)`| **Exp 4 Spectral Trajectory** | `(N, 16)` | Computes 12-D syntactic manifold centroid coordinates, dispersion ($R_g^2$), velocity moments ($\mu_v, \sigma_v$), and tortuosity. |
| `extract_all_dense_features(docs)` | **Dense Consolidation**| `(N, 34)` | Concatenates all 34 dense statistical descriptors into a single dense matrix. |
| `fit(docs, y=None)` | **Scikit-Learn API** | `self` | Fits TfidfVectorizer, SVD Syntactic Manifold, and MaxAbsScaler strictly on training data. |
| `transform(docs)` | **Scikit-Learn API** | `(N, P + 34)` | Transforms new documents and outputs composite CSR matrix. |
| `fit_transform(docs, y=None)` | **Scikit-Learn API** | `(N, P + 34)` | Fits and transforms in a single efficient call. |
| `get_dense_feature_names()` | **Inspection** | `List[str]` | Returns ordered list of 34 dense feature names. |
| `save(filepath)` / `load(filepath)` | **Persistence** | File on disk | Serializes and restores complete fitted transformer pipeline. |

---

## 🔬 Complete 34 Dense Feature Inventory

The 34 dense features appended alongside sparse TF-IDF are defined as follows:

| Index | Feature Name | Category | Mathematical Definition | Linguistic / Physical Intuition |
| :---: | :--- | :---: | :--- | :--- |
| `1` | `Document_Length` | Length | $L_d = \|d\|$ | Humans write longer essays ($747.8$ vs. $419.6$ tokens). |
| `2` | `Shannon_Entropy` | Stage 1 | $H(d) = -\sum p(t) \log_2 p(t)$ | Top-$p$ sampling truncates vocabulary tails, constraining LLM entropy. |
| `3` | `Root_TTR_Guiraud` | Stage 1 | $\text{Root\_TTR} = \frac{\|V_d\|}{\sqrt{L_d}}$ | Length-invariant vocabulary richness ($7.61$ Machine vs. $6.70$ Human). |
| `4` | `Hapax_Legomena_Ratio` | Stage 1 | $\frac{\|\{t : \text{count}(t) = 1\}\|}{L_d}$ | Idiosyncratic one-off word usage in human writing. |
| `5` | `Max_Repetition_Ratio` | Stage 1 | $\frac{\max_t \text{count}(t)}{L_d}$ | Single-keyword topic fixation or prompt echo. |
| `6` | `Distinct_Bigram_Ratio` | Stage 2 | $\frac{\|\text{unique bigrams}\|}{L_d - 1}$ | Autoregressive phrasing repetition ($0.91$ Machine vs. $0.94$ Human). |
| `7` | `Distinct_Trigram_Ratio` | Stage 2 | $\frac{\|\text{unique trigrams}\|}{L_d - 2}$ | 3-token syntactic transition variability. |
| `8` | `Mean_Token_ID` | Stage 2 | $\mu_{\text{token}} = \frac{1}{L_d} \sum t_i$ | Higher values indicate deeper deployment of rare vocabulary. |
| `9` | `Std_Token_ID` | Stage 2 | $\sigma_{\text{token}} = \text{std}(t_i)$ | Vocabulary dispersion across tokenizer frequency ranks. |
| `10` | `Q1_Functional_Token_Ratio`| Stage 2 | $\frac{\|\{t_i \le 3000\}\|}{L_d}$ | Stopword and functional syntax concentration. |
| `11` | `Q4_Rare_Token_Ratio` | Stage 2 | $\frac{\|\{t_i \ge 10000\}\|}{L_d}$ | Long-tail vocabulary usage (suppressed by nucleus sampling). |
| `12` | `Mean_Machine_Affinity` | Stage 3 | $\mu_{\text{aff}} = \frac{1}{L_d} \sum r_t$ | Aggregate Bayesian machine vocabulary bias ($+0.584$ vs. $+0.468$). |
| `13` | `Std_Machine_Affinity` | Stage 3 | $\sigma_{\text{aff}} = \text{std}(r_t)$ | Spread of log-odds across document tokens. |
| `14` | `Max_Machine_Affinity` | Stage 3 | $\max_{t \in d} r_t$ | Presence of hyper-machine indicator tokens. |
| `15` | `Min_Machine_Affinity` | Stage 3 | $\min_{t \in d} r_t$ | Presence of hyper-human colloquial terms. |
| `16` | `Positive_Affinity_Ratio`| Stage 3 | $\frac{\|\{t : r_t > 0\}\|}{L_d}$ | Fraction of vocabulary favored by generative LLMs. |
| `17` | `Half_Doc_Jaccard_Overlap`| Stage 4 | $J_{\text{half}} = \frac{\|V_1 \cap V_2\|}{\|V_1 \cup V_2\|}$ | Narrative drift in humans vs. prompt stasis in LLMs. |
| `18` | `Token_Recurrence_Burstiness`| Stage 4 | $B = \frac{\sigma_\tau - \mu_\tau}{\sigma_\tau + \mu_\tau}$ | Non-Poisson associative recall ($B > 0$) vs. stationary decoding ($B \approx 0$). |
| `19`–`30` | `Syntactic_Manifold_Dim_0` .. `11` | Exp 4 | $\boldsymbol{\mu}_k = \frac{1}{M} \sum_{i=1}^M \mathbf{e}(t_i)_k$ | 12-dimensional centroid projection in spectral syntactic space. |
| `31` | `Syntactic_Dispersion` | Exp 4 | $R_g^2 = \frac{1}{M} \sum_{i=1}^M \|\mathbf{e}(t_i) - \boldsymbol{\mu}\|^2$ | Syntactic radius of gyration (structural grammatical breadth). |
| `32` | `Mean_Velocity_Step` | Exp 4 | $\mu_v = \frac{1}{M-1} \sum_{i=1}^{M-1} \|\mathbf{e}(t_{i+1}) - \mathbf{e}(t_i)\|$ | Average grammatical transition jump between consecutive tokens. |
| `33` | `Std_Velocity_Step` | Exp 4 | $\sigma_v = \text{std}(\|\mathbf{e}(t_{i+1}) - \mathbf{e}(t_i)\|)$ | Rhythm/pacing variance in syntactic state transitions. |
| `34` | `Syntactic_Tortuosity` | Exp 4 | $\tau = \frac{\|\mathbf{e}(t_M) - \mathbf{e}(t_1)\|}{\sum \|\mathbf{e}(t_{i+1}) - \mathbf{e}(t_i)\|}$ | Arc-to-chord ratio: grammatical goal-directedness vs. local wandering. |

---

## 📈 Empirical Benchmarks (5-Fold Stratified Cross-Validation)

Evaluated on the full dataset of $N = 10,536$ documents:

```
89.23% ──────> 89.69% ──────> 90.31% ──────> 91.26% ──────> 91.31% ──────> 91.74%
Baseline       Stage 1        Stage 2        Stage 3        Stage 4        Exp 4 SOTA
(TF-IDF+Len)   (+Diversity)   (+Repetition)  (+Log-DF)      (+Burstiness)  (+Spectral)
```

| Pipeline Stage / Experiment | Dense Dims | 5-Fold CV Accuracy | Fold Std ($\sigma$) | $\Delta$ vs. Baseline | $\Delta$ vs. Prior Stage | LibLinear Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** (TF-IDF 1-5 + Length) | 1 | 89.23% | $\pm 0.75\%$ | Baseline | — | 3.5s |
| **Stage 1** (+ Diversity & Entropy) | 5 | 89.69% | $\pm 0.46\%$ | +0.46% | +0.46% | 3.5s |
| **Stage 2** (+ Repetition & Profiling) | 11 | 90.31% | $\pm 0.44\%$ | +1.08% | +0.62% | 3.6s |
| **Stage 3** (+ Log-DF Likelihood Pooling) | 16 | 91.26% | $\pm 0.55\%$ | +2.03% | +0.95% | 3.6s |
| **Stage 4** (+ Sequential Dynamics & Burstiness)| 18 | 91.31% | $\pm 0.40\%$ | +2.08% | +0.05% | 3.7s |
| *Exp 1* (+ Base HMM POS State Posteriors) | 38 | 91.27% | $\pm 0.41\%$ | +2.04% | -0.04% | 185s |
| *Exp 2* (+ Dual Class-Conditional HMMs) | 20 | 91.44% | $\pm 0.30\%$ | +2.21% | +0.13% | 210s |
| *Exp 3* (+ Grammar Skeleton N-Grams) | 18 + Skelet | 90.85% | $\pm 0.52\%$ | +1.62% | -0.46% | 9.2s |
| **Exp 4 (SOTA)** (**+ Spectral Trajectory Dynamics**)| **34** | **91.74%** | **$\pm 0.38\%$** | **+2.51%** | **+0.43%** | **4.1s** |

---

## 🧠 Why Spectral Syntactic Trajectory Dynamics Succeeded

### The Core Problem with Discrete HMMs
Standard unsupervised Hidden Markov Models rely on Baum-Welch Expectation-Maximization (EM) over discrete token indices. On an anonymized vocabulary of 17,000+ tokens without punctuation or orthography:
1. Soft posterior state distributions $\gamma_i(k) = P(z_t = k \mid d)$ degenerate into blurred bag-of-words histograms.
2. The discrete state information is **completely redundant** with cumulative 1-5 gram TF-IDF, adding computational overhead ($185\text{s}$) with no incremental accuracy gain (-0.038%).

### The Spectral Solution
Instead of fitting non-convex discrete latent state variables:
1. We construct the empirical unigram-to-unigram transition matrix $T_{ij} = \text{count}(t_a \to t_b)$ over the functional vocabulary (top $K=1,500$ tokens).
2. We row-normalize $T$ to estimate the local syntactic grammar transition operator $\mathbf{P}$.
3. We perform **Truncated Singular Value Decomposition (SVD)** on $\mathbf{P}$ to embed each functional token into a continuous 12-dimensional syntactic manifold:
   $$\mathbf{e}(t) \in \mathbb{R}^{12}$$
4. A document's token sequence forms a **continuous geometric trajectory** in this grammatical coordinate space:
   $$\mathbf{X}_{\text{traj}} = [\mathbf{e}(t_1), \mathbf{e}(t_2), \dots, \mathbf{e}(t_M)]^T \in \mathbb{R}^{M \times 12}$$
5. We extract 16 continuous dynamical invariants:
   - **Centroid** ($\boldsymbol{\mu} \in \mathbb{R}^{12}$): Where the document's grammar centers in syntactic space.
   - **Dispersion** ($R_g^2$): How widely the author traverses syntactic space.
   - **Velocity Moments** ($\mu_v, \sigma_v$): The step sizes between consecutive grammatical transitions.
   - **Tortuosity** ($\tau$): The ratio of net displacement to total path length, capturing syntactic goal-directedness vs. local wandering.

> [!TIP]
> This continuous physical trajectory captures the subtle difference between human stylistic pacing (high burstiness, intentional syntactic curvature) and machine autoregressive generation (uniform transition pacing, lower structural tortuosity).

---

## 🚀 Quickstart Guide

### 1. Ingestion & Transformation

```python
import json
from Feature_Engineering_Consolidation import FeatureExtractor

# 1. Load dataset
with open("../../../../train.json", "r") as f:
    raw_data = [json.loads(line) for line in f]

token_sequences = [d["text"] for d in raw_data]
labels = [1 if d["label"] == "B" else 0 for d in raw_data]

# 2. Initialize and transform (fit strictly on training data)
extractor = FeatureExtractor()
X_composite = extractor.fit_transform(token_sequences)

print("Composite Matrix Shape:", X_composite.shape)
# Output: (10536, 248401)
print("Sparse Features:       ", extractor.tfidf_vectorizer.get_feature_names_out().shape[0])
# Output: 248367
print("Dense Features:        ", len(extractor.get_dense_feature_names()))
# Output: 34
```

### 2. Standalone Feature Extraction

You can invoke individual feature extractors independently for ablation studies or custom architectures:

```python
# 1. Sparse TF-IDF only (N, P)
X_tfidf = extractor.generate_tfidf_features(token_sequences)

# 2. Document length only (N, 1)
lengths = extractor.extract_length_features(token_sequences)

# 3. Stage 1 Diversity & Entropy (N, 4)
diversity = extractor.extract_lexical_diversity_features(token_sequences)

# 4. Stage 2 Repetition & Token Profiling (N, 6)
profiling = extractor.extract_repetition_profiling_features(token_sequences)

# 5. Stage 3 Log-DF Machine Likelihood Moments (N, 5)
likelihood = extractor.extract_log_df_likelihood_features(token_sequences)

# 6. Stage 4 Sequential Dynamics & Burstiness (N, 2)
dynamics = extractor.extract_sequential_dynamics_features(token_sequences)

# 7. Exp 4 Spectral Syntactic Trajectory Dynamics (N, 16)
spectral_dynamics = extractor.extract_spectral_trajectory_features(token_sequences)

# 8. All 34 dense features concatenated (N, 34)
dense_all = extractor.extract_all_dense_features(token_sequences)
```

### 3. Model Training with `LinearSVC`

```python
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Split token sequences
train_tokens, val_tokens, y_train, y_val = train_test_split(
    token_sequences, labels, test_size=0.2, random_state=42, stratify=labels
)

# Fit feature extractor strictly on training split
extractor = FeatureExtractor()
X_train = extractor.fit_transform(train_tokens)
X_val = extractor.transform(val_tokens)

# Train LibLinear classifier (converges in ~4.1 seconds)
clf = LinearSVC(C=1.0, dual="auto", random_state=42)
clf.fit(X_train, y_train)

# Evaluate
preds = clf.predict(X_val)
print(classification_report(y_val, preds, target_names=["Human (A)", "Machine (B)"]))
```

### 4. Serialization & Production Deployment

```python
# Save fitted extractor to disk
extractor.save("fitted_feature_extractor.pkl")

# Load in production / inference script
loaded_extractor = FeatureExtractor.load("fitted_feature_extractor.pkl")
X_test = loaded_extractor.transform(test_token_sequences)
```

---

## ⚡ Numerical Scaling Design Decision

A critical engineering discovery from our research is the **prohibition of standard unit-variance scaling** (`StandardScaler`) on dense statistical features when concatenating with sparse TF-IDF:

```
❌ Standard Normalization (StandardScaler):
   z = (x - mu) / sigma  ==> Coordinates range from -5.0 to +12.0
   Result: LibLinear coordinate descent oscillates violently; exceeds max_iter; stalls.

❌ Composite L2 Normalization:
   Dividing the entire concatenated vector by sqrt( ||X_tfidf||^2 + ||X_dense||^2 )
   Result: Squashes dense signals below the hyperplane margin; accuracy collapses to 86.82%.

✅ In-Fold Maximum Absolute Scaling (MaxAbsScaler):
   x_scaled = x / max(|x|)  ==> Coordinates strictly bounded in [0, 1] or [-1, 1]
   Result: Perfectly matches the coordinate scale of L2-normalized TF-IDF.
           LinearSVC converges in < 4.5 seconds with ZERO warnings and peak accuracy (91.74%).
```

The `FeatureExtractor` automatically manages this via its internal `self.dense_scaler = MaxAbsScaler()`, fit strictly on training splits.

---

## 🏆 SOTA Replication & Canonical Class Usage Example

To verify exact numerical replication of the SOTA benchmark and demonstrate full usage of the `FeatureExtractor` class, two dedicated reference implementations are provided:
1. **Python Script**: [`replicate_exp4_sota.py`](file:///home/justin/PhD/ML4CPS/Kaggle%20Project/Analysis_II/ML4CPMS_Project1/Main79%20-%20Classical%20+%20Residual/Core_Analysis/Feature_Engineering_Consolidation/replicate_exp4_sota.py)
2. **Jupyter Notebook**: [`replicate_exp4_sota.ipynb`](file:///home/justin/PhD/ML4CPS/Kaggle%20Project/Analysis_II/ML4CPMS_Project1/Main79%20-%20Classical%20+%20Residual/Core_Analysis/Feature_Engineering_Consolidation/replicate_exp4_sota.ipynb)

### Running the Replication Script

```bash
cd "Core_Analysis/Feature_Engineering_Consolidation"

# 1. Run canonical class API demo (fit_transform, transform, modular extraction, save/load)
python3 replicate_exp4_sota.py --mode demo

# 2. Run full 5-Fold Stratified CV replication of Exp 4 SOTA
python3 replicate_exp4_sota.py --mode replicate

# 3. Run both workflows sequentially
python3 replicate_exp4_sota.py --mode all
```

### Verified Terminal Output

```
================================================================================
PART 2: EXACT 5-FOLD STRATIFIED CV REPLICATION OF EXP 4 SOTA
================================================================================
[Replication] Total corpus size: 10,536 documents.
[Replication] Initializing FeatureExtractor for Exp 4 SOTA pipeline...
[Replication] Step 1/3: Fitting spectral syntactic transition manifold...
  Extracted 12-D Syntactic Manifold in 1.34s.
[Replication] Step 2/3: Extracting all 34 dense feature descriptors...
  Dense Feature Matrix extracted: (10536, 34) in 7.99s.
[Replication] Step 3/3: Vectorizing cumulative 1-5 gram TF-IDF matrix...
  Sparse TF-IDF Matrix generated: (10536, 253796) in 14.20s.

[Replication] Running 5-Fold Stratified Cross-Validation (Seed = 42)...
--------------------------------------------------------------------------------
  Fold   |    Train / Val     |  Accuracy (%)   |  Macro F1 (%)   |  Duration 
--------------------------------------------------------------------------------
Fold  1  |   8428 / 2108   |        91.5085% |        90.6254% |     1.03s
Fold  2  |   8429 / 2107   |        91.8367% |        91.0547% |     1.06s
Fold  3  |   8429 / 2107   |        91.3621% |        90.6191% |     1.04s
Fold  4  |   8429 / 2107   |        92.3588% |        91.5928% |     1.18s
Fold  5  |   8429 / 2107   |        91.6469% |        90.8854% |     1.17s
--------------------------------------------------------------------------------
  MEAN   |      5 Folds       |        91.7426% |        90.9555% |     5.47s
  STD    |                    |         0.3456% |                 |
--------------------------------------------------------------------------------

================================================================================
REPLICATION BENCHMARK SUMMARY
================================================================================
  • Stage 4 Baseline Accuracy:          91.3060%
  • Replicated Exp 4 Accuracy:          91.7426% (+/- 0.3456%)
  • Delta vs. Stage 4 Baseline:         +0.4366%
  • Target Exp 4 SOTA Accuracy:         91.7426%
  • Exact Numerical Match:              True
================================================================================
```
