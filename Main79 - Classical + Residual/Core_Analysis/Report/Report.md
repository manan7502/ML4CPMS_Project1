# CP219 Project 1 — Machine vs. Human Text Classification
> **IISc | August–December 2026** &nbsp;|&nbsp; **Team:** Project Gabriel &nbsp;|&nbsp; **Members:** Manan, Sujit Justine Barwa, Aniketh Waghmare

---

## 1. Exploratory Data Analysis (EDA)

### 1.1 Dataset Overview & Schema
The dataset consists of anonymized integer token sequences:
* **Corpus Volume:** **10,536** training documents (`train.json`) and **3,000** test documents (`test.json`) with zero missing values or duplicate records.
* **Schema:** `id` (int, arbitrary row identifier), `text` (list[int], token sequences from vocabulary $\mathcal{V} \in \{0, \dots, 18437\}$), and `label` (string, `"A"` = Human, `"B"` = Machine). Token `0` denotes rare / out-of-vocabulary (OOV) words.

### 1.2 Quantitative Comparison & Key Statistical Insights

| Dimension | Human (Class A) | Machine (Class B) | Statistical Evidence & Modeling Takeaway |
| :--- | :--- | :--- | :--- |
| **Class Distribution** | 3,699 samples (**35.11%**) | 6,837 samples (**64.89%**) | ~1.85:1 class imbalance (majority baseline = 64.89%). Mandates **Stratified 5-Fold Cross-Validation** and balanced class weights. |
| **Document Length** | Mean: **137.5** ($\sigma = 101.4$, median: 114) | Mean: **213.2** ($\sigma = 237.9$, median: 107) | Machine text displays high variance and a heavy tail ($p < 10^{-46}$, KS test). Human text is unimodal, whereas machine text is bimodal (short snippets alongside long passages). |
| **Rare Words (Token 0)** | **81.43%** docs contain `0` (mean: 2.65%) | **64.60%** docs contain `0` (mean: 1.87%) | Strongly discriminative ($p = 2.72 \times 10^{-87}$, Mann-Whitney U). Humans use rare words freely, while LLM decoding suppresses the probability tail. |
| **Consecutive Repetition** | Ratio: **0.0201** (mean repeats: 1.90) | Ratio: **0.0080** (mean repeats: 1.48) | Humans repeat identical consecutive tokens **>2.5×** more frequently (expressive punctuation such as `..`, `??`), filtered out by LLM top-$p$/beam search. |
| **Boundary Tokens** | First token `17175`: **10.5%** | First token `12787`: **11.4%** | Starting tokens exhibit strong class-conditional priors (Token `17175` is 3.6× more frequent in human). Terminal token `8294` ends 55.6% of all texts (period proxy). |
| **Lexical Richness** | Bigram uniqueness: **0.938** | Bigram uniqueness: **0.906** | Machine text exhibits more repetitive local n-gram structures. At the unigram vocabulary level, 459 individual token IDs appear exclusively in human texts, while 2,479 appear exclusively in machine texts. |


---

## 2. Feature Engineering & Multi-Stage Formulation

To capture the distinct stylistic, lexical, sequential, and syntactic fingerprints uncovered during EDA, we engineered a composite feature space combining a high-dimensional sparse n-gram substrate with **34 domain-specific dense statistical descriptors** developed across 5 incremental stages.

### 2.1 Sparse Substrate: Cumulative N-Gram TF-IDF
* **Representation:** Cumulative unigrams to 5-grams (1–5 grams) extracted across token ID sequences.
* **Configuration:** Sublinear term frequency scaling (`sublinear_tf=True`), document frequency clipping ($\text{min\_df}=2, \text{max\_df}=0.9$), and $L_2$ normalization, yielding **248,367 sparse dimensions**.
* **EDA Alignment:** Captures high-frequency transition markers, boundary tokens (e.g., start tokens `17175` / `12787`), and the class-exclusive vocabulary identified in EDA.

### 2.2 Dense Feature Taxonomy & EDA Problem Mapping (Stages 1 to 5)

| Stage & Feature Group | Extracted Features (Count) | Mathematical Definition | EDA Phenomenon / Problem Addressed |
| :--- | :--- | :--- | :--- |
| **Document Geometry** | `Document_Length` (1) | $L_d = \|d\|$ | **Length Variance & Skew:** Addresses the massive disparity in sequence length distributions ($\sigma=237.9$ Machine vs. $101.4$ Human, $p < 10^{-46}$) and provides length context. |
| **Stage 1: Lexical Diversity & Information Theory** | `Shannon_Entropy`<br>`Root_TTR_Guiraud`<br>`Hapax_Legomena_Ratio`<br>`Max_Repetition_Ratio` (4) | $H(d) = -\sum p(t)\log_2 p(t)$<br>$\text{Root\_TTR} = \frac{\|V_d\|}{\sqrt{L_d}}$<br>$\text{Hapax} = \frac{\|\{t:\text{cnt}(t)=1\}\|}{L_d}$<br>$\text{MaxRep} = \frac{\max_t \text{cnt}(t)}{L_d}$ | **Vocabulary Tail Truncation:** LLM nucleus sampling (top-$p$) suppresses low-probability vocabulary tails, lowering entropy and hapax legomena. Guiraud's Root TTR provides a length-invariant metric that isolates lexical richness from document length. |
| **Stage 2: Repetition & Frequency Profiling** | `Distinct_Bigram_Ratio`<br>`Distinct_Trigram_Ratio`<br>`Mean_Token_ID`<br>`Std_Token_ID`<br>`Q1_Functional_Ratio`<br>`Q4_Rare_Ratio` (6) | $\text{Distinct\_Bi} = \frac{\|\text{unique bigrams}\|}{L_d - 1}$<br>$\text{Distinct\_Tri} = \frac{\|\text{unique trigrams}\|}{L_d - 2}$<br>$\mu_{\text{token}} = \frac{1}{L_d} \sum t_i$<br>$\sigma_{\text{token}} = \text{std}(t_i)$<br>$\text{Q1} = \frac{\|\{t_i \le 3000\}\|}{L_d}$<br>$\text{Q4} = \frac{\|\{t_i \ge 10000\}\|}{L_d}$ | **Repetitive Phrasing & Token Hierarchy:** Machine text displays lower bigram uniqueness (0.906 vs. 0.938) due to autoregressive loops. Token ID moments and quartiles quantify depth of vocabulary deployment (Q1 functional syntax vs. Q4 rare tokens). |
| **Stage 3: Log-DF Machine Likelihood Pooling** | `Mean_Machine_Affinity`<br>`Std_Machine_Affinity`<br>`Max_Machine_Affinity`<br>`Min_Machine_Affinity`<br>`Positive_Affinity_Ratio` (5) | $r_t = \log \frac{DF_B(t) + 1}{DF_A(t) + 1}$<br>$\mu_{\text{aff}} = \frac{1}{L_d} \sum r_t$<br>$\sigma_{\text{aff}} = \text{std}(r_t)$<br>$\max r_t, \; \min r_t$<br>$\text{PosRatio} = \frac{\|\{t: r_t > 0\}\|}{L_d}$ | **Class-Exclusive & Asymmetric Vocabulary:** Directly leverages the 459 human-exclusive and 2,479 machine-exclusive vocabulary tokens. Pooling document-level moments summarizes whether an essay predominantly employs machine-favored or human-favored tokens. |
| **Stage 4: Sequential Dynamics & Burstiness** | `Half_Doc_Jaccard_Overlap`<br>`Token_Recurrence_Burstiness` (2) | $J_{\text{half}} = \frac{\|V_1 \cap V_2\|}{\|V_1 \cup V_2\|}$<br>$B = \frac{\sigma_\tau - \mu_\tau}{\sigma_\tau + \mu_\tau} \in [-1, 1]$ | **Non-Poisson Human Style vs. Stationary Generation:** Humans exhibit associative, bursty writing ($B > 0$, >2.5× consecutive repetition rate) and narrative thematic drift ($J_{\text{half}}$), whereas LLMs generate text via stationary, memoryless autoregressive sampling ($B \approx 0$). |
| **Stage 5: Spectral Syntactic Trajectory Dynamics** | `Syntactic_Centroid_0`..`11`<br>`Syntactic_Dispersion`<br>`Mean_Velocity_Step`<br>`Std_Velocity_Step`<br>`Syntactic_Tortuosity` (16) | $\boldsymbol{\mu}_k = \frac{1}{M}\sum_{i=1}^M \mathbf{e}(t_i)_k$<br>$R_g^2 = \frac{1}{M}\sum \|\mathbf{e}(t_i) - \boldsymbol{\mu}\|^2$<br>$\mu_v = \frac{1}{M-1}\sum \|\mathbf{e}(t_{i+1}) - \mathbf{e}(t_i)\| $<br>$\sigma_v = \text{std}(\Delta \mathbf{e}_i)$<br>$\tau = \frac{\|\mathbf{e}(t_M) - \mathbf{e}(t_1)\|}{\sum \|\Delta \mathbf{e}_i\|}$ | **Grammatical Flow in Anonymized Text:** Overcomes the failure of discrete HMMs (which suffered posterior collapse). Projects functional transition matrices into a continuous 12-D spectral manifold via SVD to capture grammatical curvature, structural tortuosity, and transition pacing jitter. |

### 2.3 Numerical Scaling Design: MaxAbsScaler Protocol
* **Problem:** Concatenating unbounded dense features (`StandardScaler`, ranges $[-5, +12]$) with $L_2$-normalized sparse TF-IDF ($[0, 1]$) causes LibLinear coordinate descent to oscillate wildly and stall, while composite $L_2$ normalization squashes dense feature variance below margin boundaries (accuracy collapses to $86.82\%$).
* **Solution:** Applying **In-Fold `MaxAbsScaler`** bounds all 34 dense features strictly within $[-1, 1]$, harmonizing coordinate magnitudes with sparse TF-IDF. This enables rapid, stable convergence in $<4.5$ seconds with peak accuracy.

### 2.4 Empirical Validation & Incremental Progression Across Stages
All stages were evaluated using a strict **5-Fold Stratified Cross-Validation** protocol on a `LinearSVC` classifier with zero fold leakage:

| Stage | Cumulative Feature Set | Total Features | 5-Fold Mean Accuracy | Std Dev | $\Delta$ vs. Prev Stage |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Base** | Cumulative TF-IDF (1–5) + `Document_Length` | 248,368 | **89.23%** | $\pm 0.75\%$ | — |
| **Stage 1** | + Lexical Diversity & Shannon Entropy | 248,372 | **89.69%** | $\pm 0.46\%$ | $+0.46\%$ |
| **Stage 2** | + N-Gram Repetition & Token ID Profiling | 248,378 | **90.31%** | $\pm 0.44\%$ | $+0.62\%$ |
| **Stage 3** | + Document Log-DF Machine Likelihood Pooling | 248,383 | **91.26%** | $\pm 0.55\%$ | $+0.95\%$ |
| **Stage 4** | + Sequential Dynamics & Token Burstiness | 248,385 | **91.31%** | $\pm 0.40\%$ | $+0.05\%$ |
| **Stage 5** | + Spectral Syntactic Trajectory Dynamics | 248,401 | **91.74%** | $\pm 0.35\%$ | $+0.43\%$ |
| **Stage 5+** | Stage 5 with Balanced Class Weights | 248,401 | **91.94%** | $\pm 0.35\%$ | $+0.20\%$ |


---

## 3. Misclassification Analysis & Systematic Error Elimination

```
                     ┌──────────────────────────────────────────┐
                     │          STAGE 5 DUAL PIPELINE           │
                     │  248,367 Sparse TF-IDF ⊕ 34 Dense Invars │
                     └────────────────────┬─────────────────────┘
                                          │
                                          ▼
                     ┌──────────────────────────────────────────┐
                     │       STAGE 5 LINEAR BASELINE (91.89%)   │
                     │  LinearSVC(class_weight='balanced')      │
                     └────────────────────┬─────────────────────┘
                                          │
                                          ▼ (Forensic Audit: 854 Errors)
                ┌─────────────────────────┴─────────────────────────┐
                ▼                                                   ▼
┌───────────────────────────────┐                   ┌───────────────────────────────┐
│     FEATURE & PROB BRANCH     │                   │     LEVEL-2 STACKER BRANCH    │
│  • Fix 1: Platt Prob Calib    │                   │  • Fix 3: LightGBM Stacker    │
│  • Fix 2: tanh(L/50) Gating   │                   │  • Fix 4: PyTorch ResNet Head │
│  Accuracy: 91.94% - 92.06%    │                   │  Accuracy: 93.65% - 94.06%    │
└───────────────────────────────┘                   └───────────────────────────────┘
```

A rigorous forensic audit of all **10,536 out-of-fold (OOF) predictions** from the Stage 5 baseline revealed three structural failure modes, which were systematically resolved across four progressive fixes:

### 3.1 Failure Mode Diagnosis
1. **Class-Imbalance Penalty Distortion (521 vs. 333):**
   * *Finding:* Over **61.0% of all mistakes** (521 out of 854) were Machine texts misclassified as Human (False Negatives).
   * *Cause:* `class_weight='balanced'` assigns an artificial $1.85\times$ penalty ($1.424$ vs. $0.770$) to human mistakes, shifting the hyperplane into the Machine distribution and forcing borderline Machine texts across the zero margin into Human predictions.
2. **Sequence Length Epistemic Fragility:**
   * *Finding:* For long documents ($L \in [244, 994]$), accuracy is **97.80%** (only 46 errors). In contrast, ultra-short documents ($L \le 61$) suffer a **12.70% error rate** (273 errors).
   * *Cause:* Ultra-short texts suffer extreme n-gram sparsity and high estimation variance in statistical metrics (entropy, burstiness, tortuosity).
3. **Pathological Short-Text Outliers:**
   * *Finding:* Linear SVM overreacted to isolated cue tokens in short documents (e.g. `Doc 8821`, Length 33, Margin $+1.930$; `Doc 207`, Length 35, Margin $-1.487$).

### 3.2 The Four Progressive Fixes
* **Fix 1 (Threshold & Platt Calibration):** Fits in-fold Platt scaling ($P(y=1\|\mathbf{x}) = \sigma(A \cdot f(\mathbf{x}) + B)$) and tunes decision threshold $\tau^*$ on validation splits, realigning with the true ~65/35 test prior (**91.94% accuracy**).
* **Fix 2 (Length-Conditioned Gating):** Soft-dampens high-variance statistical features via $\mathbf{f}_{\text{gated}} = \mathbf{f}_{\text{dense}} \cdot \tanh(L_d / 50)$, dropping the short-text error rate from **12.70% to 12.10%** and elevating accuracy to **92.06%**.
* **Fix 3 (Non-Linear LightGBM Stacker):** Combines Stage 5 linear margins + 34 dense physical descriptors + 30 SVD components into a LightGBM decision tree ensemble that learns conditional length routing. Accuracy leaps to **93.65%**, eliminating **185 errors** (False Negatives drop from 521 to 328).
* **Fix 4 (Deep Neural Residual Stacker):** Implements a PyTorch Deep Residual Stacker with 2 Residual Blocks with skip connections, LayerNorm, and GELU on GPU. Reaches **94.06% CV Accuracy** (Macro F1: **93.54%**), eliminating **228 errors** and balancing Human Recall (**93.76%**) and Machine Recall (**94.22%**).

### 3.3 Master Benchmark Comparison Table

| Model / Fix Configuration | 5-Fold Accuracy | Macro F1 | Human (A) Recall | Machine (B) Recall | False Pos (A $\rightarrow$ B) | False Neg (B $\rightarrow$ A) | Total Errors | Net Errors Eliminated |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 5 Baseline (`LinearSVC`)** | 91.89% | 91.21% | 91.00% | 92.38% | 333 | 521 | 854 | *Baseline* |
| **Fix 1: Threshold & Platt Calibration** | 91.94% | 91.25% | 91.78% | 92.03% | 304 | 545 | 849 | -5 errors |
| **Fix 2: Length-Conditioned Gating** | 92.06% | 91.39% | 91.43% | 92.39% | 317 | 520 | 837 | -17 errors |
| **Fix 3: Tuned LightGBM Stacker** | **93.65%** | **93.18%** | 90.78% | **95.20%** | 341 | **328** | **669** | **-185 errors** |
| **Fix 4: Neural Residual Stacker (PyTorch)** | **94.06%** | **93.54%** | **93.76%** | 94.22% | **231** | 395 | **626** | **-228 errors** |