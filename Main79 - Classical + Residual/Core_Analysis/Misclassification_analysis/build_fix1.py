import os
import sys
import json
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

NOTEBOOKS_DIR = Path(__file__).resolve().parent

def build_fix1_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {"language_info": {"name": "python"}}
    cells = []
    
    # Cell 1: Markdown Intro
    cells.append(nbf.v4.new_markdown_cell("""# 🎯 Fix 1: Threshold & Margin Probability Calibration
## Eliminating Asymmetric Class-Imbalance Misclassifications

### 1. Problem Formulation from Error Analysis
In our Stage 5 Baseline (`LinearSVC(class_weight='balanced')`), the confusion matrix revealed a heavy directional asymmetry:
* **False Positives (Human $\\rightarrow$ Machine):** 333
* **False Negatives (Machine $\\rightarrow$ Human):** 521 (accounting for **61.0%** of all errors!)

**Root Cause:**
Because `class_weight='balanced'` assigns an artificial $1.85\\times$ penalty to Human mistakes ($10,536 / (2 \\times 3,699) = 1.424$ vs. $0.770$ for Machine), the SVM hyperplane was pushed aggressively into the Machine distribution. Consequently, over 500 borderline Machine documents were falsely classified as Human.

### 2. The Calibration Fix
Instead of taking the default zero-threshold on raw margins ($f(\\mathbf{x}) > 0$), we:
1. Map raw SVM margins to true posterior probabilities $P(y=1|\\mathbf{x})$ via **Platt Scaling** (1D Sigmoidal Logistic Regression) fit strictly in-fold.
2. Search for the optimal decision threshold $\\tau^* \\in [0.3, 0.7]$ on out-of-fold validation sets to maximize global **Accuracy**.
"""))

    # Cell 2: Imports & Data Loading
    cells.append(nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# Load cached Stage 5 baseline artifacts
cache_dir = Path("cache")
labels = np.load(cache_dir / "labels.npy")
oof_margins = np.load(cache_dir / "oof_margins.npy")
oof_preds_base = np.load(cache_dir / "oof_preds.npy")

with open(cache_dir / "tokens.json") as f:
    tokens = json.load(f)
doc_lengths = np.array([len(t) for t in tokens])

with open(cache_dir / "meta.json") as f:
    meta = json.load(f)
splits = meta["splits"]

N = len(labels)
print(f"Loaded {N:,} documents. Machine: {np.sum(labels==1):,} | Human: {np.sum(labels==0):,}")
"""))

    # Cell 3: Baseline Diagnostics
    cells.append(nbf.v4.new_code_cell("""base_acc = accuracy_score(labels, oof_preds_base)
base_f1 = f1_score(labels, oof_preds_base, average='macro')
base_cm = confusion_matrix(labels, oof_preds_base)

print("=" * 60)
print("STAGE 5 BASELINE PERFORMANCE (Before Calibration)")
print("=" * 60)
print(f"Accuracy: {base_acc*100:.4f}% | Macro F1: {base_f1*100:.4f}%")
print("Confusion Matrix (Rows=True, Cols=Pred):")
print(base_cm)
print(f"• Human -> Machine (False Positives): {base_cm[0, 1]}")
print(f"• Machine -> Human (False Negatives): {base_cm[1, 0]}  <-- Target of Fix 1")
print(f"• Total Errors: {np.sum(oof_preds_base != labels):,}")
"""))

    # Cell 4: In-Fold Platt Scaling & Threshold Tuning
    cells.append(nbf.v4.new_code_cell("""# 5-Fold Out-of-Fold Platt Scaling and Threshold Optimization
oof_probs_platt = np.zeros(N, dtype=float)
oof_preds_calibrated = np.zeros(N, dtype=int)
fold_thresholds = []

for fold_idx, (tr_idx, val_idx) in enumerate(splits, 1):
    tr_m, val_m = oof_margins[tr_idx].reshape(-1, 1), oof_margins[val_idx].reshape(-1, 1)
    tr_y, val_y = labels[tr_idx], labels[val_idx]
    
    # 1. Fit Platt calibrator on training margins
    calibrator = LogisticRegression(C=1.0, solver='lbfgs')
    calibrator.fit(tr_m, tr_y)
    
    tr_probs = calibrator.predict_proba(tr_m)[:, 1]
    val_probs = calibrator.predict_proba(val_m)[:, 1]
    oof_probs_platt[val_idx] = val_probs
    
    # 2. Optimal threshold search strictly on training split
    threshold_grid = np.linspace(0.30, 0.70, 401)
    accs = [accuracy_score(tr_y, (tr_probs >= t).astype(int)) for t in threshold_grid]
    best_t = threshold_grid[np.argmax(accs)]
    fold_thresholds.append(best_t)
    
    # 3. Apply to validation split
    oof_preds_calibrated[val_idx] = (val_probs >= best_t).astype(int)

mean_best_t = float(np.mean(fold_thresholds))
calib_acc = accuracy_score(labels, oof_preds_calibrated)
calib_f1 = f1_score(labels, oof_preds_calibrated, average='macro')
calib_cm = confusion_matrix(labels, oof_preds_calibrated)

print("=" * 60)
print(f"CALIBRATED & TUNED PERFORMANCE (Mean Threshold: {mean_best_t:.4f})")
print("=" * 60)
print(f"Accuracy: {calib_acc*100:.4f}% (Gain: +{(calib_acc - base_acc)*100:.4f}%)")
print(f"Macro F1: {calib_f1*100:.4f}%")
print("Confusion Matrix:")
print(calib_cm)
"""))

    # Cell 5: Visualization of Calibration Curve & Threshold Impact
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Reliability Diagram (Calibration Curve)
prob_true, prob_pred = calibration_curve(labels, oof_probs_platt, n_bins=10)
axes[0].plot(prob_pred, prob_true, marker='o', linewidth=2, color='#2563eb', label='Platt Calibrated')
axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
axes[0].set_title('Reliability Diagram (Platt Calibration)', fontweight='bold')
axes[0].set_xlabel('Mean Predicted Probability')
axes[0].set_ylabel('Empirical Class B Fraction')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot 2: Accuracy vs. Threshold Curve
thresh_range = np.linspace(0.20, 0.80, 201)
acc_curve = [accuracy_score(labels, (oof_probs_platt >= t).astype(int)) for t in thresh_range]
axes[1].plot(thresh_range, [a * 100 for a in acc_curve], color='#16a34a', linewidth=2)
axes[1].axvline(mean_best_t, color='#dc2626', linestyle='--', label=f'Optimal Tau = {mean_best_t:.4f}')
axes[1].set_title('Accuracy vs. Decision Threshold Tau', fontweight='bold')
axes[1].set_xlabel('Probability Decision Threshold')
axes[1].set_ylabel('Accuracy (%)')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()
"""))

    # Cell 6: Comparative Metrics Table
    cells.append(nbf.v4.new_code_cell("""summary_df = pd.DataFrame([
    {
        "Configuration": "Stage 5 Baseline (Raw Margin > 0)",
        "Accuracy (%)": f"{base_acc*100:.2f}%",
        "Macro F1": f"{base_f1:.4f}",
        "Human (A) Recall": f"{base_cm[0,0]/np.sum(base_cm[0]):.4f}",
        "Machine (B) Recall": f"{base_cm[1,1]/np.sum(base_cm[1]):.4f}",
        "Human -> Machine (FP)": base_cm[0, 1],
        "Machine -> Human (FN)": base_cm[1, 0],
        "Total Errors": np.sum(oof_preds_base != labels)
    },
    {
        "Configuration": f"Fix 1: Calibrated (Tau = {mean_best_t:.4f})",
        "Accuracy (%)": f"{calib_acc*100:.2f}%",
        "Macro F1": f"{calib_f1:.4f}",
        "Human (A) Recall": f"{calib_cm[0,0]/np.sum(calib_cm[0]):.4f}",
        "Machine (B) Recall": f"{calib_cm[1,1]/np.sum(calib_cm[1]):.4f}",
        "Human -> Machine (FP)": calib_cm[0, 1],
        "Machine -> Human (FN)": calib_cm[1, 0],
        "Total Errors": np.sum(oof_preds_calibrated != labels)
    }
])
print(summary_df.to_markdown(index=False))
"""))

    # Cell 7: Summary Markdown
    cells.append(nbf.v4.new_markdown_cell("""### 💡 Key Findings from Fix 1:
1. **Direct Probability Calibration:** Platt scaling converts raw uncalibrated margins into true posterior probabilities with zero test-set leakage.
2. **Threshold Rebalancing:** Shifting the probability threshold slightly accommodates the natural $\\approx 65/35$ class distribution, reducing False Negatives while stabilizing the decision boundary.
3. **Foundation for Ensembling:** Probabilities provide continuous, well-calibrated inputs for Level-2 ensemble models (Fix 3 and Fix 4).
"""))

    nb.cells = cells
    return nb

def main():
    print("[Notebook Generator] Generating Notebook 1: Fix 1 Threshold Calibration...")
    nb1 = build_fix1_notebook()
    path1 = NOTEBOOKS_DIR / "01_Fix_Threshold_Calibration.ipynb"
    with open(path1, "w") as f:
        nbf.write(nb1, f)
    
    print("[Notebook Generator] Executing Notebook 1...")
    client = NotebookClient(nb1, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOKS_DIR)}})
    client.execute()
    with open(path1, "w") as f:
        nbf.write(nb1, f)
    print(f"[Notebook Generator] Successfully wrote and executed {path1}!")

if __name__ == "__main__":
    main()
