#!/usr/bin/env python3
"""
MAIN115 — KAGGLE SUBMISSION GENERATOR

This is a submission-only companion to main115.py.

It expects:
  1. corrected_main114_results.npz
       feature_oof, feature_val, y_train, y_val, teacher_oof, teacher_val
       and a 62-D test feature matrix under one of:
       feature_test / X_test / test_features / feature_te
  2. Main64 test predictions as a .npy file, discovered automatically from:
       main64_test_predictions.npy
       main64_test_pred.npy
       main64_test_labels.npy
  3. test.json

It does NOT retrain Main64. It retrains the five Main115 specialists on
the complete Main64 OOF feature set, applies the already-selected Main115
consensus rule, and writes main115_submission.csv.

The Main115 rule is selected only from the OOF rows, exactly as in the
original experiment.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "corrected_main114_results.npz"
TEST_JSON = ROOT / "test.json"
OUT_SUBMISSION = ROOT / "main115_submission.csv"

N_SPLITS = 5
SEED = 115

print("=" * 78)
print("MAIN115 — KAGGLE SUBMISSION GENERATOR")
print("=" * 78)

if not RESULTS.exists():
    raise FileNotFoundError(f"Missing {RESULTS}")

z = np.load(RESULTS, allow_pickle=True)

required = ["feature_oof", "feature_val", "y_train", "y_val",
            "teacher_oof", "teacher_val"]
missing = [k for k in required if k not in z.files]
if missing:
    raise RuntimeError(f"Missing keys in {RESULTS.name}: {missing}")

X_oof = z["feature_oof"].astype(np.float32)
X_val = z["feature_val"].astype(np.float32)
y_oof = z["y_train"].astype(int)
y_val = z["y_val"].astype(int)
teacher_oof = z["teacher_oof"].astype(int)
teacher_val = z["teacher_val"].astype(int)

# -------------------------------------------------------------------------
# Locate 62-D Main114 test features.
# -------------------------------------------------------------------------
X_test = None
test_feature_source = None

for key in ["feature_test", "X_test", "test_features", "feature_te"]:
    if key in z.files:
        candidate = z[key]
        if candidate.ndim == 2 and candidate.shape[1] == X_oof.shape[1]:
            X_test = candidate.astype(np.float32)
            test_feature_source = f"{RESULTS.name}:{key}"
            break

# Also check common separate feature files.
if X_test is None:
    for fname in [
        "main114_test_features.npz",
        "corrected_main114_test_features.npz",
        "main115_test_features.npz",
    ]:
        f = ROOT / fname
        if not f.exists():
            continue
        zz = np.load(f, allow_pickle=True)
        for key in ["feature_test", "X_test", "test_features", "feature_te"]:
            if key in zz.files:
                candidate = zz[key]
                if candidate.ndim == 2 and candidate.shape[1] == X_oof.shape[1]:
                    X_test = candidate.astype(np.float32)
                    test_feature_source = f"{fname}:{key}"
                    break
        if X_test is not None:
            break

if X_test is None:
    raise RuntimeError(
        "Could not find the 62-D Main114 test feature matrix. "
        "Your Main114 feature-generation code must save the test matrix "
        "as 'feature_test' (same 62 columns/order as feature_oof) in "
        "corrected_main114_results.npz or main114_test_features.npz."
    )

print(f"OOF features : {X_oof.shape}")
print(f"TEST features: {X_test.shape} <- {test_feature_source}")

# -------------------------------------------------------------------------
# Exact Main115 specialist definitions.
# -------------------------------------------------------------------------
models = {
    "LR_C1": LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=5000, solver="lbfgs"
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

names = list(models)
M = len(names)

# -------------------------------------------------------------------------
# Recover the frozen Main115 rule from OOF, exactly as in main115.py.
# -------------------------------------------------------------------------
skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
p_oof = np.zeros((len(y_oof), M), dtype=np.float64)

print("\n--- Recomputing cross-fitted specialist probabilities ---")
for fold, (tr, te) in enumerate(skf.split(X_oof, y_oof), 1):
    print(f"Fold {fold}/{N_SPLITS}")
    for j, name in enumerate(names):
        mdl = clone(models[name])
        mdl.fit(X_oof[tr], y_oof[tr])
        p_oof[te, j] = mdl.predict_proba(X_oof[te])[:, 1]

hard_o = (p_oof >= 0.5).astype(int)
disagree_o = hard_o != teacher_oof[:, None]
conf_o = np.maximum(p_oof, 1.0 - p_oof)

vote_o = disagree_o.sum(axis=1)
sum_dis_o = np.sum(np.where(disagree_o, conf_o - 0.5, 0.0), axis=1)
mean_dis_o = np.divide(
    sum_dis_o, vote_o, out=np.zeros_like(sum_dis_o), where=vote_o > 0
)
weak_dis_o = np.where(disagree_o, conf_o, 1.0).min(axis=1)

rules = []
base_o = accuracy_score(y_oof, teacher_oof)

for k in [2, 3, 4, 5]:
    for margin in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        for conf in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
            change = (
                (vote_o >= k)
                & (mean_dis_o >= margin)
                & (weak_dis_o >= conf)
            )
            n = int(change.sum())
            if n < 8:
                continue

            after = teacher_oof.copy()
            after[change] = 1 - after[change]

            acc = accuracy_score(y_oof, after)
            correct = int(np.sum(after[change] == y_oof[change]))
            wrong = n - correct
            precision = correct / n

            rules.append({
                "k": k,
                "margin": margin,
                "confidence": conf,
                "oof_changes": n,
                "oof_correct": correct,
                "oof_wrong": wrong,
                "oof_change_precision": precision,
                "oof_accuracy_after": float(acc),
                "oof_delta": float(acc - base_o),
            })

safe = [
    r for r in rules
    if r["oof_change_precision"] >= 0.65 and r["oof_delta"] > 0
]

if safe:
    selected = max(
        safe,
        key=lambda r: (
            r["oof_delta"] - 0.00002 * r["oof_changes"],
            r["oof_change_precision"],
            -r["oof_changes"],
        ),
    )
elif any(r["oof_delta"] > 0 for r in rules):
    positive = [r for r in rules if r["oof_delta"] > 0]
    selected = max(
        positive,
        key=lambda r: (r["oof_change_precision"], r["oof_delta"], -r["oof_changes"])
    )
else:
    selected = {
        "k": 5,
        "margin": 0.30,
        "confidence": 0.90,
    }

k = selected["k"]
margin = selected["margin"]
conf = selected["confidence"]

print(
    f"Frozen rule: >= {k}/{M} specialists disagree, "
    f"mean margin >= {margin:.2f}, weakest confidence >= {conf:.2f}"
)
print(f"OOF teacher accuracy: {base_o:.6f}")
print(f"OOF corrected accuracy: {selected.get('oof_accuracy_after', base_o):.6f}")

# -------------------------------------------------------------------------
# Fit specialists on ALL OOF rows and predict TEST.
# -------------------------------------------------------------------------
p_test = np.zeros((len(X_test), M), dtype=np.float64)

print("\n--- Fitting specialists on all OOF rows -> TEST ---")
for j, name in enumerate(names):
    print(f"Fitting {name}...")
    mdl = clone(models[name])
    mdl.fit(X_oof, y_oof)
    p_test[:, j] = mdl.predict_proba(X_test)[:, 1]

# Main64 final test predictions.
teacher_candidates = [
    "main64_test_predictions.npy",
    "main64_test_pred.npy",
    "main64_test_labels.npy",
]

teacher_test = None
teacher_source = None

for fname in teacher_candidates:
    f = ROOT / fname
    if not f.exists():
        continue
    arr = np.asarray(np.load(f, allow_pickle=True)).reshape(-1)
    if len(arr) == len(X_test):
        # Main64 final predictions must already be binary labels.
        vals = set(np.unique(arr).tolist())
        if vals.issubset({0, 1}):
            teacher_test = arr.astype(int)
            teacher_source = fname
            break

if teacher_test is None:
    raise RuntimeError(
        "Could not find Main64 binary test predictions. Expected one of: "
        + ", ".join(teacher_candidates)
    )

hard_t = (p_test >= 0.5).astype(int)
disagree_t = hard_t != teacher_test[:, None]
conf_t = np.maximum(p_test, 1.0 - p_test)

vote_t = disagree_t.sum(axis=1)
sum_dis_t = np.sum(np.where(disagree_t, conf_t - 0.5, 0.0), axis=1)
mean_dis_t = np.divide(
    sum_dis_t, vote_t, out=np.zeros_like(sum_dis_t), where=vote_t > 0
)
weak_dis_t = np.where(disagree_t, conf_t, 1.0).min(axis=1)

change_t = (
    (vote_t >= k)
    & (mean_dis_t >= margin)
    & (weak_dis_t >= conf)
)

final_test = teacher_test.copy()
final_test[change_t] = 1 - final_test[change_t]

print(f"Main64 test predictions: {teacher_source}")
print(f"Test rows: {len(final_test)}")
print(f"Main115 corrections: {int(change_t.sum())} ({change_t.mean()*100:.3f}%)")
print(f"Final label counts [A,B]: {np.bincount(final_test, minlength=2).tolist()}")

# -------------------------------------------------------------------------
# Write Kaggle CSV.
# -------------------------------------------------------------------------
if not TEST_JSON.exists():
    raise FileNotFoundError(f"Missing {TEST_JSON}")

test_df = pd.read_json(TEST_JSON)

# Try common ID column names. If the competition expects a different schema,
# this will stop rather than silently guessing.
id_col = next((c for c in ["id", "ID", "index"] if c in test_df.columns), None)

if id_col is None:
    raise RuntimeError(
        f"Could not find an ID column in {TEST_JSON.name}. "
        f"Columns found: {list(test_df.columns)}"
    )

if len(test_df) != len(final_test):
    raise RuntimeError(
        f"Test length mismatch: test.json={len(test_df)}, "
        f"predictions={len(final_test)}"
    )

# Use the same two-column convention as a standard Kaggle classification
# submission: identifier + label.
submission = pd.DataFrame({
    id_col: test_df[id_col].to_numpy(),
    "label": final_test,
})

submission.to_csv(OUT_SUBMISSION, index=False)

print(f"\nWROTE: {OUT_SUBMISSION}")
print("First 5 rows:")
print(submission.head())
print("=" * 78)
