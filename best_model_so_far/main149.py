#!/usr/bin/env python3
"""
MAIN140 / EXPERIMENT 29
EMBEDDING LINEAR SEPARABILITY

Tracker-aligned Experiment 29.

Purpose:
    Measure how linearly separable the strongest saved learned
    representations are using ONLY the latent vector z.

Tracker definition:
    29. EMBEDDING LINEAR SEPARABILITY
        SVM/LR accuracy and AUC using only z.

Strong representations explicitly identified by the tracker:
    - Main133 / Exp13
    - Main131 / Exp11
    - Any complementary representation retained in later phases is
      reported only if its saved embedding artifact is available.

IMPORTANT:
    This is an AUDIT experiment, not a fusion experiment.
    Do NOT use Main115 specialist probabilities, Main64 features,
    or validation-fitted fusion models here.

For each available representation:
    - load saved validation latent z
    - use the canonical validation labels
    - fit SVM and Logistic Regression on z ONLY

The original saved embedding files contain validation embeddings only.
Therefore, to avoid pretending that a validation-only probe is a
strict generalization estimate, this script reports the same
diagnostic-style linear separability requested by the tracker.

No Kaggle submission is performed.
"""

from pathlib import Path
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC


ROOT = Path("/home/manu/Desktop/ML4CPMS/project_1")
TRAIN_PATH = ROOT / "train.json"

SEED = 140

# Canonical validation order is reconstructed from the dataset.
# This must remain exactly train_test_split(..., random_state=42).
from sklearn.model_selection import train_test_split


def load_jsonl(path):
    import json as _json
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(_json.loads(line))
    return rows


def canonical_validation_labels():
    rows = load_jsonl(TRAIN_PATH)
    y = np.array([0 if r["label"] == "A" else 1 for r in rows], dtype=np.int64)
    idx = np.arange(len(y))

    tr_idx, va_idx = train_test_split(
        idx,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    return y[va_idx], va_idx


def load_embedding_artifact(path, expected_val_idx):
    data = np.load(path, allow_pickle=True)

    # Prefer explicit val_labels if saved, but always verify alignment
    # against the canonical split whenever val_indices are present.
    if "z_val" not in data:
        raise KeyError(f"{path} does not contain z_val")

    z = np.asarray(data["z_val"], dtype=np.float32)

    if "val_indices" in data:
        saved_idx = np.asarray(data["val_indices"], dtype=np.int64)
        if saved_idx.shape != expected_val_idx.shape or not np.array_equal(
            saved_idx, expected_val_idx
        ):
            raise RuntimeError(
                f"Validation index mismatch in {path}.\n"
                "Refusing to evaluate an embedding whose validation order "
                "does not exactly match the canonical split."
            )

    return z


def evaluate_linear_models(name, z, y):
    print(f"\n=== {name} ===")
    print("z shape:", z.shape)

    results = {}

    # Diagnostic probes exactly on the saved validation embedding.
    # This is the tracker-requested separability audit, not a submit-safe
    # estimate because the probe is fit on the same validation set.
    models = [
        (
            "LogisticRegression",
            LogisticRegression(
                C=1.0,
                max_iter=5000,
                class_weight="balanced",
                random_state=SEED,
            ),
        ),
        (
            "SVM_RBF",
            SVC(
                C=1.0,
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=SEED,
            ),
        ),
    ]

    for model_name, model in models:
        t0 = time.time()
        model.fit(z, y)
        pred = model.predict(z)
        prob = model.predict_proba(z)[:, 1]

        acc = accuracy_score(y, pred)
        auc = roc_auc_score(y, prob)

        results[model_name] = {
            "accuracy": float(acc),
            "auc": float(auc),
            "runtime_seconds": float(time.time() - t0),
        }

        print(
            f"{model_name:20s} "
            f"accuracy={acc:.6f} "
            f"AUC={auc:.6f} "
            f"time={time.time() - t0:.2f}s"
        )

    # Linear SVM is the specific linear-separability diagnostic requested.
    t0 = time.time()
    linear_svm = SVC(
        C=1.0,
        kernel="linear",
        probability=True,
        class_weight="balanced",
        random_state=SEED,
    )
    linear_svm.fit(z, y)
    pred = linear_svm.predict(z)
    prob = linear_svm.predict_proba(z)[:, 1]

    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, prob)

    results["SVM_linear"] = {
        "accuracy": float(acc),
        "auc": float(auc),
        "runtime_seconds": float(time.time() - t0),
    }

    print(
        f"{'SVM_linear':20s} "
        f"accuracy={acc:.6f} "
        f"AUC={auc:.6f} "
        f"time={time.time() - t0:.2f}s"
    )

    return results


def geometry(z, y):
    """
    Additional descriptive geometry, retained as a secondary audit.
    This does not alter the experiment definition or select a model.
    """
    z0 = z[y == 0]
    z1 = z[y == 1]

    c0 = z0.mean(axis=0)
    c1 = z1.mean(axis=0)

    between = float(np.linalg.norm(c0 - c1))

    r0 = float(np.linalg.norm(z0 - c0, axis=1).mean())
    r1 = float(np.linalg.norm(z1 - c1, axis=1).mean())
    within = (r0 + r1) / 2.0

    ratio = between / within if within > 0 else float("inf")

    return {
        "between_centroid_distance": between,
        "within_class_mean_radius": within,
        "between_within_ratio": ratio,
    }


def main():
    t0 = time.time()

    print("MAIN140 / EXPERIMENT 29")
    print("EMBEDDING LINEAR SEPARABILITY")
    print("Tracker-aligned Phase 8 audit")
    print()

    y_val, val_idx = canonical_validation_labels()

    print("Canonical validation rows:", len(val_idx))
    print("Validation label counts:", np.bincount(y_val))

    # Tracker explicitly identifies Main133/Main131 as the primary
    # representations for the detailed audit.
    candidates = [
        ("Main133 / Exp13", ROOT / "main133_embeddings.npz"),
        ("Main131 / Exp11", ROOT / "main131_embeddings.npz"),
    ]

    # Additional retained representations can be included if their exact
    # artifact exists, but the tracker-defined primary audit remains first.
    optional = [
        ("Main145 / Exp25", ROOT / "main145_embeddings.npz"),
        ("Main146 / Exp26", ROOT / "main146_embeddings.npz"),
        ("Main144 / Exp24", ROOT / "main144_embeddings.npz"),
        ("Main139 / Exp19", ROOT / "main139_embeddings.npz"),
    ]

    all_results = {}

    for name, path in candidates + optional:
        if not path.exists():
            print(f"\nSkipping {name}: artifact not found")
            print("  ", path)
            continue

        z = load_embedding_artifact(path, val_idx)

        if z.shape[0] != len(y_val):
            raise RuntimeError(
                f"{name}: z_val has {z.shape[0]} rows but canonical "
                f"validation has {len(y_val)} rows."
            )

        model_results = evaluate_linear_models(name, z, y_val)
        geom = geometry(z, y_val)

        print(
            f"Geometry: between={geom['between_centroid_distance']:.6f}, "
            f"within={geom['within_class_mean_radius']:.6f}, "
            f"ratio={geom['between_within_ratio']:.6f}"
        )

        all_results[name] = {
            "artifact": str(path),
            "z_shape": list(z.shape),
            "models": model_results,
            "geometry": geom,
        }

    if not all_results:
        raise FileNotFoundError(
            "No retained embedding artifacts were found. "
            "Expected at least main133_embeddings.npz and/or "
            "main131_embeddings.npz in the project directory."
        )

    report = {
        "experiment": 29,
        "script": "Main140",
        "title": "Embedding Linear Separability",
        "tracker_definition": (
            "SVM/LR accuracy and AUC using only z."
        ),
        "seed": SEED,
        "canonical_split": {
            "method": "train_test_split",
            "test_size": 0.20,
            "stratify": True,
            "random_state": 42,
            "validation_size": int(len(val_idx)),
        },
        "validation_probe_warning": (
            "Saved embeddings are evaluated with diagnostic probes fit "
            "directly on the same canonical validation embeddings. "
            "These numbers are representation separability diagnostics, "
            "not submit-safe generalization estimates."
        ),
        "results": all_results,
        "runtime_seconds": float(time.time() - t0),
    }

    report_path = ROOT / "main140_report.json"
    npz_path = ROOT / "main140_results.npz"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save compact machine-readable results.
    np.savez_compressed(
        npz_path,
        val_indices=val_idx.astype(np.int64),
        y_val=y_val.astype(np.int64),
    )

    print("\n" + "=" * 60)
    print("MAIN140 / EXPERIMENT 29 COMPLETE")
    print("=" * 60)

    for name, r in all_results.items():
        lr = r["models"].get("LogisticRegression", {})
        svm = r["models"].get("SVM_linear", {})
        print(
            f"{name}: "
            f"LR={lr.get('accuracy', float('nan')):.4f} "
            f"(AUC {lr.get('auc', float('nan')):.4f}), "
            f"LinearSVM={svm.get('accuracy', float('nan')):.4f} "
            f"(AUC {svm.get('auc', float('nan')):.4f})"
        )

    print("\nSaved:")
    print(report_path)
    print(npz_path)
    print("No Kaggle submission performed.")


if __name__ == "__main__":
    main()
