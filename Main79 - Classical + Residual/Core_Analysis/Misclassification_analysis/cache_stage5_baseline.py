import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from Feature_Engineering import (
    FeatureExtractor,
    resolve_dataset_paths,
    load_dataset,
)

def main():
    cache_dir = Path(__file__).resolve().parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    print("[1/4] Loading dataset...")
    data_path, log_df_path = resolve_dataset_paths()
    tokens, labels = load_dataset(data_path)
    N = len(tokens)
    
    print("[2/4] Initializing and extracting Stage 5 features...")
    t0 = time.time()
    extractor = FeatureExtractor(
        log_df_path=log_df_path,
        ngram_range=(1, 5),
        min_df=3,
        sublinear_tf=True,
        random_state=42,
    )
    extractor._fit_spectral_syntactic_manifold(tokens)
    dense_matrix = extractor.extract_all_dense_features(tokens)
    X_tfidf = extractor.generate_tfidf_features(tokens, is_training=True)
    dense_feature_names = extractor.get_dense_feature_names()
    print(f"Features extracted in {time.time()-t0:.2f}s.")
    print(f"  Sparse TF-IDF shape: {X_tfidf.shape}")
    print(f"  Dense matrix shape:  {dense_matrix.shape}")
    
    print("[3/4] Running 5-fold Stratified CV for Stage 5 Baseline...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(N, dtype=int)
    oof_margins = np.zeros(N, dtype=float)
    splits = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tfidf, labels), 1):
        splits.append((tr_idx, val_idx))
        scaler = MaxAbsScaler()
        tr_dense = scaler.fit_transform(dense_matrix[tr_idx])
        val_dense = scaler.transform(dense_matrix[val_idx])
        
        X_tr = sp.hstack([X_tfidf[tr_idx], sp.csr_matrix(tr_dense)]).tocsr()
        X_val = sp.hstack([X_tfidf[val_idx], sp.csr_matrix(val_dense)]).tocsr()
        
        clf = LinearSVC(C=1.0, class_weight='balanced', dual="auto", random_state=42)
        clf.fit(X_tr, labels[tr_idx])
        
        oof_preds[val_idx] = clf.predict(X_val)
        oof_margins[val_idx] = clf.decision_function(X_val)
        acc = accuracy_score(labels[val_idx], oof_preds[val_idx])
        print(f"  Fold {fold}: Acc = {acc*100:.2f}%")
        
    overall_acc = accuracy_score(labels, oof_preds)
    overall_f1 = f1_score(labels, oof_preds, average='macro')
    print(f"\nStage 5 Baseline OOF Accuracy: {overall_acc*100:.4f}% | Macro F1: {overall_f1*100:.4f}%")
    print(confusion_matrix(labels, oof_preds))
    
    print("[4/4] Saving cached matrices to disk...")
    sp.save_npz(cache_dir / "X_tfidf.npz", X_tfidf)
    np.save(cache_dir / "dense_matrix.npy", dense_matrix)
    np.save(cache_dir / "labels.npy", labels)
    np.save(cache_dir / "oof_preds.npy", oof_preds)
    np.save(cache_dir / "oof_margins.npy", oof_margins)
    
    with open(cache_dir / "tokens.json", "w") as f:
        json.dump(tokens, f)
        
    with open(cache_dir / "meta.json", "w") as f:
        json.dump({
            "dense_feature_names": dense_feature_names,
            "overall_acc": overall_acc,
            "overall_f1": overall_f1,
            "splits": [(tr.tolist(), val.tolist()) for tr, val in splits]
        }, f)
        
    print(f"All cache artifacts saved successfully in {cache_dir}.")

if __name__ == "__main__":
    main()
