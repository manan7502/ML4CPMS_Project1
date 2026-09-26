import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import classification_report, confusion_matrix

# Ensure project root directory (Core_Analysis) is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from Feature_Engineering import (
    FeatureExtractor,
    resolve_dataset_paths,
    load_dataset,
)

# 1. Load data
data_path, log_df_path = resolve_dataset_paths()  # or pass train.json path
tokens, labels = load_dataset(data_path)
extractor = FeatureExtractor(
    log_df_path=log_df_path,
    ngram_range=(1, 5),
    min_df=3,
    sublinear_tf=True,
    random_state=42,
)
N = len(tokens)

# 2. Extract features
print("Fitting features...")
extractor._fit_spectral_syntactic_manifold(tokens)
dense_matrix = extractor.extract_all_dense_features(tokens)
X_tfidf = extractor.generate_tfidf_features(tokens, is_training=True)
dense_feature_names = extractor.get_dense_feature_names()

# 3. 5-Fold Stratified OOF Generation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(N, dtype=int)
oof_margins = np.zeros(N, dtype=float)

print("Running 5-fold Stratified OOF Generation...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tfidf, labels), 1):
    print(f"  Processing Fold {fold}/5...")
    scaler = MaxAbsScaler()
    tr_dense = scaler.fit_transform(dense_matrix[tr_idx])
    val_dense = scaler.transform(dense_matrix[val_idx])

    X_tr = sp.hstack([X_tfidf[tr_idx], sp.csr_matrix(tr_dense)]).tocsr()
    X_val = sp.hstack([X_tfidf[val_idx], sp.csr_matrix(val_dense)]).tocsr()

    clf = LinearSVC(C=1.0, class_weight='balanced', dual="auto", random_state=42)
    clf.fit(X_tr, labels[tr_idx])

    oof_preds[val_idx] = clf.predict(X_val)
    oof_margins[val_idx] = clf.decision_function(X_val)

# 4. Error Identification
doc_lengths = np.array([len(t) for t in tokens])
is_error = (oof_preds != labels)
df_analysis = pd.DataFrame({
    'doc_id': np.arange(N),
    'length': doc_lengths,
    'true_label': ['Human (A)' if y == 0 else 'Machine (B)' for y in labels],
    'pred_label': ['Human (A)' if y == 0 else 'Machine (B)' for y in oof_preds],
    'margin': oof_margins,
    'abs_margin': np.abs(oof_margins),
    'is_error': is_error
})

print("\n" + "=" * 60)
print("1. OVERALL CONFUSION MATRIX & ACCURACY")
print("=" * 60)
print(classification_report(labels, oof_preds, target_names=['Human (A)', 'Machine (B)']))
print("Confusion Matrix (Rows=True, Cols=Pred):")
print(confusion_matrix(labels, oof_preds))

print("\n" + "=" * 60)
print("2. ERROR RATE STRATIFIED BY SEQUENCE LENGTH")
print("=" * 60)
df_analysis['length_bin'] = pd.qcut(df_analysis['length'], q=5, duplicates='drop')
len_summary = df_analysis.groupby('length_bin', observed=False)['is_error'].agg(
    Total='count',
    Errors='sum',
    Error_Rate=lambda x: f"{x.mean()*100:.2f}%"
)
print(len_summary)

print("\n" + "=" * 60)
print("3. TOP 5 MOST CONFIDENT ERRORS (Pathological / Outliers)")
print("=" * 60)
confident_errors = df_analysis[df_analysis['is_error']].sort_values(by='abs_margin', ascending=False).head(5)
for _, row in confident_errors.iterrows():
    print(f"Doc {int(row['doc_id'])} | True: {row['true_label']} -> Pred: {row['pred_label']} | Margin: {row['margin']:+.3f} | Length: {row['length']}")