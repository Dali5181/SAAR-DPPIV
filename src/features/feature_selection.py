"""
Top-300 feature selection for the ranking head.

The ranking model does not use the full 1097-D feature space (617-D
handcrafted sequence descriptors + 480-D ESM-2 embedding). Instead, an
initial XGBoost classifier is trained on the full feature space purely to
rank features by importance, and only the top 300 (by mean feature
importance across a 3-fold stratified CV on the classification training
split) are kept for the LambdaRank ranker. This both regularises the ranker
(IC50-labelled data is much scarcer than classification-labelled data) and
keeps a fixed, versioned feature subset — see
``processed/top300_feature_indices.npy`` (the exact indices used by the
shipped checkpoint).
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

TOP_K = 300
SEED = 42


def select_top_k_features(X_full: np.ndarray, y: np.ndarray, top_k: int = TOP_K,
                           n_splits: int = 3, seed: int = SEED) -> np.ndarray:
    """Return the (sorted, ascending) indices of the top-k most important
    features in ``X_full`` (shape (n_samples, 1097): columns 0-616 are the
    617-D sequence features, columns 617-1096 are the 480-D ESM-2 embedding).

    Importance is averaged over an n-fold stratified CV to reduce
    sensitivity to a single train/validation split.
    """
    xgb_init = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=seed, verbosity=0, eval_metric="logloss",
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    importances = np.zeros(X_full.shape[1])
    for tr, _ in skf.split(X_full, y):
        xgb_init.fit(X_full[tr], y[tr])
        importances += xgb_init.feature_importances_
    importances /= n_splits

    top_idx = np.argsort(importances)[::-1][:top_k]
    return np.sort(top_idx)
