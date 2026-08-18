"""
Classification head: a 4-model tree ensemble (XGBoost, LightGBM, CatBoost,
RandomForest) predicting DPP-IV inhibitory activity from the 617-D sequence
feature vector (see ``src/features/sequence_features.py``).

At inference time (``pipeline_scoring.score_sequences``) the four models'
positive-class probabilities are simply averaged:

    P(active) = mean(xgb_cls.predict_proba, lgb_cls.predict_proba,
                      cat_cls.predict_proba, rf_cls.predict_proba)

This module defines the training configuration used to produce the
checkpoints shipped in ``checkpoints/classification/classifiers.pkl``. It is
provided for methodological transparency and to let the ensemble be
retrained from scratch; it is NOT imported at inference time (the shipped
checkpoint already contains the fitted, pickled model objects).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Hyperparameters selected via randomised search (5-fold stratified CV, ROC-AUC)
# on the classification training split (iDPP-IV-CV, n=1848). These are the
# defaults; the exact fitted parameter values baked into the shipped
# checkpoint are visible via `model.get_params()` on the unpickled objects.
DEFAULT_CONFIGS = {
    "XGBoost": dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    ),
    "LightGBM": dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    ),
    "CatBoost": dict(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
    ),
    "RandomForest": dict(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
}

_MODEL_CLASSES = {
    "XGBoost": XGBClassifier,
    "LightGBM": LGBMClassifier,
    "CatBoost": CatBoostClassifier,
    "RandomForest": RandomForestClassifier,
}


class ClassificationEnsemble:
    """Trains four tree-based classifiers and averages their probabilities.

    Example
    -------
    >>> ens = ClassificationEnsemble().fit(X_train, y_train)
    >>> p_active = ens.predict_proba(X_test)   # shape (n_samples,)
    """

    def __init__(self, configs: dict[str, dict] | None = None):
        cfgs = configs or DEFAULT_CONFIGS
        self.models: dict[str, object] = {
            name: _MODEL_CLASSES[name](**cfgs.get(name, DEFAULT_CONFIGS[name]))
            for name in _MODEL_CLASSES
        }
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassificationEnsemble":
        for model in self.models.values():
            model.fit(X, y)
        self._fitted = True
        return self

    def predict_proba_per_model(self, X: np.ndarray) -> np.ndarray:
        """Return stacked positive-class probabilities, shape (n_samples, 4)."""
        if not self._fitted:
            raise RuntimeError("ClassificationEnsemble has not been fitted yet.")
        cols = [self.models[name].predict_proba(X)[:, 1] for name in _MODEL_CLASSES]
        return np.column_stack(cols)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Mean-averaged P(active) across the four experts, shape (n_samples,)."""
        return self.predict_proba_per_model(X).mean(axis=1)
