"""
Auxiliary regression head: an XGBoost regressor predicting pIC50
(= -log10(IC50 in M)) from the 617-D sequence feature vector.

Only sequences with an experimentally measured IC50 are used for training
(``has_ic50 == True`` in the source dataset; a minority of the classification
set). The regressor is explicitly AUXILIARY: candidate prioritisation for
wet-lab validation is driven by ``Combined_Score`` (classification probability
+ source-aware rank score, see ``ranking_lambdarank.py`` /
``pipeline_scoring.py``), not by this regressor's raw pIC50 estimate, because
IC50 magnitudes are not comparable across the heterogeneous literature
sources in the dataset (different assay/enzyme/substrate conditions).

This module documents the training configuration used to produce
``checkpoints/regression/regressor.pkl``. It is not imported at inference
time (the shipped checkpoint already contains the fitted model).
"""
from __future__ import annotations

import numpy as np
from xgboost import XGBRegressor

# Matches the hyperparameters of the shipped `xgb_reg` checkpoint
# (confirmed via `model.get_params()` on the unpickled object).
DEFAULT_CONFIG = dict(
    objective="reg:squarederror",
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
)


class RegressionExpert:
    """Thin wrapper around an XGBoost pIC50 regressor.

    Example
    -------
    >>> reg = RegressionExpert().fit(X_train_ic50, pic50_train)
    >>> pic50_hat = reg.predict(X_test)
    """

    def __init__(self, config: dict | None = None):
        self.model = XGBRegressor(**(config or DEFAULT_CONFIG))
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressionExpert":
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("RegressionExpert has not been fitted yet.")
        return self.model.predict(X)
