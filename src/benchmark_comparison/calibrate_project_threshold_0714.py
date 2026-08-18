# -*- coding: utf-8 -*-
"""Select the proposed model's classification threshold from training-set
out-of-fold (OOF) predictions only; the independent test set never
participates in threshold selection (avoids test-set leakage).
"""
from __future__ import annotations

import io
import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# NOTE (archival script, see src/benchmark_comparison/README.md): this path
# pointed at the original development machine and is not portable. Edit it
# to point at your own working directory before running.
ROOT = Path(r"C:\path\to\your\workdir")
OUT = ROOT / "processed" / "benchmark_0714"
CKPT = ROOT / "checkpoints" / "benchmark_0714"
sys.path.insert(0, str(ROOT / "scripts" / "benchmark_models"))
from run_fair_benchmark_0714 import run_project_optimized

SEED = 42


def row(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "ACC": float(accuracy_score(y, pred)),
        "MCC": float(matthews_corrcoef(y, pred)),
        "Sn": float(tp / (tp + fn)),
        "Sp": float(tn / (tn + fp)),
        "F1": float(f1_score(y, pred)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def main():
    X = np.load(OUT / "X_train_617.npy")
    y = np.load(OUT / "y_train.npy")
    Xt = np.load(OUT / "X_test_617.npy")
    yt = np.load(OUT / "y_test.npy")
    pred = pd.read_csv(OUT / "fair_benchmark_partial_predictions.csv")
    ptest = pred["Proposed_prob"].to_numpy()
    with open(CKPT / "proposed_optimized_ensemble.pkl", "rb") as f:
        package = pickle.load(f)
    weights = np.asarray(package["weights"])
    names = package["base_names"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_cols = []
    for name in names:
        print("OOF", name)
        base = package["base_models"][name]
        oof = cross_val_predict(
            clone(base), X, y, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        oof_cols.append(oof)
    oof = np.column_stack(oof_cols) @ weights
    grid = np.linspace(0.20, 0.80, 601)
    rows = [row(y, oof, t) for t in grid]
    table = pd.DataFrame(rows)
    # MCC primary; tie by ACC then distance to 0.5.
    table["distance_05"] = np.abs(table.threshold - 0.5)
    best = table.sort_values(
        ["MCC", "ACC", "distance_05"], ascending=[False, False, True]
    ).iloc[0]
    threshold = float(best.threshold)
    result = {
        "selection": "training OOF only; maximize MCC, tie by ACC and closeness to 0.5",
        "selected_threshold": threshold,
        "train_oof_at_selected": row(y, oof, threshold),
        "independent_test_at_0.5": row(yt, ptest, 0.5),
        "independent_test_at_oof_threshold": row(yt, ptest, threshold),
    }
    table.to_csv(OUT / "proposed_oof_threshold_grid.csv", index=False, encoding="utf-8-sig")
    np.save(OUT / "proposed_train_oof_probability.npy", oof)
    with open(OUT / "proposed_threshold_calibration.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
