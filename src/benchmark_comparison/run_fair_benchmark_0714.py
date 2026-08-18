# -*- coding: utf-8 -*-
"""
Fair classification benchmark (2026-07-14).

Fixed on the public DPP-IV data released with the BERT-DPPIV /
StructuralDPPIV repositories:
  - benchmark train: 532 positive + 532 negative (conflicts and cross-split
    duplicates removed)
  - independent test: 133 positive + 133 negative (fully frozen, evaluated
    only once at the end)

This script runs:
  1) SVM (AAC+DPC features, hyperparameters tuned by training-set CV)
  2) iDPPIV-SCM method re-implementation (AAC/DPC propensity scoring)
  3) StackDPPIV method re-implementation (10 feature views x 5 base models +
     OOF stacking)
  4) This project's optimized ensemble (617-D features, OOF stacking)

The official StructuralDPPIV architecture and the BERT-DPPIV transfer model
are run by separate GPU scripts; results are finally aggregated by
build_fair_benchmark_delivery_0714.py.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# NOTE (archival script, see src/benchmark_comparison/README.md): this path
# pointed at the original development machine and is not portable. Edit it
# to point at your own working directory before running.
ROOT = Path(r"C:\path\to\your\workdir")
MODEL_ROOT = ROOT / "scripts" / "benchmark_models"
BERT_DATA = MODEL_ROOT / "BERT-DPPIV" / "Fine_tune_data" / "DPP-IV_Dataset"
OUT = ROOT / "processed" / "benchmark_0714"
CKPT = ROOT / "checkpoints" / "benchmark_0714"
OUT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from src.features.sequence_features import extract_sequence_features

SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def read_fasta_sequences(path: Path) -> list[str]:
    seqs: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().upper()
        if line and not line.startswith(">"):
            seqs.append(line)
    return seqs


def load_raw_dataset() -> pd.DataFrame:
    rows: list[dict] = []
    for split in ("train", "test"):
        for name, label in (("positive", 1), ("negative", 0)):
            path = BERT_DATA / f"{split}-{name}.txt"
            for sequence in read_fasta_sequences(path):
                rows.append({"split_original": split, "label": label, "sequence": sequence})
    return pd.DataFrame(rows)


def clean_and_freeze(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_raw = raw[raw["split_original"] == "train"].copy()
    test_raw = raw[raw["split_original"] == "test"].copy()

    train_label_n = train_raw.groupby("sequence")["label"].nunique()
    conflict_train = set(train_label_n[train_label_n > 1].index)
    overlap = set(train_raw["sequence"]) & set(test_raw["sequence"])

    # Test is the frozen external target. Any sequence appearing in test is removed
    # from training, regardless of label. Conflicting training sequences are removed.
    train = train_raw[
        ~train_raw["sequence"].isin(conflict_train | overlap)
    ].drop_duplicates(["sequence", "label"]).reset_index(drop=True)
    test = test_raw.drop_duplicates(["sequence", "label"]).reset_index(drop=True)

    test_conflict = test.groupby("sequence")["label"].nunique()
    if (test_conflict > 1).any():
        raise RuntimeError("Frozen independent test contains conflicting labels.")
    if set(train["sequence"]) & set(test["sequence"]):
        raise RuntimeError("Leakage remains after cleaning.")

    audit = {
        "source": "Official guanchangge/BERT-DPPIV repository; same files bundled by StructuralDPPIV",
        "raw_train_rows": int(len(train_raw)),
        "raw_test_rows": int(len(test_raw)),
        "raw_train_class_counts": {
            str(k): int(v) for k, v in train_raw["label"].value_counts().sort_index().items()
        },
        "raw_test_class_counts": {
            str(k): int(v) for k, v in test_raw["label"].value_counts().sort_index().items()
        },
        "train_conflicting_sequences_removed": sorted(conflict_train),
        "train_test_overlap_sequences_removed_from_train": sorted(overlap),
        "clean_train_rows": int(len(train)),
        "frozen_test_rows": int(len(test)),
        "clean_train_class_counts": {
            str(k): int(v) for k, v in train["label"].value_counts().sort_index().items()
        },
        "frozen_test_class_counts": {
            str(k): int(v) for k, v in test["label"].value_counts().sort_index().items()
        },
        "postclean_exact_overlap": int(
            len(set(train["sequence"]) & set(test["sequence"]))
        ),
        "policy": (
            "Independent test frozen. Removed all train rows whose sequence appears "
            "in test and all train sequences carrying conflicting labels."
        ),
    }
    return train, test, audit


def metrics_row(
    model: str,
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float = 0.5,
    implementation: str = "retrained",
) -> dict:
    pred = (np.asarray(prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Model": model,
        "ACC": accuracy_score(y_true, pred),
        "MCC": matthews_corrcoef(y_true, pred),
        "Sn": tp / (tp + fn) if tp + fn else np.nan,
        "Sp": tn / (tn + fp) if tn + fp else np.nan,
        "AUC": roc_auc_score(y_true, prob),
        "F1": f1_score(y_true, pred),
        "PR_AUC": average_precision_score(y_true, prob),
        "Threshold": threshold,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Implementation": implementation,
    }


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def make_model_pool() -> dict[str, object]:
    return {
        "LR": Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0, max_iter=4000, class_weight="balanced", random_state=SEED
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=2.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "RF": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "ET": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "XGB": XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.05,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=8,
        ),
    }


def run_svm(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, object, dict]:
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("svc", SVC(probability=True, class_weight="balanced", random_state=SEED)),
        ]
    )
    grid = {
        "svc__C": [0.25, 0.5, 1, 2, 4, 8],
        "svc__gamma": ["scale", 0.0005, 0.001, 0.002, 0.005],
        "svc__kernel": ["rbf"],
    }
    gs = GridSearchCV(
        pipe, grid, scoring="roc_auc", cv=CV, n_jobs=-1, refit=True, verbose=0
    )
    gs.fit(X_train[:, :420], y_train)
    prob = gs.predict_proba(X_test[:, :420])[:, 1]
    return prob, gs.best_estimator_, {
        "best_params": gs.best_params_,
        "best_cv_auc": float(gs.best_score_),
    }


class SCMScorer:
    """Interpretable AAC+DPC propensity scoring-card reimplementation."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_: np.ndarray | None = None
        self.scale_: float = 1.0
        self.bias_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Smoothed class means transformed to log propensity ratios.
        pos = X[y == 1].mean(axis=0)
        neg = X[y == 0].mean(axis=0)
        eps = self.alpha / max(len(y), 1)
        raw = np.log((pos + eps) / (neg + eps))
        # Scoring card: cap outliers and quantise to integer points.
        cap = np.percentile(np.abs(raw), 98)
        raw = np.clip(raw, -cap, cap)
        scale = np.std(raw) or 1.0
        self.coef_ = np.round(raw / scale * 10.0) / 10.0
        scores = X @ self.coef_
        calibrator = LogisticRegression(max_iter=2000, random_state=SEED)
        calibrator.fit(scores.reshape(-1, 1), y)
        self.scale_ = float(calibrator.coef_[0, 0])
        self.bias_ = float(calibrator.intercept_[0])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        score = X @ self.coef_
        p = expit(self.scale_ * score + self.bias_)
        return np.column_stack([1 - p, p])


def run_scm(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, SCMScorer, dict]:
    Xtr = X_train[:, :420]
    Xte = X_test[:, :420]
    oof = np.zeros(len(y_train))
    for tr, va in CV.split(Xtr, y_train):
        m = SCMScorer(alpha=1.0).fit(Xtr[tr], y_train[tr])
        oof[va] = m.predict_proba(Xtr[va])[:, 1]
    model = SCMScorer(alpha=1.0).fit(Xtr, y_train)
    return model.predict_proba(Xte)[:, 1], model, {
        "oof_auc": float(roc_auc_score(y_train, oof)),
        "method": "AAC+DPC smoothed propensity log-ratio, integer scoring card, train-only calibration",
    }


def feature_views(X: np.ndarray) -> dict[str, np.ndarray]:
    # 617 = AAC20 | DPC400 | CTD147 | PAAC50
    aac = X[:, :20]
    dpc = X[:, 20:420]
    ctd = X[:, 420:567]
    paac = X[:, 567:617]
    return {
        "AAC": aac,
        "DPC": dpc,
        "CTD": ctd,
        "PAAC": paac,
        "AAC_DPC": np.hstack([aac, dpc]),
        "AAC_CTD": np.hstack([aac, ctd]),
        "AAC_PAAC": np.hstack([aac, paac]),
        "DPC_CTD": np.hstack([dpc, ctd]),
        "CTD_PAAC": np.hstack([ctd, paac]),
        "ALL617": X,
    }


def run_stackdppiv_reimplementation(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, dict, dict]:
    train_views = feature_views(X_train)
    test_views = feature_views(X_test)
    pool = make_model_pool()
    oof_cols, test_cols, col_names = [], [], []
    fitted: dict[str, object] = {}

    for view_name, Xv in train_views.items():
        for model_name, base in pool.items():
            key = f"{view_name}__{model_name}"
            print(f"  Stack base: {key}")
            oof = cross_val_predict(
                clone(base),
                Xv,
                y_train,
                cv=CV,
                method="predict_proba",
                n_jobs=1,
            )[:, 1]
            model = clone(base).fit(Xv, y_train)
            ptest = model.predict_proba(test_views[view_name])[:, 1]
            oof_cols.append(oof)
            test_cols.append(ptest)
            col_names.append(key)
            fitted[key] = model

    O = np.column_stack(oof_cols)
    T = np.column_stack(test_cols)

    # Stable feature selection performed only with OOF data.
    aucs = np.array([roc_auc_score(y_train, O[:, j]) for j in range(O.shape[1])])
    top = np.argsort(aucs)[::-1][:20]
    meta = LogisticRegression(C=0.2, max_iter=4000, random_state=SEED)
    meta.fit(O[:, top], y_train)
    ptest = meta.predict_proba(T[:, top])[:, 1]

    package = {
        "base_models": fitted,
        "meta_model": meta,
        "selected_indices": top,
        "column_names": col_names,
    }
    info = {
        "implementation": (
            "Faithful-method reimplementation: 10 sequence descriptor views × "
            "5 base algorithms, OOF probabilistic features, top-20 OOF selection, "
            "logistic meta-predictor. Original StackDPPIV source was not public."
        ),
        "selected_base_models": [col_names[i] for i in top],
        "selected_oof_auc": [float(aucs[i]) for i in top],
    }
    return ptest, package, info


def run_project_optimized(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, dict, dict]:
    models: dict[str, object] = {
        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.7,
            reg_alpha=0.05,
            reg_lambda=1.2,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=8,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            num_leaves=31,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.05,
            reg_lambda=0.8,
            random_state=SEED,
            n_jobs=8,
            verbosity=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.035,
            l2_leaf_reg=4.0,
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
            thread_count=8,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "SVM": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "svc",
                    SVC(
                        C=2.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }

    oof_cols, test_cols, fitted = [], [], {}
    for name, base in models.items():
        print(f"  Project base: {name}")
        oof = cross_val_predict(
            clone(base),
            X_train,
            y_train,
            cv=CV,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        model = clone(base).fit(X_train, y_train)
        ptest = model.predict_proba(X_test)[:, 1]
        oof_cols.append(oof)
        test_cols.append(ptest)
        fitted[name] = model

    O = np.column_stack(oof_cols)
    T = np.column_stack(test_cols)
    meta = LogisticRegression(C=0.15, max_iter=4000, random_state=SEED)
    meta.fit(O, y_train)
    p_meta = meta.predict_proba(T)[:, 1]

    # Also calculate OOF-AUC weighted soft voting; select between stacking/voting
    # using OOF only (never independent test).
    base_aucs = np.array([roc_auc_score(y_train, O[:, i]) for i in range(O.shape[1])])
    weights = np.maximum(base_aucs - 0.5, 1e-4)
    weights /= weights.sum()
    oof_weighted = O @ weights
    oof_meta = cross_val_predict(
        LogisticRegression(C=0.15, max_iter=4000, random_state=SEED),
        O,
        y_train,
        cv=CV,
        method="predict_proba",
    )[:, 1]
    if roc_auc_score(y_train, oof_meta) >= roc_auc_score(y_train, oof_weighted):
        selected = "OOF stacking"
        ptest = p_meta
    else:
        selected = "OOF-AUC weighted soft voting"
        ptest = T @ weights

    package = {
        "base_models": fitted,
        "meta_model": meta,
        "weights": weights,
        "base_names": list(models),
        "selected": selected,
    }
    info = {
        "selected": selected,
        "base_oof_auc": {
            name: float(v) for name, v in zip(models.keys(), base_aucs)
        },
        "soft_voting_weights": {
            name: float(v) for name, v in zip(models.keys(), weights)
        },
        "selection_oof_auc": {
            "stacking": float(roc_auc_score(y_train, oof_meta)),
            "weighted_voting": float(roc_auc_score(y_train, oof_weighted)),
        },
    }
    return ptest, package, info


def main() -> None:
    t0 = time.time()
    raw = load_raw_dataset()
    train, test, audit = clean_and_freeze(raw)
    train.to_csv(OUT / "benchmark_train_clean.csv", index=False, encoding="utf-8-sig")
    test.to_csv(OUT / "independent_test_frozen.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "data_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=2))

    X_train = extract_sequence_features(train["sequence"].tolist())
    X_test = extract_sequence_features(test["sequence"].tolist())
    y_train = train["label"].to_numpy(dtype=int)
    y_test = test["label"].to_numpy(dtype=int)
    np.save(OUT / "X_train_617.npy", X_train)
    np.save(OUT / "X_test_617.npy", X_test)
    np.save(OUT / "y_train.npy", y_train)
    np.save(OUT / "y_test.npy", y_test)

    results: list[dict] = []
    pred_df = test[["sequence", "label"]].copy()
    run_info: dict[str, dict] = {"data_audit": audit}

    print("\n=== SVM ===")
    p, model, info = run_svm(X_train, y_train, X_test)
    pred_df["SVM_prob"] = p
    row = metrics_row("SVM (uniformly retrained)", y_test, p)
    row["AUC_CI_low"], row["AUC_CI_high"] = bootstrap_auc_ci(y_test, p)
    results.append(row)
    run_info["SVM"] = info
    with open(CKPT / "svm.pkl", "wb") as f:
        pickle.dump(model, f)
    print(row)

    print("\n=== iDPPIV-SCM reimplementation ===")
    p, model, info = run_scm(X_train, y_train, X_test)
    pred_df["iDPPIV_SCM_prob"] = p
    row = metrics_row(
        "iDPPIV-SCM (same-method re-implementation)",
        y_test,
        p,
        implementation="method reimplementation; original source unavailable",
    )
    row["AUC_CI_low"], row["AUC_CI_high"] = bootstrap_auc_ci(y_test, p)
    results.append(row)
    run_info["iDPPIV-SCM"] = info
    with open(CKPT / "idppiv_scm_reimplementation.pkl", "wb") as f:
        pickle.dump(model, f)
    print(row)

    print("\n=== StackDPPIV reimplementation ===")
    p, package, info = run_stackdppiv_reimplementation(X_train, y_train, X_test)
    pred_df["StackDPPIV_prob"] = p
    row = metrics_row(
        "StackDPPIV (same-method re-implementation)",
        y_test,
        p,
        implementation="method reimplementation; original source unavailable",
    )
    row["AUC_CI_low"], row["AUC_CI_high"] = bootstrap_auc_ci(y_test, p)
    results.append(row)
    run_info["StackDPPIV"] = info
    with open(CKPT / "stackdppiv_reimplementation.pkl", "wb") as f:
        pickle.dump(package, f)
    print(row)

    print("\n=== Proposed optimized ensemble ===")
    p, package, info = run_project_optimized(X_train, y_train, X_test)
    pred_df["Proposed_prob"] = p
    row = metrics_row("Proposed optimized ensemble", y_test, p)
    row["AUC_CI_low"], row["AUC_CI_high"] = bootstrap_auc_ci(y_test, p)
    results.append(row)
    run_info["Proposed"] = info
    with open(CKPT / "proposed_optimized_ensemble.pkl", "wb") as f:
        pickle.dump(package, f)
    print(row)

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT / "fair_benchmark_partial_results.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(OUT / "fair_benchmark_partial_predictions.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "run_info_partial.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    print("\nRESULTS")
    print(result_df.to_string(index=False))
    print(f"Elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
