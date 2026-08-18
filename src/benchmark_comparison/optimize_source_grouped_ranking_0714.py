# -*- coding: utf-8 -*-
"""Alternative-algorithm search for the within-source-group ranking task:
model selection uses GroupKFold CV restricted to the training-source group
only, and the held-out test-source group is evaluated exactly once, after
selection, as a frozen final check.
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
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

# NOTE (archival script, see src/benchmark_comparison/README.md): this path
# pointed at the original development machine and is not portable. Edit it
# to point at your own working directory before running.
ROOT = Path(r"C:\path\to\your\workdir")
IN_DIR = ROOT / "processed" / "ranking_source_grouped_0714"
OUT = IN_DIR
CKPT = ROOT / "checkpoints" / "ranking_source_grouped_0714"
sys.path.insert(0, str(ROOT))
from src.features.sequence_features import extract_sequence_features

SEED = 42


def pair_accuracy(frame: pd.DataFrame, score: np.ndarray) -> tuple[float, int, int]:
    work = frame.copy()
    work["_score"] = score
    c = t = 0
    for _, g in work.groupby("source_norm"):
        y = g.pic50_model.to_numpy()
        p = g._score.to_numpy()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if y[i] == y[j]:
                    continue
                t += 1
                c += int((y[i] > y[j]) == (p[i] > p[j]))
    return c / t if t else np.nan, c, t


def scc_summary(frame: pd.DataFrame, score: np.ndarray) -> tuple[float, float]:
    work = frame.copy()
    work["_score"] = score
    vals, weights = [], []
    for _, g in work.groupby("source_norm"):
        s = spearmanr(g.pic50_model, g._score).correlation
        if np.isfinite(s):
            vals.append(float(s))
            weights.append(len(g))
    return float(np.mean(vals)), float(np.average(vals, weights=weights))


def regressor_pool():
    return {
        "XGB_centered": XGBRegressor(
            n_estimators=450, max_depth=4, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.75, reg_alpha=0.1,
            reg_lambda=1.2, objective="reg:squarederror",
            random_state=SEED, n_jobs=8,
        ),
        "LGB_centered": LGBMRegressor(
            n_estimators=450, num_leaves=23, max_depth=5,
            learning_rate=0.03, subsample=0.85, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=SEED,
            n_jobs=8, verbosity=-1,
        ),
        "Cat_centered": CatBoostRegressor(
            iterations=450, depth=5, learning_rate=0.035,
            l2_leaf_reg=5, loss_function="RMSE", random_seed=SEED,
            verbose=False, allow_writing_files=False, thread_count=8,
        ),
        "RF_centered": RandomForestRegressor(
            n_estimators=700, max_features=0.7, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1,
        ),
        "ET_centered": ExtraTreesRegressor(
            n_estimators=700, max_features=0.8, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1,
        ),
    }


def make_pair_data(frame: pd.DataFrame, X: np.ndarray, idx: np.ndarray):
    diffs, labels, pair_meta = [], [], []
    for source, g in frame.iloc[idx].groupby("source_norm"):
        ids = g.index.to_numpy()
        y = g.pic50_model.to_numpy()
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                if y[a] == y[b]:
                    continue
                d = X[ids[a]] - X[ids[b]]
                lab = int(y[a] > y[b])
                diffs.extend([d, -d])
                labels.extend([lab, 1 - lab])
                pair_meta.extend([(source, ids[a], ids[b]), (source, ids[b], ids[a])])
    return np.asarray(diffs, dtype=np.float32), np.asarray(labels), pair_meta


def score_items_from_pair_model(model, frame: pd.DataFrame, X: np.ndarray, idx: np.ndarray):
    scores = np.zeros(len(idx), dtype=float)
    pos = {orig: k for k, orig in enumerate(idx)}
    for _, g in frame.iloc[idx].groupby("source_norm"):
        ids = g.index.to_numpy()
        if len(ids) == 1:
            scores[pos[ids[0]]] = 0.5
            continue
        for i in ids:
            others = [j for j in ids if j != i]
            diff = np.asarray([X[i] - X[j] for j in others], dtype=np.float32)
            scores[pos[i]] = model.predict_proba(diff)[:, 1].mean()
    return scores


def main():
    frame = pd.read_csv(IN_DIR / "ranking_eligible_n4_predictions.csv")
    # Restore deterministic row index used by feature matrix.
    frame = frame.reset_index(drop=True)
    X = extract_sequence_features(frame.sequence.tolist())
    tr_idx = frame.index[frame.split == "train_source_group"].to_numpy()
    te_idx = frame.index[frame.split == "test_source_group"].to_numpy()
    train = frame.iloc[tr_idx].copy()
    groups = train.source_norm.to_numpy()
    cv = GroupKFold(n_splits=5)

    # Source-centered pIC50 target removes laboratory/source offsets.
    centered = frame.pic50_model - frame.groupby("source_norm").pic50_model.transform("mean")
    cv_rows = []
    oof_predictions: dict[str, np.ndarray] = {}
    for name, base in regressor_pool().items():
        print("CV", name)
        oof = np.full(len(tr_idx), np.nan)
        for a, b in cv.split(train, centered.iloc[tr_idx], groups=groups):
            model = clone(base)
            model.fit(X[tr_idx[a]], centered.iloc[tr_idx[a]])
            oof[b] = model.predict(X[tr_idx[b]])
        pa, c, t = pair_accuracy(train, oof)
        sm, sw = scc_summary(train, oof)
        cv_rows.append({
            "method": name, "CV_pair_accuracy": pa,
            "CV_SCC_macro": sm, "CV_SCC_weighted": sw,
            "pairs": t,
        })
        oof_predictions[name] = oof
        print(name, pa, sw)

    # Pairwise classifiers directly optimize order.
    pair_models = {
        "PairLogistic": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=0.1, max_iter=4000, class_weight="balanced",
                random_state=SEED,
            )),
        ]),
        "PairXGB": XGBClassifier(
            n_estimators=450, max_depth=4, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.75,
            reg_alpha=0.1, reg_lambda=1.2,
            objective="binary:logistic", eval_metric="logloss",
            random_state=SEED, n_jobs=8,
        ),
    }
    for name, base in pair_models.items():
        print("CV", name)
        item_oof = np.full(len(tr_idx), np.nan)
        # Map local train rows to original frame indices.
        for a, b in cv.split(train, train.rank_label, groups=groups):
            train_orig = tr_idx[a]
            val_orig = tr_idx[b]
            PX, Py, _ = make_pair_data(frame, X, train_orig)
            model = clone(base).fit(PX, Py)
            scores = score_items_from_pair_model(model, frame, X, val_orig)
            item_oof[b] = scores
        pa, c, t = pair_accuracy(train, item_oof)
        sm, sw = scc_summary(train, item_oof)
        cv_rows.append({
            "method": name, "CV_pair_accuracy": pa,
            "CV_SCC_macro": sm, "CV_SCC_weighted": sw,
            "pairs": t,
        })
        oof_predictions[name] = item_oof
        print(name, pa, sw)

    # Ensembles selected solely from train-group CV.
    cv_df = pd.DataFrame(cv_rows).sort_values(
        ["CV_pair_accuracy", "CV_SCC_weighted"], ascending=False
    )
    top3 = cv_df.head(3).method.tolist()
    ensemble_oof = np.mean([oof_predictions[x] for x in top3], axis=0)
    pa, _, t = pair_accuracy(train, ensemble_oof)
    sm, sw = scc_summary(train, ensemble_oof)
    cv_df = pd.concat([
        cv_df,
        pd.DataFrame([{
            "method": "TrainCV_top3_ensemble",
            "CV_pair_accuracy": pa,
            "CV_SCC_macro": sm,
            "CV_SCC_weighted": sw,
            "pairs": t,
        }]),
    ], ignore_index=True).sort_values(
        ["CV_pair_accuracy", "CV_SCC_weighted"], ascending=False
    )
    # Select best method on training CV only, including ensemble.
    best = cv_df.iloc[0].method
    print("Selected by train CV:", best, "top3", top3)

    fitted = {}
    test_scores = {}
    # Fit all alternatives once; report all frozen-test results transparently.
    for name, base in regressor_pool().items():
        model = clone(base).fit(X[tr_idx], centered.iloc[tr_idx])
        fitted[name] = model
        test_scores[name] = model.predict(X[te_idx])
    for name, base in pair_models.items():
        PX, Py, _ = make_pair_data(frame, X, tr_idx)
        model = clone(base).fit(PX, Py)
        fitted[name] = model
        test_scores[name] = score_items_from_pair_model(model, frame, X, te_idx)
    test_scores["TrainCV_top3_ensemble"] = np.mean(
        [test_scores[x] for x in top3], axis=0
    )

    test = frame.iloc[te_idx].copy()
    test_rows = []
    for name, score in test_scores.items():
        pa, c, t = pair_accuracy(test, score)
        sm, sw = scc_summary(test, score)
        test_rows.append({
            "method": name,
            "selected_by_train_CV": name == best,
            "test_pair_accuracy": pa,
            "pair_correct": c,
            "pair_total": t,
            "test_SCC_macro": sm,
            "test_SCC_weighted": sw,
            "global_SCC_control": spearmanr(test.pic50_model, score).correlation,
        })
        frame.loc[te_idx, f"{name}_score"] = score

    # Deployment score: train best component(s) on all sources after evaluation.
    all_idx = frame.index.to_numpy()
    all_centered = centered
    deploy_scores = None
    deploy_package = {"selected_by_train_CV": best, "top3": top3}
    if best == "TrainCV_top3_ensemble":
        all_component_scores = []
        all_models = {}
        for name in top3:
            if name in regressor_pool():
                model = clone(regressor_pool()[name]).fit(X, all_centered)
                score = model.predict(X)
            else:
                PX, Py, _ = make_pair_data(frame, X, all_idx)
                model = clone(pair_models[name]).fit(PX, Py)
                score = score_items_from_pair_model(model, frame, X, all_idx)
            all_models[name] = model
            all_component_scores.append(score)
        deploy_scores = np.mean(all_component_scores, axis=0)
        deploy_package["models"] = all_models
    elif best in regressor_pool():
        model = clone(regressor_pool()[best]).fit(X, all_centered)
        deploy_scores = model.predict(X)
        deploy_package["model"] = model
    else:
        PX, Py, _ = make_pair_data(frame, X, all_idx)
        model = clone(pair_models[best]).fit(PX, Py)
        deploy_scores = score_items_from_pair_model(model, frame, X, all_idx)
        deploy_package["model"] = model

    frame["optimized_final_deploy_score"] = deploy_scores
    frame["optimized_within_source_rank"] = frame.groupby("source_norm")[
        "optimized_final_deploy_score"
    ].rank(ascending=False, method="dense")

    cv_df.to_csv(OUT / "ranking_alternative_train_groupcv.csv", index=False, encoding="utf-8-sig")
    test_df = pd.DataFrame(test_rows).sort_values(
        "selected_by_train_CV", ascending=False
    )
    test_df.to_csv(OUT / "ranking_alternative_frozen_test.csv", index=False, encoding="utf-8-sig")
    frame.to_csv(OUT / "ranking_optimized_all_predictions.csv", index=False, encoding="utf-8-sig")
    with open(CKPT / "ranking_optimized_deploy.pkl", "wb") as f:
        pickle.dump(deploy_package, f)
    summary = {
        "selection_rule": "best within-source pair accuracy on training-source GroupCV only",
        "selected_method": best,
        "top3_for_ensemble": top3,
        "selected_test_metrics": next(x for x in test_rows if x["method"] == best),
        "all_test_results": test_rows,
        "warning": (
            "Frozen test was evaluated only after train-CV selection. Deployment model "
            "is retrained on all sources and is not the source of held-out metrics."
        ),
    }
    with open(OUT / "ranking_optimized_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
