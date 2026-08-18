"""
Ablation utilities for the SAAR-DPPIV ranking + fusion pipeline.

These are the two ablations reported in the manuscript's ranking section:

  Ablation 1 — Global vs. source-aware ranking
      Model A ("Global"): ignore source boundaries, train LambdaRank on the
                           whole training pool as a single list.
      Model B ("Source-aware"): train LambdaRank with one group per
                           literature source (SAAR-DPPIV's actual design).

  Ablation 2 — Does the rank-score fusion help over classification alone?
      "Classification only": Score = P(active)
      "Classification + Ranking": Score = 0.65 * P(active) + 0.35 * RankScore

Both ablations are evaluated with the same Top-k enrichment / NDCG metrics
used throughout the manuscript (percentile-normalised within source for
EF@k; macro-averaged per-source NDCG@k). This module operates on
already-extracted feature matrices / dataframes so it has no dependency on
the (removed) legacy multi-branch deep-learning trainer — only LightGBM and
the shipped tree-ensemble classifiers are required.

For the exact frozen 35-train-source / 9-test-source split and cached
features used to produce the manuscript's ablation table, see the archival
script in ``src/benchmark_comparison/`` (it additionally needs the
intermediate ESM-2 feature cache, which is not shipped in this slim
package — see that folder's README).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

LAMBDARANK_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5, 10],
    "learning_rate": 0.04, "num_leaves": 31, "max_depth": -1,
    "min_data_in_leaf": 4, "feature_fraction": 0.85, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l1": 0.05, "lambda_l2": 0.8,
    "verbosity": -1, "seed": 42,
}
NUM_BOOST_ROUND = 300


def train_source_aware_ranker(frame: pd.DataFrame, X: np.ndarray, mask: np.ndarray):
    """LambdaRank with one group per ``source_norm`` (SAAR-DPPIV's design)."""
    import lightgbm as lgb

    sub = frame[mask].copy()
    sub["_orig_idx"] = np.where(mask)[0]
    sub = sub.sort_values(["source_norm", "pic50_model"]).reset_index(drop=True)
    idx = sub["_orig_idx"].to_numpy(dtype=int)
    groups = sub.groupby("source_norm", sort=False).size().to_numpy(dtype=int)
    ds = lgb.Dataset(
        X[idx], label=sub["rank_label"].to_numpy(dtype=int),
        weight=sub["sample_weight"].to_numpy(dtype=float), group=groups,
    )
    return lgb.train(LAMBDARANK_PARAMS, ds, num_boost_round=NUM_BOOST_ROUND)


def train_global_ranker(frame: pd.DataFrame, X: np.ndarray, mask: np.ndarray):
    """Ablation: LambdaRank with source boundaries ignored (single group)."""
    import lightgbm as lgb

    idx = np.where(mask)[0]
    order = np.argsort(-frame.loc[idx, "pic50_model"].to_numpy())
    idx_sorted = idx[order]
    ds = lgb.Dataset(
        X[idx_sorted], label=frame.loc[idx_sorted, "rank_label"].to_numpy(dtype=int),
        weight=frame.loc[idx_sorted, "sample_weight"].to_numpy(dtype=float),
        group=np.array([len(idx_sorted)]),
    )
    return lgb.train(LAMBDARANK_PARAMS, ds, num_boost_round=NUM_BOOST_ROUND)


def ndcg_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    k = min(k, len(y))
    order = np.argsort(score)[::-1][:k]
    ideal = np.argsort(y)[::-1][:k]
    gains = np.power(2.0, y) - 1.0
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains[order] / discounts)
    idcg = np.sum(gains[ideal] / discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def topk_enrichment_metrics(test: pd.DataFrame, score: np.ndarray) -> dict:
    """EF@1 / EF@5 (pooled, within-source percentile-normalised), macro NDCG@5/10,
    within-source pairwise accuracy, and a global Spearman correlation (control only).
    """
    work = test.reset_index(drop=True).copy()
    work["_score"] = score
    work["_pct"] = work.groupby("source_norm")["_score"].rank(pct=True, ascending=True)
    work_sorted = work.sort_values("_pct", ascending=False)
    strong = (work_sorted.rank_label >= 3).to_numpy(dtype=int)  # IC50 < 50 uM
    prevalence = strong.mean()
    ef = {k: float(strong[:k].mean() / prevalence) if prevalence > 0 else np.nan for k in (1, 5)}

    ndcg_vals = {5: [], 10: []}
    for _, g in work.groupby("source_norm"):
        y = g.rank_label.to_numpy(dtype=float)
        s = g["_score"].to_numpy(dtype=float)
        for k in (5, 10):
            ndcg_vals[k].append(ndcg_at_k(y, s, k))
    macro_ndcg = {k: float(np.mean(v)) if v else np.nan for k, v in ndcg_vals.items()}

    correct = total = 0
    for _, g in work.groupby("source_norm"):
        y = g.pic50_model.to_numpy()
        s = g["_score"].to_numpy(dtype=float)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if y[i] == y[j]:
                    continue
                total += 1
                correct += int((y[i] > y[j]) == (s[i] > s[j]))

    return {
        "EF@1": ef[1], "EF@5": ef[5],
        "NDCG@5": macro_ndcg[5], "NDCG@10": macro_ndcg[10],
        "PairAcc": correct / total if total else np.nan,
        "Global_SCC": float(spearmanr(test.pic50_model, score).correlation),
    }


def run_ranking_ablation(frame: pd.DataFrame, X617: np.ndarray, X_top300: np.ndarray | None,
                          has_esm: np.ndarray | None, tr_mask: np.ndarray, te_mask: np.ndarray,
                          cls_p_active_test: np.ndarray | None = None,
                          w_pactive: float = 0.65, w_rank: float = 0.35) -> pd.DataFrame:
    """Reproduce the manuscript's ranking + fusion ablation table.

    Parameters
    ----------
    frame : one row per IC50 peptide, columns ``source_norm``, ``pic50_model``,
        ``rank_label`` (0-3, see ``ranking_lambdarank.activity_level``),
        ``sample_weight``.
    X617 : 617-D sequence feature matrix aligned with ``frame``.
    X_top300, has_esm : optional Top-300 [617+ESM-2] features and an ESM-2
        cache-coverage boolean mask, for the "Full SAAR-DPPIV (+ESM-2)" row.
    cls_p_active_test : optional classification P(active) for the test rows
        (same order as ``frame[te_mask]``), for the classification-fusion ablation.
    """
    rows = []
    te_idx = frame.index[te_mask].to_numpy()
    test_frame = frame[te_mask].copy()

    model_global = train_global_ranker(frame, X617, tr_mask)
    model_source = train_source_aware_ranker(frame, X617, tr_mask)
    score_global = model_global.predict(X617[te_idx])
    score_source = model_source.predict(X617[te_idx])

    rows.append({"Model": "Global ranking (source boundary ignored)",
                 **topk_enrichment_metrics(test_frame, score_global)})
    rows.append({"Model": "Source-aware ranking (617-D, no ESM-2)",
                 **topk_enrichment_metrics(test_frame, score_source)})

    if X_top300 is not None and has_esm is not None:
        has_esm_train = has_esm & tr_mask
        has_esm_test = has_esm & te_mask
        model_full = train_source_aware_ranker(frame, X_top300, has_esm_train)
        test_full = frame[has_esm_test].copy()
        score_full = model_full.predict(X_top300[has_esm_test.nonzero()[0]])
        rows.append({"Model": "Full SAAR-DPPIV (617 + ESM-2 Top-300)",
                     "n_test": int(has_esm_test.sum()),
                     **topk_enrichment_metrics(test_full, score_full)})

    if cls_p_active_test is not None:
        rank_score_sigmoid = 1.0 / (1.0 + np.exp(-score_source / 2.0))
        combined = w_pactive * cls_p_active_test + w_rank * rank_score_sigmoid
        rows.append({"Model": "Classification only (Score = P(active))",
                     **topk_enrichment_metrics(test_frame, cls_p_active_test)})
        rows.append({"Model": f"Classification + Ranking ({w_pactive:.2f}*P + {w_rank:.2f}*RankScore)",
                     **topk_enrichment_metrics(test_frame, combined)})

    return pd.DataFrame(rows)
