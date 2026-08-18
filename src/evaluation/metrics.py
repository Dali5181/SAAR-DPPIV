"""
Evaluation metrics for DPP-IV inhibitor prediction.
Classification, regression, and enrichment-factor metrics.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    r2_score,
    mean_squared_error,
)
from scipy.stats import kendalltau, pearsonr, spearmanr


def enrichment_factor(y_true: np.ndarray, y_scores: np.ndarray,
                      fraction: float) -> float:
    """Enrichment factor at a given top-fraction (e.g. 0.01 for EF@1%)."""
    n = len(y_true)
    n_top = max(int(n * fraction), 1)
    total_pos = int(y_true.sum())
    if total_pos == 0 or n_top == 0:
        return 0.0
    order = np.argsort(-y_scores)
    hits = int(y_true[order][:n_top].sum())
    return (hits / n_top) / (total_pos / n)


def ndcg_at_k(y_true_score: np.ndarray, y_pred_score: np.ndarray,
              k: int = 10) -> float:
    """Normalized discounted cumulative gain for continuous activity labels."""
    y = np.asarray(y_true_score, dtype=float)
    s = np.asarray(y_pred_score, dtype=float)
    if y.size == 0 or s.size == 0:
        return 0.0

    mask = np.isfinite(y) & np.isfinite(s)
    y, s = y[mask], s[mask]
    if y.size == 0:
        return 0.0

    k = int(min(max(k, 1), y.size))
    rel = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-12)
    order = np.argsort(s)[::-1][:k]
    ideal = np.argsort(rel)[::-1][:k]
    discount = np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(rel[order] / discount))
    idcg = float(np.sum(rel[ideal] / discount))
    return dcg / idcg if idcg > 0 else 0.0


def topk_activity_stats(y_true_score: np.ndarray, y_pred_score: np.ndarray,
                        k: int = 10, hit_quantile: float = 0.75) -> dict:
    """Top-k activity enrichment statistics for pIC50-style ranking labels."""
    y = np.asarray(y_true_score, dtype=float)
    s = np.asarray(y_pred_score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y, s = y[mask], s[mask]
    if y.size == 0:
        return {
            f"top{k}_mean": 0.0,
            f"top{k}_hit_q{int(hit_quantile * 100)}": 0.0,
            f"top{k}_overlap": 0.0,
        }

    k = int(min(max(k, 1), y.size))
    pred_top = np.argsort(s)[::-1][:k]
    ideal_top = np.argsort(y)[::-1][:k]
    threshold = float(np.quantile(y, hit_quantile))
    return {
        f"top{k}_mean": float(np.mean(y[pred_top])),
        f"top{k}_hit_q{int(hit_quantile * 100)}": float(np.mean(y[pred_top] >= threshold)),
        f"top{k}_overlap": float(len(set(pred_top).intersection(ideal_top)) / k),
    }


def compute_ranking_metrics(
    y_true_score: np.ndarray,
    y_pred_score: np.ndarray,
    ks: tuple[int, ...] = (5, 10, 20),
    hit_quantile: float = 0.75,
) -> dict:
    """Compute rank-correlation, NDCG, and top-k enrichment metrics."""
    y = np.asarray(y_true_score, dtype=float)
    s = np.asarray(y_pred_score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y, s = y[mask], s[mask]

    metrics = {"n_rank": int(y.size)}
    if y.size < 2:
        metrics.update({"spearman": 0.0, "kendall": 0.0})
        for k in ks:
            metrics[f"ndcg@{k}"] = 0.0
            metrics.update(topk_activity_stats(y, s, k, hit_quantile))
        return metrics

    scc = spearmanr(y, s).correlation
    kt = kendalltau(y, s).correlation
    metrics["spearman"] = float(0.0 if np.isnan(scc) else scc)
    metrics["kendall"] = float(0.0 if np.isnan(kt) else kt)
    for k in ks:
        metrics[f"ndcg@{k}"] = ndcg_at_k(y, s, k)
        metrics.update(topk_activity_stats(y, s, k, hit_quantile))
    return metrics


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_reg: np.ndarray | None = None,
    y_true_reg: np.ndarray | None = None,
) -> dict:
    """Compute full suite of classification and (optionally) regression metrics.

    Parameters
    ----------
    y_true : (n,) binary labels
    y_pred_proba : (n,) predicted positive-class probabilities
    y_pred_reg : (n,) predicted pIC50, optional
    y_true_reg : (n,) true pIC50, optional

    Returns
    -------
    dict  — metric_name -> float (confusion_matrix -> list-of-lists)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    y_pred_cls = (y_pred_proba >= 0.5).astype(int)

    m: dict = {}

    # --- Classification --------------------------------------------------
    m["accuracy"] = float(accuracy_score(y_true, y_pred_cls))
    m["precision"] = float(precision_score(y_true, y_pred_cls, zero_division=0))
    m["recall"] = float(recall_score(y_true, y_pred_cls, zero_division=0))
    m["f1"] = float(f1_score(y_true, y_pred_cls, zero_division=0))

    try:
        m["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
    except ValueError:
        m["roc_auc"] = 0.0

    try:
        m["pr_auc"] = float(average_precision_score(y_true, y_pred_proba))
    except ValueError:
        m["pr_auc"] = 0.0

    # --- Enrichment factors -----------------------------------------------
    for frac in (0.01, 0.05, 0.10):
        m[f"ef_{int(frac * 100)}pct"] = enrichment_factor(y_true, y_pred_proba, frac)

    # --- Confusion matrix -------------------------------------------------
    m["confusion_matrix"] = confusion_matrix(y_true, y_pred_cls, labels=[0, 1]).tolist()

    # --- Regression -------------------------------------------------------
    if y_pred_reg is not None and y_true_reg is not None:
        yr = np.asarray(y_true_reg, dtype=float)
        yp = np.asarray(y_pred_reg, dtype=float)
        mask = ~np.isnan(yr)
        if mask.sum() > 1:
            yr_m, yp_m = yr[mask], yp[mask]
            m["rmse"] = float(np.sqrt(mean_squared_error(yr_m, yp_m)))
            m["r2"] = float(r2_score(yr_m, yp_m))
            pcc, pval = pearsonr(yr_m, yp_m)
            m["pcc"] = float(pcc)
            m["pcc_pvalue"] = float(pval)

    return m
