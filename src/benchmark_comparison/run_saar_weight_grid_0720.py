# -*- coding: utf-8 -*-
"""
Follow-up question (2026-07-20): the Classification+Ranking (0.65/0.35)
fusion does not clearly beat pure classification -- should the fusion
formula, or the weight, be adjusted?

Under the exact same strict source-group split used for the 0718 ablation
(35 training sources / 9 frozen test sources, zero source or sequence
overlap), this script runs a weight grid search over
combined_score = (1-w)*P(active) + w*RankScore for
w = 0, 0.05, 0.10, ..., 1.00, and reports whether EF@5 / NDCG@10 / PairAcc
have a better weight point than 0.35 -- reported honestly, with no
assumption that a better weight will necessarily be found.
"""
from __future__ import annotations

import io
import json
import pickle
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# NOTE (archival script, see src/benchmark_comparison/README.md): this path
# pointed at the original development machine and is not portable. Edit it
# to point at your own working directory before running.
ROOT = Path(r"C:\path\to\your\workdir")
PROC = ROOT / "processed"
RANK = PROC / "ranking_source_grouped_0714"
CKPT = ROOT / "checkpoints"
OUT = RANK
SEED = 42

sys.path.insert(0, str(ROOT))
from src.features.sequence_features import extract_sequence_features

# ── 1. Load the strict split data (identical to 0718) ──────────────────
frame = pd.read_csv(RANK / "ranking_eligible_n4_predictions.csv")
tr_mask = frame.split == "train_source_group"
te_mask = frame.split == "test_source_group"
print(f"train={tr_mask.sum()} ({frame[tr_mask].source_norm.nunique()} sources) "
      f"test={te_mask.sum()} ({frame[te_mask].source_norm.nunique()} sources)")
assert not (set(frame[tr_mask].source_norm) & set(frame[te_mask].source_norm))

X617 = extract_sequence_features(frame.sequence.tolist())

PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5, 10],
    "learning_rate": 0.04, "num_leaves": 31, "max_depth": -1,
    "min_data_in_leaf": 4, "feature_fraction": 0.85, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l1": 0.05, "lambda_l2": 0.8,
    "verbosity": -1, "seed": SEED, "feature_fraction_seed": SEED, "bagging_seed": SEED,
}
ROUNDS = 300


def train_source_aware(X: np.ndarray, mask: np.ndarray) -> lgb.Booster:
    sub = frame[mask].copy()
    sub["_orig_idx"] = np.where(mask)[0]
    sub = sub.sort_values(["source_norm", "pic50_model"]).reset_index(drop=True)
    idx = sub["_orig_idx"].to_numpy(dtype=int)
    groups = sub.groupby("source_norm", sort=False).size().to_numpy(dtype=int)
    ds = lgb.Dataset(X[idx], label=sub["rank_label"].to_numpy(dtype=int),
                      weight=sub["sample_weight"].to_numpy(dtype=float), group=groups)
    return lgb.train(PARAMS, ds, num_boost_round=ROUNDS)


def ndcg_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    k = min(k, len(y))
    order = np.argsort(score)[::-1][:k]
    ideal = np.argsort(y)[::-1][:k]
    gains = np.power(2.0, y) - 1.0
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains[order] / discounts)
    idcg = np.sum(gains[ideal] / discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def topk_metrics(test: pd.DataFrame, score: np.ndarray) -> dict:
    test = test.reset_index(drop=True)
    work = test.copy()
    work["_score"] = score
    work["_pct"] = work.groupby("source_norm")["_score"].rank(pct=True, ascending=True)
    work_sorted = work.sort_values("_pct", ascending=False)
    strong = (work_sorted.rank_label >= 3).to_numpy(dtype=int)
    prevalence = strong.mean()
    ef = {}
    for k in (1, 5):
        ef[k] = float(strong[:k].mean() / prevalence) if prevalence > 0 else np.nan

    ndcg_vals = {5: [], 10: []}
    for _, g in work.groupby("source_norm"):
        y = g.rank_label.to_numpy(dtype=float)
        s = g["_score"].to_numpy(dtype=float)
        for k in (5, 10):
            if len(g) >= 1:
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


te_idx = frame.index[te_mask].to_numpy()
test_frame = frame[te_mask].copy()

# ── 2. Classification probability (same as 0718: average of the cls models
# in v3_tree_models.pkl) ────────────────────────────────────────────────
with open(CKPT / "v3_tree_models.pkl", "rb") as f:
    cls_models = {k: v for k, v in pickle.load(f).items() if "cls" in k}
p_active_test = np.mean([m_.predict_proba(X617[te_idx])[:, 1] for m_ in cls_models.values()], axis=0)

# ── 3. Ranking score (source-aware LambdaRank, sigmoid normalization as in 0718) ──
model_source = train_source_aware(X617, tr_mask.to_numpy())
score_source = model_source.predict(X617[te_idx])
rank_score_sigmoid = 1.0 / (1.0 + np.exp(-score_source / 2.0))

# ── 4. Weight grid search ──────────────────────────────────────────────
weights = [round(w, 2) for w in np.arange(0.0, 1.01, 0.05)]
rows = []
for w in weights:
    combined = (1 - w) * p_active_test + w * rank_score_sigmoid
    m = topk_metrics(test_frame, combined)
    rows.append({"weight_rank": w, **m})

grid_df = pd.DataFrame(rows)
grid_df.to_csv(OUT / "SAAR_DPPIV_weight_grid_0720.csv", index=False, encoding="utf-8-sig")

best_ef5 = grid_df.loc[grid_df["EF@5"].idxmax()]
best_ndcg10 = grid_df.loc[grid_df["NDCG@10"].idxmax()]
best_pairacc = grid_df.loc[grid_df["PairAcc"].idxmax()]
cur = grid_df[grid_df.weight_rank == 0.35].iloc[0]

print("\n" + grid_df.to_string(index=False))
print(f"\nCurrent weight w=0.35: EF@5={cur['EF@5']:.3f} NDCG@10={cur['NDCG@10']:.4f} PairAcc={cur['PairAcc']:.4f}")
print(f"Best EF@5 point:    w={best_ef5.weight_rank:.2f} EF@5={best_ef5['EF@5']:.3f}")
print(f"Best NDCG@10 point: w={best_ndcg10.weight_rank:.2f} NDCG@10={best_ndcg10['NDCG@10']:.4f}")
print(f"Best PairAcc point: w={best_pairacc.weight_rank:.2f} PairAcc={best_pairacc['PairAcc']:.4f}")

summary = {
    "split": "same as 0714/0718 strict source-group split: 35 train sources / 9 frozen test sources, zero source or sequence overlap",
    "n_test": int(te_mask.sum()),
    "weight_grid": weights,
    "current_weight_0.35": {k: cur[k] for k in ["EF@1", "EF@5", "NDCG@5", "NDCG@10", "PairAcc", "Global_SCC"]},
    "best_by_EF@5": {"weight": float(best_ef5.weight_rank), "EF@5": float(best_ef5["EF@5"])},
    "best_by_NDCG@10": {"weight": float(best_ndcg10.weight_rank), "NDCG@10": float(best_ndcg10["NDCG@10"])},
    "best_by_PairAcc": {"weight": float(best_pairacc.weight_rank), "PairAcc": float(best_pairacc["PairAcc"])},
}
with open(OUT / "SAAR_DPPIV_weight_grid_0720_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
