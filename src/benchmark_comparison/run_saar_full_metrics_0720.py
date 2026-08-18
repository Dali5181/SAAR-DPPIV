# -*- coding: utf-8 -*-
"""
Follow-up review round (2026-07-20), addressing four questions raised about
the ablation table:

1) Fill in real numbers in the requested table format (Model | EF@5 |
   NDCG@10, with "Without ESM-2" as its own row).
2) Whether "within-source EF" is the metric actually used -- a third-party
   formula EF_k = (Hits_topk/k) / (TotalHits/N) was proposed for comparison.
   Cross-checking shows this is mathematically equivalent to the
   "within-source percentile normalization, then global pooled" EF we
   already compute; it is NOT literally "compute per source, then
   macro-average". This script additionally computes a genuine
   "within-source macro-average" version as a side-by-side control.
3) Hit@1/Hit@5/Hit@10 values -- computed here as well (both the pooled and
   the within-source macro-average variants).
4) Whether trying more fusion-weight combinations can push EF@5/NDCG@10
   higher -- a full 0-1 weight grid search was already run earlier the same
   day (SAAR_DPPIV_weight_grid_0720.csv) and found no clearly better point;
   training is not repeated here, the conclusion is simply cited in the
   summary table.

All variants reuse the same 0714 strict source-group split (35 training
sources / 9 frozen test sources, zero source or sequence overlap) and the
exact same model-training configuration as the earlier runs on this date,
so the numbers are comparable and reproducible.
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

frame = pd.read_csv(RANK / "ranking_eligible_n4_predictions.csv")
tr_mask = frame.split == "train_source_group"
te_mask = frame.split == "test_source_group"
assert not (set(frame[tr_mask].source_norm) & set(frame[te_mask].source_norm))

X617 = extract_sequence_features(frame.sequence.tolist())

train_df = pd.read_csv(PROC / "iDPP-IV-CV_v3.csv").dropna(subset=["sequence"])
test_df = pd.read_csv(PROC / "iDPP-IV-TS_v3.csv").dropna(subset=["sequence"])
pat = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
train_df = train_df[train_df["sequence"].str.match(pat)].reset_index(drop=True)
test_df = test_df[test_df["sequence"].str.match(pat)].reset_index(drop=True)
seq2idx = {s: i for i, s in enumerate(train_df["sequence"].tolist() + test_df["sequence"].tolist())}
X_full_cache = np.vstack([
    np.load(PROC / "v4_X_full_train.npy"),
    np.load(PROC / "v4_X_full_test.npy"),
])
top_idx = np.load(PROC / "v4_top300_idx.npy")
has_esm = frame.sequence.isin(seq2idx).to_numpy()

X1097 = np.zeros((len(frame), 1097), dtype=np.float32)
for i, seq in enumerate(frame.sequence):
    if seq in seq2idx:
        X1097[i] = X_full_cache[seq2idx[seq]]
X_top300 = X1097[:, top_idx]

PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5, 10],
    "learning_rate": 0.04, "num_leaves": 31, "max_depth": -1,
    "min_data_in_leaf": 4, "feature_fraction": 0.85, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l1": 0.05, "lambda_l2": 0.8,
    "verbosity": -1, "seed": SEED, "feature_fraction_seed": SEED, "bagging_seed": SEED,
}
ROUNDS = 300


def train_source_aware(X, mask):
    sub = frame[mask].copy()
    sub["_orig_idx"] = np.where(mask)[0]
    sub = sub.sort_values(["source_norm", "pic50_model"]).reset_index(drop=True)
    idx = sub["_orig_idx"].to_numpy(dtype=int)
    groups = sub.groupby("source_norm", sort=False).size().to_numpy(dtype=int)
    ds = lgb.Dataset(X[idx], label=sub["rank_label"].to_numpy(dtype=int),
                      weight=sub["sample_weight"].to_numpy(dtype=float), group=groups)
    return lgb.train(PARAMS, ds, num_boost_round=ROUNDS)


def train_global(X, mask):
    idx = np.where(mask)[0]
    order = np.argsort(-frame.loc[idx, "pic50_model"].to_numpy())
    idx_sorted = idx[order]
    ds = lgb.Dataset(X[idx_sorted], label=frame.loc[idx_sorted, "rank_label"].to_numpy(dtype=int),
                      weight=frame.loc[idx_sorted, "sample_weight"].to_numpy(dtype=float),
                      group=np.array([len(idx_sorted)]))
    return lgb.train(PARAMS, ds, num_boost_round=ROUNDS)


def ndcg_at_k(y, score, k):
    k = min(k, len(y))
    order = np.argsort(score)[::-1][:k]
    ideal = np.argsort(y)[::-1][:k]
    gains = np.power(2.0, y) - 1.0
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains[order] / discounts)
    idcg = np.sum(gains[ideal] / discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def pooled_metrics(test: pd.DataFrame, score: np.ndarray, ks=(1, 5, 10)) -> dict:
    """Existing protocol: within-source percentile normalization, then EF/Hit
    computed on the globally pooled ranking (mathematically equivalent to the
    third-party formula)."""
    work = test.reset_index(drop=True).copy()
    work["_score"] = score
    work["_pct"] = work.groupby("source_norm")["_score"].rank(pct=True, ascending=True)
    work_sorted = work.sort_values("_pct", ascending=False)
    strong = (work_sorted.rank_label >= 3).to_numpy(dtype=int)
    prevalence = strong.mean()
    out = {}
    for k in ks:
        kk = min(k, len(strong))
        hit_rate = strong[:kk].mean()
        out[f"Hit@{k}(pooled)"] = float(hit_rate)
        out[f"EF@{k}(pooled)"] = float(hit_rate / prevalence) if prevalence > 0 else np.nan

    ndcg_vals = {5: [], 10: []}
    for _, g in work.groupby("source_norm"):
        y = g.rank_label.to_numpy(dtype=float)
        s = g["_score"].to_numpy(dtype=float)
        for k in (5, 10):
            ndcg_vals[k].append(ndcg_at_k(y, s, k))
    out["NDCG@5"] = float(np.mean(ndcg_vals[5]))
    out["NDCG@10"] = float(np.mean(ndcg_vals[10]))

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
    out["PairAcc"] = correct / total if total else np.nan
    out["Global_SCC"] = float(spearmanr(test.pic50_model, score).correlation)
    return out


def within_source_macro_metrics(test: pd.DataFrame, score: np.ndarray, ks=(1, 5, 10)) -> dict:
    """A genuine "compute EF/Hit per source, then macro-average" variant
    (matches the third-party wording literally, not what its formula
    actually computes)."""
    work = test.reset_index(drop=True).copy()
    work["_score"] = score
    per_source = {k: [] for k in ks}
    for _, g in work.groupby("source_norm"):
        strong = (g.rank_label.to_numpy() >= 3).astype(int)
        prevalence = strong.mean()
        order = np.argsort(-g["_score"].to_numpy())
        strong_sorted = strong[order]
        for k in ks:
            kk = min(k, len(strong_sorted))
            hit_rate = strong_sorted[:kk].mean()
            ef = hit_rate / prevalence if prevalence > 0 else np.nan
            per_source[k].append(ef)
    return {f"EF@{k}(within-source macro)": float(np.nanmean(v)) for k, v in per_source.items()}


te_idx = frame.index[te_mask].to_numpy()
test_frame = frame[te_mask].copy()

model_global = train_global(X617, tr_mask.to_numpy())
model_source = train_source_aware(X617, tr_mask.to_numpy())
score_global = model_global.predict(X617[te_idx])
score_source = model_source.predict(X617[te_idx])

has_esm_train = has_esm & tr_mask.to_numpy()
has_esm_test = has_esm & te_mask.to_numpy()
model_full = train_source_aware(X_top300, has_esm_train)
score_full = model_full.predict(X_top300[has_esm_test])
test_full = frame[has_esm_test].copy()

with open(CKPT / "v3_tree_models.pkl", "rb") as f:
    cls_models = {k: v for k, v in pickle.load(f).items() if "cls" in k}
p_active_test = np.mean([m_.predict_proba(X617[te_idx])[:, 1] for m_ in cls_models.values()], axis=0)
rank_score_sigmoid = 1.0 / (1.0 + np.exp(-score_source / 2.0))
combined_score = 0.65 * p_active_test + 0.35 * rank_score_sigmoid

variants = [
    ("Global ranking", test_frame, score_global),
    ("Source-aware ranking (=Without ESM-2)", test_frame, score_source),
    ("Classification only (Score=P(active))", test_frame, p_active_test),
    ("Classification + Ranking (0.65P+0.35RankScore)", test_frame, combined_score),
    ("Full SAAR-DPPIV (617+ESM-2 Top300)", test_full, score_full),
]

rows = []
for name, tf, sc in variants:
    m1 = pooled_metrics(tf, sc)
    m2 = within_source_macro_metrics(tf, sc)
    rows.append({"Model": name, "n_test": len(tf), **m1, **m2})

result = pd.DataFrame(rows)
result.to_csv(OUT / "SAAR_DPPIV_full_metrics_0720.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200)
print(result.to_string(index=False))

# Compact 6-row table in the requested Model|EF@5|NDCG@10 format (Source-aware
# and Without ESM-2 each get a row, but are clearly noted as the same experiment).
display6 = pd.DataFrame([
    {"Model": "Global ranking", "EF@5": result.loc[0, "EF@5(pooled)"], "NDCG@10": result.loc[0, "NDCG@10"]},
    {"Model": "Source-aware ranking", "EF@5": result.loc[1, "EF@5(pooled)"], "NDCG@10": result.loc[1, "NDCG@10"]},
    {"Model": "Classification only", "EF@5": result.loc[2, "EF@5(pooled)"], "NDCG@10": result.loc[2, "NDCG@10"]},
    {"Model": "Classification + Ranking", "EF@5": result.loc[3, "EF@5(pooled)"], "NDCG@10": result.loc[3, "NDCG@10"]},
    {"Model": "Without ESM-2 (=Source-aware, same experiment)",
     "EF@5": result.loc[1, "EF@5(pooled)"], "NDCG@10": result.loc[1, "NDCG@10"]},
    {"Model": "Full SAAR-DPPIV", "EF@5": result.loc[4, "EF@5(pooled)"], "NDCG@10": result.loc[4, "NDCG@10"]},
])
display6.to_csv(OUT / "SAAR_DPPIV_display6_0720.csv", index=False, encoding="utf-8-sig")
print("\nCompact 6-row table:\n" + display6.to_string(index=False))

best_ef5_row = display6.loc[display6["EF@5"].idxmax(), "Model"]
best_ndcg_row = display6.loc[display6["NDCG@10"].idxmax(), "Model"]
print(f"\nActual best: highest EF@5={best_ef5_row}({display6['EF@5'].max():.2f}), "
      f"highest NDCG@10={best_ndcg_row}({display6['NDCG@10'].max():.4f})")
print("Note: the reference figure pre-labelled Full SAAR-DPPIV as best; this does not "
      "match the actual result of this strict test and must be reported honestly.")
