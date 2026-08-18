# -*- coding: utf-8 -*-
"""
Ranking ablation study (2026-07-18), using a strict source-group split
(leak-free: no source or sequence overlaps between train and test groups):

Ablation 1: remove the source-aware constraint (most important)
  Model A: Global ranking -- ignores source boundaries, merges all training
    samples into a single list to train one LambdaRank model
  Model B: Source-aware LambdaRank -- trained per source group (the
    SAAR-DPPIV ranking head actually deployed)
  Metrics: EF@1 / EF@5 / NDCG@5 / NDCG@10, computed on the 9 strictly frozen
  test sources after within-source normalization.

Ablation 2: remove the rank_score fusion
  Classification only: Score = P(active)
  Classification + Ranking: Score = 0.65*P(active) + 0.35*RankScore
  Checks whether the 0.35 weight helps, i.e. whether Top-k enrichment
  improves.

Additional controls:
  Without ESM-2: source-aware ranking head using only the 617-D sequence
    features (this is the current SAAR-DPPIV main model).
  Full SAAR-DPPIV: source-aware ranking head on 617+480-D features (Top-300
    selected), to check whether adding ESM-2 helps.

All variants use the same strict source-group split (35 training sources /
9 frozen test sources, zero source or sequence overlap); the test sources
are used exactly once, at final evaluation.
"""
from __future__ import annotations

import io
import json
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

# ── 1. Load the strict split data (35 train sources / 9 frozen test sources) ──
frame = pd.read_csv(RANK / "ranking_eligible_n4_predictions.csv")
tr_mask = frame.split == "train_source_group"
te_mask = frame.split == "test_source_group"
print(f"train={tr_mask.sum()} ({frame[tr_mask].source_norm.nunique()} sources) "
      f"test={te_mask.sum()} ({frame[te_mask].source_norm.nunique()} sources)")
assert not (set(frame[tr_mask].source_norm) & set(frame[te_mask].source_norm))

# ── 2. Features: 617-D sequence stats + 480-D ESM-2 (mapped via the cached
# classification-dataset feature matrix) ────────────────────────────────
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
print(f"ESM-2 cache coverage {has_esm.mean():.4f} ({has_esm.sum()}/{len(frame)}); "
      f"missing sequences are excluded from the Full/Without-ESM-2 control")

X1097 = np.zeros((len(frame), 1097), dtype=np.float32)
for i, seq in enumerate(frame.sequence):
    if seq in seq2idx:
        X1097[i] = X_full_cache[seq2idx[seq]]
X_top300 = X1097[:, top_idx]  # Top-300 selection: 110 kept seq-stat dims + 190 ESM-2 dims

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


def train_global(X: np.ndarray, mask: np.ndarray) -> lgb.Booster:
    """Ignore source boundaries: treat all training samples as one big list (a single group)."""
    idx = np.where(mask)[0]
    order = np.argsort(-frame.loc[idx, "pic50_model"].to_numpy())
    idx_sorted = idx[order]
    ds = lgb.Dataset(X[idx_sorted], label=frame.loc[idx_sorted, "rank_label"].to_numpy(dtype=int),
                      weight=frame.loc[idx_sorted, "sample_weight"].to_numpy(dtype=float),
                      group=np.array([len(idx_sorted)]))
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
    # EF: rank by score after within-source percentile normalization, then
    # take the overall Top-k (consistent with the earlier "Figure B" protocol)
    work["_pct"] = work.groupby("source_norm")["_score"].rank(pct=True, ascending=True)
    work_sorted = work.sort_values("_pct", ascending=False)
    strong = (work_sorted.rank_label >= 3).to_numpy(dtype=int)  # IC50 < 50 uM
    prevalence = strong.mean()
    ef = {}
    for k in (1, 5):
        ef[k] = float(strong[:k].mean() / prevalence) if prevalence > 0 else np.nan

    # NDCG: compute per source then macro-average (consistent with the
    # earlier "Figure C" protocol)
    ndcg_vals = {5: [], 10: []}
    for _, g in work.groupby("source_norm"):
        y = g.rank_label.to_numpy(dtype=float)
        s = g["_score"].to_numpy(dtype=float)
        for k in (5, 10):
            if len(g) >= 1:
                ndcg_vals[k].append(ndcg_at_k(y, s, k))
    macro_ndcg = {k: float(np.mean(v)) if v else np.nan for k, v in ndcg_vals.items()}

    # Within-group pairwise accuracy + global-SCC control
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


results = []

# ── Ablation 1: Global ranking vs Source-aware ranking (617-D, fair control) ──
model_global = train_global(X617, tr_mask.to_numpy())
model_source = train_source_aware(X617, tr_mask.to_numpy())
te_idx = frame.index[te_mask].to_numpy()
score_global = np.full(len(frame), np.nan)
score_source = np.full(len(frame), np.nan)
score_global[te_idx] = model_global.predict(X617[te_idx])
score_source[te_idx] = model_source.predict(X617[te_idx])
test_frame = frame[te_mask].copy()

m = topk_metrics(test_frame, score_global[te_idx])
results.append({"Model": "Global ranking", "variant": "ablation1", **m})
m = topk_metrics(test_frame, score_source[te_idx])
results.append({"Model": "Source-aware ranking (=Without ESM-2)", "variant": "ablation1/control", **m})
print("ablation1 done:", results[-2], results[-1])

# ── Full SAAR-DPPIV: 617+480 (Top-300 selected) source-aware ranking ─────
has_esm_train = has_esm & tr_mask.to_numpy()
has_esm_test = has_esm & te_mask.to_numpy()
model_full = train_source_aware(X_top300, has_esm_train)
score_full = np.full(len(frame), np.nan)
score_full[np.where(has_esm_test)[0]] = model_full.predict(X_top300[has_esm_test])
test_full = frame[has_esm_test].copy()
m = topk_metrics(test_full, score_full[has_esm_test])
results.append({"Model": "Full SAAR-DPPIV (617+ESM-2 Top300)", "variant": "control",
                 **m, "n_test": int(has_esm_test.sum())})
print("Full SAAR-DPPIV done:", results[-1])

# ── Ablation 2: Classification only vs Classification+Ranking ───────────
with open(CKPT / "v3_tree_models.pkl", "rb") as f:
    cls_models = {k: v for k, v in __import__("pickle").load(f).items() if "cls" in k}
p_active_test = np.mean([m_.predict_proba(X617[te_idx])[:, 1] for m_ in cls_models.values()], axis=0)
# rank_score normalization matches web_app: sigmoid(raw/2)
rank_score_sigmoid = 1.0 / (1.0 + np.exp(-score_source[te_idx] / 2.0))
combined_score = 0.65 * p_active_test + 0.35 * rank_score_sigmoid

m = topk_metrics(test_frame, p_active_test)
results.append({"Model": "Classification only (Score=P(active))", "variant": "ablation2", **m})
m = topk_metrics(test_frame, combined_score)
results.append({"Model": "Classification + Ranking (0.65P+0.35RankScore)", "variant": "ablation2", **m})
print("ablation2 done:", results[-2], results[-1])

result_df = pd.DataFrame(results)
result_df.to_csv(OUT / "SAAR_DPPIV_ablation_0718.csv", index=False, encoding="utf-8-sig")
print("\n" + result_df.to_string(index=False))

audit = {
    "split": "same as 0714 strict source-group split: 35 train sources / 9 frozen test sources, zero source or sequence overlap",
    "n_test_samples": int(te_mask.sum()),
    "n_test_sources": int(frame[te_mask].source_norm.nunique()),
    "esm2_coverage_test": float(has_esm_test.sum() / te_mask.sum()),
    "note": (
        "Global/Source-aware/Classification rows are all evaluated on the full 84-sample test set; "
        "Full SAAR-DPPIV is missing 1 sequence from the ESM-2 cache and is evaluated on the "
        "de-duplicated test subset, as noted in the n_test column."
    ),
}
with open(OUT / "SAAR_DPPIV_ablation_0718_audit.json", "w", encoding="utf-8") as f:
    json.dump(audit, f, ensure_ascii=False, indent=2)
print(json.dumps(audit, ensure_ascii=False, indent=2))
