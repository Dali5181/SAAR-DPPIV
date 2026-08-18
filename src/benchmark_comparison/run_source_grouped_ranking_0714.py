# -*- coding: utf-8 -*-
"""
IC50 ranking experiment with a strict source-group split (2026-07-14).

Rules:
1. Peptides from the same literature source never cross the train/test split.
2. Of the original 615 rows with a numeric IC50, only sequences made of the
   20 standard amino acids enter modelling (609 rows).
3. The main ranking experiment uses sources with >=4 same-source samples;
   the remaining rows are still documented in the exported table.
4. Hyperparameters are selected via GroupKFold within the training sources
   only; the frozen test sources are evaluated exactly once.
5. Both an "evaluation model" and a "deployment model" (retrained on all
   eligible sources) are saved.
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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# NOTE (archival script, see src/benchmark_comparison/README.md): this path
# pointed at the original development machine and is not portable. Edit it
# to point at your own working directory before running.
ROOT = Path(r"C:\path\to\your\workdir")
RAW = ROOT / "raw_data" / "positive_ranking_ic50.xlsx"
OUT = ROOT / "processed" / "ranking_source_grouped_0714"
CKPT = ROOT / "checkpoints" / "ranking_source_grouped_0714"
OUT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from src.features.sequence_features import extract_sequence_features

SEED = 42
STANDARD = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


def normalize_source(value: str) -> str:
    s = str(value).strip().lower()
    s = s.replace("：", ":")
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi\s*:\s*", "", s)
    s = re.sub(r"\s+", "", s)
    s = s.rstrip("/.,;")
    return s


def activity_level(ic50: float) -> int:
    if ic50 < 50:
        return 3
    if ic50 <= 200:
        return 2
    if ic50 <= 1000:
        return 1
    return 0


def activity_weight(level: int) -> float:
    return {3: 3.0, 2: 2.0, 1: 1.0, 0: 0.5}[int(level)]


def within_metrics(frame: pd.DataFrame, score_col: str) -> dict:
    correct = total = 0
    sccs, ns = [], []
    details = []
    for source, g in frame.groupby("source_norm"):
        if len(g) < 2:
            continue
        y = g["pic50_model"].to_numpy()
        p = g[score_col].to_numpy()
        c = t = 0
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if y[i] == y[j]:
                    continue
                t += 1
                if (y[i] > y[j]) == (p[i] > p[j]):
                    c += 1
        scc = spearmanr(y, p).correlation
        if np.isfinite(scc):
            sccs.append(float(scc))
            ns.append(len(g))
        correct += c
        total += t
        details.append(
            {
                "source": source,
                "n": len(g),
                "pair_correct": c,
                "pair_total": t,
                "pair_accuracy": c / t if t else np.nan,
                "SCC": scc,
            }
        )
    return {
        "pair_accuracy": correct / total if total else np.nan,
        "pair_correct": int(correct),
        "pair_total": int(total),
        "SCC_macro": float(np.mean(sccs)) if sccs else np.nan,
        "SCC_weighted": float(np.average(sccs, weights=ns)) if sccs else np.nan,
        "n_groups": len(details),
        "details": details,
    }


def ndcg_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    k = min(k, len(y))
    order = np.argsort(score)[::-1][:k]
    ideal = np.argsort(y)[::-1][:k]
    gains = np.power(2.0, y) - 1.0
    discount = np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains[order] / discount)
    idcg = np.sum(gains[ideal] / discount)
    return float(dcg / idcg) if idcg > 0 else 0.0


def prepare_lgb(frame: pd.DataFrame, X: np.ndarray, indices: np.ndarray):
    sub = frame.iloc[indices].copy()
    sub["_original_idx"] = indices
    sub = sub.sort_values(["source_norm", "pic50_model"]).reset_index(drop=True)
    idx = sub["_original_idx"].to_numpy(dtype=int)
    groups = sub.groupby("source_norm", sort=False).size().to_numpy(dtype=int)
    return (
        X[idx],
        sub["rank_label"].to_numpy(dtype=int),
        sub["sample_weight"].to_numpy(dtype=float),
        groups,
        sub,
    )


def train_model(
    frame: pd.DataFrame,
    X: np.ndarray,
    indices: np.ndarray,
    params: dict,
    rounds: int,
):
    Xt, yt, wt, groups, _ = prepare_lgb(frame, X, indices)
    ds = lgb.Dataset(Xt, label=yt, weight=wt, group=groups, free_raw_data=False)
    model = lgb.train(params, ds, num_boost_round=rounds)
    return model


def candidate_params() -> list[tuple[str, dict, int]]:
    base = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "verbosity": -1,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
    }
    specs = [
        ("P1", 0.02, 15, 4, 5, 0.8, 0.8, 0.1, 0.5, 500),
        ("P2", 0.03, 31, 5, 3, 0.8, 0.8, 0.1, 0.5, 350),
        ("P3", 0.05, 31, 6, 2, 0.9, 0.85, 0.0, 1.0, 250),
        ("P4", 0.03, 63, 6, 2, 0.75, 0.85, 0.1, 1.0, 350),
        ("P5", 0.01, 31, 5, 5, 0.9, 0.9, 0.2, 1.5, 700),
        ("P6", 0.05, 15, 4, 3, 1.0, 0.9, 0.0, 0.5, 250),
        ("P7", 0.02, 63, 7, 2, 0.7, 0.8, 0.2, 2.0, 500),
        ("P8", 0.04, 31, -1, 4, 0.85, 0.8, 0.05, 0.8, 300),
        ("P9", 0.025, 47, 6, 3, 0.8, 0.9, 0.1, 1.2, 450),
        ("P10", 0.03, 23, 5, 2, 1.0, 0.8, 0.0, 1.0, 350),
    ]
    out = []
    for name, lr, leaves, depth, min_leaf, ff, bf, l1, l2, rounds in specs:
        p = {
            **base,
            "learning_rate": lr,
            "num_leaves": leaves,
            "max_depth": depth,
            "min_data_in_leaf": min_leaf,
            "feature_fraction": ff,
            "bagging_fraction": bf,
            "bagging_freq": 1,
            "lambda_l1": l1,
            "lambda_l2": l2,
        }
        out.append((name, p, rounds))
    return out


def choose_group_split(big: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    # Search deterministic group splits, using only group sizes and level proportions
    # to match an ~80/20 sample ratio. No model score/test outcome is consulted.
    splitter = GroupShuffleSplit(n_splits=500, test_size=0.2, random_state=SEED)
    all_levels = big["rank_label"].value_counts(normalize=True).sort_index()
    best = None
    for n, (tr, te) in enumerate(
        splitter.split(big, big.rank_label, groups=big.source_norm)
    ):
        tr_sources = big.iloc[tr].source_norm.nunique()
        te_sources = big.iloc[te].source_norm.nunique()
        test_ratio = len(te) / len(big)
        test_levels = (
            big.iloc[te]["rank_label"]
            .value_counts(normalize=True)
            .reindex(range(4), fill_value=0)
        )
        target_levels = all_levels.reindex(range(4), fill_value=0)
        cost = abs(test_ratio - 0.2) + 0.35 * float(
            np.abs(test_levels - target_levels).sum()
        )
        # Prefer at least 8 held-out sources.
        if te_sources < 8:
            cost += 1.0
        rec = (cost, n, tr, te, tr_sources, te_sources, test_ratio)
        if best is None or rec[0] < best[0]:
            best = rec
    _, split_no, tr, te, tr_sources, te_sources, ratio = best
    return tr, te, {
        "candidate_split_no": int(split_no),
        "train_sources": int(tr_sources),
        "test_sources": int(te_sources),
        "test_sample_ratio": float(ratio),
        "selection": "closest 20% sample ratio + activity-level distribution; no model/test metric used",
    }


def main():
    raw = pd.read_excel(RAW)
    raw["ic50_numeric"] = pd.to_numeric(raw["ic50_um"], errors="coerce")
    raw["source_norm"] = raw["source"].map(normalize_source)
    raw["is_standard"] = raw["sequence"].astype(str).str.match(STANDARD)
    numeric = raw[raw.ic50_numeric.notna()].copy()
    valid = numeric[numeric.is_standard].copy().reset_index(drop=True)
    valid["pic50_model"] = -np.log10(valid["ic50_numeric"].to_numpy() * 1e-6)
    valid["rank_label"] = valid["ic50_numeric"].map(activity_level).astype(int)
    valid["sample_weight"] = valid["rank_label"].map(activity_weight)
    valid["source_n"] = valid.groupby("source_norm")["source_norm"].transform("size")
    valid["eligible_n4"] = valid["source_n"] >= 4

    big = valid[valid.eligible_n4].copy().reset_index(drop=True)
    X = extract_sequence_features(big.sequence.tolist())
    tr_idx, te_idx, split_info = choose_group_split(big)
    train_sources = set(big.iloc[tr_idx].source_norm)
    test_sources = set(big.iloc[te_idx].source_norm)
    assert not (train_sources & test_sources)

    big["split"] = "excluded"
    big.loc[tr_idx, "split"] = "train_source_group"
    big.loc[te_idx, "split"] = "test_source_group"

    # Group-CV hyperparameter selection on training sources only.
    train_frame = big.iloc[tr_idx].copy().reset_index(drop=True)
    X_train_pool = X[tr_idx]
    groups = train_frame.source_norm.to_numpy()
    cv = GroupKFold(n_splits=5)
    cv_rows = []
    print(
        f"Raw={len(raw)}, numeric={len(numeric)}, standard={len(valid)}, "
        f"n>=4={len(big)} / {big.source_norm.nunique()} sources"
    )
    print(split_info)

    for name, params, rounds in candidate_params():
        fold_pair, fold_scc, fold_ndcg = [], [], []
        print("CV", name)
        for fold, (a, b) in enumerate(
            cv.split(train_frame, train_frame.rank_label, groups=groups), 1
        ):
            model = train_model(train_frame, X_train_pool, a, params, rounds)
            pred = model.predict(X_train_pool[b])
            val = train_frame.iloc[b].copy()
            val["score"] = pred
            met = within_metrics(val, "score")
            fold_pair.append(met["pair_accuracy"])
            fold_scc.append(met["SCC_weighted"])
            fold_ndcg.append(
                ndcg_at_k(
                    val.rank_label.to_numpy(),
                    pred,
                    min(10, len(val)),
                )
            )
            print(
                f"  fold={fold} pair={met['pair_accuracy']:.4f} "
                f"scc_w={met['SCC_weighted']:.4f}"
            )
        cv_rows.append(
            {
                "config": name,
                "pair_accuracy_mean": float(np.nanmean(fold_pair)),
                "pair_accuracy_std": float(np.nanstd(fold_pair)),
                "SCC_weighted_mean": float(np.nanmean(fold_scc)),
                "NDCG10_mean": float(np.nanmean(fold_ndcg)),
                "params": json.dumps(params, ensure_ascii=False),
                "rounds": rounds,
            }
        )

    cv_df = pd.DataFrame(cv_rows).sort_values(
        ["pair_accuracy_mean", "SCC_weighted_mean"], ascending=False
    )
    cv_df.to_csv(OUT / "ranking_groupcv_hyperparameters.csv", index=False, encoding="utf-8-sig")
    best_name = cv_df.iloc[0]["config"]
    best_spec = next(x for x in candidate_params() if x[0] == best_name)
    _, best_params, best_rounds = best_spec

    # Frozen held-out source evaluation.
    eval_model = train_model(big, X, tr_idx, best_params, best_rounds)
    big["eval_rank_score"] = eval_model.predict(X)
    test_eval = big.iloc[te_idx].copy()
    metrics = within_metrics(test_eval, "eval_rank_score")
    metrics["NDCG@5_level"] = ndcg_at_k(
        test_eval.rank_label.to_numpy(),
        test_eval.eval_rank_score.to_numpy(),
        5,
    )
    metrics["NDCG@10_level"] = ndcg_at_k(
        test_eval.rank_label.to_numpy(),
        test_eval.eval_rank_score.to_numpy(),
        10,
    )
    metrics["global_SCC_control"] = float(
        spearmanr(test_eval.pic50_model, test_eval.eval_rank_score).correlation
    )
    metrics["n_test_samples"] = int(len(test_eval))
    metrics["n_test_sources"] = int(test_eval.source_norm.nunique())
    metrics["best_config"] = best_name
    details = pd.DataFrame(metrics.pop("details"))

    # Production model trained on all eligible large sources.
    all_idx = np.arange(len(big))
    final_model = train_model(big, X, all_idx, best_params, best_rounds)
    big["final_rank_score"] = final_model.predict(X)
    big["within_source_pred_rank"] = big.groupby("source_norm")[
        "final_rank_score"
    ].rank(ascending=False, method="dense")
    big["within_source_true_rank"] = big.groupby("source_norm")[
        "pic50_model"
    ].rank(ascending=False, method="dense")

    # Full 615-row export. Six non-standard sequences are retained but not scored.
    export = raw.copy()
    export["source_norm"] = export["source"].map(normalize_source)
    export = export.merge(
        big[
            [
                "sequence", "source_norm", "source_n", "rank_label",
                "sample_weight", "split", "eval_rank_score", "final_rank_score",
                "within_source_pred_rank", "within_source_true_rank",
            ]
        ],
        on=["sequence", "source_norm"],
        how="left",
    )
    export["modeling_status"] = np.select(
        [
            export["ic50_numeric"].isna(),
            ~export["is_standard"],
            export["source_n"].fillna(0).lt(4),
            export["split"].eq("train_source_group"),
            export["split"].eq("test_source_group"),
        ],
        [
            "excluded_non_numeric_ic50",
            "excluded_nonstandard_sequence",
            "excluded_source_n_lt4",
            "evaluation_train_source",
            "evaluation_test_source",
        ],
        default="not_modeled",
    )

    audit = {
        "raw_rows": int(len(raw)),
        "numeric_ic50_rows": int(len(numeric)),
        "standard_sequence_rows": int(len(valid)),
        "eligible_n4_rows": int(len(big)),
        "eligible_n4_sources": int(big.source_norm.nunique()),
        "train_rows": int(len(tr_idx)),
        "test_rows": int(len(te_idx)),
        "train_sources": int(len(train_sources)),
        "test_sources": int(len(test_sources)),
        "source_overlap": int(len(train_sources & test_sources)),
        "sequence_overlap": int(
            len(set(big.iloc[tr_idx].sequence) & set(big.iloc[te_idx].sequence))
        ),
        "split_info": split_info,
        "best_config": best_name,
        "best_params": best_params,
        "best_rounds": best_rounds,
        "test_metrics": metrics,
        "note": (
            "Evaluation model uses train sources only. Production model is retrained "
            "on all n>=4 sources after evaluation and must not be used to claim test performance."
        ),
    }
    with open(OUT / "ranking_source_grouped_audit_metrics.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    with open(CKPT / "lambdarank_eval_source_holdout.pkl", "wb") as f:
        pickle.dump(eval_model, f)
    with open(CKPT / "lambdarank_final_all_n4_sources.pkl", "wb") as f:
        pickle.dump(final_model, f)

    big.to_csv(
        OUT / "ranking_eligible_n4_predictions.csv",
        index=False, encoding="utf-8-sig"
    )
    details.to_csv(
        OUT / "ranking_test_per_source_metrics.csv",
        index=False, encoding="utf-8-sig"
    )
    export.to_excel(OUT / "all_ic50_ranking_results_source_grouped_split.xlsx", index=False)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
