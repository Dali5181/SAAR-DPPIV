"""
Source-aware ranking head: a LightGBM LambdaRank model that prioritises
candidate peptides WITHIN the same literature source/experimental batch,
rather than across heterogeneous IC50 assay conditions (see the module
docstring at the bottom for the rationale).

This module documents and reproduces the exact training procedure used for
the checkpoint shipped at ``checkpoints/ranking/lambdarank_model.txt``
(hyperparameters cross-checked against the live checkpoint's
``Booster.params``). It answers, in executable form, the two review
questions raised about the ranking head:

    1. What does LambdaRank use as its label / relevance grade?
    2. How is RankScore normalised to the [0, 1] range?

--------------------------------------------------------------------------
1) Relevance label ("what LambdaRank optimises for")
--------------------------------------------------------------------------
Raw pIC50 is NOT used directly as the LambdaRank relevance label. Instead,
each peptide with a measured IC50 is discretised into one of four ordinal
activity levels (function ``activity_level`` below), and samples are
additionally weighted so the ranker pays more attention to getting the
high-activity end of each list right (function ``activity_weight``):

    level 3 (weight 3.0)  IC50  < 50 uM   (strong)
    level 2 (weight 2.0)  50 <= IC50 <= 200 uM   (moderate)
    level 1 (weight 1.0)  200 < IC50 <= 1000 uM  (weak)
    level 0 (weight 0.5)  IC50 > 1000 uM         (inactive / very weak)

Rationale: LambdaRank/NDCG relevance grades are meant to be a small number
of ordinal buckets (as in standard learning-to-rank benchmarks), not a
continuous regression target — using the continuous pIC50 directly as the
label produced an unstable ranker in early iterations (see
``src/benchmark_comparison/`` for the full history of that ablation).

--------------------------------------------------------------------------
2) Group structure (this is what prevents cross-source leakage)
--------------------------------------------------------------------------
LightGBM's "ranking group" is defined as ALL peptides sharing the same
*literature source* (the "source" column in the IC50 table, normalised by
``normalize_source``-style string cleanup of the DOI/citation string).
LambdaRank only ever compares two peptides that are in the SAME group, i.e.
it never has to compare an IC50 measured under one assay/enzyme/substrate
protocol against an IC50 measured under a different one. This is the
central design decision documented in the manuscript's ranking section.

Two additional leakage-control rules are applied at training time:

    * ``MIN_SOURCE_GROUP_SIZE = 4`` — a source is only used for TRAINING if
      it contributes >= 4 peptides (very small groups give unstable / undefined
      pairwise comparisons and were excluded from the training signal).
    * The preference pairs used for training are built ONLY from the IC50
      subset of the classification TRAIN split (``iDPP-IV-CV`` / ``train.csv``).
      No sequence from the classification TEST split (``iDPP-IV-TS`` /
      ``test.csv``) or its IC50 entries is used to fit the ranker, so the
      ranking head's evaluation on the held-out IC50 test subset is leak-free
      with respect to the classification split. See ``checkpoints/ranking/
      training_config.json`` for the exact fitted group/sample counts.

A stricter, fully SOURCE-DISJOINT evaluation protocol (train and test share
*zero* literature sources, via ``GroupShuffleSplit``/``GroupKFold`` on the
source label) is additionally used for the paper's headline Top-k enrichment
numbers; that independent benchmark lives in
``src/benchmark_comparison/`` (archival, see its own README) and is kept
separate from this production training script because it uses a different,
larger IC50 collection sheet and its own held-out hyperparameter search.

--------------------------------------------------------------------------
3) RankScore normalisation (raw LambdaRank margin -> [0, 1])
--------------------------------------------------------------------------
LightGBM's LambdaRank objective outputs an unbounded real-valued score
(a leaf-value margin, not a probability). To turn this into a bounded,
monotonic "RankScore" in [0, 1] that can be linearly combined with the
classification probability, we apply a fixed-temperature sigmoid:

    RankScore = 1 / (1 + exp(-raw_score / T)),   T = 2.0

T = 2.0 was fixed from the empirical standard deviation of raw training
scores (a raw margin of about +/-2 already saturates towards 0/1, which
matches the spread of scores seen on the training groups). This is NOT a
percentile/rank transform — two different runs on the same peptide always
map to the same RankScore for a given trained model, and the mapping is
monotonic in the raw score, so relative ordering within a batch is
preserved exactly. See ``sigmoid_rank_score`` below and
``src/models/pipeline_scoring.py`` (the single place this is actually
evaluated at inference time).

--------------------------------------------------------------------------
4) EF@k / Hit@k: what counts as a "high-activity hit"
--------------------------------------------------------------------------
A peptide counts as a hit for the manuscript's Top-k enrichment table
(EF@1, EF@5, Hit@1, Hit@5, ...) iff its discretised relevance level is the
top ordinal bucket, i.e. ``activity_level(ic50_um) >= 3`` (equivalently
IC50 < 50 uM, see section 1 above) -- see ``is_high_activity_hit`` below.
Enrichment factor at rank k is then

    EF@k = Hit@k / prevalence,
    Hit@k = (fraction of hits among the top-k ranked candidates),
    prevalence = (fraction of hits in the whole evaluated pool),

so EF@k = 1.0 means "no better than ranking at random" and EF@k > 1.0
quantifies how much more concentrated true high-activity peptides are in
the model's top-k versus a random draw from the same pool. Both the
in-package ablation (``src/evaluation/ablation.py::topk_enrichment_metrics``)
and the archival benchmark scripts
(``src/benchmark_comparison/run_saar_ablation_0718.py``,
``run_saar_full_metrics_0720.py``) use this same >=3 (IC50 < 50 uM)
threshold for "hit".
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

MIN_SOURCE_GROUP_SIZE = 4
RANK_SCORE_TEMPERATURE = 2.0

# Matches Booster.params of the shipped checkpoint
# (checkpoints/ranking/lambdarank_model.txt), confirmed by inspection.
LAMBDARANK_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5, 10],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 1,
    "max_depth": 6,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}
NUM_BOOST_ROUND = 300


def activity_level(ic50_um: float) -> int:
    """Discretise an IC50 (in uM) into an ordinal 0-3 LambdaRank relevance grade."""
    if ic50_um < 50:
        return 3
    if ic50_um <= 200:
        return 2
    if ic50_um <= 1000:
        return 1
    return 0


def is_high_activity_hit(level: int) -> bool:
    """"Hit" definition used by every EF@k / Hit@k number in this project:
    the top relevance bucket only, i.e. level 3 (IC50 < 50 uM)."""
    return int(level) >= 3


_LEVEL_WEIGHT = {3: 3.0, 2: 2.0, 1: 1.0, 0: 0.5}


def activity_weight(level: int) -> float:
    """Per-sample training weight for a given relevance level (favours high activity)."""
    return _LEVEL_WEIGHT[int(level)]


def sigmoid_rank_score(raw_score) -> np.ndarray:
    """Map raw LambdaRank margins to a bounded, monotonic RankScore in (0, 1)."""
    raw = np.asarray(raw_score, dtype=float)
    return 1.0 / (1.0 + np.exp(-raw / RANK_SCORE_TEMPERATURE))


def build_training_groups(ic50_df: pd.DataFrame, feature_matrix: np.ndarray,
                           min_group_size: int = MIN_SOURCE_GROUP_SIZE):
    """Assemble (X, relevance_label, sample_weight, group_sizes) for lgb.Dataset.

    Parameters
    ----------
    ic50_df : DataFrame with columns ``sequence``, ``ic50_um``, ``source``,
        already restricted to the classification TRAIN split.
    feature_matrix : precomputed per-sequence feature rows (same row order
        and index alignment as ``ic50_df``), e.g. the Top-300 selection of
        [617-D sequence features + 480-D ESM-2] (see ``feature_selection.py``).
    min_group_size : sources with fewer than this many IC50 entries are
        dropped from training (see module docstring, section 2).
    """
    import lightgbm as lgb  # local import: optional heavy dependency

    Xs, ys, gs, ws = [], [], [], []
    for _, grp in ic50_df.groupby("source"):
        if len(grp) < min_group_size:
            continue
        feats, labels, weights = [], [], []
        for row_pos in grp.index:
            level = activity_level(float(ic50_df.loc[row_pos, "ic50_um"]))
            feats.append(feature_matrix[ic50_df.index.get_loc(row_pos)])
            labels.append(level)
            weights.append(activity_weight(level))
        Xs.append(np.array(feats))
        ys.append(np.array(labels, dtype=int))
        gs.append(len(labels))
        ws.append(np.array(weights))

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    group_sizes = np.array(gs)
    w = np.concatenate(ws)
    dataset = lgb.Dataset(X, label=y, group=group_sizes, weight=w)
    return dataset, {"n_groups": len(group_sizes), "n_samples": len(y)}


def train_lambdarank(ic50_train_df: pd.DataFrame, feature_matrix: np.ndarray,
                      params: dict | None = None, num_boost_round: int = NUM_BOOST_ROUND):
    """Train the source-aware LambdaRank ranker (reproduces the shipped checkpoint)."""
    import lightgbm as lgb

    dataset, info = build_training_groups(ic50_train_df, feature_matrix)
    model = lgb.train(params or LAMBDARANK_PARAMS, dataset, num_boost_round=num_boost_round)
    return model, info


def within_source_pairwise_accuracy(y_true_level: Sequence[float], y_pred_score: Sequence[float],
                                     source: Sequence) -> float:
    """Fraction of within-source pairs correctly ordered (ties on the label excluded)."""
    df = pd.DataFrame({"s": source, "y": y_true_level, "p": y_pred_score})
    correct = total = 0
    for _, g in df.groupby("s"):
        if len(g) < 2:
            continue
        y = g["y"].to_numpy()
        p = g["p"].to_numpy()
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                if y[i] == y[j]:
                    continue
                total += 1
                correct += int((y[i] > y[j]) == (p[i] > p[j]))
    return correct / total if total else float("nan")
