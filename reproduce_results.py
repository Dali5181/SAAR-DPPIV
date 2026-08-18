"""
Reproduce the headline classification + ranking numbers reported for the
deployed SAAR-DPPIV model, using ONLY the checkpoints and data shipped in
this repository (no retraining).

Usage:
    python reproduce_results.py

What this script does and does NOT reproduce
----------------------------------------------
* Classification (Accuracy / F1 / ROC-AUC / PR-AUC / MCC) is computed on
  `data/test.csv` (n=463, held-out, never used for training) using the
  4-model tree ensemble in `checkpoints/classification/classifiers.pkl`.
  These numbers should match the manuscript's classification table exactly
  (+/- floating point / library-version noise).

* Ranking (within-source Spearman correlation + pairwise accuracy) is
  computed on the IC50-labelled subset of `data/test.csv` using the
  LambdaRank checkpoint in `checkpoints/ranking/lambdarank_model.txt`.
  This uses the SAME leak-free training protocol documented in
  `src/models/ranking_lambdarank.py` (train sources != test peptides), but
  it is a *different, smaller* evaluation set than the manuscript's
  headline strict source-disjoint Top-k enrichment numbers (EF@1, EF@5,
  NDCG@5/10), which were computed on a dedicated, larger IC50 collection
  with a literature-source-disjoint train/test split
  (see `src/benchmark_comparison/README.md` for that protocol; it needs an
  additional data file not shipped in this slim package).

* If `torch` + `transformers` are installed (see
  `requirements_full_ranking.txt`), real ESM-2 embeddings are used for the
  ranking section, matching the deployed web app / CLI exactly. Otherwise
  the ESM-2 part of the feature vector is zero-padded and the printed
  ranking numbers are only an approximation (a warning is printed).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, matthews_corrcoef,
)
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

REQUIRED = {
    "data/test.csv": ROOT / "data" / "test.csv",
    "data/peptide_ic50.csv": ROOT / "data" / "peptide_ic50.csv",
    "checkpoints/classification/classifiers.pkl": ROOT / "checkpoints" / "classification" / "classifiers.pkl",
    "checkpoints/ranking/lambdarank_model.txt": ROOT / "checkpoints" / "ranking" / "lambdarank_model.txt",
    "processed/top300_feature_indices.npy": ROOT / "processed" / "top300_feature_indices.npy",
}
missing = [name for name, p in REQUIRED.items() if not p.exists()]
if missing:
    print("[ERROR] Missing required file(s):")
    for m in missing:
        print(f"    {m}")
    sys.exit(1)

print("=" * 64)
print("SAAR-DPPIV -- reproducing reported results from shipped checkpoints")
print("=" * 64)

from src.models.pipeline_scoring import score_sequences  # noqa: E402

# ── 1. Classification on the held-out test split ─────────────────────────
print("\n[1/2] Classification metrics on data/test.csv ...")
test_df = pd.read_csv(REQUIRED["data/test.csv"]).dropna(subset=["sequence"])
test_df = test_df[test_df["sequence"].str.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$")].reset_index(drop=True)
y_test = test_df["label"].to_numpy(dtype=int)

res = score_sequences(test_df["sequence"].tolist(), use_esm=False)  # classification doesn't need ESM-2
p_active = res["p_active"]
preds = (p_active >= 0.5).astype(int)

acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)
auc = roc_auc_score(y_test, p_active)
prauc = average_precision_score(y_test, p_active)
mcc = matthews_corrcoef(y_test, preds)
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

print(f"  n = {len(y_test)}  (positive={int(y_test.sum())}, negative={int((y_test == 0).sum())})")
print(f"  Accuracy : {acc:.4f}")
print(f"  F1-Score : {f1:.4f}")
print(f"  ROC-AUC  : {auc:.4f}")
print(f"  PR-AUC   : {prauc:.4f}")
print(f"  MCC      : {mcc:.4f}")
print(f"  Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

# ── 2. Ranking on the IC50-labelled subset of the test split ─────────────
print("\n[2/2] Ranking metrics (source-aware LambdaRank) on the IC50 subset of "
      "data/test.csv ...")
try:
    from src.models.ranking_lambdarank import activity_level, within_source_pairwise_accuracy

    ic50_df = pd.read_csv(REQUIRED["data/peptide_ic50.csv"]).dropna(subset=["pic50"])
    test_ic50 = ic50_df[ic50_df["sequence"].isin(set(test_df["sequence"]))].reset_index(drop=True)

    if len(test_ic50) < 5:
        print("  Too few IC50-labelled test sequences to evaluate ranking -- skipped.")
    else:
        rres = score_sequences(test_ic50["sequence"].tolist(), use_esm=True)
        if not rres["used_esm"]:
            print("  WARNING: torch/transformers not available -- ESM-2 part was zero-padded. "
                  "Ranking numbers below are an approximation only (install "
                  "requirements_full_ranking.txt for an exact match).")
        rank_score = rres["rank_score"]
        pic50 = test_ic50["pic50"].to_numpy(dtype=float)
        levels = np.array([activity_level(float(v)) for v in test_ic50["ic50_um"]])
        source = test_ic50["source"].to_numpy()

        scc = float(spearmanr(pic50, rank_score).correlation)
        pair_acc = within_source_pairwise_accuracy(levels, rank_score, source)

        print(f"  n = {len(test_ic50)} IC50-labelled peptides, "
              f"{test_ic50['source'].nunique()} literature sources")
        print(f"  Global Spearman correlation (control, NOT the primary metric): {scc:.4f}")
        print(f"  Within-source pairwise accuracy: {pair_acc:.4f}")
        print("  Note: with this few peptides per source, results are noisy; the "
              "manuscript's headline EF@k / NDCG@k numbers come from a larger, "
              "dedicated source-disjoint evaluation (see src/benchmark_comparison/).")
except Exception as exc:
    print(f"  Ranking evaluation skipped: {exc}")

print("\nDone. For batch scoring of your own sequences, use:")
print("    python full_ranking_predict.py --input data/example_input.csv --output examples/example_output.csv")
print("For the interactive web app:")
print("    streamlit run src/app/web_app.py")
