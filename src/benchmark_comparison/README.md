# Benchmark / ablation scripts (archival, NOT one-click reproducible)

This folder contains the experiment-orchestration scripts used to produce
the manuscript's fair-comparison and ablation numbers (against public
baselines BERT-DPPIV and Structural-DPP-IV, plus the ranking ablation and
fusion-weight grid search). They are kept for **methodological transparency
and provenance**, not as a push-button pipeline.

**Please read before running anything here:**

- These scripts hard-code an absolute Windows development path
  (`ROOT = Path(r"...")`) near the top of each file and were written
  incrementally during the project (filenames carry the date they were
  written, e.g. `_0714`, `_0718`, `_0720`). You will need to edit `ROOT`
  (and any other hard-coded paths) to point at your own checkout before
  running them.
- The third-party baselines (**BERT-DPPIV**, **Structural-DPP-IV**) are
  *not* bundled here (their own weights/environments are large and have
  their own licenses) — `run_bert_dppiv_fair_0714.py` /
  `run_structuraldppiv_fair_0714.py` will not run without separately
  obtaining those projects.
- Intermediate feature caches (e.g. the ESM-2 embedding cache under
  `processed/ranking_source_grouped_0714/`) referenced by these scripts are
  **not** included in this slim package to keep it GitHub-sized; only the
  final Top-300 indices and checkpoints needed for inference
  (`processed/top300_feature_indices.npy`, `checkpoints/`) are shipped.

## What each script produced (for provenance)

| Script | Produces |
|---|---|
| `calibrate_project_threshold_0714.py` | Calibrates this project's classification decision threshold for a fair head-to-head comparison |
| `run_bert_dppiv_fair_0714.py` | Runs the BERT-DPPIV baseline (needs its own weights/environment) |
| `run_structuraldppiv_fair_0714.py` | Runs the Structural-DPP-IV baseline |
| `run_fair_benchmark_0714.py` | Aggregates all three models' ROC/PR/metric tables |
| `run_source_grouped_ranking_0714.py` | Strictly source-disjoint LambdaRank training/eval (Top-k enrichment, macro NDCG@k) — the headline ranking numbers in the manuscript |
| `optimize_source_grouped_ranking_0714.py` | Hyperparameter search for the above (GroupKFold on training sources only) |
| `run_saar_ablation_0718.py` | Global-vs-source-aware ranking ablation + classification-vs-classification+ranking ablation (see `src/evaluation/ablation.py` for a clean, re-usable version of the same methodology) |
| `run_saar_full_metrics_0720.py` | Full metrics table across all model variants |
| `run_saar_weight_grid_0720.py` | Grid search over the `Combined_Score` fusion weight (see README.md "Fusion weight selection") |

## Reproducing the manuscript's headline numbers without this folder

You do **not** need to run anything in this folder to reproduce the
reported classification/ranking numbers for the deployed model — use
`reproduce_results.py` at the repository root, which loads the shipped
checkpoints directly. This folder is only relevant if you want to re-run
the baseline comparison or the hyperparameter/weight search from scratch.
