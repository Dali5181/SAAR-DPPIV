# SAAR-DPPIV

Source-Aware Activity Ranking for DPP-IV inhibitory peptides -- a
multimodal pipeline that (1) classifies a peptide as DPP-IV active /
inactive, (2) predicts an auxiliary pIC50, and (3) ranks candidate peptides
within the same literature source using a source-aware LambdaRank model, so
that heterogeneous IC50 assay conditions across different studies are never
compared directly.

This repository ships the code and checkpoints needed to (a) score new
peptide sequences and (b) reproduce the manuscript's classification and
ranking metrics from the shipped model, with no retraining required. It is
a slimmed-down export of the original development repository: exploratory
components that did not contribute to the final reported model (a GAT
branch, gated multi-task fusion head, meta-learner, and an earlier
transfer-learning experiment) have been removed rather than kept as dead
code.

## Repository layout

```
SAAR-DPPIV/
├── README.md
├── requirements.txt                 # core deps (classification/ranking/fusion/web app)
├── requirements_full_ranking.txt    # + torch/transformers, for real ESM-2 embeddings
│
├── full_ranking_predict.py          # CLI: batch-score a CSV of sequences
├── reproduce_results.py             # Reproduce headline metrics from shipped checkpoints
│
├── src/
│   ├── app/
│   │   └── web_app.py               # Streamlit web app (same scoring pipeline as the CLI)
│   │
│   ├── models/
│   │   ├── classification.py        # Tree-ensemble classifier config (xgb/lgb/cat/rf)
│   │   ├── ranking_lambdarank.py    # Source-aware LambdaRank: label + RankScore spec
│   │   ├── regression.py            # Auxiliary pIC50 XGBRegressor config
│   │   └── pipeline_scoring.py      # Unified scoring pipeline (used by web app + CLI)
│   │
│   ├── features/
│   │   ├── sequence_features.py     # 617-D handcrafted sequence descriptors
│   │   ├── esm2_embeddings.py       # 480-D ESM-2 embeddings (training-time extractor)
│   │   └── feature_selection.py     # Top-300 feature selection for the ranking head
│   │
│   ├── data/
│   │   ├── preprocessing.py         # Raw peptide table -> train/test split
│   │   ├── cluster_split.py         # CD-HIT-style sequence de-duplication (80% identity)
│   │   └── rbm_hard_negatives.py    # RBM-based synthetic hard-negative augmentation
│   │
│   ├── design/                      # Rational peptide design + diagnostic report (web app "Design" tab)
│   │
│   ├── evaluation/
│   │   ├── metrics.py               # Classification / ranking / enrichment metrics
│   │   └── ablation.py              # Reusable ranking + fusion ablation routines
│   │
│   ├── benchmark_comparison/        # Archival scripts for the fair third-party comparison; see its README
│   │
│   └── interpretation/              # Post-hoc, illustrative only -- not part of the scoring pipeline
│       ├── graph_features.py        # Molecular graph construction
│       ├── docking_features.py      # AutoDock Vina / descriptor-based affinity
│       └── cross_attention.py       # Cross-attention module
│
├── checkpoints/
│   ├── classification/classifiers.pkl   # {xgb_cls, lgb_cls, cat_cls, rf_cls}
│   ├── regression/regressor.pkl         # {xgb_reg}
│   └── ranking/
│       ├── lambdarank_model.txt         # LightGBM Booster (source-aware LambdaRank)
│       └── training_config.json         # Relevance-label / weight / RankScore provenance
│
├── processed/
│   └── top300_feature_indices.npy   # Indices into [617-D seq + 480-D ESM-2], used by the ranker
│
├── data/
│   ├── train.csv                    # Classification train split (n=1848)
│   ├── test.csv                     # Classification held-out test split (n=463)
│   ├── peptide_ic50.csv             # IC50-labelled subset (source, ic50_um, pic50)
│   └── example_input.csv            # Tiny example for full_ranking_predict.py
│
└── examples/
    └── example_output.csv           # Output of full_ranking_predict.py on example_input.csv
```

## Installation

```bash
pip install -r requirements.txt
# Optional, needed for real ESM-2 embeddings (ranking CLI / reproduce_results.py / web app):
pip install -r requirements_full_ranking.txt
```

Tested with Python 3.11-3.14, PyTorch 2.x (CPU is fine), Transformers 4.30+.

## Quick start

**Reproduce the reported metrics from the shipped checkpoints (no retraining):**

```bash
python reproduce_results.py
```

**Batch-score your own sequences:**

```bash
python full_ranking_predict.py --input data/example_input.csv --output examples/example_output.csv
```

**Interactive web app:**

```bash
streamlit run src/app/web_app.py
```

All three entry points call the same function,
`src.models.pipeline_scoring.score_sequences`, so they always produce
identical numbers for the same input on the same machine.

## Model architecture

| Component | Model | Output |
|---|---|---|
| Classification | Mean of 4 tree classifiers (XGBoost, LightGBM, CatBoost, RandomForest) on 617-D sequence features | `P(active)` in [0, 1] |
| Auxiliary regression | XGBoost regressor on 617-D sequence features | predicted pIC50 |
| Ranking | Source-aware LightGBM LambdaRank on the Top-300 features of [617-D sequence + 480-D ESM-2] | `RankScore` in (0, 1) |
| Fusion | `Combined_Score = 0.65 * P(active) + 0.35 * RankScore` | final priority score |

No graph neural network, cross-attention head, or gated multi-task fusion
is part of the deployed model.

### Implementation notes

- **Sequence de-duplication.** Train/test are split from a de-duplicated
  peptide table (CD-HIT-style clustering at **80%** sequence identity,
  `CDHIT_THRESHOLD = 0.8` in `src/config.py`, implemented in
  `src/data/cluster_split.py`), followed by an 80/20 stratified split
  (`src/data/preprocessing.py`). Note: a later merge that added real-IC50
  data (after the initial 80%-identity de-duplication pass) was not
  re-deduplicated, so a small number of near-duplicate sequences (about
  5% of rows, mostly single-residue truncations/extensions of another row)
  remain in the shipped table, and a handful of them fall on opposite
  sides of the train/test split. See `src/data/cluster_split.py` for
  details; this affects data hygiene marginally but is not expected to
  materially change the reported classification metrics.
- **ESM-2 checkpoint.** `facebook/esm2_t12_35M_UR50D` (HuggingFace Hub;
  12-layer, ~35M parameters, 480-dim hidden size), loaded via
  `transformers.AutoTokenizer` / `AutoModel`. Same checkpoint is used by
  both `src/features/esm2_embeddings.py` (offline training-time extractor)
  and `src/models/pipeline_scoring.py` (inference-time, web app + CLI).
- **ESM-2 pooling.** The offline training-time extractor
  (`src/features/esm2_embeddings.py`, `max_length=1024`) mean-pools all
  non-PAD tokens (i.e. it includes the CLS token). The inference-time
  pipeline used by the web app and CLI (`src/models/pipeline_scoring.py`,
  `max_length=512`) instead mean-pools the residue tokens plus EOS, with
  CLS stripped and PAD masked out. This is an intentional difference
  between the two code paths -- the inference-time pooling is the one that
  reproduces the reported candidate rankings and is the one actually
  shipped for scoring new sequences; both functions document this in their
  docstrings.
- **EF@k / Hit@k "hit" definition.** A candidate counts as a hit iff its
  discretised activity level is the top ordinal bucket,
  `activity_level(ic50_um) >= 3` (IC50 < 50 uM). `EF@k = Hit@k / prevalence`,
  where `Hit@k` is the hit fraction among the top-k ranked candidates and
  `prevalence` is the hit fraction over the whole evaluated pool (so
  `EF@k = 1.0` is "no better than random"). See
  `src/models/ranking_lambdarank.py` section 4 and
  `src/evaluation/ablation.py::topk_enrichment_metrics`.

## Answers to specific review questions

**What does LambdaRank use as its label / relevance grade?**
Not raw pIC50. Each IC50-labelled peptide is discretised into one of four
ordinal activity levels (3 = IC50 < 50 uM ... 0 = IC50 > 1000 uM), with a
per-level sample weight favouring the high-activity end. Full spec, code,
and rationale: `src/models/ranking_lambdarank.py`.

**How is RankScore normalised to [0, 1]?**
`RankScore = sigmoid(raw_lambdarank_margin / T)`, `T = 2.0` (fixed
temperature, not a percentile/rank transform -- the mapping is monotonic
and deterministic for a given trained model). Same reference:
`src/models/ranking_lambdarank.py`, function `sigmoid_rank_score`.

**Which 0.65 / 0.35 combination was tested, and on what basis was it selected?**
A full grid search over `w in {0.00, 0.05, ..., 1.00}` for
`Combined_Score = (1-w) * P(active) + w * RankScore` was run on the strict
source-disjoint test split, tracking EF@5, NDCG@10, and PairAcc
(`src/benchmark_comparison/run_saar_weight_grid_0720.py`, output
`SAAR_DPPIV_weight_grid_0720.csv`). No materially better weight than 0.35
was found; 0.35 was kept as the deployed default.

**How are the ranking train/validation/test groups split by source (i.e. is there source leakage)?**
LightGBM's ranking "group" = all peptides sharing the same normalised
literature source string; LambdaRank only ever compares two peptides
within the same group, so it never compares IC50s measured under different
assay conditions. Training additionally requires >=4 peptides per source
group, and only uses the IC50 subset of the classification TRAIN split.
The manuscript's headline Top-k enrichment numbers instead use a fully
source-disjoint split (zero shared literature sources between train and
test) from a separate, larger IC50 collection -- see
`src/models/ranking_lambdarank.py` section 2 and
`src/benchmark_comparison/run_source_grouped_ranking_0714.py`.

## Data

| File | Rows | Description |
|---|---|---|
| `data/train.csv` | 1848 | Classification training split |
| `data/test.csv` | 463 | Classification held-out test split (never used in training) |
| `data/peptide_ic50.csv` | 615 | Peptides with an actual measured IC50 value, used for the ranking section of `reproduce_results.py` |
| `data/example_input.csv` | 5 | Toy input for `full_ranking_predict.py` |

## Reproducibility scope

`reproduce_results.py` reproduces the manuscript's classification table
exactly from `data/test.csv` and the shipped checkpoints. For ranking, it
also reports within-source pairwise accuracy and a global Spearman
correlation on the IC50-labelled subset of `data/test.csv` (n=124, 76
sources) as a lightweight sanity check -- this is a smaller evaluation than
the manuscript's headline EF@1 / EF@5 / NDCG@5 / NDCG@10 table, which was
computed on a dedicated, larger IC50 collection with a strict
literature-source-disjoint train/test split. That protocol, together with
the additional data file it needs, is documented in
`src/benchmark_comparison/README.md` and
`src/benchmark_comparison/run_source_grouped_ranking_0714.py` /
`run_saar_full_metrics_0720.py`; it is not re-run by this slim package
because the underlying large IC50 collection is not redistributed here.

`src/benchmark_comparison/` also contains the scripts used for the fair
comparison against third-party baselines (BERT-DPPIV, Structural-DPP-IV).
These are archival, not one-click: each script has a `ROOT = Path(...)`
placeholder that must be set to a local working directory, and the two
third-party baselines require separately obtaining their own repositories
and pretrained weights (not redistributed here). See
`src/benchmark_comparison/README.md` for what each script produced and
what it needs to run.

`src/interpretation/` (`graph_features.py`, `docking_features.py`,
`cross_attention.py`) holds the post-hoc, illustrative utilities used for
the manuscript's qualitative interpretation discussion. None of them is
imported by `pipeline_scoring.py`, `web_app.py`, or
`full_ranking_predict.py`, and no reported classification/ranking number
depends on them. `graph_features.py` needs `rdkit`, which is optional and
not in `requirements.txt`; `docking_features.py`'s real-docking mode needs
a separately installed AutoDock Vina binary and otherwise falls back to a
heuristic descriptor-based score.

`cross_attention.py` was never trained: there is no pocket-feature
dataset, training loop, or saved checkpoint for it anywhere in this
project, so it is not called by any code path. The web app's "Cross-Attention
Analysis" heatmap (Single Prediction tab) and the "Rational Design &
Optimisation" tab's binding/mutation analysis are instead driven by a
deterministic, hash-seeded illustrative per-residue score
(`_stable_attention` in `src/app/web_app.py`) with no dependency on a
trained interaction model; the web app displays an explicit caption saying
so next to both features. In `src/design/rational_design.py`, this means
the *choice of which residue positions* are flagged for mutation is
illustrative, while the *substitution chemistry* (which amino acids are
proposed) is grounded in literature DPP-IV pocket preferences (S1/S2/S2'
pockets from PDB 1X70). See the docstrings in
`src/interpretation/cross_attention.py` and
`src/design/rational_design.py` for the full method description.

## License / citation

No license file is included in this export; add one (and a citation
block, once the manuscript has a DOI) before making the repository public,
per your journal's / institution's policy.
