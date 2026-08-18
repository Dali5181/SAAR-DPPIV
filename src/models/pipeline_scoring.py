# -*- coding: utf-8 -*-
"""
Unified SAAR-DPPIV scoring pipeline shared by the web app and the CLI
(``full_ranking_predict.py``).

Both entry points MUST call :func:`score_sequences` so the website and the
command-line script always produce identical numbers on the same machine
(the client asked that the Python results be authoritative).

Pipeline (identical for web + CLI):
    * Classification probability  : mean of xgb/lgb/cat/rf classifiers in
                                    ``checkpoints/classification/classifiers.pkl``
                                    (see ``classification.py``)
    * Auxiliary pIC50             : ``xgb_reg`` regressor in
                                    ``checkpoints/regression/regressor.pkl``
                                    (see ``regression.py``)
    * Source-aware rank score     : LambdaRank booster in
                                    ``checkpoints/ranking/lambdarank_model.txt``
                                    over the Top-300 features of
                                    [617D sequence + 480D ESM-2] (see
                                    ``ranking_lambdarank.py`` for the relevance
                                    label / RankScore normalisation spec, and
                                    ``feature_selection.py`` for Top-300)
    * Combined score              : W_PACTIVE * P(active) + W_RANK * Rank_Score
                                    (0.65 / 0.35 by default; see README
                                    "Fusion weight selection" for the
                                    grid-search evidence)

ESM-2 checkpoint and pooling
----------------------------
* Checkpoint : ``facebook/esm2_t12_35M_UR50D`` (HuggingFace Hub; 12-layer,
  ~35M-parameter ESM-2, 480-dim hidden size), loaded via
  ``transformers.AutoTokenizer`` / ``AutoModel``. Same checkpoint name is
  used by both this module and ``src/features/esm2_embeddings.py``.
* Tokenisation here: ``padding=True, truncation=True, max_length=512``.
  (``src/features/esm2_embeddings.py`` uses ``max_length=1024`` instead --
  harmless in practice since every peptide in this project is well under
  512 residues, but noted here for exactness.)
* Pooling here (inference time, used by the web app + CLI): mean-pool the
  residue tokens + EOS, with CLS stripped and PAD masked out via the
  attention mask (see ``esm2_embeddings()`` below, variable ``res_mask``).
  This is intentionally NOT the same pooling as the offline training-time
  extractor in ``src/features/esm2_embeddings.py``, which mean-pools ALL
  non-PAD tokens (i.e. it keeps CLS) -- that is how the ranking model's
  ESM-2 features were originally built. Both poolings are kept as-is
  because changing either would shift already-reported candidate rankings;
  the inference-time pooling in THIS file is the one actually used to
  score new sequences in the web app / CLI.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

# Combined-score fusion weights (report / deployment configuration).
W_PACTIVE = 0.65
W_RANK = 0.35

ESM_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"


def project_root() -> Path:
    """Repository root (…/  containing checkpoints/, processed/, src/)."""
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_models():
    """Load (cls_dict, xgb_reg, lgb_rank, top_idx). Cached across calls.

    Checkpoints are split by role (see README "Repository layout"):
        checkpoints/classification/classifiers.pkl   {xgb_cls, lgb_cls, cat_cls, rf_cls}
        checkpoints/regression/regressor.pkl          {xgb_reg}
        checkpoints/ranking/lambdarank_model.txt      LightGBM Booster (source-aware LambdaRank)
        processed/top300_feature_indices.npy          indices into [617-D seq + 480-D ESM-2]
    """
    import lightgbm as lgb

    ck = project_root() / "checkpoints"
    with open(ck / "classification" / "classifiers.pkl", "rb") as f:
        cls = pickle.load(f)
    with open(ck / "regression" / "regressor.pkl", "rb") as f:
        reg = pickle.load(f)["xgb_reg"]
    rank = lgb.Booster(model_file=str(ck / "ranking" / "lambdarank_model.txt"))

    top = np.load(project_root() / "processed" / "top300_feature_indices.npy")
    top = np.asarray(top, dtype=int)
    return cls, reg, rank, top


@lru_cache(maxsize=1)
def _load_esm2():
    """Load ESM-2 tokenizer + model (cached). Raises if torch/transformers absent."""
    import torch
    from transformers import EsmModel, EsmTokenizer

    tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL_NAME)
    model = EsmModel.from_pretrained(ESM_MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return tokenizer, model, device


def esm2_embeddings(seqs: list[str], batch_size: int = 32) -> np.ndarray:
    """480-D mean-pooled ESM-2 embedding per sequence: residue tokens + EOS,
    with CLS stripped and PAD masked out (padding-invariant). See the module
    docstring -- this intentionally differs from the training-time feature
    extractor in ``src/features/esm2_embeddings.py``, which pools ALL
    non-PAD tokens (i.e. it keeps CLS)."""
    import torch

    tokenizer, model, device = _load_esm2()
    out = []
    for start in range(0, len(seqs), batch_size):
        batch = seqs[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state

        # Mean-pool over residues + EOS (drop CLS at index 0, drop PAD via the
        # attention mask). This reproduces the pooling of the ORIGINAL
        # `full_ranking_predict.py`, whose positional `[1:-1]` slice over a
        # padded batch kept the EOS token for every non-longest sequence, and
        # is exactly how the delivered candidate ranking (e.g. VPIL rank 0.8878
        # / combined 0.9413) was generated. Building the mask from the true
        # sequence length instead of a positional slice makes it
        # padding-invariant, so single- and batch-mode now return the same
        # embedding for a given peptide.
        res_mask = enc["attention_mask"].float().clone()
        res_mask[:, 0] = 0.0                              # drop CLS only
        res_mask = res_mask.unsqueeze(-1)
        emb = (hidden * res_mask).sum(dim=1) / res_mask.sum(dim=1).clamp(min=1e-9)
        out.append(emb.cpu().numpy().astype(np.float32))
    return np.vstack(out)


def priority_level(score: float) -> str:
    """Combined-score -> priority label (client thresholds).

        High >= 0.75  |  Medium 0.60-0.75  |  Low < 0.60
    """
    s = float(score)
    if s >= 0.75:
        return "High"
    if s >= 0.60:
        return "Medium"
    return "Low"


def score_sequences(seqs, use_esm: bool = True, batch_size: int = 32) -> dict:
    """Score peptide sequences with the unified pipeline.

    Parameters
    ----------
    seqs : list[str]
        Cleaned upper-case peptide sequences (standard 20 AA).
    use_esm : bool
        When True, real ESM-2 embeddings are computed so the result matches
        the CLI candidate-table workflow. When False (fast web mode) the
        ESM-2 block is zero-padded — quicker but only an approximation.

    Returns
    -------
    dict with numpy arrays: ``p_active``, ``rank_score``, ``combined``,
    ``pic50`` and a bool ``used_esm``.
    """
    from src.features.sequence_features import extract_sequence_features

    seqs = list(seqs)
    cls, reg, rank_model, top = load_models()

    x617 = extract_sequence_features(seqs).astype(np.float32)
    p_active = np.mean([cls[k].predict_proba(x617)[:, 1] for k in cls], axis=0)
    pic50 = (reg.predict(x617).astype(float)
             if reg is not None else np.full(len(seqs), np.nan))

    used_esm = False
    if use_esm:
        try:
            emb = esm2_embeddings(seqs, batch_size=batch_size)
            used_esm = True
        except Exception:
            emb = np.zeros((len(seqs), 480), dtype=np.float32)
    else:
        emb = np.zeros((len(seqs), 480), dtype=np.float32)

    x1097 = np.hstack([x617, emb]).astype(np.float32)
    raw = rank_model.predict(x1097[:, top])
    rank_score = 1.0 / (1.0 + np.exp(-raw / 2.0))
    combined = W_PACTIVE * p_active + W_RANK * rank_score

    return {
        "p_active": np.asarray(p_active, dtype=float),
        "rank_score": np.asarray(rank_score, dtype=float),
        "combined": np.asarray(combined, dtype=float),
        "pic50": np.asarray(pic50, dtype=float),
        "used_esm": used_esm,
    }
