"""
ESM-2 protein language model embeddings (Branch B), offline training-time
feature extractor.

Uses the ``facebook/esm2_t12_35M_UR50D`` model (35M params, 480-dim)
via HuggingFace Transformers to produce mean-pooled per-sequence embeddings.
Tokenisation: ``padding=True, truncation=True, max_length=1024``. Pooling:
mean over ALL non-PAD tokens (i.e. CLS is included in the mean).

This is the extractor that was used offline to build the ESM-2 features
the shipped LambdaRank ranking model was originally trained on. It is NOT
called at inference time by the web app / CLI -- those instead call
``src.models.pipeline_scoring.esm2_embeddings``, which uses the same
checkpoint but a different ``max_length`` (512) and a different pooling
(residues + EOS, CLS stripped); see that module's docstring for why the two
pooling methods are intentionally different.
"""
import logging
import numpy as np
from typing import List

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"
_EMBED_DIM = 480


def _load_model(model_name: str):
    """Lazy-load the ESM-2 tokenizer and model (downloads on first call)."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as exc:
        raise ImportError(
            "ESM-2 embeddings require `torch` and `transformers`. "
            "Install with: pip install torch transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device


def extract_esm2_embeddings(
    sequences: List[str],
    batch_size: int = 32,
    model_name: str = _DEFAULT_MODEL,
) -> np.ndarray:
    """Mean-pooled ESM-2 embeddings for peptide sequences.

    Parameters
    ----------
    sequences : list of str
        Amino acid sequences (single-letter code).
        Empty / None entries receive zero vectors.
    batch_size : int
        Number of sequences per forward pass.
    model_name : str
        HuggingFace model identifier.

    Returns
    -------
    np.ndarray of shape (len(sequences), 480), dtype float32.
    """
    import torch

    tokenizer, model, device = _load_model(model_name)

    n = len(sequences)
    embeddings = np.zeros((n, _EMBED_DIM), dtype=np.float32)

    valid_indices: list[int] = []
    valid_seqs: list[str] = []
    for i, seq in enumerate(sequences):
        if seq and isinstance(seq, str) and len(seq.strip()) > 0:
            valid_indices.append(i)
            valid_seqs.append(seq.strip().upper())

    if not valid_seqs:
        return embeddings

    for start in range(0, len(valid_seqs), batch_size):
        batch_seqs = valid_seqs[start : start + batch_size]
        batch_idx = valid_indices[start : start + batch_size]

        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state  # (B, L, 480)

        mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled_np = pooled.cpu().numpy()

        for j, idx in enumerate(batch_idx):
            embeddings[idx] = pooled_np[j]

        if (start // batch_size) % 10 == 0:
            logger.info(
                "ESM-2 progress: %d / %d sequences",
                min(start + batch_size, len(valid_seqs)),
                len(valid_seqs),
            )

    return embeddings
