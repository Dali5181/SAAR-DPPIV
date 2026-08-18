"""
POST-HOC ONLY -- not part of the trained scoring pipeline, and NEVER
FITTED as part of this project.

Training / generation status
-----------------------------
This module was never trained. There is no training script, no loss
function, no pocket-feature dataset, and no saved checkpoint for this
class anywhere in the development history of this project -- it exists
only as reference code for a cross-attention architecture that was
explored on paper (peptide-features-attend-to-pocket-features) but never
fitted to data. Concretely, none of the following exist for this module:
    * A dataset of (peptide, DPP-IV pocket) pairs with a supervision
      signal for what the attention SHOULD look like.
    * A training loop / optimizer / loss for ``CrossAttentionModule``.
    * A saved ``state_dict`` / checkpoint file (verify: nothing under
      ``checkpoints/`` matches this class).

Because it is untrained, ``CrossAttentionModule`` is not invoked anywhere
in this repository. The deployed classifier/ranker/regressor are tree
ensembles + LambdaRank (see ``src/models/``), which have no attention
mechanism at all.

What the web app actually shows instead
-----------------------------------------
The "Cross-Attention Analysis" heatmap / bar chart and the downstream
"Mutation Suggestions" in the web app's Design tab, and the
"attention heatmap" placeholder in the Model Interpretation tab, are all
driven by ``_stable_attention()`` in ``src/app/web_app.py``: a
deterministic, MD5-hash-seeded Dirichlet-random array with no dependency
on peptide chemistry, docking, or any trained model -- it exists only so
the UI has a stable, reproducible-looking per-residue distribution to
visualise, not to represent a real learned interaction score. The UI
carries an explicit caption stating this (see ``src/app/web_app.py``,
Design tab). If this repository is used to regenerate a manuscript figure
that shows a "cross-attention map", that figure must be labelled as an
illustrative mock-up, not a measured or trained quantity.

Cross-attention module for peptide-enzyme pocket interaction modelling
(untrained reference implementation).
Query: peptide sequence features; Key/Value: pocket residue features.
Returns attended features and per-head attention weights for interpretability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionModule(nn.Module):
    """
    Multi-head cross-attention: peptide features attend to pocket features.

    The attention weights reveal which pocket residues are most relevant
    to each peptide position, enabling interaction-site visualisation.
    """

    def __init__(self, peptide_dim: int, pocket_dim: int, hidden_dim: int = 256,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim

        self.q_proj = nn.Linear(peptide_dim, hidden_dim)
        self.k_proj = nn.Linear(pocket_dim, hidden_dim)
        self.v_proj = nn.Linear(pocket_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm_q = nn.LayerNorm(peptide_dim)
        self.norm_kv = nn.LayerNorm(pocket_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)

    def forward(self, peptide_features: torch.Tensor,
                pocket_features: torch.Tensor,
                pocket_mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            peptide_features: (batch, Lp, peptide_dim)
            pocket_features:  (batch, Lk, pocket_dim)
            pocket_mask:      (batch, Lk) True for padded pocket positions
        Returns:
            output:           (batch, Lp, hidden_dim)
            attn_weights:     (batch, num_heads, Lp, Lk)
        """
        B, Lp, _ = peptide_features.shape
        Lk = pocket_features.size(1)

        q = self.q_proj(self.norm_q(peptide_features))
        k = self.k_proj(self.norm_kv(pocket_features))
        v = self.v_proj(self.norm_kv(pocket_features))

        # Reshape to (B, num_heads, L, head_dim)
        q = q.view(B, Lp, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, Lp, Lk)

        if pocket_mask is not None:
            attn = attn.masked_fill(pocket_mask[:, None, None, :], float("-inf"))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights_dropped, v)  # (B, H, Lp, head_dim)
        attended = attended.transpose(1, 2).contiguous().view(B, Lp, self.hidden_dim)
        attended = self.out_proj(attended)
        attended = self.dropout(attended)

        # Residual requires projecting peptide_features to hidden_dim
        # Use the q projection output (pre-reshape) as the residual stream
        residual = self.q_proj(peptide_features)
        x = self.norm_out(attended + residual)

        # Feed-forward with residual
        x = self.norm_ff(x + self.ff(x))

        return x, attn_weights
