"""
Streamlit web application for DPP-IV inhibitory peptide discovery.

Launch:
    streamlit run src/app/web_app.py
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Resolve project root so local imports work when run via `streamlit run`
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.design.rational_design import RationalDesigner, AA_PROPERTIES
from src.design.diagnostic_report import generate_report_text, generate_report_docx
from src.models.pipeline_scoring import score_sequences, priority_level

# Priority-level display colours (deep-blue / teal palette; red reserved for
# risk warnings only, per the client's SCI-figure style request).
_PRIORITY_COLORS = {
    "High":   "#1a6e5a",   # teal-green  (top priority)
    "Medium": "#2c7fb8",   # blue
    "Low":    "#8a94a6",   # muted grey-blue
}

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DPP-IV Predictor",
    page_icon="\U0001f9ec",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
_MODEL_VERSION = "1.0.0"
_DESIGNER = RationalDesigner()


# ===================================================================
# Model loading helpers
# ===================================================================

def _model_available() -> bool:
    try:
        from src.models.pipeline_scoring import load_models
        load_models()
        return True
    except Exception:
        return False


# ===================================================================
# Mock / demo prediction (used when no checkpoint is present)
# ===================================================================

def _stable_attention(seq: str) -> np.ndarray:
    """Deterministic, sequence-dependent illustrative attention (8 heads x L x 12).

    Uses a stable md5 seed (not Python's process-randomised hash) so the same
    sequence always yields the same figure, and scales each residue by a
    per-position strength so the per-residue bar chart actually varies across
    residues (otherwise every dirichlet row sums to 1.0 and all bars collapse
    to a uniform value).
    """
    seed = int(hashlib.md5(seq.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    pos_strength = rng.uniform(0.25, 1.0, size=len(seq))
    pocket_pref = rng.dirichlet(np.ones(12), size=(8, len(seq)))
    return pocket_pref * pos_strength[None, :, None]


def _activity_score_to_grade(score: float | None) -> str:
    """Map the 0-1 combined Activity Score to a grade key.

    Grades are derived directly from the displayed Activity Score so the
    label is always self-consistent with the number shown (the pIC50
    regression is unreliable for some peptides and produced contradictory
    "inactive" labels for clearly active sequences). Thresholds:
        >= 0.7  -> strong    (high activity)
        >= 0.6  -> medium    (medium activity)
        >= 0.5  -> weak      (weak activity)
        <  0.5  -> inactive  (inactive)
    """
    if score is None or np.isnan(float(score)):
        return "inactive"
    s = float(score)
    if s >= 0.7:
        return "strong"
    if s >= 0.6:
        return "medium"
    if s >= 0.5:
        return "weak"
    return "inactive"


# English activity-grade labels for the web UI (the site is English-only).
# Grade keys map to the four qualitative activity levels: high/medium/weak/none.
_GRADE_LABELS_EN = {
    "strong":   "High",
    "medium":   "Medium",
    "weak":     "Weak",
    "inactive": "Inactive",
}


def _mock_predict_single(seq: str) -> dict:
    """Deterministic mock prediction seeded by the sequence hash."""
    seed = int(hashlib.md5(seq.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    prob_active = rng.beta(2, 2)
    pic50 = rng.normal(6.8, 0.8)

    return {
        "cls_prob": np.array([1 - prob_active, prob_active]),
        "reg_pIC50": float(pic50),
        "attention_weights": _stable_attention(seq),
    }


def _predict_single(seq: str) -> dict:
    """Run a real or mock prediction for a single sequence (design tab only).

    Uses the same unified ``score_sequences`` pipeline as the Single/Batch
    prediction tabs (ESM-2 disabled here since the design tab only needs
    classification probability and auxiliary pIC50, not the source-aware
    rank score). Falls back to deterministic mock output only when the
    checkpoints are missing.
    """
    try:
        from src.models.pipeline_scoring import score_sequences
        res = score_sequences([seq], use_esm=False)
        p_active = float(res["p_active"][0])
        pic50 = float(res["pic50"][0])
        return {
            "cls_prob": np.array([1 - p_active, p_active]),
            "reg_pIC50": pic50,
            "attention_weights": _stable_attention(seq),
        }
    except Exception:
        return _mock_predict_single(seq)


# ===================================================================
# Validation
# ===================================================================

def _validate_sequence(seq: str) -> tuple[str | None, str | None]:
    """Return (cleaned_seq, error_msg). One of them is None."""
    seq = seq.strip().upper()
    seq = re.sub(r"\s+", "", seq)
    if not seq:
        return None, "Please enter a peptide sequence."
    if len(seq) < 2:
        return None, "Sequence must be at least 2 residues."
    if len(seq) > 100:
        return None, "Sequence exceeds 100 residues — please use batch mode for long sequences."
    invalid = set(seq) - _VALID_AA
    if invalid:
        return None, f"Invalid residue(s): {', '.join(sorted(invalid))}. Use standard 20 amino acids."
    return seq, None


# ===================================================================
# Visualisation helpers
# ===================================================================

def _attention_heatmap(seq: str, attn: np.ndarray):
    """Render an attention heatmap via matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
    })

    if attn.ndim == 3:
        avg_attn = attn.mean(axis=0)
    else:
        avg_attn = attn

    seq_len = min(len(seq), avg_attn.shape[0])
    avg_attn = avg_attn[:seq_len]

    fig, ax = plt.subplots(figsize=(max(6, avg_attn.shape[1] * 0.5), max(3, seq_len * 0.35)))
    im = ax.imshow(avg_attn, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_yticks(range(seq_len))
    ax.set_yticklabels([f"{seq[i]}{i+1}" for i in range(seq_len)], fontsize=9)
    ax.set_xlabel("Pocket residue index", fontsize=10)
    ax.set_ylabel("Peptide position", fontsize=10)
    ax.set_title("Cross-Attention Weights (peptide × pocket)", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Attention weight")
    fig.tight_layout()
    return fig


def _bar_chart_scores(seq: str, scores: list[float]):
    """Render per-position attention scores as a bar chart."""
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
    })

    seq_len = len(scores)
    labels = [f"{seq[i]}{i+1}" for i in range(seq_len)]

    colors = []
    for s in scores:
        if s >= 0.7:
            colors.append("#2ecc71")
        elif s >= 0.3:
            colors.append("#f39c12")
        else:
            colors.append("#e74c3c")

    fig, ax = plt.subplots(figsize=(max(6, seq_len * 0.45), 3.5))
    ax.bar(range(seq_len), scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Normalised attention score", fontsize=10)
    ax.set_title("Per-Residue Binding Contribution", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.7, color="#2ecc71", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(0.3, color="#e74c3c", linestyle="--", linewidth=0.8, alpha=0.6)
    fig.tight_layout()
    return fig


# ===================================================================
# CSS theming
# ===================================================================

def _inject_css():
    st.markdown("""
    <style>
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }
    h1 {
        color: #1a3c6e;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.05rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .metric-card h3 {
        margin: 0 0 0.3rem 0;
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card p {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a3c6e;
    }
    .metric-card-primary {
        background: linear-gradient(135deg, #e8f2f0 0%, #d3e6e0 100%);
        border: 2px solid #1a6e5a;
    }
    .badge-active {
        background-color: #2ecc71;
        color: white;
        padding: 0.25em 0.8em;
        border-radius: 999px;
        font-weight: 600;
    }
    .badge-inactive {
        background-color: #e74c3c;
        color: white;
        padding: 0.25em 0.8em;
        border-radius: 999px;
        font-weight: 600;
    }
    .badge-grade-strong {
        background-color: #27ae60;
        color: white;
        padding: 0.25em 0.8em;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95em;
    }
    .badge-grade-medium {
        background-color: #f39c12;
        color: white;
        padding: 0.25em 0.8em;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95em;
    }
    .badge-grade-weak {
        background-color: #e67e22;
        color: white;
        padding: 0.25em 0.8em;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95em;
    }
    .badge-grade-inactive {
        background-color: #e74c3c;
        color: white;
        padding: 0.25em 0.8em;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95em;
    }
    </style>
    """, unsafe_allow_html=True)


# ===================================================================
# Sidebar
# ===================================================================

def _sidebar():
    with st.sidebar:
        st.markdown("### SAAR-DPPIV")

        st.divider()
        if _model_available():
            st.success("Model status: Loaded", icon="\u2705")
        else:
            st.info("Model status: Demo mode", icon="\u2139\ufe0f")
        st.caption(f"Version {_MODEL_VERSION}")

        st.divider()
        st.caption(
            "**Disclaimer**  \nFor academic research use only. "
            "Not intended for clinical decision-making."
        )


# ===================================================================
# Tab 1 — Single Prediction
# ===================================================================

def _tab_single():
    st.header("Single Peptide Prediction")
    st.markdown(
        "Enter a peptide sequence (standard single-letter amino acid code) "
        "to predict DPP-IV inhibitory activity."
    )

    col_input, col_example = st.columns([3, 1])
    with col_input:
        seq_input = st.text_input(
            "Peptide sequence",
            placeholder="e.g. IPAVFK",
            help="2–100 standard amino acids (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)",
        )
    with col_example:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Use example", key="example_single"):
            seq_input = "IPAVFK"
            st.session_state["_single_seq"] = seq_input

    if "_single_seq" in st.session_state:
        seq_input = st.session_state.pop("_single_seq")

    use_full_ranking = st.checkbox(
        "Enable ESM-2-enhanced ranking",
        value=True,
        key="esm2_single",
        help="Uses ESM-2 embeddings for source-aware prioritization. "
             "Results then match the Python (full_ranking_predict.py) workflow. "
             "ESM-2 may be downloaded the first time you run the program.",
    )
    st.caption("Uses ESM-2 embeddings for source-aware prioritization.")

    if st.button("Predict", type="primary", key="btn_single"):
        seq, err = _validate_sequence(seq_input)
        if err:
            st.error(err)
            return

        with st.spinner("Running prediction..."):
            res = score_sequences([seq], use_esm=use_full_ranking)
            attn = _stable_attention(seq)

        p_active_val   = float(res["p_active"][0])
        rank_score_val = float(res["rank_score"][0])
        combined_val   = float(res["combined"][0])
        pic50_val      = float(res["pic50"][0])
        used_real_esm2 = res["used_esm"]
        predicted_class = "Active" if p_active_val >= 0.5 else "Inactive"
        prio = priority_level(combined_val)

        # Report/attention artefacts reuse the same probabilities shown above.
        result = {
            "cls_prob": np.array([1 - p_active_val, p_active_val]),
            "reg_pIC50": pic50_val,
            "attention_weights": attn,
        }
        report = _DESIGNER.generate_diagnostic_report(seq, result, attn)

        if use_full_ranking and not used_real_esm2:
            st.warning(
                "ESM-2 model unavailable on this machine (torch/transformers not installed) — "
                "the source-aware rank score falls back to sequence-only features and will "
                "differ from the Python workflow. Run `pip install torch transformers` and restart.",
                icon="\u26a0\ufe0f",
            )

        # --- Metrics row: Predicted class -> Classification probability ->
        #     Source-aware rank score -> Auxiliary pIC50 -> Combined score ---
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            badge = "active" if predicted_class == "Active" else "inactive"
            st.markdown(
                f'<div class="metric-card"><h3>Predicted Class</h3>'
                f'<p><span class="badge-{badge}">{predicted_class}</span></p></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-card"><h3>Classification Probability</h3>'
                f'<p>{p_active_val:.4f}</p></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="metric-card"><h3>Source-aware Rank Score</h3>'
                f'<p>{rank_score_val:.4f}</p>'
                f'<p style="font-size:0.75em;color:#666;">LambdaRank 0–1</p></div>',
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f'<div class="metric-card"><h3>Auxiliary pIC\u2085\u2080</h3>'
                f'<p>{pic50_val:.3f}</p>'
                f'<p style="font-size:0.75em;color:#666;">reference only</p></div>',
                unsafe_allow_html=True,
            )
        with c5:
            pcol = _PRIORITY_COLORS[prio]
            st.markdown(
                f'<div class="metric-card metric-card-primary"><h3>Combined Score</h3>'
                f'<p style="font-size:1.7em;font-weight:800;">{combined_val:.4f}</p>'
                f'<p style="font-size:0.9em;font-weight:700;color:{pcol};">{prio} priority</p></div>',
                unsafe_allow_html=True,
            )

        st.caption("Combined score integrates classification probability and source-aware "
                   "ranking score  (0.65 \u00d7 P(active) + 0.35 \u00d7 rank score).")
        st.caption("Priority level:  High \u2265 0.75  |  Medium 0.60\u20130.75  |  Low < 0.60.")

        st.divider()

        # --- Attention visualisation ---
        st.subheader("Cross-Attention Analysis")
        st.caption(
            "Illustrative only: this heatmap is a deterministic, sequence-seeded "
            "mock-up, not the output of a trained interaction model -- the deployed "
            "classifier/ranker/regressor (above) are tree ensembles + LambdaRank, "
            "which have no attention mechanism. See `src/interpretation/cross_attention.py`."
        )
        tab_heat, tab_bar = st.tabs(["Heatmap", "Per-residue scores"])
        with tab_heat:
            fig = _attention_heatmap(seq, result["attention_weights"])
            st.pyplot(fig, use_container_width=True)
        with tab_bar:
            analysis = _DESIGNER.analyze_attention(seq, result["attention_weights"])
            fig2 = _bar_chart_scores(seq, analysis["per_position_scores"])
            st.pyplot(fig2, use_container_width=True)

        st.divider()

        # --- Mutation suggestions ---
        st.subheader("Mutation Suggestions")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**S1 Pocket** (hydrophobic)")
            muts_s1 = report["mutation_suggestions"]["S1_pocket"]
            if muts_s1:
                st.dataframe(
                    pd.DataFrame(muts_s1)[["position", "original", "suggested", "expected_impact", "rationale"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No S1 pocket mutations suggested.")
        with col_s2:
            st.markdown("**S2 Pocket** (electrostatic)")
            muts_s2 = report["mutation_suggestions"]["S2_pocket"]
            if muts_s2:
                st.dataframe(
                    pd.DataFrame(muts_s2)[["position", "original", "suggested", "expected_impact", "rationale"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No S2 pocket mutations suggested.")

        st.session_state["_last_report"] = report


# ===================================================================
# Tab 2 — Batch Prediction
# ===================================================================

def _tab_batch():
    st.header("Batch Prediction")
    st.markdown(
        "Upload a CSV file with a `sequence` column to screen multiple peptides."
    )

    uploaded = st.file_uploader(
        "Upload CSV / Excel", type=["csv", "xlsx", "xls"], key="batch_upload",
        help="File must contain a column named `sequence`.",
    )

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded)
            else:
                df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Failed to read file: {exc}")
            return

        seq_col = None
        for candidate in ("sequence", "Sequence", "SEQUENCE", "seq", "peptide"):
            if candidate in df.columns:
                seq_col = candidate
                break
        if seq_col is None:
            st.error("CSV must contain a column named `sequence`.")
            return

        st.info(f"Found **{len(df)}** sequences in column `{seq_col}`.")

        use_full_ranking = st.checkbox(
            "Enable ESM-2-enhanced ranking",
            value=True,
            key="esm2_batch",
            help="Uses ESM-2 embeddings for source-aware prioritization. "
                 "Results then match the Python (full_ranking_predict.py) workflow. "
                 "For large batches this is significantly slower and may download "
                 "ESM-2 on first run.",
        )
        st.caption("Uses ESM-2 embeddings for source-aware prioritization.")

        if st.button("Run Batch Prediction", type="primary", key="btn_batch"):
            # Validate every input row first, then score all valid sequences in
            # one batched call (identical pipeline to full_ranking_predict.py).
            valid_seqs, valid_pos, error_rows = [], [], []
            for raw_seq in df[seq_col]:
                seq, err = _validate_sequence(str(raw_seq))
                if err:
                    error_rows.append({
                        "sequence": raw_seq, "status": "error", "error": err,
                        "predicted_class": None, "priority_level": None,
                        "combined_score": None, "classification_probability": None,
                        "source_aware_rank_score": None, "auxiliary_pIC50": None,
                        "confidence": None,
                    })
                else:
                    valid_seqs.append(seq)
                    valid_pos.append(raw_seq)

            with st.spinner(f"Scoring {len(valid_seqs)} sequences..."):
                res = score_sequences(valid_seqs, use_esm=use_full_ranking) if valid_seqs else None

            if res is not None and use_full_ranking and not res["used_esm"]:
                st.warning(
                    "ESM-2 unavailable (torch/transformers not installed) — rank scores "
                    "fell back to sequence-only features and will differ from the Python workflow.",
                    icon="\u26a0\ufe0f",
                )

            rows = list(error_rows)
            if res is not None:
                for i, seq in enumerate(valid_seqs):
                    p_act = float(res["p_active"][i])
                    rs = float(res["rank_score"][i])
                    comb = float(res["combined"][i])
                    pic50 = float(res["pic50"][i])
                    cls = "Active" if p_act >= 0.5 else "Inactive"
                    rows.append({
                        "sequence": seq,
                        "status": "ok",
                        "error": None,
                        "predicted_class": cls,
                        "priority_level": priority_level(comb),
                        "combined_score": round(comb, 4),
                        "classification_probability": round(p_act, 4),
                        "source_aware_rank_score": round(rs, 4),
                        "auxiliary_pIC50": round(pic50, 3),
                        "confidence": round(max(p_act, 1 - p_act), 4),
                    })

            res_df = pd.DataFrame(rows)
            # Sort by combined score (high -> low); error rows sink to the bottom.
            res_df["_sort_key"] = res_df["combined_score"].fillna(-1.0)
            res_df = res_df.sort_values("_sort_key", ascending=False).drop(columns="_sort_key").reset_index(drop=True)
            res_df.insert(0, "rank", np.arange(1, len(res_df) + 1))

            # --- Priority-level distribution ---
            ok_mask = res_df["status"] == "ok"
            levels = res_df.loc[ok_mask, "priority_level"].tolist()
            dist = {lv: levels.count(lv) for lv in ("High", "Medium", "Low")}
            n_err = int((res_df["status"] == "error").sum())

            st.subheader("Priority-Level Distribution")
            gcols = st.columns(4)
            metric_map = [
                ("High",   dist["High"],   _PRIORITY_COLORS["High"]),
                ("Medium", dist["Medium"], _PRIORITY_COLORS["Medium"]),
                ("Low",    dist["Low"],    _PRIORITY_COLORS["Low"]),
                ("Errors", n_err,          "#c0392b"),
            ]
            for col, (label, count, color) in zip(gcols, metric_map):
                col.markdown(
                    f'<div class="metric-card"><h3>{label}</h3>'
                    f'<p style="color:{color};">{count}</p></div>',
                    unsafe_allow_html=True,
                )
            st.caption("Priority level by combined score:  High \u2265 0.75  |  "
                       "Medium 0.60\u20130.75  |  Low < 0.60.")

            st.divider()
            st.caption("Results sorted by combined score (high \u2192 low).")
            display_cols = ["rank", "sequence", "predicted_class", "priority_level",
                            "combined_score", "classification_probability",
                            "source_aware_rank_score", "auxiliary_pIC50", "confidence"]
            if n_err > 0:
                display_cols.append("error")
            st.dataframe(res_df[display_cols], use_container_width=True, hide_index=True)

            csv_buf = res_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download results CSV",
                data=csv_buf,
                file_name="dppiv_batch_results.csv",
                mime="text/csv",
            )


# ===================================================================
# Tab 3 — Rational Design
# ===================================================================

def _tab_design():
    st.header("Rational Design & Optimisation")
    st.markdown(
        "Input a peptide to receive a full AI diagnostic report covering "
        "binding analysis, mutation suggestions, and stability modifications."
    )
    st.caption(
        "How this report is built: the Classification Probability / Auxiliary "
        "pIC\u2085\u2080 shown here come from the same trained models used elsewhere in "
        "this app. Which positions are flagged as \"weak binding\" (and therefore "
        "mutated) is instead driven by an illustrative, deterministic per-residue "
        "score, not a trained interaction model -- see "
        "`src/interpretation/cross_attention.py` for details. The suggested "
        "substitution residues themselves are grounded in literature DPP-IV "
        "pocket preferences (S1/S2/S2' pockets, `src/design/rational_design.py`)."
    )

    seq_input = st.text_input(
        "Peptide sequence",
        placeholder="e.g. LKPNM",
        key="design_seq",
    )

    col_btn, col_ex = st.columns([1, 3])
    with col_btn:
        run = st.button("Generate Report", type="primary", key="btn_design")
    with col_ex:
        if st.button("Use example", key="example_design"):
            seq_input = "LKPNM"
            st.session_state["design_seq"] = seq_input
            run = True

    if run:
        seq, err = _validate_sequence(seq_input)
        if err:
            st.error(err)
            return

        with st.spinner("Analysing peptide..."):
            result = _predict_single(seq)
            report = _DESIGNER.generate_diagnostic_report(
                seq, result, result["attention_weights"],
            )

        # --- Assessment overview ---
        oa = report["overall_assessment"]
        score_pct = oa["score"] / oa["max_score"]
        score_colour = "#2ecc71" if score_pct >= 0.7 else ("#f39c12" if score_pct >= 0.4 else "#e74c3c")
        st.markdown(
            f'<div style="background:{score_colour}15;border-left:4px solid {score_colour};'
            f'padding:1rem 1.5rem;border-radius:8px;margin-bottom:1rem;">'
            f'<strong style="font-size:1.1rem;">Overall Assessment: '
            f'{oa["score"]:.1f} / {oa["max_score"]:.0f}</strong><br/>'
            f'{oa["tier"]}<br/><span style="color:#555;">{oa["summary"]}</span></div>',
            unsafe_allow_html=True,
        )

        # --- Tabs inside design report ---
        t1, t2, t3, t4 = st.tabs([
            "Binding Analysis",
            "Mutation Suggestions",
            "Stability Modifications",
            "Full Report",
        ])

        with t1:
            analysis = _DESIGNER.analyze_attention(seq, result["attention_weights"])
            fig = _bar_chart_scores(seq, analysis["per_position_scores"])
            st.pyplot(fig, use_container_width=True)

            if report["binding_analysis"]["key_binding_residues"]:
                st.markdown("**Key Binding Residues**")
                st.dataframe(
                    pd.DataFrame(report["binding_analysis"]["key_binding_residues"]),
                    use_container_width=True,
                    hide_index=True,
                )

        with t2:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("**S1 Pocket Mutations**")
                muts = report["mutation_suggestions"]["S1_pocket"]
                if muts:
                    st.dataframe(pd.DataFrame(muts), use_container_width=True, hide_index=True)
                else:
                    st.info("No mutations suggested.")
            with col_s2:
                st.markdown("**S2 Pocket Mutations**")
                muts = report["mutation_suggestions"]["S2_pocket"]
                if muts:
                    st.dataframe(pd.DataFrame(muts), use_container_width=True, hide_index=True)
                else:
                    st.info("No mutations suggested.")

        with t3:
            mods = report.get("stability_modifications", [])
            if mods:
                st.dataframe(pd.DataFrame(mods), use_container_width=True, hide_index=True)
            else:
                st.info("No stability modifications identified.")

        with t4:
            report_text = generate_report_text(report)
            st.code(report_text, language="text")

            c_dl_txt, c_dl_docx = st.columns(2)
            with c_dl_txt:
                st.download_button(
                    "Download report (.txt)",
                    data=report_text.encode("utf-8"),
                    file_name=f"dppiv_report_{seq}.txt",
                    mime="text/plain",
                )
            with c_dl_docx:
                try:
                    tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
                    generate_report_docx(report, tmp.name)
                    with open(tmp.name, "rb") as f:
                        docx_bytes = f.read()
                    st.download_button(
                        "Download report (.docx)",
                        data=docx_bytes,
                        file_name=f"dppiv_report_{seq}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                except ImportError:
                    st.warning("Install `python-docx` for Word export.")
                finally:
                    try:
                        os.unlink(tmp.name)
                    except Exception:
                        pass


# ===================================================================
# Tab 4 — About
# ===================================================================

def _tab_about():
    st.header("About This Platform")

    st.subheader("Model Architecture")
    st.markdown("""
SAAR-DPPIV scores each peptide with three components that are trained and
evaluated independently, then combined at inference time:

| Component | Input | Model |
|-----------|-------|-------|
| Classification | 617-D sequence descriptors (AAC, DPC, CTD, PAAC) | Ensemble of XGBoost, LightGBM, CatBoost, RandomForest (mean-averaged P(active)) |
| Auxiliary regression | Same 617-D sequence descriptors | XGBoost regressor -> auxiliary pIC50 estimate |
| Source-aware ranking | Top-300 features selected from [617-D sequence + 480-D ESM-2] | LightGBM LambdaRank, trained to prioritise candidates **within** the same literature source (see `src/models/ranking_lambdarank.py`) |

The final `Combined_Score = 0.65 x P(active) + 0.35 x RankScore` drives
candidate prioritisation (see `src/models/pipeline_scoring.py`). There is no
graph neural network, cross-attention head, or gated multi-task fusion in
the deployed model — the attention/graph/docking material shown under
**Model Interpretation** is a **post-hoc, illustrative** analysis layer
(see `src/interpretation/`), not part of the trained scoring pipeline.
""")

    st.subheader("Dataset")
    st.markdown("""
- **Peptide inhibitors**: aggregated from published literature, with
  CD-HIT-style de-duplication at 80% sequence identity
  (`CDHIT_THRESHOLD = 0.8` in `src/config.py`).
- **Negative samples**: augmented via Restricted Boltzmann Machine (RBM)
  generation to address class imbalance (`src/data/rbm_hard_negatives.py`).
- **IC50 subset**: a minority of peptides have a measured IC50 value; only
  these are used to train/evaluate the auxiliary regressor and the ranking
  head.
""")

    st.subheader("Citation")
    st.code(
        "If you use this platform in your research, please cite:\n\n"
        "[Your publication details here]",
        language="text",
    )

    st.subheader("Technology Stack")
    cols = st.columns(4)
    tech = [
        ("Python", "3.11+"),
        ("PyTorch", "2.x"),
        ("Streamlit", "1.55"),
        ("ESM-2", "35M params"),
    ]
    for col, (name, ver) in zip(cols, tech):
        col.metric(name, ver)

    st.subheader("Quick Links")
    st.markdown("- [UniProt P27487](https://www.uniprot.org/uniprot/P27487)")
    st.markdown("- [PDB 1X70](https://www.rcsb.org/structure/1X70)")
    st.markdown("- [ChEMBL DPP-IV](https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL284/)")


# ===================================================================
# Main
# ===================================================================

def _find_figure(*names: str):
    """Return the first existing figure path under `figures/`, if shipped."""
    figures_dir = _PROJECT_ROOT / "figures"
    for n in names:
        p = figures_dir / n
        if p.exists():
            return str(p)
    return None


def _tab_interpretation():
    st.header("Model Interpretation")
    st.markdown(
        "Global explainability of the SAAR-DPPIV model: which features drive "
        "the classifier, how the learned representation separates active vs "
        "inactive peptides, and how cross-attention maps peptide residues onto "
        "the DPP-IV binding pocket."
    )

    fig_fi = _find_figure("fig7_feature_importance.png", "fig7_feature_importance_top20.png")
    if fig_fi:
        st.subheader("Feature Importance")
        st.image(fig_fi, use_container_width=True)

    fig_tsne = _find_figure("fig6_tsne_visualization.png", "fig6_tsne.png")
    if fig_tsne:
        st.subheader("Feature-Space vs. Learned Representation (t-SNE)")
        st.image(fig_tsne, use_container_width=True)

    fig_attn = _find_figure("fig10_cross_attention.png")
    if fig_attn:
        st.subheader("Cross-Attention: Peptide Residues \u00d7 DPP-IV Pocket")
        st.image(fig_attn, use_container_width=True)

    if not any([fig_fi, fig_tsne, fig_attn]):
        st.info(
            "No pre-generated interpretation figures were found under `figures/`. "
            "This code release ships the trained checkpoints and scoring pipeline "
            "only; the feature-importance, t-SNE, and cross-attention figures shown "
            "in the manuscript were generated separately and are not required to "
            "reproduce the reported classification/ranking metrics."
        )


def main():
    _inject_css()
    _sidebar()

    st.title("SAAR-DPPIV: Source-Aware AI Platform for DPP-IV Inhibitory Peptide Prioritization")
    st.caption("Classification \u00b7 Source-aware ranking \u00b7 Activity prioritization \u00b7 Rational design")

    if not _model_available():
        st.warning(
            "No model checkpoint found — running in **demo mode** with "
            "mock predictions. Place `v4_final_models.pkl` in `checkpoints/` "
            "to enable real inference.",
            icon="\u26a0\ufe0f",
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Single Prediction",
        "Batch Prediction",
        "Rational Design",
        "Model Interpretation",
        "About",
    ])

    with tab1:
        _tab_single()
    with tab2:
        _tab_batch()
    with tab3:
        _tab_design()
    with tab4:
        _tab_interpretation()
    with tab5:
        _tab_about()


if __name__ == "__main__":
    main()
