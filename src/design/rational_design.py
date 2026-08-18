"""
Rational peptide design and optimization module for DPP-IV inhibitors.

Method
------
This module takes a per-position "attention" score array (shape
``(num_heads, seq_len, pocket_len)`` or ``(seq_len, pocket_len)``) and:

1. ``analyze_attention`` -- averages over heads/pocket positions to get one
   score per residue, then thresholds it into "weak" vs "strong" binding
   positions (``WEAK_ATTENTION_THRESHOLD`` / ``STRONG_ATTENTION_THRESHOLD``).
2. ``suggest_mutations`` -- for each WEAK position, proposes substitutions
   drawn from a literature-derived pocket-preference list
   (``S1_POCKET_PREFERRED`` / ``S2_POCKET_PREFERRED`` /
   ``S2_EXTENDED_PREFERRED``, sourced from the DPP-IV crystal structure
   PDB: 1X70), scored by a hydrophobicity/charge-based heuristic
   (``_estimate_impact``).

Important caveat on the input: in this repository, the per-position score
array passed into this module is NOT the output of a trained interaction
model. `src/app/web_app.py` always supplies a deterministic, hash-seeded
illustrative placeholder (`_stable_attention`), because no trained
cross-attention checkpoint exists (see
`src/interpretation/cross_attention.py`). That means WHICH positions get
flagged as "weak" (and therefore mutated) in the shipped web app is
illustrative, not learned; the pocket-preference substitution lists and
the physicochemical scoring in this module are the only parts of the
suggestion grounded in real DPP-IV structural biology. If this module is
ever wired to a real per-position confidence score (e.g. from a trained
model, or a structural/docking-derived metric), no other code in this
file needs to change -- it operates purely on whatever score array it is
given.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

HYDROPHOBIC_RESIDUES = {"W": "Trp", "F": "Phe", "L": "Leu",
                        "I": "Ile", "V": "Val", "M": "Met"}
POSITIVE_RESIDUES = {"K": "Lys", "R": "Arg", "H": "His"}
AROMATIC_RESIDUES = {"W": "Trp", "F": "Phe", "Y": "Tyr", "H": "His"}

AA_PROPERTIES = {
    "A": {"mw":  89.1, "hydrophobicity":  1.8, "charge":  0},
    "C": {"mw": 121.2, "hydrophobicity":  2.5, "charge":  0},
    "D": {"mw": 133.1, "hydrophobicity": -3.5, "charge": -1},
    "E": {"mw": 147.1, "hydrophobicity": -3.5, "charge": -1},
    "F": {"mw": 165.2, "hydrophobicity":  2.8, "charge":  0},
    "G": {"mw":  75.0, "hydrophobicity": -0.4, "charge":  0},
    "H": {"mw": 155.2, "hydrophobicity": -3.2, "charge":  0},
    "I": {"mw": 131.2, "hydrophobicity":  4.5, "charge":  0},
    "K": {"mw": 146.2, "hydrophobicity": -3.9, "charge":  1},
    "L": {"mw": 131.2, "hydrophobicity":  3.8, "charge":  0},
    "M": {"mw": 149.2, "hydrophobicity":  1.9, "charge":  0},
    "N": {"mw": 132.1, "hydrophobicity": -3.5, "charge":  0},
    "P": {"mw": 115.1, "hydrophobicity": -1.6, "charge":  0},
    "Q": {"mw": 146.2, "hydrophobicity": -3.5, "charge":  0},
    "R": {"mw": 174.2, "hydrophobicity": -4.5, "charge":  1},
    "S": {"mw": 105.1, "hydrophobicity": -0.8, "charge":  0},
    "T": {"mw": 119.1, "hydrophobicity": -0.7, "charge":  0},
    "V": {"mw": 117.1, "hydrophobicity":  4.2, "charge":  0},
    "W": {"mw": 204.2, "hydrophobicity": -0.9, "charge":  0},
    "Y": {"mw": 181.2, "hydrophobicity": -1.3, "charge":  0},
}

# DPP-IV pocket residue preferences (from crystal structure PDB: 1X70)
S1_POCKET_PREFERRED = ["W", "F", "L", "Y", "I", "V"]
S2_POCKET_PREFERRED = ["K", "R", "P", "A"]
S2_EXTENDED_PREFERRED = ["E", "D", "Q", "N"]


@dataclass
class MutationSuggestion:
    position: int
    original_residue: str
    suggested_residue: str
    pocket_type: str
    expected_impact: float
    rationale: str


@dataclass
class StabilityModification:
    modification_type: str
    target_positions: list[int]
    description: str
    expected_benefit: str
    feasibility: str


@dataclass
class BindingResidue:
    position: int
    residue: str
    attention_score: float
    role: str


class RationalDesigner:
    """Rational peptide design engine driven by cross-attention analysis."""

    WEAK_ATTENTION_THRESHOLD = 0.3
    STRONG_ATTENTION_THRESHOLD = 0.7

    def __init__(self, weak_threshold: float | None = None,
                 strong_threshold: float | None = None):
        if weak_threshold is not None:
            self.WEAK_ATTENTION_THRESHOLD = weak_threshold
        if strong_threshold is not None:
            self.STRONG_ATTENTION_THRESHOLD = strong_threshold

    # ------------------------------------------------------------------
    # Attention analysis
    # ------------------------------------------------------------------

    def analyze_attention(
        self,
        peptide_seq: str,
        attention_weights: np.ndarray,
    ) -> dict:
        """Identify weak and strong binding positions from attention weights.

        Parameters
        ----------
        peptide_seq : str
            Single-letter amino acid sequence.
        attention_weights : np.ndarray
            Attention matrix of shape ``(num_heads, seq_len, pocket_len)``
            or ``(seq_len, pocket_len)``.  Multi-head weights are averaged.

        Returns
        -------
        dict with keys:
            per_position_scores : list[float]
            weak_positions      : list[int]
            strong_positions    : list[int]
            binding_residues    : list[BindingResidue]
        """
        attn = np.array(attention_weights, dtype=np.float64)
        if attn.ndim == 3:
            attn = attn.mean(axis=0)  # average over heads

        seq_len = min(len(peptide_seq), attn.shape[0])
        scores = attn[:seq_len].sum(axis=-1)

        if scores.max() > 0:
            scores = scores / scores.max()

        weak = [i for i, s in enumerate(scores) if s < self.WEAK_ATTENTION_THRESHOLD]
        strong = [i for i, s in enumerate(scores) if s >= self.STRONG_ATTENTION_THRESHOLD]

        binding_residues = []
        for i in strong:
            binding_residues.append(BindingResidue(
                position=i,
                residue=peptide_seq[i],
                attention_score=float(scores[i]),
                role=self._classify_residue_role(peptide_seq[i]),
            ))

        return {
            "per_position_scores": [float(s) for s in scores],
            "weak_positions": weak,
            "strong_positions": strong,
            "binding_residues": binding_residues,
        }

    # ------------------------------------------------------------------
    # Mutation suggestions
    # ------------------------------------------------------------------

    def suggest_mutations(
        self,
        peptide_seq: str,
        attention_weights: np.ndarray,
        pocket_type: str = "S1",
    ) -> list[MutationSuggestion]:
        """Suggest pocket-aware mutations at weak binding positions.

        Parameters
        ----------
        pocket_type : str
            ``"S1"`` — favour hydrophobic / aromatic residues.
            ``"S2"`` — favour positively charged residues.
            ``"S2_extended"`` or ``"S2'"`` — favour negatively charged / polar.
        """
        analysis = self.analyze_attention(peptide_seq, attention_weights)
        weak_positions = analysis["weak_positions"]

        pocket_upper = pocket_type.upper().replace("'", "_EXTENDED")
        if pocket_upper == "S1":
            preferred = S1_POCKET_PREFERRED
            pocket_label = "S1"
        elif pocket_upper in ("S2", "S2_PRIME"):
            preferred = S2_POCKET_PREFERRED
            pocket_label = "S2"
        else:
            preferred = S2_EXTENDED_PREFERRED
            pocket_label = "S2'"

        suggestions: list[MutationSuggestion] = []
        scores = analysis["per_position_scores"]

        for pos in weak_positions:
            orig = peptide_seq[pos]
            if orig in preferred:
                continue

            for sub in preferred:
                if sub == orig:
                    continue
                impact = self._estimate_impact(orig, sub, scores[pos], pocket_label)
                rationale = self._build_rationale(orig, sub, pos, pocket_label)
                suggestions.append(MutationSuggestion(
                    position=pos,
                    original_residue=orig,
                    suggested_residue=sub,
                    pocket_type=pocket_label,
                    expected_impact=round(impact, 3),
                    rationale=rationale,
                ))

        suggestions.sort(key=lambda m: m.expected_impact, reverse=True)
        return suggestions

    # ------------------------------------------------------------------
    # Stability modifications
    # ------------------------------------------------------------------

    def suggest_stability_modifications(
        self, peptide_seq: str,
    ) -> list[StabilityModification]:
        """Propose stability-enhancing modifications for a peptide."""
        seq = peptide_seq.upper()
        modifications: list[StabilityModification] = []

        # N-terminal methylation
        modifications.append(StabilityModification(
            modification_type="N-methylation",
            target_positions=[0],
            description=(
                f"N-methylation of {seq[0]}1 to block aminopeptidase cleavage"
            ),
            expected_benefit="Improved proteolytic stability; ~2-5x half-life increase",
            feasibility="high",
        ))

        # D-amino acid substitution at termini
        terminal_positions = [0]
        if len(seq) > 1:
            terminal_positions.append(len(seq) - 1)
        modifications.append(StabilityModification(
            modification_type="D-amino acid substitution",
            target_positions=terminal_positions,
            description=(
                "Replace terminal residue(s) "
                + ", ".join(f"{seq[p]}{p+1}" for p in terminal_positions)
                + " with D-enantiomer(s)"
            ),
            expected_benefit="Resistance to exopeptidase degradation",
            feasibility="high",
        ))

        # Cysteine-based disulfide cyclisation
        cys_positions = [i for i, aa in enumerate(seq) if aa == "C"]
        if len(cys_positions) >= 2:
            for j in range(len(cys_positions)):
                for k in range(j + 1, len(cys_positions)):
                    p1, p2 = cys_positions[j], cys_positions[k]
                    modifications.append(StabilityModification(
                        modification_type="Disulfide cyclisation",
                        target_positions=[p1, p2],
                        description=(
                            f"Disulfide bridge between Cys{p1+1} and Cys{p2+1}"
                        ),
                        expected_benefit="Conformational rigidity and protease resistance",
                        feasibility="high" if abs(p2 - p1) >= 3 else "moderate",
                    ))

        # Lactam bridge candidates (i, i+3 or i, i+4 Lys–Asp/Glu pairs)
        for i in range(len(seq)):
            for offset in (3, 4):
                j = i + offset
                if j >= len(seq):
                    break
                pair = {seq[i], seq[j]}
                if pair & {"K"} and pair & {"D", "E"}:
                    modifications.append(StabilityModification(
                        modification_type="Lactam bridge",
                        target_positions=[i, j],
                        description=(
                            f"Lactam cyclisation between {seq[i]}{i+1} "
                            f"and {seq[j]}{j+1} (i, i+{offset} spacing)"
                        ),
                        expected_benefit="Enhanced helical stability and protease resistance",
                        feasibility="moderate",
                    ))

        # C-terminal amidation
        modifications.append(StabilityModification(
            modification_type="C-terminal amidation",
            target_positions=[len(seq) - 1],
            description=(
                f"Amidation of C-terminal {seq[-1]}{len(seq)} "
                f"to neutralise charge and resist carboxypeptidases"
            ),
            expected_benefit="Improved metabolic stability and membrane permeability",
            feasibility="high",
        ))

        return modifications

    # ------------------------------------------------------------------
    # Diagnostic report
    # ------------------------------------------------------------------

    def generate_diagnostic_report(
        self,
        peptide_seq: str,
        model_output: dict,
        attention_weights: np.ndarray,
    ) -> dict:
        """Compile a full diagnostic report for a single peptide.

        Parameters
        ----------
        peptide_seq : str
            Amino acid sequence.
        model_output : dict
            Must contain ``cls_logits`` (or ``cls_prob``) and ``reg_pIC50``.
        attention_weights : np.ndarray
            Cross-attention weights from the ensemble model.

        Returns
        -------
        dict   Structured report ready for ``diagnostic_report`` module.
        """
        analysis = self.analyze_attention(peptide_seq, attention_weights)

        # Classification result
        if "cls_prob" in model_output:
            prob = model_output["cls_prob"]
        elif "cls_logits" in model_output:
            logits = np.array(model_output["cls_logits"], dtype=np.float64)
            exp_l = np.exp(logits - logits.max())
            prob = exp_l / exp_l.sum()
        else:
            prob = np.array([0.5, 0.5])

        predicted_class = int(np.argmax(prob))
        confidence = float(prob[predicted_class])

        pic50 = float(model_output.get("reg_pIC50", 0.0))

        # Mutations for both pockets
        s1_mutations = self.suggest_mutations(
            peptide_seq, attention_weights, pocket_type="S1")
        s2_mutations = self.suggest_mutations(
            peptide_seq, attention_weights, pocket_type="S2")

        stability_mods = self.suggest_stability_modifications(peptide_seq)

        assessment = self._compute_assessment(
            predicted_class, confidence, pic50,
            analysis, s1_mutations, s2_mutations,
        )

        return {
            "peptide_sequence": peptide_seq,
            "sequence_length": len(peptide_seq),
            "molecular_weight_approx": sum(
                AA_PROPERTIES.get(aa, {}).get("mw", 110.0)
                for aa in peptide_seq.upper()
            ),
            "activity_prediction": {
                "predicted_class": "Active" if predicted_class == 1 else "Inactive",
                "class_probabilities": {
                    "inactive": round(float(prob[0]), 4),
                    "active": round(float(prob[1]), 4),
                },
                "confidence": round(confidence, 4),
                "predicted_pIC50": round(pic50, 3),
            },
            "binding_analysis": {
                "per_position_scores": analysis["per_position_scores"],
                "key_binding_residues": [
                    {
                        "position": br.position + 1,
                        "residue": br.residue,
                        "attention_score": round(br.attention_score, 4),
                        "role": br.role,
                    }
                    for br in analysis["binding_residues"]
                ],
                "weak_positions": [p + 1 for p in analysis["weak_positions"]],
                "strong_positions": [p + 1 for p in analysis["strong_positions"]],
            },
            "mutation_suggestions": {
                "S1_pocket": [
                    {
                        "position": m.position + 1,
                        "original": m.original_residue,
                        "suggested": m.suggested_residue,
                        "expected_impact": m.expected_impact,
                        "rationale": m.rationale,
                    }
                    for m in s1_mutations[:5]
                ],
                "S2_pocket": [
                    {
                        "position": m.position + 1,
                        "original": m.original_residue,
                        "suggested": m.suggested_residue,
                        "expected_impact": m.expected_impact,
                        "rationale": m.rationale,
                    }
                    for m in s2_mutations[:5]
                ],
            },
            "stability_modifications": [
                {
                    "type": mod.modification_type,
                    "positions": [p + 1 for p in mod.target_positions],
                    "description": mod.description,
                    "expected_benefit": mod.expected_benefit,
                    "feasibility": mod.feasibility,
                }
                for mod in stability_mods
            ],
            "overall_assessment": assessment,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_residue_role(aa: str) -> str:
        aa = aa.upper()
        if aa in HYDROPHOBIC_RESIDUES:
            return "hydrophobic contact"
        if aa in POSITIVE_RESIDUES:
            return "electrostatic interaction"
        if aa in AROMATIC_RESIDUES:
            return "pi-stacking / cation-pi"
        if aa in {"S", "T", "N", "Q"}:
            return "hydrogen bonding"
        if aa in {"D", "E"}:
            return "salt bridge (negative)"
        if aa in {"P"}:
            return "conformational constraint"
        if aa in {"G"}:
            return "backbone flexibility"
        return "van der Waals contact"

    @staticmethod
    def _estimate_impact(
        original: str, substitute: str, current_score: float, pocket: str,
    ) -> float:
        """Heuristic impact score in [0, 1] based on property change."""
        orig_props = AA_PROPERTIES.get(original, {"hydrophobicity": 0, "charge": 0})
        sub_props = AA_PROPERTIES.get(substitute, {"hydrophobicity": 0, "charge": 0})

        if pocket == "S1":
            delta_h = sub_props["hydrophobicity"] - orig_props["hydrophobicity"]
            raw = max(0.0, delta_h / 9.0 + 0.5)
        elif pocket == "S2":
            delta_c = sub_props["charge"] - orig_props["charge"]
            raw = max(0.0, delta_c / 2.0 + 0.5)
        else:
            delta_c = orig_props["charge"] - sub_props["charge"]
            raw = max(0.0, delta_c / 2.0 + 0.5)

        weakness_bonus = 1.0 - current_score
        return min(1.0, raw * 0.7 + weakness_bonus * 0.3)

    @staticmethod
    def _build_rationale(
        original: str, substitute: str, position: int, pocket: str,
    ) -> str:
        sub_props = AA_PROPERTIES.get(substitute, {})
        if pocket == "S1":
            h = sub_props.get("hydrophobicity", 0)
            return (
                f"Position {position+1}: {original} -> {substitute} "
                f"increases hydrophobicity (Kyte-Doolittle={h:+.1f}) "
                f"for S1 pocket van der Waals contacts"
            )
        elif pocket == "S2":
            c = sub_props.get("charge", 0)
            return (
                f"Position {position+1}: {original} -> {substitute} "
                f"introduces {'positive' if c > 0 else 'neutral'} charge "
                f"for S2 pocket electrostatic interactions"
            )
        else:
            return (
                f"Position {position+1}: {original} -> {substitute} "
                f"enhances polar interactions for S2' extended sub-site"
            )

    @staticmethod
    def _compute_assessment(
        pred_class: int,
        confidence: float,
        pic50: float,
        analysis: dict,
        s1_mut: list,
        s2_mut: list,
    ) -> dict:
        score = 0.0
        max_score = 100.0

        if pred_class == 1:
            score += 30.0 * confidence
        else:
            score += 10.0

        if pic50 >= 8.0:
            score += 25.0
        elif pic50 >= 7.0:
            score += 20.0
        elif pic50 >= 6.0:
            score += 15.0
        else:
            score += 5.0

        n_strong = len(analysis["strong_positions"])
        score += min(20.0, n_strong * 5.0)

        n_weak = len(analysis["weak_positions"])
        optimisable = min(len(s1_mut), 5) + min(len(s2_mut), 5)
        if n_weak > 0 and optimisable > 0:
            score += 10.0
        elif n_weak == 0:
            score += 15.0

        score = min(max_score, score)

        if score >= 80:
            tier = "Excellent candidate"
        elif score >= 60:
            tier = "Good candidate with optimisation potential"
        elif score >= 40:
            tier = "Moderate candidate — significant optimisation needed"
        else:
            tier = "Weak candidate — consider alternative scaffold"

        return {
            "score": round(score, 1),
            "max_score": max_score,
            "tier": tier,
            "summary": (
                f"Assessment score {score:.1f}/{max_score:.0f}. "
                f"Predicted {'active' if pred_class == 1 else 'inactive'} "
                f"(confidence {confidence:.1%}), pIC50={pic50:.2f}. "
                f"{n_strong} strong binding position(s), "
                f"{n_weak} position(s) amenable to optimisation."
            ),
        }
