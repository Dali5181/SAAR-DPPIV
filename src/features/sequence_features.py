"""
Sequence-based feature extraction for peptide sequences (Branch A).

Descriptors:
  - AAC : Amino Acid Composition          (20-dim)
  - DPC : Dipeptide Composition            (400-dim)
  - CTD : Composition/Transition/Distribution (147-dim)
  - PAAC: Pseudo Amino Acid Composition    (50-dim)
Total: 617-dimensional feature vector per sequence.
"""
import math
import numpy as np
from typing import List

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
_STANDARD_AA_SET = set(AMINO_ACIDS)
N_AA = len(AMINO_ACIDS)

# ---------------------------------------------------------------------------
# CTD: 7 physicochemical property groupings (Dubchak et al., 1995)
# Each property partitions the 20 standard amino acids into 3 groups.
# ---------------------------------------------------------------------------
_CTD_PROPERTIES = {
    "hydrophobicity": {
        "1": set("RKEDQN"),
        "2": set("GASTPHY"),
        "3": set("CLVIMFW"),
    },
    "normalized_vdw_volume": {
        "1": set("GASCTPD"),
        "2": set("NVEQIL"),
        "3": set("MHKFRYW"),
    },
    "polarity": {
        "1": set("LIFWCMVY"),
        "2": set("PATGS"),
        "3": set("HQRKNED"),
    },
    "polarizability": {
        "1": set("GASDT"),
        "2": set("CPNVEQIL"),
        "3": set("KMHFRYW"),
    },
    "charge": {
        "1": set("KR"),
        "2": set("ANCQGHILMFPSTWYV"),
        "3": set("DE"),
    },
    "secondary_structure": {
        "1": set("EALMQKRH"),
        "2": set("VIYCWFT"),
        "3": set("GNPSD"),
    },
    "solvent_accessibility": {
        "1": set("ALFCGIVW"),
        "2": set("RKQEND"),
        "3": set("MPSTHY"),
    },
}

# ---------------------------------------------------------------------------
# PAAC: 3 physicochemical properties for sequence-order correlation
#   H1 – Hydrophobicity  (Tanford, 1962)
#   H2 – Hydrophilicity  (Hopp & Woods, 1981)
#   H3 – Side-chain mass
# ---------------------------------------------------------------------------
_PAAC_RAW = {
    "hydrophobicity": {
        "A": 0.62, "C": 0.29, "D": -0.90, "E": -0.74, "F": 1.19,
        "G": 0.48, "H": -0.40, "I": 1.38, "K": -1.50, "L": 1.06,
        "M": 0.64, "N": -0.78, "P": 0.12, "Q": -0.85, "R": -2.53,
        "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
    },
    "hydrophilicity": {
        "A": -0.5, "C": -1.0, "D": 3.0, "E": 3.0, "F": -2.5,
        "G": 0.0, "H": -0.5, "I": -1.8, "K": 3.0, "L": -1.8,
        "M": -1.3, "N": 0.2, "P": 0.0, "Q": 0.2, "R": 3.0,
        "S": 0.3, "T": -0.4, "W": -3.4, "Y": -2.3, "V": -1.5,
    },
    "sidechain_mass": {
        "A": 15.0, "C": 47.0, "D": 59.0, "E": 73.0, "F": 91.0,
        "G": 1.0, "H": 82.0, "I": 57.0, "K": 73.0, "L": 57.0,
        "M": 75.0, "N": 58.0, "P": 42.0, "Q": 72.0, "R": 101.0,
        "S": 31.0, "T": 45.0, "W": 130.0, "Y": 107.0, "V": 43.0,
    },
}

# Pre-compute standardised (zero-mean, unit-variance) property vectors
_PAAC_STD: list[dict[str, float]] = []
for _prop_name, _raw in _PAAC_RAW.items():
    _vals = np.array([_raw[aa] for aa in AMINO_ACIDS])
    _mean, _std = _vals.mean(), _vals.std()
    _std = _std if _std > 1e-10 else 1.0
    _PAAC_STD.append({aa: (_raw[aa] - _mean) / _std for aa in AMINO_ACIDS})


def _clean_sequence(seq: str) -> str:
    """Keep only the 20 standard amino acids."""
    return "".join(c for c in seq.upper() if c in _STANDARD_AA_SET)


# ---- AAC (20-dim) ---------------------------------------------------------

def _compute_aac(seq: str) -> np.ndarray:
    n = len(seq)
    counts = np.zeros(N_AA, dtype=np.float64)
    for c in seq:
        idx = AA_TO_IDX.get(c)
        if idx is not None:
            counts[idx] += 1
    return counts / n if n > 0 else counts


# ---- DPC (400-dim) --------------------------------------------------------

def _compute_dpc(seq: str) -> np.ndarray:
    counts = np.zeros(N_AA * N_AA, dtype=np.float64)
    for i in range(len(seq) - 1):
        a, b = AA_TO_IDX.get(seq[i]), AA_TO_IDX.get(seq[i + 1])
        if a is not None and b is not None:
            counts[a * N_AA + b] += 1
    denom = max(len(seq) - 1, 1)
    return counts / denom


# ---- CTD (147-dim) --------------------------------------------------------

def _ctd_distribution(positions: list, seq_len: int) -> list:
    """5 distribution descriptors for a set of residue positions."""
    if not positions:
        return [0.0] * 5
    n = len(positions)
    return [
        (positions[0] + 1) / seq_len,
        (positions[math.ceil(0.25 * n) - 1] + 1) / seq_len,
        (positions[math.ceil(0.50 * n) - 1] + 1) / seq_len,
        (positions[math.ceil(0.75 * n) - 1] + 1) / seq_len,
        (positions[-1] + 1) / seq_len,
    ]


def _compute_ctd(seq: str) -> np.ndarray:
    features: list[float] = []

    for groups in _CTD_PROPERTIES.values():
        group_seq: list[str] = []
        for c in seq:
            for gid, members in groups.items():
                if c in members:
                    group_seq.append(gid)
                    break

        n = len(group_seq)
        if n == 0:
            features.extend([0.0] * 21)
            continue

        # Composition (3 values)
        for gid in ("1", "2", "3"):
            features.append(group_seq.count(gid) / n)

        # Transition (3 values): symmetric pairs (1↔2, 1↔3, 2↔3)
        for g_a, g_b in (("1", "2"), ("1", "3"), ("2", "3")):
            t_count = 0
            for i in range(n - 1):
                if (group_seq[i] == g_a and group_seq[i + 1] == g_b) or \
                   (group_seq[i] == g_b and group_seq[i + 1] == g_a):
                    t_count += 1
            features.append(t_count / (n - 1) if n > 1 else 0.0)

        # Distribution (3 × 5 = 15 values)
        for gid in ("1", "2", "3"):
            positions = [i for i, x in enumerate(group_seq) if x == gid]
            features.extend(_ctd_distribution(positions, n))

    return np.array(features, dtype=np.float64)


# ---- PAAC (50-dim, lambda=30) ---------------------------------------------

def _compute_paac(seq: str, lamda: int = 30, w: float = 0.05) -> np.ndarray:
    n = len(seq)
    if n == 0:
        return np.zeros(N_AA + lamda, dtype=np.float64)

    aac = _compute_aac(seq)

    effective_lamda = min(lamda, n - 1) if n > 1 else 0
    theta = np.zeros(lamda, dtype=np.float64)

    for j in range(1, effective_lamda + 1):
        s = 0.0
        for i in range(n - j):
            ri, rj = seq[i], seq[i + j]
            if ri in AA_TO_IDX and rj in AA_TO_IDX:
                s += sum((p[ri] - p[rj]) ** 2 for p in _PAAC_STD) / len(_PAAC_STD)
        theta[j - 1] = s / (n - j)

    denom = aac.sum() + w * theta.sum()
    if denom < 1e-15:
        return np.zeros(N_AA + lamda, dtype=np.float64)

    return np.concatenate([aac / denom, w * theta / denom])


# ---- Public API -----------------------------------------------------------

def extract_sequence_features(sequences: List[str]) -> np.ndarray:
    """Extract AAC+DPC+CTD+PAAC for each peptide sequence.

    Parameters
    ----------
    sequences : list of str
        Amino acid sequences (single-letter code).

    Returns
    -------
    np.ndarray of shape (len(sequences), 617), dtype float32.
    """
    dim = N_AA + N_AA * N_AA + 147 + N_AA + 30  # 20+400+147+50 = 617
    out = np.zeros((len(sequences), dim), dtype=np.float32)

    for i, raw_seq in enumerate(sequences):
        seq = _clean_sequence(raw_seq)
        if len(seq) == 0:
            continue
        aac = _compute_aac(seq)
        dpc = _compute_dpc(seq)
        ctd = _compute_ctd(seq)
        paac = _compute_paac(seq)
        out[i] = np.concatenate([aac, dpc, ctd, paac])

    return out
