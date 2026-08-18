"""
POST-HOC ONLY -- not part of the trained scoring pipeline.

This module is retained purely as an optional, illustrative interpretability
utility (e.g. for exploratory small-molecule/graph visualisation). It is NOT
imported by ``pipeline_scoring.py``, ``web_app.py``, or
``full_ranking_predict.py``, and no Graph Attention Network weights ship
with this release -- the deployed classification/ranking/regression models
are the tree ensembles and LambdaRank described in ``src/models/``. See
README "Model architecture" for what is actually used to produce the
reported results.

Molecular graph construction for Graph Attention Networks (Branch C).

Converts SMILES strings to molecular graphs with:
  - Node features (34-dim): atomic number, degree, charge, Hs, hybridization,
    aromaticity, ring membership.
  - Edge features (6-dim): bond type, conjugation, ring membership.

Returns ``torch_geometric.data.Data`` objects when PyG is available,
otherwise falls back to plain dict/numpy representation.
"""
import logging
import numpy as np
from typing import List, Optional, Union

from rdkit import Chem
from rdkit.Chem import rdchem

logger = logging.getLogger(__name__)

# --- Check for PyG availability ---
try:
    import torch
    from torch_geometric.data import Data as PyGData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    try:
        import torch
        _HAS_TORCH = True
    except ImportError:
        _HAS_TORCH = False

# --- Feature dimensions ---
ATOM_FEAT_DIM = 34
BOND_FEAT_DIM = 6

_ATOM_NUMS = [6, 7, 8, 16, 9, 15, 17, 35, 53]  # C N O S F P Cl Br I

_HYBRIDIZATIONS = [
    rdchem.HybridizationType.S,
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

_BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _one_hot(val, allowed: list) -> list:
    """One-hot encode *val* against *allowed* set; last position = 'other'."""
    vec = [0] * (len(allowed) + 1)
    try:
        vec[allowed.index(val)] = 1
    except ValueError:
        vec[-1] = 1
    return vec


def _atom_features(atom: rdchem.Atom) -> list:
    """34-dimensional atom feature vector."""
    feats: list[int] = []

    # Atomic number (10): 9 common elements + other
    feats.extend(_one_hot(atom.GetAtomicNum(), _ATOM_NUMS))

    # Degree (6): 0-5, clamped
    deg = min(atom.GetDegree(), 5)
    oh = [0] * 6
    oh[deg] = 1
    feats.extend(oh)

    # Formal charge (5): clipped to [-2, 2]
    charge = max(-2, min(2, atom.GetFormalCharge()))
    oh = [0] * 5
    oh[charge + 2] = 1
    feats.extend(oh)

    # Total number of Hs (5): 0-4, clamped
    nhs = min(atom.GetTotalNumHs(), 4)
    oh = [0] * 5
    oh[nhs] = 1
    feats.extend(oh)

    # Hybridization (6): 6 known types, all-zero if unrecognised
    hyb = atom.GetHybridization()
    oh = [0] * 6
    if hyb in _HYBRIDIZATIONS:
        oh[_HYBRIDIZATIONS.index(hyb)] = 1
    feats.extend(oh)

    # Boolean flags (2)
    feats.append(int(atom.GetIsAromatic()))
    feats.append(int(atom.IsInRing()))

    return feats  # 10+6+5+5+6+1+1 = 34


def _bond_features(bond: rdchem.Bond) -> list:
    """6-dimensional bond feature vector."""
    feats: list[int] = []

    bt = bond.GetBondType()
    oh = [0] * 4
    if bt in _BOND_TYPES:
        oh[_BOND_TYPES.index(bt)] = 1
    feats.extend(oh)

    feats.append(int(bond.GetIsConjugated()))
    feats.append(int(bond.IsInRing()))

    return feats  # 4+1+1 = 6


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def smiles_to_graph(smiles: str) -> Optional[Union["PyGData", dict]]:
    """Convert a SMILES string into a molecular graph.

    Returns
    -------
    ``torch_geometric.data.Data`` when PyG is installed, otherwise a dict
    with keys ``x``, ``edge_index``, ``edge_attr``, ``num_nodes``, ``smiles``.
    Returns ``None`` for invalid SMILES.
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    atom_feats = [_atom_features(a) for a in mol.GetAtoms()]
    x = np.array(atom_feats, dtype=np.float32)

    # Edge features (undirected → duplicate each bond)
    src_dst: list[list[int]] = []
    edge_feats: list[list[int]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        src_dst.extend([[i, j], [j, i]])
        edge_feats.extend([bf, bf])

    if src_dst:
        edge_index = np.array(src_dst, dtype=np.int64).T
        edge_attr = np.array(edge_feats, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, BOND_FEAT_DIM), dtype=np.float32)

    if HAS_PYG:
        return PyGData(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr),
            smiles=smiles,
        )

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "num_nodes": x.shape[0],
        "smiles": smiles,
    }


def build_graph_dataset(smiles_list: List[Optional[str]]) -> list:
    """Build a list of molecular graphs from SMILES.

    Invalid / None entries are represented as ``None`` in the output list.
    """
    graphs: list = []
    n_fail = 0
    for smi in smiles_list:
        g = smiles_to_graph(smi) if smi else None
        if g is None and smi:
            n_fail += 1
        graphs.append(g)

    if n_fail:
        logger.warning(
            "%d / %d SMILES failed graph conversion.", n_fail, len(smiles_list)
        )
    return graphs
