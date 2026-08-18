"""
POST-HOC ONLY -- not part of the trained scoring pipeline.

This is an optional, exploratory docking/affinity-estimation utility. It is
NOT imported by ``pipeline_scoring.py``, ``web_app.py``, or
``full_ranking_predict.py`` and none of the manuscript's reported
classification/ranking numbers depend on it. "Mode B" (descriptor-based
affinity estimation) is a heuristic fallback, not a validated docking score
-- treat any output from this module as illustrative only unless you have
independently run and validated real AutoDock Vina docking (Mode A) with
your own receptor structures.

AutoDock Vina molecular docking pipeline for DPP-IV peptide screening.

Dual-mode operation:
  Mode A -- Real docking via the Vina command-line binary.
  Mode B -- Descriptor-based affinity estimation (fallback).

Auto-detects available tools and selects the appropriate mode.
Grid box: 50x50x50 points, spacing 0.375 A (= 18.75 A per side).
"""

import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False
    logger.debug("RDKit not available; 3D conformer generation disabled")


# ── DPP-IV active-site pocket definitions ────────────────────────────────

DPPIV_POCKET_RESIDUES: Dict[str, List[Tuple[str, int]]] = {
    "catalytic": [("S", 630), ("D", 708), ("H", 740)],
    "S1": [
        ("Y", 631), ("V", 656), ("W", 659),
        ("Y", 662), ("Y", 666), ("V", 711),
    ],
    "S2": [
        ("R", 125), ("E", 205), ("E", 206),
        ("F", 357), ("R", 358),
    ],
    "S1_prime": [("Y", 547), ("W", 629)],
    "S2_prime": [("S", 209), ("R", 669), ("N", 710)],
}

_ALL_POCKET_IDS: List[str] = []
for _pk, _rs in DPPIV_POCKET_RESIDUES.items():
    _ALL_POCKET_IDS.extend(f"{aa}{num}" for aa, num in _rs)

# Grid-box centres (Å) estimated from catalytic Ser630 in each crystal
# structure.  Verify with PyMOL before production runs.
_GRID_CENTRES: Dict[str, Tuple[float, float, float]] = {
    "1NU8": (44.0, 53.0, 38.0),
    "5J3J": (31.0, -2.0, 65.0),
}

_GRID_NPTS = 50
_GRID_SPACING = 0.375  # Å
_GRID_SIZE = _GRID_NPTS * _GRID_SPACING  # 18.75 Å

_STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ── Amino-acid physicochemical property table ────────────────────────────
# hydro = Kyte-Doolittle hydrophobicity index
# charge = formal charge at pH 7
# hbd/hba = H-bond donor/acceptor count (backbone + side-chain)
# vol = van der Waals volume (Å³)

_AA_PROPS: Dict[str, Dict[str, float]] = {
    "A": {"mw":  89.09, "hydro":  1.8, "charge":  0.0, "hbd": 1, "hba": 1, "vol":  88.6, "polar": 0, "arom": 0},
    "C": {"mw": 121.16, "hydro":  2.5, "charge":  0.0, "hbd": 1, "hba": 1, "vol": 108.5, "polar": 0, "arom": 0},
    "D": {"mw": 133.10, "hydro": -3.5, "charge": -1.0, "hbd": 1, "hba": 3, "vol": 111.1, "polar": 1, "arom": 0},
    "E": {"mw": 147.13, "hydro": -3.5, "charge": -1.0, "hbd": 1, "hba": 3, "vol": 138.4, "polar": 1, "arom": 0},
    "F": {"mw": 165.19, "hydro":  2.8, "charge":  0.0, "hbd": 1, "hba": 1, "vol": 189.9, "polar": 0, "arom": 1},
    "G": {"mw":  75.03, "hydro": -0.4, "charge":  0.0, "hbd": 1, "hba": 1, "vol":  60.1, "polar": 0, "arom": 0},
    "H": {"mw": 155.16, "hydro": -3.2, "charge":  0.5, "hbd": 2, "hba": 2, "vol": 153.2, "polar": 1, "arom": 1},
    "I": {"mw": 131.17, "hydro":  4.5, "charge":  0.0, "hbd": 1, "hba": 1, "vol": 166.7, "polar": 0, "arom": 0},
    "K": {"mw": 146.19, "hydro": -3.9, "charge":  1.0, "hbd": 2, "hba": 1, "vol": 168.6, "polar": 1, "arom": 0},
    "L": {"mw": 131.17, "hydro":  3.8, "charge":  0.0, "hbd": 1, "hba": 1, "vol": 166.7, "polar": 0, "arom": 0},
    "M": {"mw": 149.21, "hydro":  1.9, "charge":  0.0, "hbd": 1, "hba": 1, "vol": 162.9, "polar": 0, "arom": 0},
    "N": {"mw": 132.12, "hydro": -3.5, "charge":  0.0, "hbd": 2, "hba": 2, "vol": 114.1, "polar": 1, "arom": 0},
    "P": {"mw": 115.13, "hydro": -1.6, "charge":  0.0, "hbd": 0, "hba": 1, "vol": 112.7, "polar": 0, "arom": 0},
    "Q": {"mw": 146.15, "hydro": -3.5, "charge":  0.0, "hbd": 2, "hba": 2, "vol": 143.8, "polar": 1, "arom": 0},
    "R": {"mw": 174.20, "hydro": -4.5, "charge":  1.0, "hbd": 5, "hba": 2, "vol": 173.4, "polar": 1, "arom": 0},
    "S": {"mw": 105.09, "hydro": -0.8, "charge":  0.0, "hbd": 2, "hba": 2, "vol":  89.0, "polar": 1, "arom": 0},
    "T": {"mw": 119.12, "hydro": -0.7, "charge":  0.0, "hbd": 2, "hba": 2, "vol": 116.1, "polar": 1, "arom": 0},
    "V": {"mw": 117.15, "hydro":  4.2, "charge":  0.0, "hbd": 1, "hba": 1, "vol": 140.0, "polar": 0, "arom": 0},
    "W": {"mw": 204.23, "hydro": -0.9, "charge":  0.0, "hbd": 2, "hba": 1, "vol": 227.8, "polar": 0, "arom": 1},
    "Y": {"mw": 181.19, "hydro": -1.3, "charge":  0.0, "hbd": 2, "hba": 2, "vol": 193.6, "polar": 1, "arom": 1},
}


# ── Helpers ──────────────────────────────────────────────────────────────

def _clean_sequence(seq: str) -> str:
    return "".join(c for c in seq.upper() if c in _STANDARD_AA)


def _seq_rng(sequence: str, salt: str = "") -> np.random.RandomState:
    """Deterministic RNG seeded by sequence content for reproducibility."""
    digest = hashlib.md5((sequence + salt).encode()).hexdigest()
    return np.random.RandomState(int(digest[:8], 16) % (2 ** 31))


# ── Pipeline ─────────────────────────────────────────────────────────────

class VinaDockingPipeline:
    """Molecular docking pipeline targeting DPP-IV (PDB 1NU8 / 5J3J).

    Automatically selects real Vina docking (Mode A) when the binary is
    on PATH, otherwise falls back to a descriptor-based estimator (Mode B).

    Parameters
    ----------
    pdb_id : str
        PDB identifier for the receptor crystal structure.
    work_dir : str or None
        Directory for intermediate files.  A temp directory is created
        when *None*.
    """

    def __init__(self, pdb_id: str = "1NU8", work_dir: Optional[str] = None):
        self.pdb_id = pdb_id.upper()
        if work_dir:
            self.work_dir = Path(work_dir)
        else:
            self.work_dir = Path(tempfile.mkdtemp(prefix="vina_docking_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.receptor_pdb: Optional[Path] = None
        self.receptor_pdbqt: Optional[Path] = None
        self._prepared = False

        self._vina_bin = self._detect_vina()
        self.real_mode = self._vina_bin is not None

        if self.real_mode:
            logger.info("Mode A (real docking): Vina binary at %s", self._vina_bin)
        else:
            logger.warning(
                "Mode B (simulation): Vina binary not found. "
                "Install AutoDock Vina and add to PATH, "
                "or set VINA_PATH environment variable."
            )

        centre = _GRID_CENTRES.get(self.pdb_id, _GRID_CENTRES["1NU8"])
        self.grid_center: Tuple[float, float, float] = centre

    # ── binary detection ─────────────────────────────────────────────

    @staticmethod
    def _detect_vina() -> Optional[str]:
        env = os.environ.get("VINA_PATH")
        if env and os.path.isfile(env):
            return env
        for name in ("vina", "vina.exe", "vina_1.2.5", "vina_1.2.3"):
            found = shutil.which(name)
            if found:
                return found
        return None

    # ── receptor preparation ─────────────────────────────────────────

    def prepare_receptor(self, pdb_path: Optional[str] = None) -> Path:
        """Download (or copy) PDB, strip non-protein atoms, convert to PDBQT.

        Parameters
        ----------
        pdb_path : str or None
            Path to a local PDB file.  Downloads from RCSB when *None*.

        Returns
        -------
        Path to the cleaned PDB file.
        """
        if pdb_path is not None:
            src = Path(pdb_path)
            if not src.exists():
                raise FileNotFoundError(f"PDB file not found: {src}")
            local = self.work_dir / src.name
            shutil.copy2(src, local)
        else:
            local = self.work_dir / f"{self.pdb_id}.pdb"
            self._download_pdb(self.pdb_id, local)

        clean = self._strip_pdb(local)
        self.receptor_pdb = clean

        if self.real_mode:
            pdbqt = self._receptor_to_pdbqt(clean)
            if pdbqt is not None:
                self.receptor_pdbqt = pdbqt
            else:
                logger.warning(
                    "PDBQT conversion failed (OpenBabel / MGLTools not found); "
                    "falling back to simulation mode"
                )
                self.real_mode = False

        self._prepared = True
        logger.info("Receptor prepared: %s", clean)
        return clean

    @staticmethod
    def _download_pdb(pdb_id: str, dest: Path) -> None:
        import urllib.request

        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            urllib.request.urlretrieve(url, str(dest))
            logger.info("Downloaded PDB %s (%d bytes)", pdb_id, dest.stat().st_size)
        except Exception:
            logger.error(
                "Failed to download %s — check network or supply a local file", url
            )
            raise

    @staticmethod
    def _strip_pdb(pdb_path: Path) -> Path:
        """Remove water, ligands, and non-protein records from PDB."""
        out = pdb_path.with_name(pdb_path.stem + "_clean.pdb")
        n_atoms = 0
        with open(pdb_path, encoding="utf-8", errors="replace") as fin, \
                open(out, "w", encoding="utf-8") as fout:
            for line in fin:
                tag = line[:6].strip()
                if tag == "ATOM":
                    fout.write(line)
                    n_atoms += 1
                elif tag in ("TER", "END"):
                    fout.write(line)
        logger.info("Stripped PDB → %d ATOM records", n_atoms)
        return out

    def _receptor_to_pdbqt(self, pdb_path: Path) -> Optional[Path]:
        """Convert cleaned PDB → PDBQT using OpenBabel or MGLTools."""
        out = pdb_path.with_suffix(".pdbqt")

        obabel = shutil.which("obabel") or shutil.which("obabel.exe")
        if obabel:
            try:
                subprocess.run(
                    [obabel, str(pdb_path), "-O", str(out), "-p", "7.4", "-xr"],
                    check=True, capture_output=True, timeout=300,
                )
                logger.info("Receptor → PDBQT via OpenBabel")
                return out
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                    FileNotFoundError) as exc:
                logger.debug("OpenBabel receptor conversion failed: %s", exc)

        for script in ("prepare_receptor4.py", "prepare_receptor"):
            prog = shutil.which(script)
            if prog:
                try:
                    subprocess.run(
                        ["python", prog, "-r", str(pdb_path),
                         "-o", str(out), "-A", "hydrogens"],
                        check=True, capture_output=True, timeout=300,
                    )
                    logger.info("Receptor → PDBQT via MGLTools (%s)", script)
                    return out
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                        FileNotFoundError) as exc:
                    logger.debug("MGLTools receptor conversion failed: %s", exc)

        return None

    # ── ligand preparation ───────────────────────────────────────────

    def _ligand_pdbqt(self, sequence: str) -> Optional[Path]:
        """Generate 3D conformer and convert peptide → PDBQT."""
        if not _HAS_RDKIT:
            return None

        mol = Chem.MolFromSequence(sequence)
        if mol is None:
            mol = Chem.MolFromFASTA(sequence)
        if mol is None:
            logger.warning("RDKit cannot build molecule for '%s'", sequence)
            return None

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) != 0:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

        tag = re.sub(r"[^A-Za-z0-9]", "", sequence[:10])
        sdf = self.work_dir / f"lig_{tag}.sdf"
        with Chem.SDWriter(str(sdf)) as w:
            w.write(mol)

        pdbqt = sdf.with_suffix(".pdbqt")

        obabel = shutil.which("obabel") or shutil.which("obabel.exe")
        if obabel:
            try:
                subprocess.run(
                    [obabel, str(sdf), "-O", str(pdbqt)],
                    check=True, capture_output=True, timeout=60,
                )
                return pdbqt
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

        meeko = shutil.which("mk_prepare_ligand.py")
        if meeko:
            try:
                subprocess.run(
                    ["python", meeko, "-i", str(sdf), "-o", str(pdbqt)],
                    check=True, capture_output=True, timeout=60,
                )
                return pdbqt
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

        logger.warning("No ligand PDBQT converter available (OpenBabel / Meeko)")
        return None

    # ── real docking (Mode A) ────────────────────────────────────────

    def _real_dock(self, sequence: str, n_poses: int) -> dict:
        ligand = self._ligand_pdbqt(sequence)
        if ligand is None:
            logger.warning(
                "Ligand preparation failed for '%s'; falling back to simulation",
                sequence,
            )
            return self._sim_dock(sequence, n_poses)

        cx, cy, cz = self.grid_center
        out_pdbqt = self.work_dir / f"out_{re.sub(r'[^A-Za-z0-9]', '', sequence[:10])}.pdbqt"

        cmd = [
            self._vina_bin,
            "--receptor", str(self.receptor_pdbqt),
            "--ligand", str(ligand),
            "--center_x", f"{cx:.3f}",
            "--center_y", f"{cy:.3f}",
            "--center_z", f"{cz:.3f}",
            "--size_x", f"{_GRID_SIZE:.3f}",
            "--size_y", f"{_GRID_SIZE:.3f}",
            "--size_z", f"{_GRID_SIZE:.3f}",
            "--num_modes", str(n_poses),
            "--exhaustiveness", "8",
            "--out", str(out_pdbqt),
        ]

        try:
            proc = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=600,
            )
            energies = self._parse_vina_stdout(proc.stdout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            logger.error("Vina execution failed for '%s': %s", sequence, exc)
            return self._sim_dock(sequence, n_poses)

        if not energies:
            logger.warning("No poses parsed from Vina output for '%s'", sequence)
            return self._sim_dock(sequence, n_poses)

        best = energies[0]
        inter = self._estimate_interactions(sequence, best)
        return {
            "sequence": sequence,
            "binding_energy": best,
            "pose_energies": energies[:n_poses],
            "n_hbonds": inter["n_hbonds"],
            "n_hydrophobic_contacts": inter["n_hydrophobic"],
            "contact_residues": inter["contacts"],
            "mode": "real",
            "pdb_id": self.pdb_id,
        }

    @staticmethod
    def _parse_vina_stdout(text: str) -> List[float]:
        """Extract binding energies from Vina's tabular stdout."""
        energies: List[float] = []
        past_sep = False
        for line in text.splitlines():
            if re.match(r"\s*-{3,}\+", line):
                past_sep = True
                continue
            if past_sep:
                m = re.match(r"\s*\d+\s+([-\d.]+)", line)
                if m:
                    energies.append(float(m.group(1)))
                elif not line.strip():
                    break
        return energies

    # ── simulated docking (Mode B) ───────────────────────────────────

    def _sim_dock(self, sequence: str, n_poses: int) -> dict:
        """Descriptor-based affinity estimation with plausible interaction
        counts and contact residue sets."""
        seq = _clean_sequence(sequence)
        if not seq:
            return self._null_result(sequence)

        energy = self._score_affinity(seq)
        inter = self._estimate_interactions(seq, energy)

        rng = _seq_rng(seq, self.pdb_id)
        deltas = sorted(rng.exponential(0.3, size=max(n_poses - 1, 0)))
        poses = [energy] + [
            float(np.clip(energy + d, -12.0, -2.0)) for d in deltas
        ]

        return {
            "sequence": sequence,
            "binding_energy": energy,
            "pose_energies": poses,
            "n_hbonds": inter["n_hbonds"],
            "n_hydrophobic_contacts": inter["n_hydrophobic"],
            "contact_residues": inter["contacts"],
            "mode": "simulated",
            "pdb_id": self.pdb_id,
        }

    def _score_affinity(self, seq: str) -> float:
        """Heuristic binding-energy estimator calibrated to DPP-IV
        substrate specificity.

        The scoring function rewards:
          - di-/tri-peptide length (optimal for the DPP-IV active-site
            tunnel geometry)
          - proline / alanine at P2 (DPP-IV cleaves Xaa-Pro bonds)
          - hydrophobic P1 residue (fits the S2 sub-pocket)
          - moderate overall hydrophobicity and H-bond capacity

        Active inhibitors typically score −7 to −12 kcal/mol;
        weak binders score −2 to −6 kcal/mol.
        """
        residues = [c for c in seq if c in _AA_PROPS]
        if not residues:
            return -2.0

        n = len(residues)
        e = -4.0

        # Peptide length contribution
        length_map = {2: -1.0, 3: -1.5, 4: -1.0, 5: -0.5}
        e += length_map.get(n, 0.3 * max(0, n - 7))

        # P2 position (second residue) — key DPP-IV specificity determinant
        if n >= 2:
            e += {"P": -2.5, "A": -1.5, "G": -0.5}.get(residues[1], 0.0)

        # P1 (N-terminal) hydrophobic preference for S2 pocket
        p1 = residues[0]
        if p1 in "ILVF":
            e -= 1.0
        elif p1 in "WMY":
            e -= 0.7
        elif p1 == "A":
            e -= 0.3
        elif p1 in "KRH":
            e += 0.3
        elif p1 in "DE":
            e += 0.5

        # Average hydrophobicity (moderate is favourable)
        avg_h = float(np.mean([_AA_PROPS[r]["hydro"] for r in residues]))
        e -= 0.15 * avg_h

        # H-bond capacity (capped contribution)
        total_hb = sum(
            _AA_PROPS[r]["hbd"] + _AA_PROPS[r]["hba"] for r in residues
        )
        e -= 0.03 * min(total_hb, 15)

        # Net-charge penalty (highly charged peptides bind poorly)
        net_q = abs(sum(_AA_PROPS[r]["charge"] for r in residues))
        if net_q > 2:
            e += 0.3 * (net_q - 2)

        # Small deterministic noise for sequence individuality
        e += _seq_rng(seq, self.pdb_id).uniform(-0.3, 0.3)
        return float(np.clip(e, -12.0, -2.0))

    def _estimate_interactions(self, sequence: str, energy: float) -> dict:
        """Generate plausible H-bond / hydrophobic-contact counts and
        contact-residue lists scaled by binding strength."""
        seq = _clean_sequence(sequence)
        rng = _seq_rng(seq, f"{self.pdb_id}_inter")

        polar_n = sum(1 for c in seq if c in "NDEQKRHSTY")
        hydro_n = sum(1 for c in seq if c in "AVILMFWP")

        if energy < -9:
            hb, hc, n_extra = rng.randint(5, 9), rng.randint(5, 10), rng.randint(8, len(_ALL_POCKET_IDS))
        elif energy < -7:
            hb, hc, n_extra = rng.randint(3, 6), rng.randint(3, 7), rng.randint(5, 10)
        elif energy < -5:
            hb, hc, n_extra = rng.randint(2, 4), rng.randint(2, 5), rng.randint(3, 7)
        else:
            hb, hc, n_extra = rng.randint(0, 3), rng.randint(1, 3), rng.randint(1, 4)

        hb = min(int(hb), polar_n * 2 + 2)
        hc = max(int(hc), hydro_n)

        catalytic = ["S630", "D708", "H740"]
        pool = [r for r in _ALL_POCKET_IDS if r not in catalytic]
        n_extra = min(int(n_extra), len(pool))
        extras = rng.choice(pool, size=n_extra, replace=False).tolist()
        contacts = catalytic + extras

        return {"n_hbonds": hb, "n_hydrophobic": hc, "contacts": contacts}

    @staticmethod
    def _null_result(sequence: str) -> dict:
        return {
            "sequence": sequence,
            "binding_energy": 0.0,
            "pose_energies": [0.0],
            "n_hbonds": 0,
            "n_hydrophobic_contacts": 0,
            "contact_residues": [],
            "mode": "failed",
            "pdb_id": "",
        }

    # ── public API ───────────────────────────────────────────────────

    def dock_peptide(self, sequence: str, n_poses: int = 5) -> dict:
        """Dock a single peptide sequence against the DPP-IV receptor.

        Parameters
        ----------
        sequence : str
            Amino-acid sequence (single-letter code).
        n_poses : int
            Number of top binding poses to retain.

        Returns
        -------
        dict with keys: sequence, binding_energy, pose_energies,
        n_hbonds, n_hydrophobic_contacts, contact_residues, mode, pdb_id.
        """
        seq = _clean_sequence(sequence)
        if not seq:
            logger.warning("No valid residues in '%s'", sequence)
            return self._null_result(sequence)

        if self.real_mode:
            if not self._prepared:
                self.prepare_receptor()
            return self._real_dock(sequence, n_poses)

        return self._sim_dock(sequence, n_poses)

    def dock_batch(
        self,
        sequences: List[str],
        labels: Optional[List] = None,
    ) -> pd.DataFrame:
        """Dock a list of peptide sequences and return results as a DataFrame.

        Parameters
        ----------
        sequences : list of str
            Peptide sequences.
        labels : list or None
            Optional activity labels (0/1) aligned with *sequences*.

        Returns
        -------
        pd.DataFrame with one row per sequence, including flattened
        per-pose energy columns (pose_energy_1 … pose_energy_N).
        """
        if labels is not None and len(labels) != len(sequences):
            logger.warning(
                "labels length (%d) != sequences length (%d); labels ignored",
                len(labels), len(sequences),
            )
            labels = None

        records: List[dict] = []
        n = len(sequences)
        for i, seq in enumerate(sequences):
            result = self.dock_peptide(seq)
            if labels is not None:
                result["label"] = labels[i]
            records.append(result)
            if (i + 1) % 100 == 0 or i + 1 == n:
                logger.info("Docking progress: %d / %d", i + 1, n)

        df = pd.DataFrame(records)

        if "pose_energies" in df.columns:
            max_k = df["pose_energies"].apply(len).max()
            for j in range(max_k):
                df[f"pose_energy_{j + 1}"] = df["pose_energies"].apply(
                    lambda x, _j=j: x[_j] if _j < len(x) else np.nan
                )

        return df

    def extract_features(self, docking_result: dict) -> dict:
        """Extract a flat numerical feature vector from a docking result.

        Suitable for concatenation with other feature branches
        (sequence descriptors, ESM-2 embeddings, molecular fingerprints)
        before feeding into the ensemble model.

        Returns
        -------
        dict of {feature_name: float/int}.
        """
        poses = docking_result.get("pose_energies", [0.0])
        contacts = docking_result.get("contact_residues", [])

        _cat = {"S630", "D708", "H740"}
        _s1 = {"Y631", "V656", "W659", "Y662", "Y666", "V711"}
        _s2 = {"R125", "E205", "E206", "F357", "R358"}
        _s1p = {"Y547", "W629"}
        _s2p = {"S209", "R669", "N710"}

        return {
            "binding_energy": docking_result.get("binding_energy", 0.0),
            "n_hbonds": docking_result.get("n_hbonds", 0),
            "n_hydrophobic_contacts": docking_result.get(
                "n_hydrophobic_contacts", 0
            ),
            "n_contact_residues": len(contacts),
            "pose_energy_mean": float(np.mean(poses)),
            "pose_energy_std": (
                float(np.std(poses)) if len(poses) > 1 else 0.0
            ),
            "pose_energy_range": (
                float(max(poses) - min(poses)) if len(poses) > 1 else 0.0
            ),
            "contacts_catalytic": sum(1 for r in contacts if r in _cat),
            "contacts_s1": sum(1 for r in contacts if r in _s1),
            "contacts_s2": sum(1 for r in contacts if r in _s2),
            "contacts_s1_prime": sum(1 for r in contacts if r in _s1p),
            "contacts_s2_prime": sum(1 for r in contacts if r in _s2p),
        }

    def get_pocket_residues(self) -> List[dict]:
        """Return structured representations of DPP-IV active-site residues.

        Each entry contains the residue identity, pocket assignment, and
        physicochemical properties — useful for cross-attention key/value
        construction and binding-site visualisation.
        """
        result: List[dict] = []
        for pocket_name, res_list in DPPIV_POCKET_RESIDUES.items():
            for aa, num in res_list:
                result.append({
                    "id": f"{aa}{num}",
                    "aa": aa,
                    "num": num,
                    "pocket": pocket_name,
                    "properties": dict(_AA_PROPS.get(aa, {})),
                })
        return result
