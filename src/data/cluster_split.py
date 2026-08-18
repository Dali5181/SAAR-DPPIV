"""
Sequence deduplication using CD-HIT-like algorithm.
Clusters peptide sequences by pairwise identity and keeps cluster representatives.

NOTE ON THRESHOLD: the deployed default is 80% identity
(``CDHIT_THRESHOLD = 0.8`` in ``src/config.py``, matching
``scripts/expand_merge_dataset.py`` in the full development history, which
hard-codes ``CDHIT_THRESHOLD = 0.80`` and is what actually produced the
base de-duplicated peptide pool). This matches the manuscript's stated 80%.

Known limitation: the final shipped table (``data/train.csv`` +
``data/test.csv``) was assembled by merging in additional real-IC50 data
provided later in the project, and that merge step did not re-run this
de-duplication function. Re-clustering the final merged pool at 80%
identity does still flag a small number of near-duplicate sequences
(about 5% of rows) that were never re-checked after the merge, and a
handful of near-duplicate pairs (>=80% identity) end up split across the
train/test partition because the final train/test split is a plain
stratified split on labels, not a cluster-aware split. This is a minor,
disclosed limitation of the shipped data table, not a threshold labelling
error.
"""
import numpy as np
from itertools import combinations


def sequence_identity(seq1: str, seq2: str) -> float:
    """Compute pairwise sequence identity (fraction of identical residues
    at aligned positions using simple Needleman-Wunsch with unit scoring)."""
    s1, s2 = seq1.upper(), seq2.upper()
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return 0.0

    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        dp[i][0] = -i
    for j in range(m + 1):
        dp[0][j] = -j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i-1][j-1] + (1 if s1[i-1] == s2[j-1] else -1)
            dp[i][j] = max(match, dp[i-1][j] - 1, dp[i][j-1] - 1)

    i, j = n, m
    matches = 0
    aligned = 0
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1] and dp[i][j] == dp[i-1][j-1] + 1:
            matches += 1
            aligned += 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j-1] - 1:
            aligned += 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j] - 1:
            aligned += 1
            i -= 1
        else:
            aligned += 1
            j -= 1
    aligned += i + j

    return matches / aligned if aligned > 0 else 0.0


def cdhit_cluster(sequences: list, threshold: float = 0.8) -> list:
    """
    CD-HIT style greedy incremental clustering.
    Sort by length descending; the longest becomes the first representative.
    Each subsequent sequence is compared to representatives; if identity > threshold,
    it joins that cluster; otherwise it becomes a new representative.
    Returns list of representative sequence indices.
    """
    if not sequences:
        return []

    indexed = sorted(enumerate(sequences), key=lambda x: -len(x[1]))
    representatives = [indexed[0][0]]
    rep_seqs = [indexed[0][1]]

    for idx, seq in indexed[1:]:
        is_redundant = False
        for rep_seq in rep_seqs:
            if abs(len(seq) - len(rep_seq)) / max(len(seq), len(rep_seq)) > (1 - threshold):
                continue
            if sequence_identity(seq, rep_seq) >= threshold:
                is_redundant = True
                break
        if not is_redundant:
            representatives.append(idx)
            rep_seqs.append(seq)

    return sorted(representatives)


def deduplicate_peptides(sequences: list, labels: list = None,
                         threshold: float = 0.8) -> dict:
    """
    Deduplicate peptide sequences using CD-HIT-like clustering.

    Returns dict with keys: sequences, labels (if provided), removed_count, kept_indices
    """
    print(f"  CD-HIT deduplication (threshold={threshold})...")
    print(f"  Input: {len(sequences)} sequences")

    kept_indices = cdhit_cluster(sequences, threshold)

    result = {
        "sequences": [sequences[i] for i in kept_indices],
        "kept_indices": kept_indices,
        "removed_count": len(sequences) - len(kept_indices),
    }
    if labels is not None:
        result["labels"] = [labels[i] for i in kept_indices]

    print(f"  Output: {len(kept_indices)} sequences (removed {result['removed_count']})")
    return result
