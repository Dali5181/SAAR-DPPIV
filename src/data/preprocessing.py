"""
Peptide dataset assembly and train/test split.

Pipeline actually used to produce ``data/train.csv`` / ``data/test.csv``:

  Step 1 (see ``cluster_split.py`` + ``rbm_hard_negatives.py``):
      literature peptides -> CD-HIT-style de-duplication (80% identity,
      ``CDHIT_THRESHOLD`` in ``src/config.py``) -> RBM-generated hard
      negatives merged in to address class imbalance -> the consolidated
      master table (``data/peptide_all.csv``: sequence, label, ic50_um,
      pic50, has_ic50, source).

  Step 2 (this module):
      stratified 80/20 split of the master table by ``label`` ->
      ``train.csv`` (n=1848) / ``test.csv`` (n=463) -> extract the
      617-D sequence feature vector per split (``sequence_features.py``).

Note: an earlier iteration of this project additionally combined a
ChEMBL small-molecule branch into the same classifier. That branch was
dropped before the reported results (the final classifier and all
manuscript numbers are peptide-only, sequence-feature-based) and its code
has been removed from this release to avoid confusion; the class balance
between "actives" and "RBM-generated hard negatives" is what accounts for
class 0 / class 1 in ``label``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
TEST_RATIO = 0.20


def split_and_extract(master_csv_path: str):
    """Reproduce the train/test split + 617-D feature extraction.

    Returns (train_df, test_df, X_train, X_test) where X_* are the 617-D
    sequence feature matrices aligned row-for-row with train_df/test_df.
    """
    from src.features.sequence_features import extract_sequence_features

    df = pd.read_csv(master_csv_path)
    df = df.dropna(subset=["sequence"]).copy()
    df["sequence"] = df["sequence"].astype(str)
    df = df[df["sequence"].str.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$")].reset_index(drop=True)

    train_df, test_df = train_test_split(
        df, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    X_train = extract_sequence_features(train_df["sequence"].tolist()).astype(np.float32)
    X_test = extract_sequence_features(test_df["sequence"].tolist()).astype(np.float32)
    return train_df, test_df, X_train, X_test


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", default="data/peptide_all.csv")
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    train_df, test_df, X_train, X_test = split_and_extract(args.master_csv)
    train_df.to_csv(f"{args.out_dir}/train.csv", index=False)
    test_df.to_csv(f"{args.out_dir}/test.csv", index=False)
    print(f"train.csv: {len(train_df)} rows  (pos={int((train_df.label == 1).sum())}, "
          f"neg={int((train_df.label == 0).sum())})")
    print(f"test.csv:  {len(test_df)} rows  (pos={int((test_df.label == 1).sum())}, "
          f"neg={int((test_df.label == 0).sum())})")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
