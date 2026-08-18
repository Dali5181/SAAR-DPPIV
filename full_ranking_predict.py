# -*- coding: utf-8 -*-
"""
Full ranking prediction with ESM-2 + LambdaRank.

This script reproduces the candidate-table scoring pipeline:
    617D sequence features + 480D ESM-2 embedding -> Top300 -> LambdaRank

Usage:
    python full_ranking_predict.py --input input.csv --output full_ranking_results.csv

Input file:
    CSV / XLSX with one column named one of:
    sequence, Sequence, SEQUENCE, seq, peptide, Peptide

Notes:
    - Requires torch and transformers.
    - The first run downloads facebook/esm2_t12_35M_UR50D if it is not cached.
    - This is slower than the web UI, but it matches the full candidate scoring workflow.
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def project_root() -> Path:
    return Path(__file__).resolve().parent


def clean_sequence(value: str) -> str:
    seq = re.sub(r"\s+", "", str(value).upper().strip())
    return "".join(c for c in seq if c in VALID_AA)


def read_input(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    seq_col = None
    for c in ["sequence", "Sequence", "SEQUENCE", "seq", "peptide", "Peptide"]:
        if c in df.columns:
            seq_col = c
            break
    if seq_col is None:
        raise ValueError("Input file must contain a sequence column.")

    out = df.copy()
    out["sequence"] = out[seq_col].apply(clean_sequence)
    out = out[out["sequence"].str.len().between(2, 100)]
    out = out[out["sequence"].str.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$")]
    out = out.drop_duplicates(subset=["sequence"], keep="first").reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV/XLSX with peptide sequences.")
    parser.add_argument("--output", default="full_ranking_results.csv", help="Output CSV/XLSX path.")
    parser.add_argument("--batch-size", type=int, default=32, help="ESM-2 batch size.")
    args = parser.parse_args()

    root = project_root()
    sys.path.insert(0, str(root))
    from src.models.pipeline_scoring import score_sequences, priority_level

    print("Loading input...")
    df = read_input(args.input)
    seqs = df["sequence"].tolist()
    print(f"Valid unique sequences: {len(seqs)}")

    print("Scoring (617D features + ESM-2 + LambdaRank)...")
    res = score_sequences(seqs, use_esm=True, batch_size=args.batch_size)
    if not res["used_esm"]:
        print("WARNING: ESM-2 could not be loaded (install torch + transformers). "
              "Rank scores fell back to zero-padded features and will differ from "
              "the candidate-table workflow.")

    p_active = res["p_active"]
    rank_score = res["rank_score"]
    combined_score = res["combined"]
    pred_pic50 = res["pic50"]
    pred_ic50_um = 10 ** (-pred_pic50) * 1e6

    out = df.copy()
    out["len"] = out["sequence"].str.len()
    out["classification_probability"] = p_active
    out["source_aware_rank_score"] = rank_score
    out["combined_score"] = combined_score
    out["priority_level"] = [priority_level(s) for s in combined_score]
    out["auxiliary_pic50"] = pred_pic50
    out["auxiliary_ic50_um"] = pred_ic50_um
    out = out.sort_values("combined_score", ascending=False).reset_index(drop=True)

    if args.output.lower().endswith(".xlsx"):
        out.to_excel(args.output, index=False)
    else:
        out.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Saved: {args.output}")
    cols = ["sequence", "len", "classification_probability", "source_aware_rank_score",
            "combined_score", "priority_level", "auxiliary_pic50"]
    print(out[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

