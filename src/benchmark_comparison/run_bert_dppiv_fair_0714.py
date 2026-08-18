# -*- coding: utf-8 -*-
"""
Reproducible transfer-learning re-implementation of the BERT-DPPIV method.

The original official repository depends on TensorFlow 1.x, and its official
pretrained-weight archive (2.89 GB) cannot be run directly in the current
Python 3.11 environment. This script preserves the core method (protein
language-model pretrained representation + downstream DPP-IV binary
fine-tuning), using ESM-2 35M -- another protein PLM -- as a runnable
backbone.

Any reported result must be labelled "BERT-DPPIV re-implementation (PLM
transfer)"; it must not be presented as a reproduction of the official
pretrained weights.
"""
from __future__ import annotations

import io
import json
import random
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# NOTE (archival script, see src/benchmark_comparison/README.md): this path
# pointed at the original development machine and is not portable. Edit it
# to point at your own working directory before running.
ROOT = Path(r"C:\path\to\your\workdir")
BERT_DATA = (
    ROOT
    / "scripts"
    / "benchmark_models"
    / "BERT-DPPIV"
    / "Fine_tune_data"
    / "DPP-IV_Dataset"
)
OUT = ROOT / "processed" / "benchmark_0714"
CKPT = ROOT / "checkpoints" / "benchmark_0714" / "bert_dppiv_plm_reimplementation"
OUT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH = 8
ACCUM = 2
MAX_EPOCHS = 20
PATIENCE = 5
MAX_LEN = 90
print(f"Using device: {DEVICE}")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_fasta(path: Path) -> list[str]:
    return [
        x.strip().upper()
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip() and not x.startswith(">")
    ]


def load_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for split in ("train", "test"):
        for name, label in (("positive", 1), ("negative", 0)):
            rows.extend(
                (split, label, seq)
                for seq in read_fasta(BERT_DATA / f"{split}-{name}.txt")
            )
    raw = pd.DataFrame(rows, columns=["split", "label", "sequence"])
    tr = raw[raw.split == "train"].copy()
    te = raw[raw.split == "test"].copy()
    conflicts = set(
        tr.groupby("sequence").label.nunique().loc[lambda s: s > 1].index
    )
    overlap = set(tr.sequence) & set(te.sequence)
    tr = tr[~tr.sequence.isin(conflicts | overlap)].drop_duplicates(
        ["sequence", "label"]
    )
    te = te.drop_duplicates(["sequence", "label"])
    assert not (set(tr.sequence) & set(te.sequence))
    return tr.reset_index(drop=True), te.reset_index(drop=True)


class SeqDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer):
        self.frame = frame.reset_index(drop=True)
        self.enc = tokenizer(
            self.frame.sequence.tolist(),
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN + 2,
            return_tensors="pt",
        )
        self.labels = torch.tensor(self.frame.label.to_numpy(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        labels.append(batch["labels"].numpy())
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            logits = model(**batch).logits
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(labels), np.concatenate(probs)


def metric_dict(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "ACC": float(accuracy_score(y, pred)),
        "MCC": float(matthews_corrcoef(y, pred)),
        "Sn": float(tp / (tp + fn)),
        "Sp": float(tn / (tn + fp)),
        "AUC": float(roc_auc_score(y, p)),
        "F1": float(f1_score(y, pred)),
        "PR_AUC": float(average_precision_score(y, p)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def main():
    seed_all(SEED)
    train_df, test_df = load_clean()
    tr_idx, va_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=0.15,
        stratify=train_df.label,
        random_state=SEED,
    )
    fit_df = train_df.iloc[tr_idx].reset_index(drop=True)
    val_df = train_df.iloc[va_idx].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(DEVICE)

    # Freeze lower layers; fine-tune last 4 transformer blocks + classifier.
    for param in model.base_model.parameters():
        param.requires_grad = False
    encoder = getattr(model.base_model, "encoder", None)
    layers = getattr(encoder, "layer", []) if encoder is not None else []
    for layer in layers[-4:]:
        for param in layer.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        "/",
        sum(p.numel() for p in model.parameters()),
    )

    train_loader = DataLoader(
        SeqDataset(fit_df, tokenizer),
        batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        SeqDataset(val_df, tokenizer),
        batch_size=BATCH * 2, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        SeqDataset(test_df, tokenizer),
        batch_size=BATCH * 2, shuffle=False, num_workers=0
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5,
        weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    best_auc, stale, best_epoch = -np.inf, 0, -1
    history = []
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                out = model(**batch)
                loss = out.loss / ACCUM
            scaler.scale(loss).backward()
            if step % ACCUM == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()) * ACCUM)

        yv, pv = predict(model, val_loader)
        val_auc = roc_auc_score(yv, pv)
        val_mcc = matthews_corrcoef(yv, (pv >= 0.5).astype(int))
        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_auc": float(val_auc),
            "val_mcc": float(val_mcc),
        })
        print(
            f"epoch={epoch:02d} loss={np.mean(losses):.5f} "
            f"val_auc={val_auc:.4f} val_mcc={val_mcc:.4f}"
        )
        if val_auc > best_auc + 1e-4:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            model.save_pretrained(CKPT)
            tokenizer.save_pretrained(CKPT)
        else:
            stale += 1
        if stale >= PATIENCE:
            print("Early stopping.")
            break

    model = AutoModelForSequenceClassification.from_pretrained(CKPT).to(DEVICE)
    yt, pt = predict(model, test_loader)
    metrics = metric_dict(yt, pt)
    metrics.update({
        "Model": "BERT-DPPIV (PLM transfer re-implementation)",
        "backbone": MODEL_NAME,
        "best_epoch": best_epoch,
        "best_val_auc": float(best_auc),
        "train_n": int(len(fit_df)),
        "val_n": int(len(val_df)),
        "test_n": int(len(test_df)),
        "elapsed_min": (time.time() - t0) / 60,
        "implementation": (
            "Method reimplementation, not official checkpoint reproduction. "
            "Official TensorFlow-1 checkpoint is incompatible with current runtime; "
            "core transfer-learning protocol retained using ESM-2 protein PLM."
        ),
    })
    pd.DataFrame(history).to_csv(
        OUT / "bert_dppiv_training_history.csv",
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({
        "sequence": test_df.sequence,
        "label": yt,
        "BERT_DPPIV_prob": pt,
    }).to_csv(
        OUT / "bert_dppiv_predictions.csv",
        index=False, encoding="utf-8-sig"
    )
    with open(OUT / "bert_dppiv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
