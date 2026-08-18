# -*- coding: utf-8 -*-
"""
Fair retraining of the official StructuralDPPIV architecture.

Important corrections vs. the official code:
- The official repository's lightning_data_module.py reuses the independent
  test set as validation as well; this script instead splits off 15%
  validation from the benchmark train set and evaluates the frozen test set
  exactly once.
- Conflicting labels within the training set and exact train/test sequence
  overlaps are removed.
- The architecture, 21-D atom encoding, TextCNN, structural residual
  blocks, and FocalLoss all follow the official implementation.
"""
from __future__ import annotations

import io
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# NOTE (archival script, see src/benchmark_comparison/README.md): this path
# pointed at the original development machine and is not portable. Edit it
# to point at your own working directory before running.
ROOT = Path(r"C:\path\to\your\workdir")
REPO = ROOT / "scripts" / "benchmark_models" / "Structural-DPP-IV"
BERT_DATA = (
    ROOT
    / "scripts"
    / "benchmark_models"
    / "BERT-DPPIV"
    / "Fine_tune_data"
    / "DPP-IV_Dataset"
)
OUT = ROOT / "processed" / "benchmark_0714"
CKPT = ROOT / "checkpoints" / "benchmark_0714"
OUT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO))
from data.StructuralEncode import (
    convert_to_graph_channel,
    convert_to_graph_channel_returning_maxSeqLenx15xfn,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
SEED = 9999
MAX_LEN = 90
BATCH = 16
MAX_EPOCHS = 150
PATIENCE = 25
AA_DICT = {
    "A": 1, "R": 2, "N": 3, "D": 4, "C": 5, "Q": 6, "E": 7,
    "G": 8, "H": 9, "I": 10, "L": 11, "K": 12, "M": 13, "F": 14,
    "P": 15, "S": 17, "T": 19, "W": 20, "Y": 21, "V": 22,
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
            for sequence in read_fasta(BERT_DATA / f"{split}-{name}.txt"):
                rows.append((split, label, sequence))
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


def encode_text(seq: str) -> np.ndarray:
    arr = np.zeros(MAX_LEN, dtype=np.int64)
    vals = [AA_DICT[x] for x in seq[:MAX_LEN]]
    arr[: len(vals)] = vals
    return arr


def encode_struct(seq: str) -> np.ndarray:
    cube = convert_to_graph_channel(seq)
    return convert_to_graph_channel_returning_maxSeqLenx15xfn(
        cube, maxSeqLen=MAX_LEN
    ).astype(np.float32)


def encode_all(df: pd.DataFrame, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text_path = OUT / f"{prefix}_structural_text.npy"
    graph_path = OUT / f"{prefix}_structural_graph.npy"
    y_path = OUT / f"{prefix}_structural_y.npy"
    if text_path.exists() and graph_path.exists() and y_path.exists():
        return np.load(text_path), np.load(graph_path), np.load(y_path)
    texts, graphs = [], []
    for i, s in enumerate(df.sequence):
        if i % 100 == 0:
            print(f"encode {prefix}: {i}/{len(df)}")
        texts.append(encode_text(s))
        graphs.append(encode_struct(s))
    xt = np.stack(texts)
    xg = np.stack(graphs)
    y = df.label.to_numpy(dtype=np.int64)
    np.save(text_path, xt)
    np.save(graph_path, xg)
    np.save(y_path, y)
    return xt, xg, y


class PeptideDataset(Dataset):
    def __init__(self, xt, xg, y):
        self.xt = torch.from_numpy(xt).long()
        self.xg = torch.from_numpy(xg).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.xt[i], self.xg[i], self.y[i]


class ResBlock(nn.Module):
    def __init__(self, input_ch, output_ch, incre_dim=False):
        super().__init__()
        self.incre_dim = incre_dim
        padding_dim1 = 2 if MAX_LEN % 2 == 0 else 1
        self.conv = nn.Conv2d(
            input_ch, output_ch, (3, 3), stride=(2, 2),
            padding=(padding_dim1, 1)
        )
        self.conv1 = nn.Conv2d(
            input_ch, output_ch, (3, 3), stride=1, padding="same"
        )
        self.bn1 = nn.BatchNorm2d(input_ch)
        self.conv2 = nn.Conv2d(
            output_ch, output_ch, (3, 3), stride=1, padding="same"
        )
        self.bn2 = nn.BatchNorm2d(output_ch)

    def forward(self, x):
        original = x
        if self.incre_dim:
            x = F.max_pool2d(x, kernel_size=(2, 2), padding=1)
            original = self.conv(original)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + original


class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(24, 100)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 90, (fsz, 100)) for fsz in [1, 2]]
        )
        self.linear = nn.Linear(180, 1024)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        xs = [F.relu(conv(x)) for conv in self.convs]
        xs = [F.max_pool2d(v, (v.size(2), v.size(3))).flatten(1) for v in xs]
        return self.linear(torch.cat(xs, dim=1))


class StructuralBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(21, 32, (3, 3), stride=1, padding="same")
        self.res1 = ResBlock(32, 32)
        self.res2 = ResBlock(32, 64, incre_dim=True)
        self.linear = nn.Linear(23552, 1024)

    def forward(self, graph):
        x = graph.transpose(2, 3).transpose(1, 2)
        x = self.conv(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.linear(x.flatten(1))


class StructuralDPPIV(nn.Module):
    def __init__(self):
        super().__init__()
        self.text = TextCNN()
        self.struct = StructuralBranch()
        self.classifier = nn.Sequential(
            nn.Linear(1024, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, text, graph):
        fused = self.text(text) * self.struct(graph)
        return self.classifier(fused)


class FocalLoss(nn.Module):
    def __init__(self, alpha=(0.2, 0.8), gamma=3.0):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        idx = torch.arange(len(targets), device=targets.device)
        pt = p[idx, targets]
        return (-self.alpha[targets] * (1 - pt).pow(self.gamma) * logp[idx, targets]).mean()


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, ys = [], []
    for xt, xg, y in loader:
        xt, xg = xt.to(DEVICE), xg.to(DEVICE)
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            logits = model(xt, xg)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


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
    xt, xg, y = encode_all(train_df, "train")
    xtest, xgtest, ytest = encode_all(test_df, "test")

    indices = np.arange(len(y))
    tr_idx, va_idx = train_test_split(
        indices, test_size=0.15, stratify=y, random_state=SEED
    )
    train_loader = DataLoader(
        PeptideDataset(xt[tr_idx], xg[tr_idx], y[tr_idx]),
        batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        PeptideDataset(xt[va_idx], xg[va_idx], y[va_idx]),
        batch_size=BATCH * 2, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        PeptideDataset(xtest, xgtest, ytest),
        batch_size=BATCH * 2, shuffle=False, num_workers=0, pin_memory=True
    )

    model = StructuralDPPIV().to(DEVICE)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = FocalLoss().to(DEVICE)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    best_auc = -np.inf
    best_epoch = -1
    stale = 0
    history = []
    ckpt_path = CKPT / "structuraldppiv_fair.pt"
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for text, graph, target in train_loader:
            text = text.to(DEVICE, non_blocking=True)
            graph = graph.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                logits = model(text, graph)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

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
            f"epoch={epoch:03d} loss={np.mean(losses):.5f} "
            f"val_auc={val_auc:.4f} val_mcc={val_mcc:.4f}"
        )
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "seed": SEED,
                    "best_epoch": best_epoch,
                    "best_val_auc": best_auc,
                    "protocol": "train-internal validation; frozen independent test",
                },
                ckpt_path,
            )
        else:
            stale += 1
        if stale >= PATIENCE:
            print("Early stopping.")
            break

    saved = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    yt, pt = predict(model, test_loader)
    metrics = metric_dict(yt, pt)
    metrics.update({
        "Model": "StructuralDPPIV (official architecture, fair retraining)",
        "best_epoch": best_epoch,
        "best_val_auc": float(best_auc),
        "train_n": int(len(tr_idx)),
        "val_n": int(len(va_idx)),
        "test_n": int(len(ytest)),
        "elapsed_min": (time.time() - t0) / 60,
        "note": (
            "Official architecture and atom encoding. Independent test was not "
            "used for validation/model selection (fixed leakage in official loader)."
        ),
    })
    pd.DataFrame(history).to_csv(
        OUT / "structuraldppiv_training_history.csv",
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({
        "sequence": test_df.sequence,
        "label": ytest,
        "StructuralDPPIV_prob": pt,
    }).to_csv(
        OUT / "structuraldppiv_predictions.csv",
        index=False, encoding="utf-8-sig"
    )
    with open(OUT / "structuraldppiv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
