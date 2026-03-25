"""
train.py — STIGN video captioning training script.

Usage::

    python train.py \\
        --train_ann  data/annotations/msrvtt_train.json \\
        --val_ann    data/annotations/msrvtt_val.json \\
        --feat_dir   data/features/msrvtt \\
        --vocab_path data/vocab.json \\
        --epochs     30 \\
        --batch_size 32

Quick smoke-test (no real data needed)::

    python train.py --epochs 3 --batch_size 4 --eval_every 1 --val_samples 10
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        label = "CUDA"
        mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        label = "MPS (Apple Silicon)"
        mem = 0  # not exposed by PyTorch MPS API
    else:
        dev = torch.device("cpu")
        label = "CPU"
        mem = 0
    return dev, label, mem


# ---------------------------------------------------------------------------
# Vocabulary builder
# ---------------------------------------------------------------------------

def _build_vocab(ann_files: list[str], min_freq: int = 2, save_path: str | None = None):
    """Build or load vocabulary from annotation files."""
    import json

    from utils.vocab import Vocabulary

    if save_path and Path(save_path).exists():
        logger.info("Loading existing vocabulary from %s …", save_path)
        return Vocabulary.load(save_path)

    logger.info("构建词汇表…")
    vocab = Vocabulary(min_freq=min_freq)
    for ann_file in ann_files:
        if not Path(ann_file).exists():
            continue
        with open(ann_file, "r", encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            for cap in entry.get("captions", []):
                vocab.count_sentence(cap)

    vocab.build()
    logger.info("词汇表大小: %d", len(vocab))

    if save_path:
        vocab.save(save_path)
        logger.info("词汇表已保存到 %s", save_path)

    return vocab


# ---------------------------------------------------------------------------
# Synthetic dataset (for smoke-testing without real data)
# ---------------------------------------------------------------------------

class _SyntheticDataset(torch.utils.data.Dataset):
    """Tiny synthetic dataset used when real data files are unavailable."""

    def __init__(
        self,
        size: int = 200,
        num_frames: int = 10,
        num_regions: int = 8,
        feat_dim: int = 512,
        vocab_size: int = 500,
        max_cap_len: int = 15,
    ):
        self.size = size
        self.T = num_frames
        self.N = num_regions
        self.D = feat_dim
        self.V = vocab_size
        self.L = max_cap_len

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        feats = torch.randn(self.T, self.N, self.D)
        length = torch.randint(4, self.L + 1, ()).item()
        cap = torch.randint(4, self.V, (length + 2,))
        cap[0] = 1   # <sos>
        cap[-1] = 2  # <eos>
        return feats, cap, len(cap)


def _synth_collate(batch):
    from torch.nn.utils.rnn import pad_sequence

    feats_list, cap_list, len_list = zip(*batch)
    T = max(f.size(0) for f in feats_list)
    N, D = feats_list[0].size(1), feats_list[0].size(2)
    feats_padded = torch.zeros(len(feats_list), T, N, D)
    for i, f in enumerate(feats_list):
        feats_padded[i, : f.size(0)] = f
    caps_padded = pad_sequence(cap_list, batch_first=True, padding_value=0)
    lengths = torch.tensor(len_list, dtype=torch.long)
    return feats_padded, caps_padded, lengths


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device, dev_label, mem_gb = _get_device()

    print()
    print("=" * 60)
    print("STIGN 训练")
    print("=" * 60)
    print(f"设备: {dev_label} | 内存: {mem_gb}GB | Batch: {args.batch_size}")
    print(f"LR: {args.lr:.0e} | β: {args.beta} | Beam: {args.beam_size}")
    print(f"数据集: {args.dataset} | Epochs: {args.epochs}")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------ data
    use_synthetic = not (
        args.train_ann and Path(args.train_ann).exists()
        and args.feat_dir and Path(args.feat_dir).exists()
    )

    if use_synthetic:
        logger.info("真实数据未找到，使用合成数据进行测试 …")
        train_ds = _SyntheticDataset(size=max(args.batch_size * 4, 32))
        val_ds = _SyntheticDataset(size=max(args.batch_size, 8))
        vocab_size = train_ds.V
        feat_dim = train_ds.D
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=_synth_collate, num_workers=0
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=_synth_collate, num_workers=0
        )
    else:
        from data.dataset import VideoDataset, collate_fn

        vocab = _build_vocab(
            [args.train_ann, args.val_ann],
            min_freq=args.min_freq,
            save_path=args.vocab_path,
        )
        vocab_size = len(vocab)
        feat_dim = args.feat_dim

        train_ds = VideoDataset(
            ann_file=args.train_ann,
            feat_dir=args.feat_dir,
            vocab=vocab,
            split="train",
            max_frames=args.max_frames,
            rand_cap=True,
        )
        val_ann = args.val_ann if (args.val_ann and Path(args.val_ann).exists()) else args.train_ann
        val_ds_full = VideoDataset(
            ann_file=val_ann,
            feat_dir=args.feat_dir,
            vocab=vocab,
            split="val",
            max_frames=args.max_frames,
            rand_cap=False,
        )
        if args.val_samples and args.val_samples < len(val_ds_full):
            val_ds = Subset(val_ds_full, list(range(args.val_samples)))
        else:
            val_ds = val_ds_full

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True
        )

    logger.info("训练集: %d 样本 | 验证集: %d 样本", len(train_ds), len(val_ds))

    # ----------------------------------------------------------------- model
    from models import STIGN

    model = STIGN(
        vocab_size=vocab_size,
        feat_dim=feat_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        embed_dim=args.embed_dim,
        max_caption_len=args.max_cap_len,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("模型参数: %.2fM", n_params / 1e6)

    # --------------------------------------------------------------- optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val_loss = math.inf
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for feats, caps, lengths in train_loader:
            feats = feats.to(device)
            caps = caps.to(device)
            lengths = lengths.to(device)

            logits = model(feats, caps, lengths)   # (B, L-1, V)
            B, Lm1, V = logits.shape
            loss = criterion(logits.reshape(-1, V), caps[:, 1:].reshape(-1))

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        elapsed = time.time() - t0

        # ---- val ----
        val_loss = _evaluate_loss(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | %.1fs",
            epoch, args.epochs, train_loss, val_loss, elapsed,
        )

        if epoch % args.eval_every == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = save_dir / "best_model.pt"
                torch.save(model.state_dict(), ckpt)
                logger.info("  ✓ 新最优模型已保存至 %s (val_loss=%.4f)", ckpt, best_val_loss)

    logger.info("训练完成。最优验证损失: %.4f", best_val_loss)


def _evaluate_loss(model, loader, criterion, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for feats, caps, lengths in loader:
            feats = feats.to(device)
            caps = caps.to(device)
            lengths = lengths.to(device)
            logits = model(feats, caps, lengths)
            B, Lm1, V = logits.shape
            loss = criterion(logits.reshape(-1, V), caps[:, 1:].reshape(-1))
            total += loss.item()
    return total / max(len(loader), 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train STIGN video captioning model.")
    # Data
    p.add_argument("--train_ann", type=str, default=None)
    p.add_argument("--val_ann", type=str, default=None)
    p.add_argument("--feat_dir", type=str, default=None)
    p.add_argument("--vocab_path", type=str, default="data/vocab.json")
    p.add_argument("--dataset", type=str, default="TBD")
    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.4,
                   help="Beta hyperparameter (informational).")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--min_freq", type=int, default=2)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--val_samples", type=int, default=None,
                   help="Use only N validation samples (for fast eval).")
    # Model
    p.add_argument("--feat_dim", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--max_cap_len", type=int, default=30)
    p.add_argument("--beam_size", type=int, default=6)
    p.add_argument("--max_frames", type=int, default=20)
    # I/O
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--num_workers", type=int, default=0)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train(args)
