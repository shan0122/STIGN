"""
PyTorch Dataset for video captioning.

Expects pre-extracted region feature `.pt` files produced by
``scripts/extract_features.py`` and caption annotations in the format
produced by ``scripts/prepare_real_data.py``.

Directory layout expected::

    data_root/
        features/
            <video_id>.pt     # torch.Tensor (T, N, feat_dim)
        annotations/
            train.json        # [{video_id, captions:[...]}]
            val.json
            test.json
        vocab.json            # saved Vocabulary file
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils.vocab import Vocabulary


class VideoDataset(Dataset):
    """Dataset that returns (region_feats, caption_ids, length) tuples.

    Args:
        ann_file:    Path to the annotation JSON (train/val/test).
        feat_dir:    Directory containing ``<video_id>.pt`` feature files.
        vocab:       Pre-built :class:`~utils.vocab.Vocabulary`.
        split:       One of ``"train"``, ``"val"``, or ``"test"``.
        max_frames:  Maximum number of frames to keep per video.
        max_cap_len: Maximum caption length (in tokens, excluding <sos>/<eos>).
        rand_cap:    If True pick a random caption per video (training);
                     if False always pick the first caption (val/test).
    """

    def __init__(
        self,
        ann_file: str | Path,
        feat_dir: str | Path,
        vocab: Vocabulary,
        split: str = "train",
        max_frames: int = 20,
        max_cap_len: int = 30,
        rand_cap: bool = True,
    ):
        self.feat_dir = Path(feat_dir)
        self.vocab = vocab
        self.split = split
        self.max_frames = max_frames
        self.max_cap_len = max_cap_len
        self.rand_cap = rand_cap

        with open(ann_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.samples: list[dict] = []
        for entry in raw:
            vid = entry["video_id"]
            feat_path = self.feat_dir / f"{vid}.pt"
            if not feat_path.exists():
                continue
            caps = entry.get("captions", [])
            if not caps:
                continue
            self.samples.append({"video_id": vid, "captions": caps})

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]
        vid = sample["video_id"]
        caps = sample["captions"]

        # Load pre-extracted features
        feat_path = self.feat_dir / f"{vid}.pt"
        feats = torch.load(feat_path, map_location="cpu")  # (T, N, D)

        # Truncate temporal dimension
        if feats.size(0) > self.max_frames:
            indices = torch.linspace(0, feats.size(0) - 1, self.max_frames).long()
            feats = feats[indices]

        # Pick caption
        cap_text = random.choice(caps) if self.rand_cap else caps[0]

        # Encode to token ids and truncate
        token_ids = self.vocab.encode(cap_text, add_special=True)
        if len(token_ids) > self.max_cap_len + 2:
            token_ids = token_ids[: self.max_cap_len + 1] + [self.vocab.eos_id]

        caption_tensor = torch.tensor(token_ids, dtype=torch.long)
        length = len(token_ids)

        return feats, caption_tensor, length


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, int]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of variable-length samples.

    Returns:
        feats:    (B, T_max, N, D) — padded with zeros
        captions: (B, L_max)       — padded with <pad>=0
        lengths:  (B,)             — actual caption lengths
    """
    feats_list, cap_list, len_list = zip(*batch)

    # Pad video features along time axis
    T_max = max(f.size(0) for f in feats_list)
    N = feats_list[0].size(1)
    D = feats_list[0].size(2)
    feats_padded = torch.zeros(len(feats_list), T_max, N, D)
    for i, f in enumerate(feats_list):
        feats_padded[i, : f.size(0)] = f

    # Pad captions
    caps_padded = pad_sequence(cap_list, batch_first=True, padding_value=0)
    lengths = torch.tensor(len_list, dtype=torch.long)

    return feats_padded, caps_padded, lengths
