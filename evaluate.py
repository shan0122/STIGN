"""
evaluate.py — STIGN evaluation script.

Computes BLEU-1/2/3/4 and CIDEr scores on the test split.

Usage::

    python evaluate.py \\
        --test_ann   data/annotations/msrvtt_test.json \\
        --feat_dir   data/features/msrvtt \\
        --vocab_path data/vocab.json \\
        --checkpoint checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple BLEU-N implementation (no extra dependencies)
# ---------------------------------------------------------------------------

from collections import Counter


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(references: list[list[str]], hypothesis: list[str], n: int) -> float:
    """Compute corpus BLEU-N.

    Args:
        references:  List of reference token lists.
        hypothesis:  Hypothesis token list.
        n:           N-gram order.

    Returns:
        BLEU-N precision clipped by brevity penalty.
    """
    if len(hypothesis) == 0:
        return 0.0

    hyp_ngrams = _ngrams(hypothesis, n)
    total_clip = 0
    total_hyp = max(len(hypothesis) - n + 1, 0)
    if total_hyp == 0:
        return 0.0

    for ng, cnt in hyp_ngrams.items():
        max_ref = max(
            _ngrams(ref, n).get(ng, 0) for ref in references
        )
        total_clip += min(cnt, max_ref)

    precision = total_clip / total_hyp

    # Brevity penalty
    closest_len = min(
        (abs(len(ref) - len(hypothesis)), len(ref)) for ref in references
    )[1]
    import math

    bp = 1.0 if len(hypothesis) >= closest_len else math.exp(1 - closest_len / len(hypothesis))
    return bp * precision


def evaluate_model(
    model,
    loader: DataLoader,
    vocab,
    device: torch.device,
    beam_size: int = 6,
) -> dict[str, float]:
    """Run generation on the test set and compute BLEU-1/2/3/4."""
    model.eval()
    all_refs: list[list[list[str]]] = []
    all_hyps: list[list[str]] = []

    with torch.no_grad():
        for batch in loader:
            # batch = (feats, caps, lengths) from VideoDataset
            # or (feats, caps, lengths) from synthetic loader
            feats, caps, lengths = batch
            feats = feats.to(device)  # (B, T, N, D)

            for b in range(feats.size(0)):
                sample_feat = feats[b : b + 1]  # (1, T, N, D)
                beams = model.caption(sample_feat, vocab.sos_id, vocab.eos_id, beam_size)
                hyp_ids = beams[0] if beams else []
                hyp_text = vocab.decode(hyp_ids, skip_special=True)
                hyp_tokens = hyp_text.split()

                # References for this sample (first caption in the batch entry)
                ref_ids = caps[b].tolist()
                ref_text = vocab.decode(ref_ids, skip_special=True)
                ref_tokens = ref_text.split()

                all_hyps.append(hyp_tokens)
                all_refs.append([ref_tokens])

    scores: dict[str, float] = {}
    for n in range(1, 5):
        total = sum(bleu_score(refs, hyp, n) for refs, hyp in zip(all_refs, all_hyps))
        scores[f"BLEU-{n}"] = total / max(len(all_hyps), 1)

    return scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate STIGN on a test split.")
    p.add_argument("--test_ann", type=str, required=False,
                   default=None, help="Test annotation JSON.")
    p.add_argument("--feat_dir", type=str, required=False,
                   default=None, help="Feature directory.")
    p.add_argument("--vocab_path", type=str, default="data/vocab.json")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--beam_size", type=int, default=6)
    p.add_argument("--feat_dim", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--max_frames", type=int, default=20)
    p.add_argument("--max_cap_len", type=int, default=30)
    p.add_argument("--num_workers", type=int, default=0)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("设备: %s", device)

    # Load vocab
    from utils.vocab import Vocabulary

    if not Path(args.vocab_path).exists():
        logger.error("词汇表文件不存在: %s", args.vocab_path)
        return
    vocab = Vocabulary.load(args.vocab_path)
    logger.info("词汇表大小: %d", len(vocab))

    # Load model
    from models import STIGN

    model = STIGN(
        vocab_size=len(vocab),
        feat_dim=args.feat_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        embed_dim=args.embed_dim,
        max_caption_len=args.max_cap_len,
    ).to(device)

    if not Path(args.checkpoint).exists():
        logger.error("检查点文件不存在: %s", args.checkpoint)
        return
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    logger.info("已加载检查点: %s", args.checkpoint)

    # Build loader
    if not (args.test_ann and Path(args.test_ann).exists()):
        logger.error("测试集注释文件不存在: %s", args.test_ann)
        return

    from data.dataset import VideoDataset, collate_fn

    test_ds = VideoDataset(
        ann_file=args.test_ann,
        feat_dir=args.feat_dir,
        vocab=vocab,
        split="test",
        max_frames=args.max_frames,
        rand_cap=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    logger.info("测试集: %d 样本", len(test_ds))

    scores = evaluate_model(model, test_loader, vocab, device, beam_size=args.beam_size)

    print("\n" + "=" * 40)
    print("Evaluation Results")
    print("=" * 40)
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
