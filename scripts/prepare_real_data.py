"""
scripts/prepare_real_data.py
============================
Parse raw MSR-VTT and MSVD annotation files and write train/val/test
annotation JSONs in the unified format expected by VideoDataset::

    [{"video_id": "video0", "captions": ["a man is ...", ...]}, ...]

Supported input formats
-----------------------
MSR-VTT
~~~~~~~
Format A (official train_val_videodatainfo.json / videodatainfo_2016.json)::

    {
      "videos": [{"video_id": "video0", "split": "train", ...}, ...],
      "sentences": [{"video_id": "video0", "caption": "...", ...}, ...]
    }

Format B (some re-distributed versions with "annotations" key)::

    {
      "annotations": [{"image_id": "video0", "caption": "...", ...}]
    }

MSVD
~~~~
Format A (list of dicts with Description field)::

    [{"videoID": "vid0", "Description": "...", "Language": "English"}, ...]

Format B (dict mapping video-id to list of captions)::

    {"vid0": ["caption one", "caption two", ...], ...}

Format C (flat list of plain strings, one per line — treated as single video)
    Rarely used; included for completeness.

Usage
-----
    python scripts/prepare_real_data.py \\
        --msrvtt_json  data/raw/MSR-VTT/train_val_videodatainfo.json \\
        --msvd_json    data/raw/MSVD/annotations.json \\
        --output_dir   data/annotations \\
        --dataset      msrvtt           # or msvd / both
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("  Wrote %d entries → %s", len(data), path)


# ---------------------------------------------------------------------------
# MSR-VTT parsers
# ---------------------------------------------------------------------------

def _parse_msrvtt(raw: Any) -> dict[str, dict[str, Any]]:
    """Return {video_id: {"split": str, "captions": [str, ...]}} for MSR-VTT.

    Handles two common JSON layouts.
    """
    videos: dict[str, dict] = {}

    # ---- Detect format ----
    if isinstance(raw, dict) and "sentences" in raw:
        # Format A: official videodatainfo JSON
        # Build split mapping from videos list
        split_map: dict[str, str] = {}
        for v in raw.get("videos", []):
            vid_id = v.get("video_id") or v.get("id") or v.get("videoID")
            if vid_id is None:
                continue
            vid_id = str(vid_id)
            split = v.get("split", "train")
            split_map[vid_id] = split

        for sent in raw["sentences"]:
            vid_id = sent.get("video_id") or sent.get("videoID") or sent.get("image_id")
            caption = sent.get("caption") or sent.get("Description") or ""
            if not vid_id or not caption:
                continue
            vid_id = str(vid_id)
            if vid_id not in videos:
                videos[vid_id] = {
                    "split": split_map.get(vid_id, "train"),
                    "captions": [],
                }
            videos[vid_id]["captions"].append(caption.strip())

    elif isinstance(raw, dict) and "annotations" in raw:
        # Format B: COCO-style annotations
        for ann in raw["annotations"]:
            vid_id = str(ann.get("image_id") or ann.get("video_id") or "")
            caption = ann.get("caption") or ann.get("Description") or ""
            if not vid_id or not caption:
                continue
            if vid_id not in videos:
                videos[vid_id] = {"split": "train", "captions": []}
            videos[vid_id]["captions"].append(caption.strip())

    elif isinstance(raw, list):
        # Format C: flat list of sentence dicts
        for item in raw:
            vid_id = str(
                item.get("video_id") or item.get("videoID") or item.get("image_id") or ""
            )
            caption = item.get("caption") or item.get("Description") or ""
            split = item.get("split", "train")
            if not vid_id or not caption:
                continue
            if vid_id not in videos:
                videos[vid_id] = {"split": split, "captions": []}
            videos[vid_id]["captions"].append(caption.strip())
    else:
        logger.warning(
            "Unrecognised MSR-VTT JSON structure (top-level keys: %s). "
            "No captions extracted.",
            list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__,
        )

    return videos


# ---------------------------------------------------------------------------
# MSVD parsers
# ---------------------------------------------------------------------------

def _parse_msvd(raw: Any) -> dict[str, list[str]]:
    """Return {video_id: [caption, ...]} for MSVD.

    Handles three common JSON layouts.
    """
    captions: dict[str, list[str]] = defaultdict(list)

    if isinstance(raw, list):
        # Format A: list of dicts with videoID / Description keys
        for item in raw:
            vid_id = str(
                item.get("videoID") or item.get("video_id") or item.get("image_id") or ""
            )
            caption = item.get("Description") or item.get("caption") or ""
            lang = item.get("Language") or item.get("language") or "English"
            if not vid_id or not caption:
                continue
            if lang.lower() != "english":
                continue
            captions[vid_id].append(caption.strip())

    elif isinstance(raw, dict):
        # Format B: {video_id: [cap1, cap2, ...]}
        for vid_id, value in raw.items():
            if isinstance(value, list):
                for cap in value:
                    if isinstance(cap, str) and cap.strip():
                        captions[vid_id].append(cap.strip())
            elif isinstance(value, str) and value.strip():
                captions[vid_id].append(value.strip())
    else:
        logger.warning(
            "Unrecognised MSVD JSON structure (%s). No captions extracted.",
            type(raw).__name__,
        )

    return dict(captions)


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def _split_ids(
    ids: list[str],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


def _build_annotation_list(video_ids: list[str], cap_map: dict[str, list[str]]) -> list[dict]:
    result = []
    for vid in video_ids:
        caps = cap_map.get(vid, [])
        if caps:
            result.append({"video_id": vid, "captions": caps})
    return result


# ---------------------------------------------------------------------------
# MSR-VTT entry point
# ---------------------------------------------------------------------------

def prepare_msrvtt(json_path: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    logger.info("Parsing MSR-VTT from %s …", json_path)

    raw = _load_json(json_path)
    videos = _parse_msrvtt(raw)

    if not videos:
        logger.error(
            "MSR-VTT: 0 videos parsed. "
            "Please check that the JSON file matches one of the supported formats "
            "(see script docstring for details)."
        )
        return

    # Group by split
    by_split: dict[str, list[dict]] = defaultdict(list)
    for vid_id, info in videos.items():
        if not info["captions"]:
            continue
        entry = {"video_id": vid_id, "captions": info["captions"]}
        by_split[info["split"]].append(entry)

    # If all captions end up in "train" (no split info in file), auto-split
    if len(by_split) == 1 and "train" in by_split:
        logger.info("No split information found — performing 80/10/10 auto-split.")
        all_ids = [e["video_id"] for e in by_split["train"]]
        cap_map = {e["video_id"]: e["captions"] for e in by_split["train"]}
        train_ids, val_ids, test_ids = _split_ids(all_ids)
        by_split = {
            "train": _build_annotation_list(train_ids, cap_map),
            "val": _build_annotation_list(val_ids, cap_map),
            "test": _build_annotation_list(test_ids, cap_map),
        }

    for split, entries in by_split.items():
        _write_json(entries, output_dir / f"msrvtt_{split}.json")
        logger.info("  MSR-VTT %s: %d 个视频的有效字幕。", split, len(entries))

    total = sum(len(v) for v in by_split.values())
    logger.info("MSR-VTT 共提取 %d 个视频的有效字幕。", total)


# ---------------------------------------------------------------------------
# MSVD entry point
# ---------------------------------------------------------------------------

def prepare_msvd(json_path: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    logger.info("Parsing MSVD from %s …", json_path)

    raw = _load_json(json_path)
    cap_map = _parse_msvd(raw)

    if not cap_map:
        logger.error(
            "MSVD: 0 videos parsed. "
            "Please check that the JSON file matches one of the supported formats "
            "(see script docstring for details)."
        )
        return

    all_ids = list(cap_map.keys())
    train_ids, val_ids, test_ids = _split_ids(all_ids)

    splits = {
        "train": _build_annotation_list(train_ids, cap_map),
        "val": _build_annotation_list(val_ids, cap_map),
        "test": _build_annotation_list(test_ids, cap_map),
    }

    for split, entries in splits.items():
        _write_json(entries, output_dir / f"msvd_{split}.json")
        logger.info("  MSVD %s 集提取了 %d 个视频的有效字幕。", split, len(entries))

    logger.info("MSVD 共提取 %d 个视频的有效字幕。", len(all_ids))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare MSR-VTT and/or MSVD annotation files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--msrvtt_json", type=str, default=None,
                   help="Path to MSR-VTT annotation JSON.")
    p.add_argument("--msvd_json", type=str, default=None,
                   help="Path to MSVD annotation JSON.")
    p.add_argument("--output_dir", type=str, default="data/annotations",
                   help="Directory to write output JSONs (default: data/annotations).")
    p.add_argument(
        "--dataset",
        type=str,
        default="both",
        choices=["msrvtt", "msvd", "both"],
        help="Which dataset to process (default: both).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    process_msrvtt = args.dataset in ("msrvtt", "both")
    process_msvd = args.dataset in ("msvd", "both")

    if process_msrvtt:
        if not args.msrvtt_json:
            logger.error("--msrvtt_json is required when --dataset is 'msrvtt' or 'both'.")
        else:
            prepare_msrvtt(args.msrvtt_json, args.output_dir)

    if process_msvd:
        if not args.msvd_json:
            logger.error("--msvd_json is required when --dataset is 'msvd' or 'both'.")
        else:
            prepare_msvd(args.msvd_json, args.output_dir)


if __name__ == "__main__":
    main()
