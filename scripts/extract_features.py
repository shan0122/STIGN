"""
scripts/extract_features.py
============================
Extract YOLOv5 region features from video frames and save them as PyTorch
tensors that can be consumed directly by VideoDataset.

Dependencies (install before running)::

    pip install torch torchvision opencv-python tqdm pandas

Note on ``pandas``
------------------
YOLOv5's internal utilities (non_max_suppression, etc.) import ``pandas``
at module load time.  If pandas is not installed you will see::

    ModuleNotFoundError: No module named 'pandas'

Install it with: ``pip install pandas``  (or ``pip install -r requirements.txt``).

Output layout::

    <feat_dir>/
        <video_id>.pt   # torch.Tensor of shape (T, N, feat_dim)

Usage::

    python scripts/extract_features.py \\
        --video_dir   data/raw/MSR-VTT/videos \\
        --feat_dir    data/features/msrvtt \\
        --frame_rate  1 \\
        --max_frames  20 \\
        --num_regions 36 \\
        --img_size    640
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# pandas is required by YOLOv5 internals; import it early so that we give
# a clear error message if it is missing.
try:
    import pandas  # noqa: F401  — imported for side-effects / early error check
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The 'pandas' package is required by YOLOv5 but is not installed.\n"
        "Install it with:  pip install pandas\n"
        "Or install all dependencies:  pip install -r requirements.txt"
    ) from exc

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def _sample_frames(
    video_path: str | Path,
    frame_rate: float = 1.0,
    max_frames: int = 20,
) -> list[np.ndarray]:
    """Sample frames from a video file.

    Args:
        video_path: Path to the video file.
        frame_rate: Frames per second to sample (1 = one frame per second).
        max_frames: Hard cap on the total number of frames returned.

    Returns:
        List of BGR uint8 numpy arrays, one per sampled frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(fps / frame_rate))

    frames: list[np.ndarray] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        frame_idx += 1

    cap.release()

    # Uniform sub-sampling if we still have too many
    if len(frames) > max_frames:
        idxs = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in idxs]

    return frames


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class RegionFeatureExtractor:
    """Wrap YOLOv5 to extract fixed-size region feature grids.

    If ``torch.hub`` cannot load YOLOv5 (e.g. offline environment), the
    extractor falls back to a lightweight CNN grid feature baseline.

    Args:
        model_name: YOLOv5 variant, e.g. ``'yolov5s'``, ``'yolov5m'``.
        num_regions: Fixed number of region proposals to keep per frame.
        img_size:    Input image size (pixels, square).
        device:      Torch device string.
    """

    def __init__(
        self,
        model_name: str = "yolov5m",
        num_regions: int = 36,
        img_size: int = 640,
        device: str = "cpu",
    ):
        self.num_regions = num_regions
        self.img_size = img_size
        self.device = torch.device(device)
        self._feat_dim: int = 2048  # ResNet backbone output channels

        logger.info("Loading YOLOv5 model '%s' …", model_name)
        try:
            self._model = torch.hub.load(
                "ultralytics/yolov5",
                model_name,
                pretrained=True,
                verbose=False,
            )
            self._model.eval().to(self.device)
            self._use_yolo = True
            # The feature dimension exposed by YOLOv5 backbone is 512 for small/medium
            self._feat_dim = 512
            logger.info("YOLOv5 loaded successfully (feat_dim=%d).", self._feat_dim)
        except Exception as e:
            logger.warning(
                "Failed to load YOLOv5 (%s). Falling back to grid features.", e
            )
            self._model = self._build_fallback_cnn()
            self._use_yolo = False

    def _build_fallback_cnn(self) -> torch.nn.Module:
        """Build a lightweight ResNet18 grid-feature extractor as fallback."""
        import torchvision.models as tvm

        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        # Remove the final avgpool + fc to keep spatial grid
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone.eval().to(self.device)
        self._feat_dim = 512
        return backbone

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    @torch.no_grad()
    def extract_frame(self, bgr_frame: np.ndarray) -> torch.Tensor:
        """Extract region features from a single BGR frame.

        Returns:
            Tensor of shape (num_regions, feat_dim).
        """
        if self._use_yolo:
            return self._extract_yolo(bgr_frame)
        return self._extract_grid(bgr_frame)

    def _extract_yolo(self, bgr_frame: np.ndarray) -> torch.Tensor:
        """Use YOLOv5 detections as regions."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self._model(rgb, size=self.img_size)
        # results.xyxy[0]: (num_dets, 6) — x1,y1,x2,y2,conf,cls
        dets = results.xyxy[0]  # Tensor on device

        h, w = bgr_frame.shape[:2]
        feat_list: list[torch.Tensor] = []

        for det in dets:
            x1, y1, x2, y2 = det[:4].cpu().numpy()
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            patch = bgr_frame[y1:y2, x1:x2]
            feat = self._pool_patch(patch)
            feat_list.append(feat)

        return self._pad_or_trim(feat_list)

    def _extract_grid(self, bgr_frame: np.ndarray) -> torch.Tensor:
        """Fallback: uniform grid sampling via CNN feature map."""
        import torchvision.transforms.functional as TF

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        from PIL import Image

        img = Image.fromarray(rgb)
        img_t = TF.resize(img, [self.img_size, self.img_size])
        img_t = TF.to_tensor(img_t).unsqueeze(0).to(self.device)
        img_t = TF.normalize(
            img_t.squeeze(0), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ).unsqueeze(0)

        feat_map = self._model(img_t)  # (1, C, H, W)
        C, H, W = feat_map.shape[1:]
        # Flatten spatial positions as "regions"
        regions = feat_map.squeeze(0).view(C, -1).T  # (H*W, C)
        return self._pad_or_trim(list(regions))

    def _pool_patch(self, patch: np.ndarray) -> torch.Tensor:
        """Average-pool a crop to a feature vector via the fallback CNN."""
        import torchvision.transforms.functional as TF
        from PIL import Image

        if patch.size == 0:
            return torch.zeros(self._feat_dim, device=self.device)

        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img_t = TF.resize(img, [64, 64])
        img_t = TF.to_tensor(img_t).unsqueeze(0).to(self.device)
        img_t = TF.normalize(
            img_t.squeeze(0), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ).unsqueeze(0)

        feat_map = self._model(img_t)  # (1, C, h, w)
        return feat_map.mean(dim=[2, 3]).squeeze(0)

    def _pad_or_trim(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Ensure exactly ``num_regions`` features (pad or trim)."""
        if len(feats) == 0:
            return torch.zeros(self.num_regions, self._feat_dim, device=self.device)
        stacked = torch.stack(
            [f.to(self.device) if isinstance(f, torch.Tensor) else torch.tensor(f, device=self.device)
             for f in feats[:self.num_regions]]
        )  # (≤num_regions, D)
        if stacked.size(0) < self.num_regions:
            pad = torch.zeros(
                self.num_regions - stacked.size(0), stacked.size(1), device=self.device
            )
            stacked = torch.cat([stacked, pad], dim=0)
        return stacked.cpu()


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract_all(
    video_dir: str | Path,
    feat_dir: str | Path,
    frame_rate: float = 1.0,
    max_frames: int = 20,
    num_regions: int = 36,
    img_size: int = 640,
    model_name: str = "yolov5m",
    device: str = "cpu",
    overwrite: bool = False,
) -> None:
    video_dir = Path(video_dir)
    feat_dir = Path(feat_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        p for p in video_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_VIDEO_EXTS
    )
    if not video_files:
        logger.error("No video files found in %s", video_dir)
        return

    logger.info("Found %d videos in %s", len(video_files), video_dir)

    extractor = RegionFeatureExtractor(
        model_name=model_name,
        num_regions=num_regions,
        img_size=img_size,
        device=device,
    )
    feat_dim = extractor.feat_dim

    skipped = 0
    for video_path in tqdm(video_files, desc="Extracting features"):
        video_id = video_path.stem
        out_path = feat_dir / f"{video_id}.pt"

        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        frames = _sample_frames(video_path, frame_rate=frame_rate, max_frames=max_frames)
        if not frames:
            logger.warning("Skipping %s — could not decode frames.", video_path.name)
            continue

        frame_feats: list[torch.Tensor] = []
        for frame in frames:
            feat = extractor.extract_frame(frame)  # (num_regions, feat_dim)
            frame_feats.append(feat)

        video_tensor = torch.stack(frame_feats)  # (T, num_regions, feat_dim)
        torch.save(video_tensor, out_path)

    logger.info(
        "Done. Processed %d videos; %d already existed (use --overwrite to redo).",
        len(video_files) - skipped,
        skipped,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract YOLOv5 region features from videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--video_dir", required=True,
                   help="Directory containing video files.")
    p.add_argument("--feat_dir", required=True,
                   help="Output directory for .pt feature files.")
    p.add_argument("--frame_rate", type=float, default=1.0,
                   help="Frames per second to sample (default: 1).")
    p.add_argument("--max_frames", type=int, default=20,
                   help="Maximum frames per video (default: 20).")
    p.add_argument("--num_regions", type=int, default=36,
                   help="Number of region proposals per frame (default: 36).")
    p.add_argument("--img_size", type=int, default=640,
                   help="YOLOv5 input image size (default: 640).")
    p.add_argument("--model_name", type=str, default="yolov5m",
                   help="YOLOv5 model variant (default: yolov5m).")
    p.add_argument("--device", type=str, default="cpu",
                   help="Torch device string, e.g. 'cpu', 'cuda', 'mps' (default: cpu).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract even if the output .pt file already exists.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    extract_all(
        video_dir=args.video_dir,
        feat_dir=args.feat_dir,
        frame_rate=args.frame_rate,
        max_frames=args.max_frames,
        num_regions=args.num_regions,
        img_size=args.img_size,
        model_name=args.model_name,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
