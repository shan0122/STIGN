"""
STIGN — Spatial-Temporal Interaction Graph Network for Video Captioning.

Architecture pipeline:
  video frames → YOLOv5 region features
  → TSIModule  (spatial interactions across regions)
  → TTCModule  (temporal transition correlations)
  → SCDModule  (scene change detection weighting)
  → DescriptionGenerator (LSTM decoder with attention)
"""

import torch
import torch.nn as nn

from models.tsi import TSIModule
from models.ttc import TTCModule
from models.scd import SCDModule
from models.decoder import DescriptionGenerator


class STIGN(nn.Module):
    """Spatial-Temporal Interaction Graph Network.

    Args:
        vocab_size (int): Number of words in the vocabulary.
        feat_dim (int): Dimensionality of input region features from YOLOv5.
        hidden_dim (int): Internal representation dimension shared across modules.
        num_heads (int): Attention heads in TSI.
        embed_dim (int): Word embedding dimension for the decoder.
        max_caption_len (int): Maximum generated caption length.
        dropout (float): Dropout probability in the decoder.
    """

    def __init__(
        self,
        vocab_size: int,
        feat_dim: int = 2048,
        hidden_dim: int = 512,
        num_heads: int = 4,
        embed_dim: int = 512,
        max_caption_len: int = 30,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.tsi = TSIModule(feat_dim=feat_dim, hidden_dim=hidden_dim, num_heads=num_heads)
        self.ttc = TTCModule(hidden_dim=hidden_dim)
        self.scd = SCDModule(hidden_dim=hidden_dim)
        self.decoder = DescriptionGenerator(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            max_len=max_caption_len,
            dropout=dropout,
        )

    def encode(self, region_feats: torch.Tensor) -> torch.Tensor:
        """Encode a video clip into a temporal feature sequence.

        Args:
            region_feats: YOLOv5 region features, shape (B, T, N, feat_dim).

        Returns:
            enc: Encoded video features, shape (B, T, hidden_dim).
        """
        x = self.tsi(region_feats)  # (B, T, H)
        x = self.ttc(x)             # (B, T, H)
        x = self.scd(x)             # (B, T, H)
        return x

    def forward(
        self,
        region_feats: torch.Tensor,
        captions: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced training pass.

        Args:
            region_feats: (B, T, N, feat_dim)
            captions:     (B, L) token ids including <sos>
            lengths:      (B,) actual caption lengths

        Returns:
            logits: (B, L-1, vocab_size)
        """
        enc = self.encode(region_feats)
        return self.decoder(enc, captions, lengths)

    @torch.no_grad()
    def caption(
        self,
        region_feats: torch.Tensor,
        sos_id: int,
        eos_id: int,
        beam_size: int = 6,
    ) -> list[list[int]]:
        """Inference caption generation with beam search.

        Args:
            region_feats: (1, T, N, feat_dim) — single video sample.
            sos_id:       Start token id.
            eos_id:       End token id.
            beam_size:    Number of beams.

        Returns:
            List of token-id sequences (best beam first).
        """
        enc = self.encode(region_feats)
        return self.decoder.generate(enc, sos_id, eos_id, beam_size)
