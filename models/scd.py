"""
Scene Change Detection (SCD) module.

Detects semantically meaningful scene boundaries and uses them as soft
weighting signals so that the decoder attends more to key frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCDModule(nn.Module):
    """Scene Change Detection module.

    Args:
        hidden_dim (int): Feature dimension from the TTC module.
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Score each frame pair for scene change probability
        self.change_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Transition-aware features from TTC, shape (B, T, hidden_dim).

        Returns:
            out: Scene-boundary-weighted features, shape (B, T, hidden_dim).
        """
        B, T, D = x.shape

        # Pair each frame with the next frame to detect scene transitions
        curr = x[:, :-1, :]  # (B, T-1, D)
        nxt = x[:, 1:, :]    # (B, T-1, D)
        pair = torch.cat([curr, nxt], dim=-1)  # (B, T-1, 2D)

        change_score = self.change_scorer(pair)  # (B, T-1, 1)
        # Pad so that it aligns with all T frames
        pad = torch.zeros(B, 1, 1, device=x.device)
        change_score = torch.cat([pad, change_score], dim=1)  # (B, T, 1)

        # Weight frame features by scene-change importance
        out = self.norm(self.proj(x) * change_score)
        return out
