"""
Temporal Transition Correlation (TTC) module.

Models transitions between consecutive video frames so that the captioning
decoder can reason about *changes* in the scene rather than just static
appearance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTCModule(nn.Module):
    """Temporal Transition Correlation module.

    Args:
        hidden_dim (int): Feature dimension coming from the TSI module.
        num_layers (int): Number of GRU layers.
    """

    def __init__(self, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Bidirectional GRU to capture temporal dependencies
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0,
        )

        # Transition gate: highlights frames where the scene changes
        self.transition_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Temporal features from TSI, shape (B, T, hidden_dim).

        Returns:
            out: Transition-aware features, shape (B, T, hidden_dim).
        """
        enc, _ = self.gru(x)  # (B, T, hidden_dim)

        # Compute transition score between neighbouring frames
        # Concatenate current and previous frame (pad with zeros at t=0)
        prev = torch.cat([enc[:, :1, :], enc[:, :-1, :]], dim=1)
        gate = self.transition_gate(torch.cat([enc, prev], dim=-1))  # (B, T, 1)

        out = self.out_norm(enc * gate)
        return out
