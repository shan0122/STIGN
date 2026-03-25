"""
Temporal-Spatial Interaction (TSI) module.

Captures cross-frame spatial relationships among detected objects using
a lightweight graph-attention layer, then aggregates the attended features
along the temporal axis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TSIModule(nn.Module):
    """Temporal-Spatial Interaction module for video captioning.

    Args:
        feat_dim (int): Dimension of input region features.
        hidden_dim (int): Projection dimension used inside the module.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, feat_dim: int = 2048, hidden_dim: int = 512, num_heads: int = 4):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.q_proj = nn.Linear(feat_dim, hidden_dim)
        self.k_proj = nn.Linear(feat_dim, hidden_dim)
        self.v_proj = nn.Linear(feat_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Region feature tensor of shape (B, T, N, feat_dim)
               B=batch, T=frames, N=regions per frame.

        Returns:
            out: Aggregated temporal feature of shape (B, T, hidden_dim).
        """
        B, T, N, D = x.shape

        # Flatten (T, N) together for spatial self-attention
        x_flat = x.view(B * T, N, D)

        Q = self.q_proj(x_flat)  # (B*T, N, H)
        K = self.k_proj(x_flat)
        V = self.v_proj(x_flat)

        scale = (self.hidden_dim // self.num_heads) ** -0.5
        head_dim = self.hidden_dim // self.num_heads

        Q = Q.view(B * T, N, self.num_heads, head_dim).transpose(1, 2)
        K = K.view(B * T, N, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(B * T, N, self.num_heads, head_dim).transpose(1, 2)

        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * scale, dim=-1)
        attn = self.dropout(attn)

        attended = torch.matmul(attn, V)  # (B*T, heads, N, head_dim)
        attended = attended.transpose(1, 2).contiguous().view(B * T, N, self.hidden_dim)

        out = self.out_proj(attended)
        out = self.norm(out.mean(dim=1))  # pool over regions → (B*T, H)
        out = out.view(B, T, self.hidden_dim)

        return out
