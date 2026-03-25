"""
Description Generator (decoder module).

An LSTM-based auto-regressive language decoder with temporal attention over
the video feature sequence produced by TSI → TTC → SCD pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DescriptionGenerator(nn.Module):
    """Auto-regressive caption decoder with temporal attention.

    Args:
        vocab_size (int): Size of the target vocabulary (including <pad>, <sos>, <eos>).
        embed_dim (int): Word embedding dimension.
        hidden_dim (int): LSTM hidden dimension; must match video feature dim.
        max_len (int): Maximum caption length during inference.
        dropout (float): Dropout rate applied to embeddings and output projection.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        max_len: int = 30,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTMCell(embed_dim + hidden_dim, hidden_dim)

        # Temporal attention
        self.attn_w = nn.Linear(hidden_dim, hidden_dim)
        self.attn_u = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, 1)

        self.out_proj = nn.Linear(hidden_dim, vocab_size)

    def _attend(self, h: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted context vector.

        Args:
            h:   Current LSTM hidden state (B, H).
            enc: Video encoder output (B, T, H).

        Returns:
            ctx: Context vector (B, H).
        """
        energy = self.attn_v(
            torch.tanh(self.attn_w(enc) + self.attn_u(h).unsqueeze(1))
        )  # (B, T, 1)
        alpha = torch.softmax(energy, dim=1)  # (B, T, 1)
        ctx = (alpha * enc).sum(dim=1)       # (B, H)
        return ctx

    def forward(
        self,
        enc: torch.Tensor,
        captions: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced forward pass.

        Args:
            enc:      Encoded video features (B, T, H).
            captions: Ground-truth token ids (B, L) — includes <sos> at position 0.
            lengths:  Actual caption length per sample (B,).

        Returns:
            logits: Log-probabilities over vocabulary (B, L-1, vocab_size).
        """
        B, T, H = enc.shape
        L = captions.size(1)

        h = enc.mean(dim=1)          # (B, H)  — init hidden with mean-pooled video
        c = torch.zeros_like(h)

        embeds = self.dropout(self.embedding(captions))  # (B, L, E)

        outputs = []
        for t in range(L - 1):
            ctx = self._attend(h, enc)
            inp = torch.cat([embeds[:, t, :], ctx], dim=-1)  # (B, E+H)
            h, c = self.lstm(inp, (h, c))
            logit = self.out_proj(self.dropout(h))           # (B, V)
            outputs.append(logit.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # (B, L-1, V)

    @torch.no_grad()
    def generate(
        self,
        enc: torch.Tensor,
        sos_id: int,
        eos_id: int,
        beam_size: int = 6,
    ) -> list[list[int]]:
        """Beam-search caption generation for a single sample.

        Args:
            enc:       Encoded video features for ONE sample (1, T, H).
            sos_id:    Start-of-sentence token id.
            eos_id:    End-of-sentence token id.
            beam_size: Number of beams.

        Returns:
            List of token-id lists (best beam first).
        """
        device = enc.device
        B = enc.size(0)
        assert B == 1, "generate() operates on a single sample (B=1)"

        h = enc.mean(dim=1)
        c = torch.zeros_like(h)

        # Each beam: (log_prob, token_list, h, c)
        beams = [(0.0, [sos_id], h, c)]
        completed = []

        for _ in range(self.max_len):
            new_beams = []
            for log_p, tokens, bh, bc in beams:
                last_tok = torch.tensor([[tokens[-1]]], device=device)
                emb = self.embedding(last_tok).squeeze(1)  # (1, E)
                ctx = self._attend(bh, enc)
                inp = torch.cat([emb, ctx], dim=-1)
                nh, nc = self.lstm(inp, (bh, bc))
                logit = self.out_proj(nh)
                log_probs = F.log_softmax(logit, dim=-1).squeeze(0)

                top_lp, top_ids = log_probs.topk(beam_size)
                for lp, tid in zip(top_lp.tolist(), top_ids.tolist()):
                    new_seq = tokens + [tid]
                    new_lp = log_p + lp
                    if tid == eos_id:
                        completed.append((new_lp, new_seq))
                    else:
                        new_beams.append((new_lp, new_seq, nh, nc))

            if not new_beams:
                break
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_size]

        if not completed:
            completed = [(lp, toks) for lp, toks, _, _ in beams]

        completed.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
        return [toks for _, toks in completed]
