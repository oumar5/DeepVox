"""BiLSTM + CTC ASR model for DeepVox Phase 2.

Architecture:
  Input (T, 48) Codec2 frames
    → Linear projection (48 → 256)
    → BiLSTM 3 layers (hidden=384, bidirectional → 768)
    → Dropout
    → Linear(768 → vocab_size)
    → Log-softmax for CTC

Total params: ~10M
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepvox.data.text import VOCAB_SIZE


class CTCASR(nn.Module):
    """BiLSTM-based ASR with CTC output.

    Args:
        input_dim: input feature dimension (48 for Codec2).
        embed_dim: input projection dimension.
        hidden_dim: LSTM hidden size.
        num_layers: number of BiLSTM layers.
        vocab_size: output vocabulary size (includes blank).
        dropout: dropout between LSTM layers and before classifier.
    """

    def __init__(
        self,
        input_dim: int = 48,
        embed_dim: int = 256,
        hidden_dim: int = 384,
        num_layers: int = 3,
        vocab_size: int = VOCAB_SIZE,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.vocab_size = vocab_size

        self.input_proj = nn.Linear(input_dim, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, T, input_dim) — Codec2 features.

        Returns:
            Log-probabilities (batch, T, vocab_size). Apply CTC loss with
            permuted shape (T, batch, vocab_size).
        """
        x = self.input_proj(x)  # (B, T, embed)
        x = F.relu(x)
        x, _ = self.lstm(x)  # (B, T, hidden*2)
        x = self.dropout(x)
        logits = self.classifier(x)  # (B, T, vocab_size)
        return F.log_softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def greedy_decode(self, x: torch.Tensor) -> list[list[int]]:
        """Greedy CTC decoding (argmax per frame, then collapse+strip blank).

        Args:
            x: (batch, T, input_dim).

        Returns:
            List of decoded token ID sequences, one per batch element.
        """
        log_probs = self.forward(x)
        preds = log_probs.argmax(dim=-1)  # (B, T)
        results = []
        for seq in preds.cpu().numpy():
            results.append(seq.tolist())
        return results
