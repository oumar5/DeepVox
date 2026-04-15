"""BiLSTM phoneme classifier for DeepVox Phase 1.

Architecture (from protocol):
  Input → BiLSTM 2 layers (hidden=256) → Linear(256 → 36) → Softmax
  ~1.5 M parameters, identical across all 4 experimental conditions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deepvox.data.preprocess import NUM_PHONEMES


class PhonemeClassifier(nn.Module):
    """BiLSTM-based phoneme classifier.

    Args:
        input_dim: feature dimension per frame (48, 96, 80, or 320).
        hidden_dim: LSTM hidden size (default 256).
        num_layers: number of BiLSTM layers (default 2).
        num_classes: number of phoneme classes (default 36).
        dropout: dropout between LSTM layers (default 0.3).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = NUM_PHONEMES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # BiLSTM outputs 2 * hidden_dim
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input features, shape (batch, seq_len, input_dim).

        Returns:
            Logits, shape (batch, seq_len, num_classes).
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim * 2)
        logits = self.classifier(lstm_out)  # (batch, seq_len, num_classes)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict phoneme classes.

        Args:
            x: input features, shape (batch, seq_len, input_dim).

        Returns:
            Predicted class indices, shape (batch, seq_len).
        """
        logits = self.forward(x)
        return logits.argmax(dim=-1)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
