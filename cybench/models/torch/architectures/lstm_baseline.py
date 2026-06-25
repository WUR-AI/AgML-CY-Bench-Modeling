"""End-to-end LSTM yield baseline (CropBench-style).

A single ``nn.LSTM`` over the full temporal tensor, last-hidden (or mean) pooling,
and a linear head. No late-fusion stack, tokenizer, or static encoder.

Adapted from vishalned/CropBench ``LSTMBaseline``; ``forward`` matches
``TorchTrainer`` / ``LateFusionNetwork`` signature ``(context, temporal, doys)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """LSTM baseline: ``(B, T, F)`` in → ``(B,)`` yield out."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        pool: str = "last",
    ):
        super().__init__()
        if pool not in {"last", "mean"}:
            raise ValueError(f"pool must be 'last' or 'mean', got {pool!r}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pool = pool

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        head_in = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(head_in, 1)

    def forward(
        self,
        context: torch.Tensor,
        temporal: torch.Tensor,
        doys: torch.Tensor,
    ) -> torch.Tensor:
        del context, doys

        out, (h_n, _c_n) = self.lstm(temporal)
        if self.pool == "last":
            if self.bidirectional:
                h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                h_last = h_n[-1]
        else:
            h_last = out.mean(dim=1)

        return self.head(h_last).squeeze(-1)
