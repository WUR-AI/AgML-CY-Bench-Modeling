from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_sequence_length(x: torch.Tensor, expected: int) -> None:
    if x.shape[1] != expected:
        raise ValueError(f"Expected sequence length {expected}, got {x.shape[1]}")


class NLinearTemporalEncoder(nn.Module):
    """
    NLinear temporal encoder for LateFusionNetwork.

    Single linear layer per channel with last-value subtraction
    (Zeng et al., AAAI 2023). Maps (B, T, C) -> (B, embed_dim).
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        seq_len: int,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.context_length = seq_len

        self.temporal_linear = nn.Linear(seq_len, 1)
        self.channel_proj = nn.Linear(in_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, doys: torch.Tensor) -> torch.Tensor:
        del doys
        _validate_sequence_length(x, self.context_length)

        last_val = x[:, -1:, :]
        x_shifted = x - last_val
        x_t = x_shifted.transpose(1, 2)
        out = self.temporal_linear(x_t) + last_val.transpose(1, 2)
        pooled = out.squeeze(-1)

        return self.norm(self.channel_proj(pooled))


class DLinearTemporalEncoder(nn.Module):
    """
    DLinear temporal encoder for LateFusionNetwork.

    Trend + remainder decomposition with separate linear layers
    (Zeng et al., AAAI 2023). Maps (B, T, C) -> (B, embed_dim).
    """

    DEFAULT_KERNEL_SIZE = 25

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        seq_len: int,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.context_length = seq_len

        if kernel_size % 2 == 0:
            kernel_size += 1
        self._kernel_size = kernel_size

        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.trend_linear = nn.Linear(seq_len, 1)
        self.remainder_linear = nn.Linear(seq_len, 1)
        self.channel_proj = nn.Linear(in_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def _extract_trend(
        self,
        x: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Symmetric moving average; x shape (B, C, T)."""
        _, _, seq_len = x.shape
        pad = (self._kernel_size - 1) // 2

        if observed_mask is not None:
            mask = observed_mask.unsqueeze(1).float()
            x_padded = F.pad(x, (pad, pad), mode="replicate")
            mask_padded = F.pad(mask, (pad, pad), mode="constant", value=0.0)
            trend_padded = self.moving_avg(x_padded * mask_padded)
            mask_sum = self.moving_avg(mask_padded).clamp(min=1e-8)
            trend = trend_padded / mask_sum
            return trend[:, :, :seq_len]

        x_padded = F.pad(x, (pad, pad), mode="replicate")
        trend = self.moving_avg(x_padded)
        return trend[:, :, :seq_len]

    def forward(self, x: torch.Tensor, doys: torch.Tensor) -> torch.Tensor:
        del doys
        _validate_sequence_length(x, self.context_length)

        x_t = x.transpose(1, 2)
        trend = self._extract_trend(x_t)
        remainder = x_t - trend
        pooled = (
            self.trend_linear(trend) + self.remainder_linear(remainder)
        ).squeeze(-1)

        return self.norm(self.channel_proj(pooled))
