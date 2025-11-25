import torch
import torch.nn as nn
from typing import Optional

"""
Temporal Encoder Module
=======================

This module implements a flexible temporal-encoding pipeline composed of:
1. **Tokenizer**: Downsamples raw temporal features and maps them into a fixed
   embedding dimension (`embed_dim`). Implemented via a high–receptive-field
   Conv1d block (`ConvTokenizer`).
2. **Processor**: Applies temporal feature transformation without changing the
   embedding dimensionality. Two processor families are provided:
       - `CNNProcessor`: Lightweight temporal convolutions with preserved shape.
       - `LSTMProcessor`: Recurrent sequence modeling with optional projection.
3. **Time Pooling**: Reduces the encoded sequence over the time axis into a
   single representation vector. Supported pooling strategies:
       - `MeanTimePooling`
       - `MaxTimePooling`
       - `LastTimeStepPooling` (CLS-token style)
       - `AttentionTimePooling` (learned temporal attention)

The main entry point is the `TemporalEncoder`, which simply composes:
    Tokenizer → Processor → Pooling.

Input  shape:  (B, T, in_dim)
Output shape:  (B, embed_dim)

The design is modular:
- Each component is a standalone `nn.Module`.
- Dimensions and architectures remain decoupled.
- Hydra instantiation is supported by clean component boundaries.

This file only contains computation logic.
The Hydra configuration (tokenizer, processor, pooling selection) should be
defined in `conf/model/torch/temporal/` YAML files and instantiated via `hydra.utils.instantiate`.

"""

# ----------------------------------------------------------------------
# 1. Tokenizer
# ----------------------------------------------------------------------

class ConvTokenizer(nn.Module):
    """
    Temporal tokenizer that down-samples the sequence with Conv1d and
    maps feature dimension -> embed_dim.
    """
    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        kernel_size: int = 7,
        stride: int = 7,
    ):
        super().__init__()

        padding = (kernel_size - 1) // 2 # keeps the time dimension

        self.proj = nn.Conv1d(
            in_channels=in_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, in_dim)
        Returns:
            (B, T', embed_dim)
        """
        x = x.transpose(1, 2)          # (B, in_dim, T)
        x = self.proj(x)               # (B, embed_dim, T')
        x = x.transpose(1, 2)          # (B, T', embed_dim)
        return x


# ----------------------------------------------------------------------
# 2. Processors
# ----------------------------------------------------------------------

class CNNProcessor(nn.Module):
    """
    Simple temporal CNN stack that preserves embed_dim and sequence length.
    """
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        padding = kernel_size // 2  # keep length

        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, embed_dim)
        Returns:
            (B, T, embed_dim)
        """
        x = x.transpose(1, 2)      # (B, embed_dim, T)
        x = self.net(x)            # (B, embed_dim, T)
        x = x.transpose(1, 2)      # (B, T, embed_dim)
        return x


class LSTMProcessor(nn.Module):
    """
    LSTM-based temporal processor that preserves embed_dim.
    """
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 4,
        dropout: float = 0.5,
        bidirectional: bool = False,
    ):
        super().__init__()
        hidden_size = embed_dim
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Project back to embed_dim if bidirectional doubles it
        if self.bidirectional:
            out_dim = hidden_size * (2 if bidirectional else 1)
            self.proj = nn.Linear(out_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, embed_dim)
        Returns:
            (B, T, embed_dim)
        """
        out, _ = self.lstm(x)      # (B, T, H or 2H)
        if self.bidirectional:
            out = self.proj(out)       # (B, T, embed_dim)
        return out


class TransformerProcessor(nn.Module):
    """
    Transformer-based temporal processor that preserves (B, T, embed_dim).
    """
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # keeps (B, T, D)
        )
        self.net = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, embed_dim)
        Returns:
            (B, T, embed_dim)
        """
        x = self.net(x)
        return x

# ----------------------------------------------------------------------
# 3. Time pooling blocks
# ----------------------------------------------------------------------

class MeanPooling(nn.Module):
    """Mean over the temporal dimension."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, D)
        """
        return x.mean(dim=1)


class MaxPooling(nn.Module):
    """Max over the temporal dimension."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, D)
        """
        return x.max(dim=1).values


class LastStepPooling(nn.Module):
    """Take the last time step (classification-token style)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, D)
        """
        return x[:, -1, :]


class AttentionPooling(nn.Module):
    """
    Attention pooling over time.
    Learns scalar attention weights for each time step.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, D)
        """
        # Compute unnormalized scores
        scores = self.score(x)             # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)

        # Weighted sum over time
        pooled = (weights * x).sum(dim=1)  # (B, D)
        return pooled


# ----------------------------------------------------------------------
# 4. TemporalEncoder: orchestrates tokenizer, processor, pooling
# ----------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """
    Full temporal encoder: Tokenizer -> Processor -> Pooling -> LayerNorm.
    Input:  (B, T, in_dim)
    Output: (B, embed_dim)
    """
    def __init__(
        self,
        tokenizer: nn.Module,
        processor: nn.Module,
        pooling: nn.Module,
        embed_dim: int,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self.pooling = pooling
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, in_dim)
        Returns:
            (B, embed_dim)
        """
        x = self.tokenizer(x)     # (B, T', embed_dim)
        x = self.processor(x)     # (B, T', embed_dim)
        x = self.pooling(x)       # (B, embed_dim)
        x = self.norm(x)          # (B, embed_dim)
        return x