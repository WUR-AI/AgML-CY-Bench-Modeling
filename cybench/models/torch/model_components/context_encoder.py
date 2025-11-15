import torch
import torch.nn as nn
from typing import List


class ContextEncoder(nn.Module):
    """
    Encodes static/contextual metadata (location, year, soil variables ...)
    into a dense embedding vector that can be injected into downstream models.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        embed_dim: int,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
    ):
        """
        Args:
            in_dim: Input feature dimension.
            hidden_dims: Sizes of intermediate hidden layers.
                         Example: [128, 128] produces in_dim→128→128→embed_dim.
            embed_dim: Final embedding dimension.
            dropout: Dropout applied between hidden layers.
            activation: Activation used after each hidden layer.
        """
        super().__init__()

        layers = []

        # First layer: in_dim → hidden_dims[0] or → embed_dim if no hidden dims
        dims = [in_dim] + hidden_dims + [embed_dim]

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Final projection to embed_dim
        layers.append(nn.Linear(dims[-2], dims[-1]))

        # Normalize embedding
        layers.append(nn.LayerNorm(embed_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return context embedding of shape (B, embed_dim)."""
        return self.net(x)
