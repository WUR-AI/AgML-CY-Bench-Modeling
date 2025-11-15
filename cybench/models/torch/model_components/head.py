import torch
import torch.nn as nn
from typing import List


class RegressionHead(nn.Module):
    def __init__(self,
                 hidden_dims: List[int],
                 dropout: float = 0.0,
                 activation: nn.Module = nn.ReLU()):
        """
        Fully-connected regression head.
        - hidden_dim: list of layer sizes including output of previous block.
          Example: [256, 128] → 256->128->1
        - dropout: applied between hidden layers.
        """
        super().__init__()

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dim must contain at least one hidden size.")

        layers = []

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Final regression output (univariate)
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
