import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvTokenizer(nn.Module):
    """
    Convolutional tokenizer for temporal aggregation.
    Converts a (B, T, F) daily time series into a (B, P, D) sequence of patch embeddings.
    """
    def __init__(self, in_features: int, d_model: int,
                 kernel_size: int = 7, stride: int = 7,
                 n_layers: int = 1, dropout: float = 0.0):
        """
        Args:
            in_features: number of input features per day (F)
            d_model: output embedding dimension per token (D)
            kernel_size: temporal patch length (e.g., 7 for weekly)
            stride: step between patches (controls overlap)
            n_layers: number of convolutional layers (>=1)
            dropout: optional dropout between layers
        """
        super().__init__()
        layers = []
        in_ch = in_features
        for i in range(n_layers):
            out_ch = d_model if i == n_layers - 1 else d_model
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (B, T, F)
        Returns:
            tokens: tensor of shape (B, P, D)
        """
        # Conv1d expects (B, C, T)
        x = x.transpose(1, 2)        # (B, F, T)
        x = self.conv(x)             # (B, D, P)
        x = x.transpose(1, 2)        # (B, P, D)
        return self.norm(x)
