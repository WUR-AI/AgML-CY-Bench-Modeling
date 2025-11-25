import torch
import torch.nn as nn
from typing import Optional


class ConcatFusion(nn.Module):
    """
    Fuse temporal and static/context embeddings by concatenation.

    Input:
        temporal_emb: (B, H_t)
        static_emb:   (B, H_s)

    Output:
        (B, out_dim) if out_dim is not None, else (B, H_t + H_s)
    """
    def __init__(
        self,
        temporal_dim: int,
        static_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()

        in_dim = temporal_dim + static_dim

        if out_dim is None or out_dim == in_dim:
            self.proj = nn.Identity()
            self.out_dim = in_dim
        else:
            layers = [nn.Linear(in_dim, out_dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            if use_layernorm:
                layers.append(nn.LayerNorm(out_dim))
            self.proj = nn.Sequential(*layers)
            self.out_dim = out_dim

    def forward(self, temporal_emb: torch.Tensor, static_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([temporal_emb, static_emb], dim=-1)
        return self.proj(x)


class AddFusion(nn.Module):
    """
    Fuse temporal and static/context embeddings by addition after projection.

    Input:
        temporal_emb: (B, H_t)
        static_emb:   (B, H_s)

    Output:
        (B, out_dim)  where out_dim defaults to H_t
    """
    def __init__(
        self,
        temporal_dim: int,
        static_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()

        self.out_dim = out_dim or temporal_dim

        # Project to common dimension
        self.temporal_proj = (
            nn.Linear(temporal_dim, self.out_dim)
            if temporal_dim != self.out_dim else
            nn.Identity()
        )
        self.static_proj = nn.Linear(static_dim, self.out_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(self.out_dim) if use_layernorm else nn.Identity()

    def forward(self, temporal_emb: torch.Tensor, static_emb: torch.Tensor) -> torch.Tensor:
        t = self.temporal_proj(temporal_emb)
        s = self.static_proj(static_emb)
        combined = t + s
        combined = self.dropout(combined)
        combined = self.norm(combined)
        return combined


class GatedFusion(nn.Module):
    """
    Gated fusion of temporal and static/context embeddings.

    Gate determines how much of temporal vs static to use:

        gate in [0,1]^D
        out = gate * temporal_proj + (1 - gate) * static_proj

    Input:
        temporal_emb: (B, H_t)
        static_emb:   (B, H_s)

    Output:
        (B, out_dim)
    """
    def __init__(
        self,
        temporal_dim: int,
        static_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()

        self.out_dim = out_dim if out_dim is not None else temporal_dim

        # Gate: function of concatenated *original* embeddings
        self.gate_net = nn.Sequential(
            nn.Linear(temporal_dim + static_dim, out_dim),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()

    def forward(self, temporal_emb: torch.Tensor, static_emb: torch.Tensor) -> torch.Tensor:
        # Gate from original concatenated embeddings
        gate_input = torch.cat([temporal_emb, static_emb], dim=-1)  # (B, H_t + H_s)
        gate = self.gate_net(gate_input)                            # (B, D)

        combined = gate * temporal_emb + (1.0 - gate) * static_emb
        combined = self.dropout(combined)
        combined = self.norm(combined)
        return combined
