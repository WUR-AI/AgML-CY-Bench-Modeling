import torch
import torch.nn as nn


class LateFusionNetwork(nn.Module):
    """
    Full late-fusion model combining:
      - ContextEncoder:    encodes static/context features  (B, C_ctx) -> (B, D_ctx)
      - TemporalEncoder:   encodes temporal features        (B, T, C_tmp) -> (B, D_tmp)
      - Fusion block:      fuses both embeddings            -> (B, D_fused)
      - RegressionHead:    maps fused embedding to target   -> (B,)

    Forward expects two *separate* inputs:
        context:  (B, C_ctx)
        temporal: (B, T, C_tmp)
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        temporal_encoder: nn.Module,
        fusion: nn.Module,
        regression_head: nn.Module,
        context_in_dim: int,
        temporal_in_dim: int,
        embed_dim: int,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.temporal_encoder = temporal_encoder
        self.fusion = fusion
        self.regression_head = regression_head
        self.context_in_dim = context_in_dim
        self.temporal_in_dim = temporal_in_dim
        self.embed_dim = embed_dim

    def forward(
        self,
        context: torch.Tensor,
        temporal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context:  Static/context features of shape (B, C_ctx).
            temporal: Temporal features of shape (B, T, C_tmp).

        Returns:
            preds: Regression output of shape (B,).
        """
        # Encode context and temporal streams
        context_emb = self.context_encoder(context)      # (B, D_ctx)
        temporal_emb = self.temporal_encoder(temporal)  # (B, D_tmp)

        # Fuse both embeddings
        fused = self.fusion(temporal_emb, context_emb)  # (B, D_fused)

        # Final regression head
        out = self.regression_head(fused)               # (B,)
        return out
