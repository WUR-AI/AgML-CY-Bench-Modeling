from __future__ import annotations

import torch
import torch.nn as nn
from transformers import (
    AutoformerConfig,
    AutoformerModel,
    InformerConfig,
    InformerModel,
    PatchTSTConfig,
    PatchTSTModel,
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerModel,
)

# HF encoder-decoder models require a lags_sequence; [0] means no within-series shift.
_HF_LAGS_SEQUENCE = [0]


def extract_hidden_state(outputs) -> torch.Tensor:
    """Extract the main hidden tensor from a Hugging Face time-series model output."""
    for attr in ("encoder_last_hidden_state", "last_hidden_state"):
        value = getattr(outputs, attr, None)
        if value is not None:
            return value

    encoder_hidden_states = getattr(outputs, "encoder_hidden_states", None)
    if encoder_hidden_states is not None:
        return encoder_hidden_states[-1]

    tensor_attrs = [name for name in dir(outputs) if hasattr(getattr(outputs, name, None), "shape")]
    raise ValueError(
        "Could not extract hidden state from model outputs. "
        f"Available tensor attributes: {tensor_attrs}"
    )


def pool_hidden_state(hidden: torch.Tensor) -> torch.Tensor:
    """Reduce HF hidden states to a single vector per batch item."""
    if hidden.dim() == 2:
        return hidden
    if hidden.dim() == 3:
        return hidden.mean(dim=1)
    if hidden.dim() == 4:
        pooled = hidden.mean(dim=2)
        return pooled.reshape(hidden.shape[0], -1)
    raise ValueError(
        f"Unexpected hidden state shape: {hidden.shape} "
        f"(expected 2D, 3D, or 4D tensor, got {hidden.dim()}D)"
    )


def _validate_sequence_length(x: torch.Tensor, expected: int) -> None:
    if x.shape[1] != expected:
        raise ValueError(f"Expected sequence length {expected}, got {x.shape[1]}")


class _HFProjectionMixin:
    embed_dim: int
    output_proj: nn.Linear
    norm: nn.LayerNorm

    def _init_output_projection(self, outputs) -> None:
        hidden = extract_hidden_state(outputs)
        pooled_dim = pool_hidden_state(hidden).shape[-1]
        self.output_proj = nn.Linear(pooled_dim, self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim)

    def _project_hidden(self, outputs) -> torch.Tensor:
        hidden = extract_hidden_state(outputs)
        pooled = pool_hidden_state(hidden)
        return self.norm(self.output_proj(pooled))


class _HFPastValuesEncoderBase(nn.Module, _HFProjectionMixin):
    """Shared forward path for HF models that take past_values (+ masks)."""

    in_dim: int
    embed_dim: int
    context_length: int

    def _hf_past_inputs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, seq_len, n_channels = x.shape
        return {
            "past_values": x,
            "past_time_features": torch.zeros(
                batch_size, seq_len, 0, device=x.device, dtype=x.dtype
            ),
            "past_observed_mask": torch.ones(
                batch_size, seq_len, n_channels, device=x.device, dtype=x.dtype
            ),
        }


class PatchTSTTemporalEncoder(nn.Module, _HFProjectionMixin):
    """
    Hugging Face PatchTST backbone for LateFusionNetwork.

    Maps (B, T, C) weather/RS channels to (B, embed_dim) via patching,
    transformer encoding, pooling, and a projection head.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        seq_len: int,
        patch_length: int = 16,
        patch_stride: int = 8,
        d_model: int = 64,
        num_attention_heads: int = 4,
        ffn_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.context_length = seq_len

        cfg = PatchTSTConfig(
            num_input_channels=in_dim,
            context_length=seq_len,
            prediction_length=1,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            ffn_dim=ffn_dim,
            num_hidden_layers=num_layers,
            attention_dropout=dropout,
            path_dropout=dropout,
            ff_dropout=dropout,
        )
        self.backbone = PatchTSTModel(cfg)

        with torch.no_grad():
            dummy = torch.zeros(1, seq_len, in_dim)
            outputs = self.backbone(past_values=dummy)
            self._init_output_projection(outputs)

    def forward(self, x: torch.Tensor, doys: torch.Tensor) -> torch.Tensor:
        del doys
        _validate_sequence_length(x, self.context_length)
        outputs = self.backbone(past_values=x)
        return self._project_hidden(outputs)


class InformerTemporalEncoder(_HFPastValuesEncoderBase):
    """Hugging Face Informer backbone for LateFusionNetwork."""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        seq_len: int,
        d_model: int = 64,
        num_attention_heads: int = 4,
        ffn_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.context_length = seq_len

        cfg = InformerConfig(
            prediction_length=1,
            context_length=seq_len,
            lags_sequence=_HF_LAGS_SEQUENCE,
            input_size=in_dim,
            num_time_features=0,
            d_model=d_model,
            encoder_attention_heads=num_attention_heads,
            encoder_ffn_dim=ffn_dim,
            encoder_layers=num_layers,
            dropout=dropout,
        )
        self.backbone = InformerModel(cfg)

        with torch.no_grad():
            dummy = torch.zeros(1, seq_len, in_dim)
            outputs = self.backbone(**self._hf_past_inputs(dummy), return_dict=True)
            self._init_output_projection(outputs)

    def forward(self, x: torch.Tensor, doys: torch.Tensor) -> torch.Tensor:
        del doys
        _validate_sequence_length(x, self.context_length)
        outputs = self.backbone(**self._hf_past_inputs(x), return_dict=True)
        return self._project_hidden(outputs)


class TSTTemporalEncoder(_HFPastValuesEncoderBase):
    """Hugging Face TimeSeriesTransformer backbone for LateFusionNetwork."""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        seq_len: int,
        d_model: int = 64,
        num_attention_heads: int = 4,
        ffn_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.context_length = seq_len

        cfg = TimeSeriesTransformerConfig(
            prediction_length=1,
            context_length=seq_len,
            lags_sequence=_HF_LAGS_SEQUENCE,
            input_size=in_dim,
            num_time_features=0,
            d_model=d_model,
            encoder_attention_heads=num_attention_heads,
            encoder_layers=num_layers,
            encoder_ffn_dim=ffn_dim,
            dropout=dropout,
            attention_dropout=dropout,
            activation_dropout=dropout,
            activation_function="gelu",
            scaling="std",
            loss="nll",
            distribution_output="student_t",
        )
        self.backbone = TimeSeriesTransformerModel(cfg)

        with torch.no_grad():
            dummy = torch.zeros(1, seq_len, in_dim)
            outputs = self.backbone(**self._hf_past_inputs(dummy), return_dict=True)
            self._init_output_projection(outputs)

    def forward(self, x: torch.Tensor, doys: torch.Tensor) -> torch.Tensor:
        del doys
        _validate_sequence_length(x, self.context_length)
        outputs = self.backbone(**self._hf_past_inputs(x), return_dict=True)
        return self._project_hidden(outputs)


class AutoformerTemporalEncoder(_HFPastValuesEncoderBase):
    """Hugging Face Autoformer backbone for LateFusionNetwork."""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        seq_len: int,
        d_model: int = 64,
        num_attention_heads: int = 4,
        ffn_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.context_length = seq_len

        cfg = AutoformerConfig(
            prediction_length=1,
            context_length=seq_len,
            lags_sequence=_HF_LAGS_SEQUENCE,
            input_size=in_dim,
            num_time_features=0,
            num_static_categorical_features=0,
            d_model=d_model,
            encoder_attention_heads=num_attention_heads,
            encoder_ffn_dim=ffn_dim,
            encoder_layers=num_layers,
            dropout=dropout,
        )
        self.backbone = AutoformerModel(cfg)

        with torch.no_grad():
            dummy = torch.zeros(1, seq_len, in_dim)
            outputs = self.backbone(
                **self._hf_past_inputs(dummy),
                future_values=torch.zeros(1, 1, in_dim),
                future_time_features=torch.zeros(1, 1, 0),
                return_dict=True,
                output_hidden_states=True,
            )
            self._init_output_projection(outputs)

    def forward(self, x: torch.Tensor, doys: torch.Tensor) -> torch.Tensor:
        del doys
        _validate_sequence_length(x, self.context_length)
        batch_size = x.shape[0]
        outputs = self.backbone(
            **self._hf_past_inputs(x),
            future_values=torch.zeros(batch_size, 1, self.in_dim, device=x.device, dtype=x.dtype),
            future_time_features=torch.zeros(batch_size, 1, 0, device=x.device, dtype=x.dtype),
            return_dict=True,
            output_hidden_states=True,
        )
        return self._project_hidden(outputs)
