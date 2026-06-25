"""Tests for Hydra config helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from cybench.util.config_utils import adjust_model_cfg_to_dataset


class _FakeTorchDataset:
    def __init__(self, seq_len: int = 100, n_channels: int = 12, n_context: int = 5):
        self._x_t = torch.randn(seq_len, n_channels)
        self._x_c = torch.randn(n_context)

    def __getitem__(self, index: int):
        return None, self._x_c, self._x_t, None


def test_adjust_model_cfg_sets_missing_placeholder_dims():
    config_dir = Path(__file__).resolve().parents[2] / "cybench" / "conf"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["model=cnn_lf", "dataset.framework=torch"],
        )

    adjusted = adjust_model_cfg_to_dataset(cfg.model, _FakeTorchDataset())
    assert adjusted.torch_model.temporal_in_dim == 12
    assert adjusted.torch_model.context_in_dim == 5
    assert not OmegaConf.is_missing(adjusted.torch_model, "temporal_in_dim")
