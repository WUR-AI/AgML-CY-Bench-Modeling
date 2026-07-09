"""Tests for Hydra config helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from cybench.util.config_utils import (
    adjust_model_cfg_to_dataset,
    apply_force_cpu_to_frozen_model_cfg,
    is_cybench_force_cpu,
    walk_forward_force_cpu,
)


class _FakeTorchDataset:
    def __init__(self, seq_len: int = 100, n_channels: int = 12, n_context: int = 5):
        self._x_t = torch.randn(seq_len, n_channels)
        self._x_c = torch.randn(n_context)

    def __getitem__(self, index: int):
        return None, self._x_c, self._x_t, None


def test_adjust_model_cfg_sets_lstm_input_size():
    config_dir = Path(__file__).resolve().parents[2] / "cybench" / "conf"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["model=lstm_baseline", "dataset.framework=torch"],
        )

    adjusted = adjust_model_cfg_to_dataset(cfg.model, _FakeTorchDataset(n_channels=6))
    assert adjusted.torch_model.input_size == 6
    assert not OmegaConf.is_missing(adjusted.torch_model, "input_size")


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


def test_is_cybench_force_cpu(monkeypatch):
    monkeypatch.delenv("CYBENCH_FORCE_CPU", raising=False)
    assert is_cybench_force_cpu() is False
    monkeypatch.setenv("CYBENCH_FORCE_CPU", "1")
    assert is_cybench_force_cpu() is True
    monkeypatch.setenv("CYBENCH_FORCE_CPU", "yes")
    assert is_cybench_force_cpu() is True


def test_walk_forward_force_cpu_from_hydra_overrides(monkeypatch):
    monkeypatch.delenv("CYBENCH_FORCE_CPU", raising=False)
    cfg = OmegaConf.create({"model": {"device": "cpu"}, "experiment": {"device": "cuda"}})
    assert walk_forward_force_cpu(cfg) is True
    cfg = OmegaConf.create({"model": {"device": "auto"}, "experiment": {"device": "cpu"}})
    assert walk_forward_force_cpu(cfg) is True
    cfg = OmegaConf.create({"model": {"device": "auto"}, "experiment": {"device": "cuda"}})
    assert walk_forward_force_cpu(cfg) is False


def test_apply_force_cpu_to_frozen_model_cfg():
    frozen = OmegaConf.create(
        {
            "_target_": "cybench.models.tabular_foundation_model.TabDPTModel",
            "name": "tabdpt",
            "device": "cuda",
            "allow_cpu_fallback": False,
        }
    )
    out = apply_force_cpu_to_frozen_model_cfg(frozen)
    assert out.device == "cpu"
    assert out.allow_cpu_fallback is True


def test_apply_force_cpu_to_frozen_torch_model_cfg():
    frozen = OmegaConf.create(
        {
            "framework": "torch",
            "device": "cuda",
            "torch_model": {"input_size": 6, "device": "cuda"},
        }
    )
    out = apply_force_cpu_to_frozen_model_cfg(frozen)
    assert out.device == "cpu"
    assert out.torch_model.device == "cpu"
