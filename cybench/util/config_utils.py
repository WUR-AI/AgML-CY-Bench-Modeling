from __future__ import annotations

import os
import pathlib
import random
from pathlib import Path
from typing import List, cast

import numpy as np
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from cybench.datasets.dataset import BaseDataset


def adjust_model_cfg_to_dataset(model_cfg: DictConfig, dataset: BaseDataset) -> DictConfig:
    # Only called for torch datasets (see run_experiments.py).
    _, x_c_sample, x_t_sample, _ = dataset[0]
    torch_model = model_cfg.torch_model
    temporal_in_dim = len(x_t_sample.T)
    seq_len = int(x_t_sample.shape[0])

    # Use .keys(): ``???`` placeholders are MISSING and ``key in cfg`` is False.
    with open_dict(torch_model):
        if "input_size" in torch_model.keys():
            torch_model.input_size = temporal_in_dim
        if "context_in_dim" in torch_model.keys():
            torch_model.context_in_dim = len(x_c_sample)
        if "temporal_in_dim" in torch_model.keys():
            torch_model.temporal_in_dim = temporal_in_dim

    temporal_encoder = torch_model.get("temporal_encoder")
    if temporal_encoder is not None and "seq_len" in temporal_encoder.keys():
        with open_dict(temporal_encoder):
            temporal_encoder.seq_len = seq_len
    return model_cfg


def remove_keys(model_cfg: DictConfig, key="_search_") -> DictConfig:
    """Recursively remove _search_ keys from config before instantiation. Only important for hyperparameter search."""
    cfg_dict = OmegaConf.to_container(model_cfg, resolve=True)

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if k != key}
        elif isinstance(obj, list):
            return [_clean(item) for item in obj]
        else:
            return obj

    return cast(DictConfig, OmegaConf.create(_clean(cfg_dict)))

def remove_search_keys(model_cfg: DictConfig) -> DictConfig:
    return remove_keys(model_cfg, key="_search_")


def reload_config_with_overrides(
        config_dir: Path,
        config_name: str,
        overrides: List[str]
) -> DictConfig:
    """
    Reload Hydra config with new overrides (e.g., processor=lstm).

    Args:
        config_dir: Path to config directory
        config_name: Name of main config file
        overrides: List of Hydra overrides like ["model/processor=lstm"]
    """
    with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg



def get_run_description(overrides_path: pathlib.Path) -> str:
    """Loads and formats the list of Hydra overrides into a unique string."""
    try:
        with open(overrides_path, 'r') as f:
            overrides = yaml.safe_load(f)
        if isinstance(overrides, list):
            # Sort for a canonical (consistent) index description
            return " | ".join(sorted(overrides))
        return "Unknown_Overrides"
    except Exception:
        return "Error_Loading_Overrides"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Safe to put these here since we checked cuda availability
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
