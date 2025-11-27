from typing import Dict
import os
from cybench.datasets.dataset import Dataset
from cybench.datasets.torch_dataset import TorchDataset
import random
import numpy as np
import torch
import pathlib
import yaml


def adjust_model_cfg_to_dataset(model_cfg: Dict, dataset: Dataset):
    if type(dataset) == TorchDataset:
        # add input-dim to first layers
        _, x_c_sample, x_t_sample = dataset[0]
        model_cfg.torch_model.context_in_dim = len(x_c_sample)
        model_cfg.torch_model.temporal_in_dim = len(x_t_sample.T)
    return model_cfg


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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Safe to put these here since we checked cuda availability
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False