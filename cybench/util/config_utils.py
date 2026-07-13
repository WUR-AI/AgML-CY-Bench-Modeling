from __future__ import annotations

import os
import pathlib
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, cast

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
        for name, value in (
            ("input_size", temporal_in_dim),
            ("context_in_dim", len(x_c_sample)),
            ("temporal_in_dim", temporal_in_dim),
        ):
            if name in torch_model.keys():
                torch_model[name] = value

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


def is_cybench_force_cpu() -> bool:
    """True when Slurm submit_array.sh --cpu exported CYBENCH_FORCE_CPU=1."""
    val = os.environ.get("CYBENCH_FORCE_CPU", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def walk_forward_force_cpu(cfg: DictConfig) -> bool:
    """Whether walk-forward should override device in frozen optimal_model.yaml."""
    if is_cybench_force_cpu():
        return True
    if OmegaConf.select(cfg, "model.device") == "cpu":
        return True
    return OmegaConf.select(cfg, "experiment.device") == "cpu"


def apply_force_cpu_to_frozen_model_cfg(model_cfg: DictConfig) -> DictConfig:
    """
    Override device fields baked into screening optimal_model.yaml for CPU runs.

    Walk-forward loads the frozen config wholesale; without this, TabDPT and other
    GPU-screened models keep device=cuda even on the main partition.
    """
    cfg_out = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(model_cfg)))
    with open_dict(cfg_out):
        if "device" in cfg_out:
            cfg_out.device = "cpu"
        if "allow_cpu_fallback" in cfg_out:
            cfg_out.allow_cpu_fallback = True
        torch_model = cfg_out.get("torch_model")
        if torch_model is not None and "device" in torch_model:
            with open_dict(torch_model):
                torch_model.device = "cpu"
    return cfg_out


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
    """Seed Python/NumPy/PyTorch RNGs for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    # Fixed cuBLAS workspace for deterministic CUDA matmuls (no perf hit on CPU).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@contextmanager
def deterministic_torch_training() -> Iterator[None]:
    """Tighten PyTorch determinism for the training loop only.

    Uses warn_only=True so unsupported ops keep working. Single-threaded CPU
    matmuls avoid BLAS races; cost is small on typical CY-Bench batch sizes.
    """
    try:
        import torch
    except ImportError:
        yield
        return

    thread_count = os.environ.get("CYBENCH_TORCH_THREADS", "1")
    blas_env = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    prev_env = {key: os.environ.get(key) for key in blas_env}
    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    for key in blas_env:
        os.environ[key] = thread_count
    torch.set_num_threads(int(thread_count))
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=True)
        torch.set_num_threads(prev_threads)
        for key, value in prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
