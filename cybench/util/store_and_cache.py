from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from cybench.datasets.dataset import BaseDataset, PandasDataset
from cybench.datasets.torch_dataset import TorchDataset


def cfg_to_hash(cfg: DictConfig, add_str: bool = True) -> str:
    """
    Create a deterministic hash from a DatasetConfig, to use it as a keys e.g. in caching

    Args:
        cfg: The dataset configuration

    Returns:
        A hex string hash that uniquely identifies this configuration
    """
    # Convert OmegaConf to a regular dict/primitive structure
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Convert to JSON string with sorted keys for deterministic ordering
    config_str = json.dumps(config_dict, sort_keys=True, default=str)

    # Create hash
    hash_obj = hashlib.sha256(config_str.encode('utf-8'))
    hash_value = hash_obj.hexdigest()
    if add_str:
        hash_value = "_".join([cfg.name, cfg.framework, cfg.temporal.season.end_of_sequence, hash_value])
    return hash_value


def make_folder(dir: str | Path, name) -> Path:
    """
    Creates a folder in a cross-platform way.
    Returns the absolute path as a Path.
    """
    # 1. Handle list inputs (e.g. [2015, 2016] -> "2015_2016")
    if isinstance(name, list):
        name_str = "_".join(str(x) for x in name)
    else:
        name_str = str(name)

    # 2. Use Path for robust joining
    # Path(directory) / name_str works on both Windows (\) and Linux (/)
    target_path = Path(dir) / name_str

    # 3. Create and return Path
    target_path.mkdir(parents=True, exist_ok=True)
    return target_path


def _dataset_indices(dataset: BaseDataset) -> pd.DataFrame:
    if isinstance(dataset, (PandasDataset, TorchDataset)):
        return dataset.indices.copy()
    raise TypeError(
        f"save_preds requires PandasDataset or TorchDataset, got {type(dataset).__name__}"
    )


def save_preds(
        path: str | Path,
        dataset: BaseDataset,
        preds: npt.NDArray[Any],
        file_name: str,
        ):
    """
    Save predictions to a csv file.
    Args:
        path:
        dataset: Dataset to get targets and indices from.
        preds: Predicted targets.
    Returns: Nada
    """
    preds = np.asarray(preds)
    targets = np.asarray(dataset.targets)
    indices_df = _dataset_indices(dataset)

    # Align on row order (not index labels) to support both MultiIndex-backed
    # and RangeIndex-backed frames without concat index-union errors.
    if isinstance(indices_df, pd.DataFrame):
        indices_df = indices_df.reset_index(drop=True)
    else:
        indices_df = pd.DataFrame(indices_df).reset_index(drop=True)

    if len(indices_df) != len(preds) or len(preds) != len(targets):
        raise ValueError(
            f"Prediction output length mismatch: indices={len(indices_df)}, "
            f"targets={len(targets)}, preds={len(preds)}"
        )

    yield_df = pd.concat(
        [indices_df, pd.DataFrame({"targets": targets, "preds": preds})],
        axis=1,
    )
    output_path = Path(path)
    print(output_path / f"{file_name}.csv")
    yield_df.to_csv(output_path / f"{file_name}.csv", index=False, float_format="%.3f")
    return yield_df


def save_meta_dict(path, dict):
    """
    Save a dict of any metadata after training.
    Args:
        path:
        dict:

    Returns: Nichts
    """
    with open(os.path.join(path, f'meta.pkl'), 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return None
