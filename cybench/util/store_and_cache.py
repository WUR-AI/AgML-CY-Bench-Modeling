import pickle

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import hashlib
import json
import os

from pathlib import Path

from cybench.datasets.dataset import Dataset
from cybench.datasets.torch_dataset import TorchDataset


def cfg_to_hash(cfg: OmegaConf, add_str: bool = True):
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
    hash = hash_obj.hexdigest()
    if add_str is not None:
        hash = "_".join([cfg.name, cfg.framework, cfg.temporal.season.end_of_sequence, hash])
    return hash


def make_folder(dir: str, name) -> str:
    """
    Creates a folder in a cross-platform way.
    Returns the absolute path as a string.
    """
    # 1. Handle list inputs (e.g. [2015, 2016] -> "2015_2016")
    if isinstance(name, list):
        name_str = "_".join(str(x) for x in name)
    else:
        name_str = str(name)

    # 2. Use Path for robust joining
    # Path(directory) / name_str works on both Windows (\) and Linux (/)
    target_path = Path(dir) / name_str

    # 3. Create and return string
    target_path.mkdir(parents=True, exist_ok=True)
    return target_path


def save_preds(
        path: str,
        dataset: Dataset,
        preds: np.ndarray[tuple[int], np.dtype[np.number]],
        file_name: str
        ):
    """
    Save predictions to a csv file.
    Args:
        path:
        dataset: Dataset to get targets and indices from.
        preds: Predicted targets.
    Returns: Nada
    """
    yield_df = pd.concat([dataset.indices, pd.DataFrame({"targets": dataset.targets, "preds": preds})], axis=1)
    yield_df.to_csv(os.path.join(path, file_name + '.csv'), index=False, float_format="%.3f")
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

