from omegaconf import OmegaConf
import hashlib
import json


def cfg_to_hash(cfg: OmegaConf, add_str: str = None):
    """
    Create a deterministic hash from a DatasetConfig, to use it as a keys e.g. in caching

    Args:
        cfg: The dataset configuration

    Returns:
        A hex string hash that uniquely identifies this configuration
    """
    # Convert OmegaConf to a regular dict/primitive structure
    if hasattr(cfg, '__dict__'):
        config_dict = cfg.__dict__
    else:
        config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Convert to JSON string with sorted keys for deterministic ordering
    config_str = json.dumps(config_dict, sort_keys=True, default=str)

    # Create hash
    hash_obj = hashlib.sha256(config_str.encode('utf-8'))
    hash = hash_obj.hexdigest()
    if add_str is not None:
        hash = config_dict["name"] + hash
    return hash
