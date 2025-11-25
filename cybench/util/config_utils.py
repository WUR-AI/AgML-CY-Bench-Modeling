from typing import Dict

from cybench.datasets.dataset import Dataset
from cybench.datasets.torch_dataset import TorchDataset


def adjust_model_cfg_to_dataset(model_cfg: Dict, dataset: Dataset):
    if type(dataset) == TorchDataset:
        # add input-dim to first layers
        _, x_c_sample, x_t_sample = dataset[0]
        model_cfg.torch_model.context_in_dim = len(x_c_sample)
        model_cfg.torch_model.temporal_in_dim = len(x_t_sample.T)
    return model_cfg
