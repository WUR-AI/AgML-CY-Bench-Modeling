import pickle
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from cybench.models.model import BaseModel
from cybench.datasets.dataset import PandasDataset

log = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost wrapper compatible with PandasDataset and the Hydra instantiation pattern.

    All constructor kwargs (except `name`) are forwarded directly to xgboost's XGBRegressor.
    """

    def __init__(self, name: str = "xgboost", verbose: bool = False, **kwargs):
        self.name = name
        if 'random_state' not in kwargs:
            kwargs['random_state'] = int(np.random.randint(2**31))
        self.model = XGBRegressor(**kwargs)
        log.info(f"Initialized {self.name}")

    def fit(self, dataset: PandasDataset, **fit_params) -> Dict[str, Any]:
        X, y = dataset.xy
        self.model.fit(X, y.values.ravel())
        return {}

    def predict(self, dataset: PandasDataset, **predict_params) -> Tuple[np.ndarray, Dict]:
        X, _ = dataset.xy
        return self.model.predict(X), {}

    def save(self, path):
        with open(Path(path) / f"{self.name}.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


class LGBMModel(BaseModel):
    """LightGBM wrapper compatible with PandasDataset and the Hydra instantiation pattern.

    All constructor kwargs (except `name`) are forwarded directly to lightgbm's LGBMRegressor.
    """

    def __init__(self, name: str = "lightgbm", verbose: bool = False, **kwargs):
        self.name = name
        if 'random_state' not in kwargs:
            kwargs['random_state'] = int(np.random.randint(2**31))
        self.model = LGBMRegressor(**kwargs)
        log.info(f"Initialized {self.name}")

    def fit(self, dataset: PandasDataset, **fit_params) -> Dict[str, Any]:
        X, y = dataset.xy
        self.model.fit(X, y.values.ravel())
        return {}

    def predict(self, dataset: PandasDataset, **predict_params) -> Tuple[np.ndarray, Dict]:
        X, _ = dataset.xy
        return self.model.predict(X), {}

    def save(self, path):
        with open(Path(path) / f"{self.name}.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
