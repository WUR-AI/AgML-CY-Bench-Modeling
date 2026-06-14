from __future__ import annotations

import pickle
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from cybench.models.model import BaseModel
from cybench.datasets.dataset import PandasDataset

log = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost wrapper compatible with PandasDataset and the Hydra instantiation pattern.

    All constructor kwargs (except `name`) are forwarded directly to xgboost's XGBRegressor.
    """

    def __init__(self, name: str = "xgboost", verbose: bool = False, framework: str | None = None, **kwargs):
        self.name = name
        if 'random_state' not in kwargs:
            kwargs['random_state'] = int(np.random.randint(2**31))
        self.model = XGBRegressor(**kwargs)
        log.info(f"Initialized {self.name}")

    def fit(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, dataset: PandasDataset, **fit_params
    ) -> tuple[Any, dict[str, Any]]:
        X, y = dataset.xy
        self.model.fit(X, y.values.ravel())
        return self, {}

    def predict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, dataset: PandasDataset, **predict_params
    ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        X, _ = dataset.xy
        return cast(npt.NDArray[Any], np.asarray(self.model.predict(X))), {}

    def save(self, model_path: str) -> None:
        with open(Path(model_path) / f"{self.name}.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, model_path: str) -> XGBoostModel:
        with open(model_path, "rb") as f:
            return pickle.load(f)


class LGBMModel(BaseModel):
    """LightGBM wrapper compatible with PandasDataset and the Hydra instantiation pattern.

    All constructor kwargs (except `name`) are forwarded directly to lightgbm's LGBMRegressor.
    """

    def __init__(self, name: str = "lightgbm", verbose: bool = False, framework: str | None = None, **kwargs):
        self.name = name
        if 'random_state' not in kwargs:
            kwargs['random_state'] = int(np.random.randint(2**31))
        self.model = LGBMRegressor(**kwargs)
        log.info(f"Initialized {self.name}")

    def fit(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, dataset: PandasDataset, **fit_params
    ) -> tuple[Any, dict[str, Any]]:
        X, y = dataset.xy
        self.model.fit(X, y.values.ravel())
        return self, {}

    def predict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, dataset: PandasDataset, **predict_params
    ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        X, _ = dataset.xy
        return cast(npt.NDArray[Any], np.asarray(self.model.predict(X))), {}

    def save(self, model_path: str) -> None:
        with open(Path(model_path) / f"{self.name}.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, model_path: str) -> LGBMModel:
        with open(model_path, "rb") as f:
            return pickle.load(f)
