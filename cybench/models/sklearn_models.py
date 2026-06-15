from __future__ import annotations

import logging
from typing import Any, cast

import numpy.typing as npt
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from cybench.models.model import BaseModel
from cybench.models.persistence import load_pickle, save_pickle
from cybench.datasets.dataset import PandasDataset

log = logging.getLogger(__name__)


class Ridge(BaseModel):
    """Ridge regression wrapper compatible with PandasDataset and the Hydra instantiation pattern.

    All constructor kwargs (except `name`) are forwarded directly to sklearn's Ridge.
    Internally wraps the estimator in a Pipeline with median imputation and standard scaling,
    since Ridge does not accept NaNs and is sensitive to feature scale.
    """

    def __init__(self, name: str = "ridge", verbose: bool = False, framework: str | None = None, **kwargs):
        self.name = name
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("estimator", SklearnRidge(**kwargs)),
        ])
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
        return cast(npt.NDArray[Any], self.model.predict(X)), {}

    def save(self, model_path: str) -> None:
        save_pickle(self, model_path, self.name)

    @classmethod
    def load(cls, model_path: str, name: str = "ridge") -> Ridge:
        return load_pickle(model_path, name)


class RandomForest(BaseModel):
    """Random forest wrapper compatible with PandasDataset and the Hydra instantiation pattern.

    All constructor kwargs (except `name`) are forwarded directly to sklearn's RandomForestRegressor.
    Internally wraps the estimator in a Pipeline with median imputation.
    RandomForest is scale-invariant so no scaler is added.
    """

    def __init__(self, name: str = "random_forest", verbose: bool = False, framework: str | None = None, **kwargs):
        self.name = name
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("estimator", RandomForestRegressor(**kwargs)),
        ])
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
        return cast(npt.NDArray[Any], self.model.predict(X)), {}

    def save(self, model_path: str) -> None:
        save_pickle(self, model_path, self.name)

    @classmethod
    def load(cls, model_path: str, name: str = "random_forest") -> RandomForest:
        return load_pickle(model_path, name)
