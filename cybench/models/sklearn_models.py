import pickle
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from cybench.models.model import BaseModel
from cybench.datasets.dataset import PandasDataset

log = logging.getLogger(__name__)


class Ridge(BaseModel):
    """Ridge regression wrapper compatible with PandasDataset and the Hydra instantiation pattern.

    All constructor kwargs (except `name`) are forwarded directly to sklearn's Ridge.
    Internally wraps the estimator in a Pipeline with median imputation and standard scaling,
    since Ridge does not accept NaNs and is sensitive to feature scale.
    """

    def __init__(self, name: str = "ridge", verbose: bool = False, **kwargs):
        self.name = name
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("estimator", SklearnRidge(**kwargs)),
        ])
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


class RandomForest(BaseModel):
    """Random forest wrapper compatible with PandasDataset and the Hydra instantiation pattern.

    All constructor kwargs (except `name`) are forwarded directly to sklearn's RandomForestRegressor.
    Internally wraps the estimator in a Pipeline with median imputation.
    RandomForest is scale-invariant so no scaler is added.
    """

    def __init__(self, name: str = "random_forest", verbose: bool = False, **kwargs):
        self.name = name
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("estimator", RandomForestRegressor(**kwargs)),
        ])
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
