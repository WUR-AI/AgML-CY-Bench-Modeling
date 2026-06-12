"""Model base class
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy.typing as npt

from cybench.datasets.dataset import BaseDataset


class BaseModel(ABC):
    @abstractmethod
    def fit(self, dataset: BaseDataset, **fit_params) -> tuple[Any, dict[str, Any]]:
        """Fit or train the model.

        Args:
          dataset: Dataset
          **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        raise NotImplementedError

    def predict(
        self, dataset: BaseDataset, **predict_params
    ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        """Run fitted model on data.

        Args:
          dataset: Dataset
          **predict_params: Additional parameters.

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, model_path: str) -> None:
        """Save model, e.g. using pickle.

        Args:
          model_path: File path that will be used to save the model.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, model_path: str) -> BaseModel:
        """Deserialize a saved model.

        Args:
          model_path: File path that was used to save the model.

        Returns:
          The deserialized model.
        """
        raise NotImplementedError
