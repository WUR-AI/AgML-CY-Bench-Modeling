from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from cybench.config import KEY_YEAR


class Normalizer:
    """
    Normalizer that unites parameters for normalizing features depending on their value distribution.
    Parameters can be given or fitted on the experiment data.
    Parameter configurations are structured by YAML files. See: conf/dataset/normalizer
    YAML Config structure:

    features:
      year:
        type: minmax
        params: null
      bulk_density:
        type: standard
        params: null

    Supports normalization types:
      - "minmax": maps [min, max] -> [-1, 1]
      - "standard" maps N(mu, sigma) to N(0,1)
      - "logsinh"
      - "none"
    """

    def __init__(self, norm_cfg: dict[str, Any]):
        self.name = norm_cfg["name"]
        self.feature_cfg: dict[str, Any] = norm_cfg["features"]

    def _fit_feature(self, series: pd.Series, ftype: str):
        """Compute needed statistics depending on normalization type."""
        if ftype == "none":
            return {}

        if ftype == "minmax":
            return {
                "min": float(series.min()),
                "max": float(series.max())
            }

        if ftype == "standard":
            return {
                "mean": float(series.mean()),
                "std": float(series.std())
            }

        if ftype == "logsinh":
            # No parameters needed — log-sinh is reversible without fitting.
            return {}

        raise ValueError(f"Unknown normalization type: {ftype}")

    def _apply_feature(self, series: pd.Series, ftype: str, params: dict[str, Any]):
        """Apply normalization using already-fitted parameters."""
        if ftype == "none":
            return series

        if ftype == "minmax":
            min, max = params["min"], params["max"]
            range = max - min
            if range == 0:
                return series * 0.0
            return (series - max / 2 - min / 2) / (range / 2)

        if ftype == "standard":
            if params["std"] == 0:
                return series * 0.0
            return (series - params["mean"]) / params["std"]

        if ftype == "logsinh":
            # x ↦ arcsinh(x)  (safe transform for skewed positive variables)
            return np.arcsinh(series)

        raise ValueError(f"Unknown normalization type: {ftype}")

    def _reverse_feature(self, value, ftype: str, params: dict[str, Any]):
        """Apply inverse normalization using fitted parameters."""
        if ftype == "none":
            return value

        if ftype == "minmax":
            min_val, max_val = params["min"], params["max"]
            range_val = max_val - min_val
            if range_val == 0:
                return value
            # Reverse: y = (x - mid) / (range/2) -> x = y * (range/2) + mid
            return value * (range_val / 2) + (max_val / 2 + min_val / 2)

        if ftype == "standard":
            if params["std"] == 0:
                return value
            # Reverse: y = (x - mean) / std -> x = y * std + mean
            return value * params["std"] + params["mean"]

        if ftype == "logsinh":
            # Reverse: y = arcsinh(x) -> x = sinh(y)
            return np.sinh(value)

        raise ValueError(f"Unknown normalization type: {ftype}")

    def _series_for_fit(
        self,
        series: pd.Series,
        df: pd.DataFrame,
        fit_years: list[int] | None,
    ) -> pd.Series:
        """Restrict fit statistics to ``fit_years`` when the frame is year-indexed."""
        if fit_years is None:
            return series
        fit_set = set(int(y) for y in fit_years)
        if isinstance(df.index, pd.MultiIndex) and KEY_YEAR in df.index.names:
            mask = df.index.get_level_values(KEY_YEAR).isin(fit_set)
            return series.loc[mask]
        if KEY_YEAR in df.columns:
            return series.loc[df[KEY_YEAR].isin(fit_set)]
        return series

    def fit_normalize(self, dfs, fit_years: list[int] | None = None):
        """
        Fits parameters across all DataFrames and returns
        normalized copies of the DataFrames.

        When ``fit_years`` is set, statistics are computed on those years only
        (e.g. screening train ∪ val); normalization is then applied to all rows.
        """
        for source_name, df in dfs.items():
            for feature, cfg in self.feature_cfg.items():
                ftype = cfg["type"]
                if ftype == "logsinh":  # logsinh has no parameter
                    continue

                params = cfg["params"]
                if params:  # parameter already set
                    continue
                if feature not in df.columns:
                    continue

                fit_series = self._series_for_fit(df[feature], df, fit_years)
                params = self._fit_feature(fit_series, ftype)
                self.feature_cfg[feature]["params"] = params
        return self.normalize(dfs)

    def normalize(self, dfs):
        """
        Normalize using already-fitted parameters.
        Returns new list of DataFrames.
        """
        for source_name, df in dfs.items():
            for feature, cfg in self.feature_cfg.items():
                if feature not in df.columns:
                    continue
                ftype = cfg["type"]
                params = cfg.get("params", {})
                df[feature] = self._apply_feature(df[feature], ftype, params)
        return dfs

    def normalize_sequence(self, series: pd.Series):
        """
        Normalize sequence.
        Args:
            series: pd.Series sequence.

        Returns: normalized sequence.
        """
        feature_name = str(series.name)
        assert feature_name in self.feature_cfg, (
            f"{feature_name} not in normalizer feature keys: {self.feature_cfg.keys()}"
        )
        cfg = self.feature_cfg[feature_name]
        ftype = cfg["type"]
        params = cfg.get("params", {})
        return self._apply_feature(series, ftype, params)

    def denormalize(self, data, feature_names):
        """
        Reverses normalization for a value, array, or matrix.
        The feature dimension must be the last dimension.

        Args:
            data: Input data (numpy array or tensor).
            feature_names: List of feature names matching the last dimension of data.

        Returns:
            Numpy array of denormalized data.
        """
        # Convert Torch tensors to numpy if necessary
        if hasattr(data, "cpu"):
            data = data.detach().cpu().numpy()

        # Ensure data is a numpy array
        data = np.array(data)

        # Validate dimensions
        if data.shape[-1] != len(feature_names):
            raise ValueError(
                f"Last dimension size ({data.shape[-1]}) does not match "
                f"the number of feature names provided ({len(feature_names)})."
            )

        # Create a copy to avoid modifying the input in-place
        denorm_data = data.copy()

        for i, feature in enumerate(feature_names):
            if feature not in self.feature_cfg:
                continue

            cfg = self.feature_cfg[feature]
            ftype = cfg["type"]
            params = cfg.get("params", {})

            # Apply inverse transformation to the specific feature slice
            # usage of [...] preserves all preceding dimensions (batch, time, etc.)
            denorm_data[..., i] = self._reverse_feature(denorm_data[..., i], ftype, params)

        return denorm_data

    def to_omegaconf(self):
        """
        Produces an OmegaConf node corresponding to normalization.yaml:

        features:
          feature_name:
            type: ...
            params: ...

        This is suitable to write back into a YAML file.
        """
        return OmegaConf.create({"features": self.feature_cfg})
