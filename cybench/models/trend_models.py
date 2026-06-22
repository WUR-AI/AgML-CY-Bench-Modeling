from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import pymannkendall as trend_mk

from cybench.models.model import BaseModel
from cybench.models.persistence import load_pickle, save_pickle
from cybench.datasets.dataset import PandasDataset
from cybench.config import KEY_LOC, KEY_YEAR, KEY_TARGET

log = logging.getLogger(__name__)


class TrendModel(BaseModel):
    """Linear trend estimator using years as the sole predictor.

    For each test location, finds an optimal trend window via the
    Mann-Kendall test and fits OLS on (year -> yield).  Falls back
    to the location mean (or global mean) when no significant trend
    is detected or when training data is insufficient.

    Operates on PandasDataset; compatible with Hydra instantiation.

    Parameters
    ----------
    name : str
        Model name, used for saving artifacts.
    min_trend_window : int
        Minimum number of years required to estimate a trend.
    max_trend_window : int
        Maximum number of years considered when searching for the
        optimal trend window via the Mann-Kendall test.
    """

    def __init__(
        self,
        name: str = "trend",
        min_trend_window: int = 5,
        max_trend_window: int = 10,
        **_ignored,
    ):
        # Hydra model configs may include metadata keys (e.g. framework).
        self.name = name
        self.min_trend_window = min_trend_window
        self.max_trend_window = max_trend_window
        self._train_df: pd.DataFrame | None = None

    def fit(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, dataset: PandasDataset, **fit_params
    ) -> tuple[Any, dict[str, Any]]:
        y = dataset.y
        self._train_df = cast(
            pd.DataFrame,
            y.reset_index()[[KEY_LOC, KEY_YEAR, KEY_TARGET]],
        )
        return self, {}

    def predict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, dataset: PandasDataset, **predict_params
    ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        y = dataset.y
        test_df = y.reset_index()
        predictions = np.empty(len(test_df))

        for i, (_, row) in enumerate(test_df.iterrows()):
            loc = row[KEY_LOC]
            test_year = row[KEY_YEAR]
            predictions[i] = self._predict_single(loc, test_year)

        return predictions, {}

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _require_train_df(self) -> pd.DataFrame:
        if self._train_df is None:
            raise RuntimeError(f"{self.name} must be fitted before predict()")
        return self._train_df

    def _predict_single(self, loc, test_year) -> float:
        train_df = self._require_train_df()
        sel = train_df[train_df[KEY_LOC] == loc]

        # Case 1: no training data for location
        if sel.empty:
            return float(train_df[KEY_TARGET].mean())

        label_df = cast(
            pd.DataFrame,
            sel.loc[:, [KEY_YEAR, KEY_TARGET]].sort_values(KEY_YEAR),
        )
        train_years = sorted(cast(pd.Series, label_df[KEY_YEAR]).unique())

        lt = [yr for yr in train_years if yr < test_year]
        gt = [yr for yr in train_years if yr > test_year]

        # Case 2: not enough years on either side
        if len(lt) < self.min_trend_window and len(gt) < self.min_trend_window:
            return float(label_df[KEY_TARGET].mean())

        trend = None

        # Case 3: trend from years before test year
        if len(lt) >= self.min_trend_window:
            window = self._find_optimal_window(label_df, lt, extend_forward=False)
            if window is not None:
                window_x, vals = self._window_series(label_df, window)
                trend = self._estimate_trend(window_x, vals, test_year)

        # Case 4: trend from years after test year
        if trend is None and len(gt) >= self.min_trend_window:
            window = self._find_optimal_window(label_df, gt, extend_forward=True)
            if window is not None:
                window_x, vals = self._window_series(label_df, window)
                trend = self._estimate_trend(window_x, vals, test_year)

        # Case 5: no significant trend — use location mean
        if trend is None:
            trend = float(label_df[KEY_TARGET].mean())

        return float(trend)

    @staticmethod
    def _window_series(
        label_df: pd.DataFrame,
        years: list[int],
    ) -> tuple[npt.NDArray[Any], npt.NDArray[np.floating[Any]]]:
        """Return chronologically ordered (years, yields) for Mann-Kendall / OLS."""
        sub = label_df[label_df[KEY_YEAR].isin(years)].sort_values(KEY_YEAR)
        return (
            sub[KEY_YEAR].to_numpy(),
            cast(npt.NDArray[np.floating[Any]], sub[KEY_TARGET].to_numpy(dtype=float)),
        )

    def _estimate_trend(self, trend_x, trend_y, test_x):
        """Fit OLS on (year, yield) and predict at test_x."""
        trend_x = add_constant(trend_x)
        model = OLS(trend_y, trend_x).fit()
        pred_x = add_constant(np.array([[test_x]]), has_constant="add")
        return model.predict(pred_x)[0]

    def _find_optimal_window(
        self,
        label_df: pd.DataFrame,
        window_years: list[int],
        extend_forward: bool = False,
    ):
        """Select the window size that yields the most significant Mann-Kendall trend."""
        min_p = float("inf")
        best = None
        upper = min(self.max_trend_window, len(window_years)) + 1

        for i in range(self.min_trend_window, upper):
            years = window_years[:i] if extend_forward else window_years[-i:]
            _, vals = self._window_series(label_df, years)
            result = trend_mk.original_test(vals)
            if result.h and result.p < min_p:
                min_p = result.p
                best = years

        return best

    def save(self, model_path: str) -> None:
        save_pickle(self, model_path, self.name)

    @classmethod
    def load(cls, model_path: str, name: str = "trend") -> TrendModel:
        return load_pickle(model_path, name)
