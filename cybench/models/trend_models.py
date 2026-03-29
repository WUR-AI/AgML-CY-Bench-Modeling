import pickle
import logging
from pathlib import Path

import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import pymannkendall as trend_mk

from cybench.models.model import BaseModel
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
    ):
        self.name = name
        self.min_trend_window = min_trend_window
        self.max_trend_window = max_trend_window
        self._train_df = None

    def fit(self, dataset: PandasDataset, **fit_params) -> dict:
        y = dataset.y
        self._train_df = y.reset_index()[[KEY_LOC, KEY_YEAR, KEY_TARGET]]
        return {}

    def predict(self, dataset: PandasDataset, **predict_params):
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

    def _predict_single(self, loc, test_year):
        sel = self._train_df[self._train_df[KEY_LOC] == loc]

        # Case 1: no training data for location
        if sel.empty:
            return self._train_df[KEY_TARGET].mean()

        train_labels = sel[[KEY_YEAR, KEY_TARGET]].values
        train_years = sorted(sel[KEY_YEAR].unique())

        lt = [yr for yr in train_years if yr < test_year]
        gt = [yr for yr in train_years if yr > test_year]

        # Case 2: not enough years on either side
        if len(lt) < self.min_trend_window and len(gt) < self.min_trend_window:
            return sel[KEY_TARGET].mean()

        trend = None

        # Case 3: trend from years before test year
        if len(lt) >= self.min_trend_window:
            window = self._find_optimal_window(train_labels, lt, extend_forward=False)
            if window is not None:
                vals = train_labels[np.isin(train_labels[:, 0], window)][:, 1]
                trend = self._estimate_trend(window, vals, test_year)

        # Case 4: trend from years after test year
        if trend is None and len(gt) >= self.min_trend_window:
            window = self._find_optimal_window(train_labels, gt, extend_forward=True)
            if window is not None:
                vals = train_labels[np.isin(train_labels[:, 0], window)][:, 1]
                trend = self._estimate_trend(window, vals, test_year)

        # Case 5: no significant trend — use location mean
        if trend is None:
            trend = sel[KEY_TARGET].mean()

        return trend

    def _estimate_trend(self, trend_x, trend_y, test_x):
        """Fit OLS on (year, yield) and predict at test_x."""
        trend_x = add_constant(trend_x)
        model = OLS(trend_y, trend_x).fit()
        pred_x = add_constant(np.array([[test_x]]), has_constant="add")
        return model.predict(pred_x)[0]

    def _find_optimal_window(self, train_labels, window_years, extend_forward=False):
        """Select the window size that yields the most significant Mann-Kendall trend."""
        min_p = float("inf")
        best = None
        upper = min(self.max_trend_window, len(window_years)) + 1

        for i in range(self.min_trend_window, upper):
            years = window_years[:i] if extend_forward else window_years[-i:]
            vals = train_labels[np.isin(train_labels[:, 0], years)][:, 1]
            result = trend_mk.original_test(vals)
            if result.h and result.p < min_p:
                min_p = result.p
                best = years

        return best

    def save(self, path):
        with open(Path(path) / f"{self.name}.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
