"""ZeeLin local-decade trend / residual ensemble, ported to PandasDataset.

Original submission (MIT License, Copyright (c) 2026 geqiulin / ZeeLin):
https://github.com/geqiulin/ZeeLin-CYBench-China-Maize

The published China-maize result uses official leave-one-year-out evaluation.
The local-decade trend therefore interpolates from the ten *nearest* training
years, including later years when they exist in the outer training fold. That
is allowed by LOYO and is **not** a forward operational forecast.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR
from cybench.datasets.dataset import PandasDataset
from cybench.models.model import BaseModel
from cybench.models.persistence import load_pickle, save_pickle
from cybench.models.trend_models import TrendModel

log = logging.getLogger(__name__)

_DEFAULT_RESIDUAL_WEIGHTS = {
    "ridge": 0.15,
    "extra": 0.35,
    "rf": 0.20,
    "xgb": 0.30,
}


class ZeeLinTrendResidualEnsemble(BaseModel):
    """Blend a local yield trend with four residual learners.

    All target-derived quantities use only the current training fold. Training
    residuals use leave-one-year-out official trend estimates, and the location
    mean is leave-one-sample-out for training rows.
    """

    def __init__(
        self,
        name: str = "zeelin_trend_residual",
        framework: str | None = None,
        random_state: int = 20260814,
        local_window: int = 10,
        trend_blend: float = 0.70,
        residual_weights: Mapping[str, float] | None = None,
        n_jobs: int = -1,
        n_estimators: int | None = None,
        extra_trees_n_estimators: int = 700,
        rf_n_estimators: int = 700,
        xgb_n_estimators: int = 500,
        verbose: bool = False,
        **_ignored,
    ):
        self.name = name
        self.random_state = int(random_state)
        self.local_window = int(local_window)
        self.trend_blend = float(trend_blend)
        weights = residual_weights or _DEFAULT_RESIDUAL_WEIGHTS
        self._residual_weights = {str(k): float(v) for k, v in dict(weights).items()}
        self.n_jobs = int(n_jobs)
        if n_estimators is not None:
            extra_trees_n_estimators = int(n_estimators)
            rf_n_estimators = int(n_estimators)
            xgb_n_estimators = int(n_estimators)
        self.extra_trees_n_estimators = int(extra_trees_n_estimators)
        self.rf_n_estimators = int(rf_n_estimators)
        self.xgb_n_estimators = int(xgb_n_estimators)
        self._trend = TrendModel()
        self._feature_cols: list[str] | None = None
        self._loc_columns: list[str] | None = None
        self._loc_means: pd.Series | None = None
        self._models: dict[str, Any] = {}
        self._train_targets: pd.DataFrame | None = None
        self._global_mean: float | None = None
        log.info("Initialized %s", self.name)

    @staticmethod
    def _target_dataset(frame: pd.DataFrame) -> PandasDataset:
        target = frame[[KEY_LOC, KEY_YEAR, KEY_TARGET]].copy()
        target = target.set_index([KEY_LOC, KEY_YEAR])
        return PandasDataset(
            cfg=None,
            y=target,
            x=pd.DataFrame(index=target.index.copy()),
        )

    @classmethod
    def _trend_predict(
        cls, train: pd.DataFrame, test: pd.DataFrame
    ) -> npt.NDArray[Any]:
        model = TrendModel()
        model.fit(cls._target_dataset(train))
        preds, _ = model.predict(cls._target_dataset(test))
        return np.asarray(preds)

    @classmethod
    def _crossfit_trend(cls, train: pd.DataFrame) -> npt.NDArray[Any]:
        result = pd.Series(index=train.index, dtype=float)
        for year in sorted(train[KEY_YEAR].unique()):
            mask = train[KEY_YEAR].eq(year)
            result.loc[train.index[mask]] = cls._trend_predict(
                train.loc[~mask], train.loc[mask]
            )
        return result.to_numpy()

    @staticmethod
    def _select_decade_years(
        train_years: npt.NDArray[Any],
        test_year: int,
        window: int,
    ) -> npt.NDArray[Any]:
        """Nearest ``window`` training years by |year - test_year|, then year."""
        years = np.asarray(train_years)
        if years.size == 0:
            return years
        order = np.lexsort((years, np.abs(years - test_year)))
        return years[order][:window]

    def _frame(self, dataset: PandasDataset) -> pd.DataFrame:
        x, y = dataset.xy
        frame = x.copy()
        frame[KEY_TARGET] = y[KEY_TARGET].to_numpy()
        return frame.reset_index()

    def _design_matrix(self, frame: pd.DataFrame, training: bool) -> pd.DataFrame:
        loc = pd.get_dummies(frame[KEY_LOC], prefix="loc", dtype=float)
        numeric = frame.drop(columns=[KEY_LOC, KEY_TARGET]).astype(float)
        matrix = pd.concat([numeric, loc], axis=1)
        if training:
            self._loc_columns = list(loc.columns)
        else:
            if self._loc_columns is None:
                raise RuntimeError(f"{self.name} must be fitted before predict()")
            for col in self._loc_columns:
                if col not in matrix:
                    matrix[col] = 0.0
            unexpected = [
                c
                for c in matrix.columns
                if str(c).startswith("loc_") and c not in self._loc_columns
            ]
            if unexpected:
                matrix = matrix.drop(columns=unexpected)

        if training:
            grouped = frame.groupby(KEY_LOC, observed=True)[KEY_TARGET]
            sums = frame[KEY_LOC].map(grouped.sum())
            counts = frame[KEY_LOC].map(grouped.count())
            denom = (counts - 1).astype(float)
            loc_mean = pd.Series(np.nan, index=frame.index, dtype=float)
            ok = denom.gt(0)
            loc_mean.loc[ok] = (sums.loc[ok] - frame.loc[ok, KEY_TARGET]) / denom.loc[ok]
            global_sum = float(frame[KEY_TARGET].sum())
            n = len(frame)
            if n > 1:
                loc_mean.loc[~ok] = (global_sum - frame.loc[~ok, KEY_TARGET]) / (n - 1)
            else:
                loc_mean.loc[~ok] = float(frame[KEY_TARGET].iloc[0])
            matrix["loc_y_mean"] = loc_mean
            self._loc_means = grouped.mean()
            self._feature_cols = list(matrix.columns)
        else:
            if self._loc_means is None or self._feature_cols is None:
                raise RuntimeError(f"{self.name} must be fitted before predict()")
            matrix["loc_y_mean"] = frame[KEY_LOC].map(self._loc_means)
            if self._global_mean is not None:
                matrix["loc_y_mean"] = matrix["loc_y_mean"].fillna(self._global_mean)
            for col in self._feature_cols:
                if col not in matrix:
                    matrix[col] = 0.0
            matrix = matrix[self._feature_cols]
        return matrix

    def _local_decade_predictions(self, dataset: PandasDataset) -> npt.NDArray[Any]:
        if self._train_targets is None or self._global_mean is None:
            raise RuntimeError(f"{self.name} must be fitted before predict()")
        index = dataset.y.index
        locs = index.get_level_values(KEY_LOC)
        years = index.get_level_values(KEY_YEAR)
        preds = np.empty(len(index), dtype=float)
        for i, (loc, year) in enumerate(zip(locs, years)):
            history = self._train_targets[self._train_targets[KEY_LOC] == loc]
            if history.empty:
                preds[i] = self._global_mean
                continue
            selected_years = self._select_decade_years(
                history[KEY_YEAR].to_numpy(), int(year), self.local_window
            )
            window = history[history[KEY_YEAR].isin(selected_years)]
            if len(window) < 2:
                preds[i] = float(window[KEY_TARGET].mean()) if len(window) else self._global_mean
                continue
            reg = LinearRegression().fit(window[[KEY_YEAR]], window[KEY_TARGET])
            preds[i] = float(reg.predict(pd.DataFrame({KEY_YEAR: [year]}))[0])
        return preds

    def fit(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, dataset: PandasDataset, **fit_params
    ) -> tuple[Any, dict[str, Any]]:
        frame = self._frame(dataset)
        x = self._design_matrix(frame, training=True)
        y = frame[KEY_TARGET].to_numpy()
        residual = y - self._crossfit_trend(frame)
        self._trend.fit(self._target_dataset(frame))
        self._train_targets = frame[[KEY_LOC, KEY_YEAR, KEY_TARGET]].copy()
        self._global_mean = float(frame[KEY_TARGET].mean())

        self._models = {
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=20.0)),
            "extra": ExtraTreesRegressor(
                n_estimators=self.extra_trees_n_estimators,
                max_features=0.65,
                min_samples_leaf=3,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ),
            "rf": RandomForestRegressor(
                n_estimators=self.rf_n_estimators,
                max_features=0.7,
                min_samples_leaf=4,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ),
            "xgb": XGBRegressor(
                n_estimators=self.xgb_n_estimators,
                learning_rate=0.025,
                max_depth=2,
                min_child_weight=8,
                subsample=0.85,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=20.0,
                objective="reg:squarederror",
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ),
        }
        x_values = x.to_numpy()
        for model in self._models.values():
            model.fit(x_values, residual)
        return self, {}

    def predict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, dataset: PandasDataset, **predict_params
    ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        if not self._models:
            raise RuntimeError(f"{self.name} must be fitted before predict()")
        frame = self._frame(dataset)
        x = self._design_matrix(frame, training=False).to_numpy()
        trend, _ = self._trend.predict(self._target_dataset(frame))
        residuals = {name: model.predict(x) for name, model in self._models.items()}
        ml_prediction = trend + (
            self._residual_weights["ridge"] * residuals["ridge"]
            + self._residual_weights["extra"] * residuals["extra"]
            + self._residual_weights["rf"] * residuals["rf"]
            + self._residual_weights["xgb"] * residuals["xgb"]
        )
        local_predictions = self._local_decade_predictions(dataset)
        prediction = (1.0 - self.trend_blend) * ml_prediction + self.trend_blend * local_predictions
        return prediction, {
            "official_trend": trend,
            "local_decade_trend": local_predictions,
            "ml_prediction": ml_prediction,
            "residual_components": residuals,
        }

    def save(self, model_path: str) -> None:
        save_pickle(self, model_path, self.name)

    @classmethod
    def load(
        cls, model_path: str, name: str = "zeelin_trend_residual"
    ) -> ZeeLinTrendResidualEnsemble:
        return load_pickle(model_path, name)
