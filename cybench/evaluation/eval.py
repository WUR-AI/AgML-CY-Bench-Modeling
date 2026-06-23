from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from scipy.stats import pearsonr

from cybench.models.model import BaseModel
from cybench.datasets.dataset import BaseDataset
from cybench.config import KEY_TARGET, KEY_LOC

implemented_metrics = {}


def metric(func):
    """Decorator to mark functions as metrics"""
    implemented_metrics[func.__name__] = func
    return func


def evaluate_model(
    cfg,
    model: BaseModel,
    dataset: BaseDataset,
):
    """
    Evaluate the performance of a model using specified metrics.

    Args:
      cfg: controlling which metrics are computed.
      model: The trained model to be evaluated.
      dataset: Dataset.

    Returns:
      A dictionary containing the calculated metrics.
    """
    y_true = dataset.targets
    y_pred, _ = model.predict(dataset)
    results = evaluate_predictions(y_true, y_pred, cfg)

    return results


def _finite_prediction_mask(
    y_true: npt.NDArray[Any],
    y_pred: npt.NDArray[Any],
) -> npt.NDArray[np.bool_]:
    """Rows where both observed and predicted yield are finite (e.g. skip missing TWSO)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.isfinite(y_true) & np.isfinite(y_pred)


def evaluate_predictions(
    y_true: npt.NDArray[Any],
    y_pred: npt.NDArray[Any],
    cfg,
):
    """
    Evaluate predictions using specified metrics.

    Args:
      y_true (numpy.ndarray): True labels for evaluation.
      y_pred (numpy.ndarray): Predicted values.
      cfg: controlling which metrics are computed.

    Returns:
      A dictionary containing the calculated metrics.
    """
    mask = _finite_prediction_mask(y_true, y_pred)
    y_true = np.asarray(y_true, dtype=float)[mask]
    y_pred = np.asarray(y_pred, dtype=float)[mask]
    if y_true.size == 0:
        return {metric_name: float("nan") for metric_name in cfg.metrics}

    results = {}
    for metric_name in cfg.metrics:
        metric_function = implemented_metrics.get(metric_name)
        if metric_function:
            result = metric_function(y_true, y_pred)
            results[metric_name] = np.round(result, 4)
        else:
            raise ValueError(f"Metric function '{metric_name}' not implemented.")

    return results


def prepare_targets_preds(df_yr, model_name, y_loc_mean=None, residual=False):
    """Prepare y_true and y_pred, optionally using residuals, dropping NaNs."""
    y_true = df_yr[KEY_TARGET].values
    y_pred = df_yr[model_name].values

    if residual and y_loc_mean is not None:
        y_true = y_true - df_yr[KEY_LOC].map(y_loc_mean)
        y_pred = y_pred - df_yr[KEY_LOC].map(y_loc_mean)

    # --- Drop NaNs ---
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    return y_true, y_pred


@metric
def mse(y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]):
    """
    Calculate the mean squared error (MSE) between true and predicted values.

    Args:
      y_true (numpy.ndarray): True values.
      y_pred (numpy.ndarray): Predicted values.

    Returns:
      float: MSE value as a percentage.
    """

    mse = mean_squared_error(y_true, y_pred)
    return mse


@metric
def normalized_rmse(y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]):
    """
    Calculate the normalized Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
      y_true (numpy.ndarray): True values.
      y_pred (numpy.ndarray): Predicted values.

    Returns:
      float: Normalized RMSE value as a percentage.
    """

    mse = mean_squared_error(y_true, y_pred)
    mean_y_true = np.mean(y_true)
    return 100 * np.sqrt(mse) / mean_y_true


@metric
def mape(y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    Note that in the provided implementation using scikit-learn, there is an absence of multiplication by 100

    Args:
    - y_true (numpy.ndarray): True values.
    - y_pred (numpy.ndarray): Predicted values.

    Returns:
    - float: Mean Absolute Percentage Error.
    """

    return mean_absolute_percentage_error(y_true, y_pred)


@metric
def r2(y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]):
    """
    Calculate coefficient of determination (R2).

    Args:
    - y_true (numpy.ndarray): True values.
    - y_pred (numpy.ndarray): Predicted values.

    Returns:
    - float: r2.
    """

    return r2_score(y_true, y_pred)


@metric
def r(y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]):
    """
    Calculate Pearson correlation coefficient.

    Args:
    - y_true (numpy.ndarray): True values.
    - y_pred (numpy.ndarray): Predicted values.

    Returns:
    - float: Pearson's R.
    """
    try:
        return pearsonr(y_true, y_pred)[0]
    except Exception:
        return float("nan")
