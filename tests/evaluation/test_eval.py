import numpy as np
from hydra import compose, initialize

from cybench.evaluation.eval import evaluate_predictions


def test_evaluate_predictions_ignores_nan_pairs():
    with initialize(version_base=None, config_path="../../cybench/conf"):
        cfg = compose(config_name="config", overrides=["dataset/crop=maize", "model=average"])

    y_true = np.array([5.0, 6.0, 7.0, 8.0])
    y_pred = np.array([4.5, np.nan, 6.5, 8.1])
    metrics = evaluate_predictions(y_true, y_pred, cfg.evaluation)

    assert np.isfinite(metrics["normalized_rmse"])
    assert metrics["normalized_rmse"] >= 0


def test_evaluate_predictions_all_nan_returns_nan_metrics():
    with initialize(version_base=None, config_path="../../cybench/conf"):
        cfg = compose(config_name="config", overrides=["dataset/crop=maize", "model=average"])

    y_true = np.array([5.0, 6.0])
    y_pred = np.array([np.nan, np.nan])
    metrics = evaluate_predictions(y_true, y_pred, cfg.evaluation)

    assert np.isnan(metrics["normalized_rmse"])
