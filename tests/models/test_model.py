import os
from typing import cast

import pandas as pd
import numpy as np
import pytest
from hydra import compose, initialize
from hydra.utils import instantiate
import copy


from cybench.datasets.dataset import PandasDataset
from cybench.models.naive_models import AverageYieldModel
from cybench.models.trend_models import TrendModel
# from cybench.models.sklearn_models import SklearnRidge
# from cybench.models.residual_models import RidgeRes
# from cybench.models.torch.nn_models import BaselineLSTM
from cybench.evaluation.eval import evaluate_predictions
from cybench.datasets.data_factory import DataFactory
from cybench.util.config_utils import remove_keys, adjust_model_cfg_to_dataset

from cybench.config import PATH_DATA_DIR
from cybench.config import (
    KEY_LOC,
    KEY_YEAR,
    KEY_TARGET,
    KEY_COMBINED_FEATURES,
)


def test_average_yield_model():
    model = AverageYieldModel()
    dummy_data = [
        ["US-01-001", 2000, 5.0],
        ["US-01-001", 2001, 5.5],
        ["US-01-001", 2002, 6.0],
        ["US-01-001", 2003, 5.2],
        ["US-01-002", 2000, 7.0],
        ["US-01-002", 2001, 7.5],
        ["US-01-002", 2002, 6.2],
        ["US-01-002", 2003, 5.8],
    ]
    yield_df = pd.DataFrame(dummy_data, columns=pd.Index([KEY_LOC, KEY_YEAR, KEY_TARGET]))
    yield_df = yield_df.set_index([KEY_LOC, KEY_YEAR])

    # test prediction for an existing item
    sel_loc = "US-01-001"
    assert sel_loc in yield_df.index.get_level_values(0)
    dataset = PandasDataset(cfg=None, y=yield_df, x = pd.DataFrame(index=yield_df.index.copy()))
    model.fit(dataset)
    sel_year = 2018
    filtered_df = yield_df[yield_df.index.get_level_values(0) == sel_loc]
    expected_pred = filtered_df[KEY_TARGET].mean()
    test_index = pd.MultiIndex.from_tuples([(sel_loc, sel_year)], names=[KEY_LOC, KEY_YEAR])
    test_y = pd.DataFrame(index=test_index, columns=pd.Index([KEY_TARGET]), data=[[0.0]])
    test_x = pd.DataFrame(index=test_index)
    test_dataset = PandasDataset(cfg=None, y=test_y, x=test_x)
    test_preds, _ = model.predict(test_dataset)
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)

    # test one more location
    sel_loc = "US-01-002"
    filtered_df = yield_df[yield_df.index.get_level_values(0) == sel_loc]
    expected_pred = filtered_df[KEY_TARGET].mean()
    test_index = pd.MultiIndex.from_tuples([(sel_loc, sel_year)], names=[KEY_LOC, KEY_YEAR])
    test_y = pd.DataFrame(index=test_index, columns=pd.Index([KEY_TARGET]), data=[[0.0]])
    test_x = pd.DataFrame(index=test_index)
    test_dataset = PandasDataset(cfg=None, y=test_y, x=test_x)
    test_preds, _ = model.predict(test_dataset)
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)

    # test prediction for a non-existent item
    sel_loc = "US-01-003"
    assert sel_loc not in yield_df.index.get_level_values(0)
    expected_pred = yield_df[KEY_TARGET].mean()
    test_index = pd.MultiIndex.from_tuples([(sel_loc, sel_year)], names=[KEY_LOC, KEY_YEAR])
    test_y = pd.DataFrame(index=test_index, columns=pd.Index([KEY_TARGET]), data=[[0.0]])
    test_x = pd.DataFrame(index=test_index)
    test_dataset = PandasDataset(cfg=None, y=test_y, x=test_x)
    test_preds, _ = model.predict(test_dataset)
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)


def test_trend_model():
    dummy_data = [
        ["US-01-001", 2000, 4.1],
        ["US-01-001", 2001, 4.2],
        ["US-01-001", 2002, 4.3],
        ["US-01-001", 2003, 4.4],
        ["US-01-001", 2004, 4.5],
        ["US-01-001", 2005, 4.6],
        ["US-01-001", 2006, 4.7],
        ["US-01-001", 2007, 4.8],
        ["US-01-001", 2008, 4.9],
        ["US-01-001", 2009, 5.0],
        ["US-01-002", 2000, 5.1],
        ["US-01-002", 2001, 5.2],
        ["US-01-002", 2002, 5.3],
        ["US-01-002", 2003, 5.4],
        ["US-01-002", 2004, 5.5],
        ["US-01-002", 2005, 5.6],
        ["US-01-002", 2006, 5.7],
        ["US-01-002", 2007, 5.8],
        ["US-01-002", 2008, 5.9],
        ["US-01-002", 2009, 6.0],
        ["US-01-003", 2000, 7.0],
        ["US-01-003", 2001, 8.0],
        ["US-01-003", 2003, 9.0],
        ["US-01-004", 2000, 6.0],
        ["US-01-004", 2001, 6.1],
        ["US-01-004", 2003, 6.0],
        ["US-01-004", 2004, 6.1],
        ["US-01-004", 2005, 6.0],
        ["US-01-004", 2006, 6.1],
        ["US-01-004", 2007, 6.0],
        ["US-01-004", 2008, 6.1],
        ["US-01-004", 2009, 6.0],
    ]
    yield_df = pd.DataFrame(dummy_data, columns=pd.Index([KEY_LOC, KEY_YEAR, KEY_TARGET]))

    for sel_loc in yield_df[KEY_LOC].unique():
        yield_loc_df = yield_df.loc[yield_df[KEY_LOC] == sel_loc]
        all_years = sorted(yield_loc_df[KEY_YEAR].unique())

        if sel_loc in ["US-01-001", "US-01-002"]:
            test_indexes = [0, 2, len(all_years) - 1]
            for idx in test_indexes:
                test_year = all_years[idx]
                train_years = [y for y in all_years if y != test_year]
                train_yields = yield_loc_df.loc[
                    yield_loc_df[KEY_YEAR].isin(train_years)
                ].set_index([KEY_LOC, KEY_YEAR])
                test_yields = yield_loc_df.loc[yield_loc_df[KEY_YEAR] == test_year]
                train_dataset = PandasDataset(cfg=None, y=train_yields, x=pd.DataFrame(index=train_yields.index.copy()))
                model = TrendModel()
                model.fit(train_dataset)
                test_index = pd.MultiIndex.from_tuples([(sel_loc, test_year)], names=[KEY_LOC, KEY_YEAR])
                test_y = pd.DataFrame(index=test_index, columns=pd.Index([KEY_TARGET]), data=[[0.0]])
                test_x = pd.DataFrame(index=test_index)
                test_dataset = PandasDataset(cfg=None, y=test_y, x=test_x)
                test_preds, _ = model.predict(test_dataset)
                expected_pred = test_yields[KEY_TARGET].values[0]
                assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)
        else:
            test_year = all_years[-1]
            train_years = [y for y in all_years if y != test_year]
            train_yields = yield_loc_df.loc[
                yield_loc_df[KEY_YEAR].isin(train_years)
            ].set_index([KEY_LOC, KEY_YEAR])
            train_dataset = PandasDataset(cfg=None, y=train_yields, x=pd.DataFrame(index=train_yields.index.copy()))

            # Expect the average due to insufficient data or no trend
            model = TrendModel()
            model.fit(train_dataset)
            test_index = pd.MultiIndex.from_tuples([(sel_loc, test_year)], names=[KEY_LOC, KEY_YEAR])
            test_y = pd.DataFrame(index=test_index, columns=pd.Index([KEY_TARGET]), data=[[0.0]])
            test_x = pd.DataFrame(index=test_index)
            test_dataset = PandasDataset(cfg=None, y=test_y, x=test_x)
            test_preds, _ = model.predict(test_dataset)
            expected_pred = train_yields[KEY_TARGET].mean()
            assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)


def test_trend_model_unchanged_when_train_rows_shuffled():
    """Mann-Kendall and OLS must use chronological yield order, not row order."""
    rows = [["US-01-001", year, 4.0 + 0.1 * (year - 2000)] for year in range(2000, 2010)]
    yield_df = pd.DataFrame(rows, columns=pd.Index([KEY_LOC, KEY_YEAR, KEY_TARGET]))

    test_year = 2009
    train_sorted = yield_df.loc[yield_df[KEY_YEAR] != test_year].set_index([KEY_LOC, KEY_YEAR])
    train_shuffled = train_sorted.sample(frac=1.0, random_state=0)

    test_index = pd.MultiIndex.from_tuples([("US-01-001", test_year)], names=[KEY_LOC, KEY_YEAR])
    test_dataset = PandasDataset(
        cfg=None,
        y=pd.DataFrame(index=test_index, columns=pd.Index([KEY_TARGET]), data=[[0.0]]),
        x=pd.DataFrame(index=test_index),
    )

    model_sorted = TrendModel()
    model_sorted.fit(PandasDataset(cfg=None, y=train_sorted, x=pd.DataFrame(index=train_sorted.index)))
    pred_sorted, _ = model_sorted.predict(test_dataset)

    model_shuffled = TrendModel()
    model_shuffled.fit(
        PandasDataset(cfg=None, y=train_shuffled, x=pd.DataFrame(index=train_shuffled.index))
    )
    pred_shuffled, _ = model_shuffled.predict(test_dataset)

    assert np.round(pred_sorted[0], 4) == np.round(pred_shuffled[0], 4)


@pytest.fixture
def cfg():
    with initialize(version_base=None, config_path="../../cybench/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset/crop=wheat",
                "dataset.country=NL",
                "dataset.framework=pandas",
                "dataset.target.filter_samples=null",
                "model=ridge",
            ],
        )
    return cfg

def test_sklearn_model(cfg):
    # Test 1: Test with raw data
    dataset_wheat = DataFactory(cfg.dataset).build()
    all_years = list(range(2001, 2019))
    test_years = [2017, 2018]
    train_years = [yr for yr in all_years if yr not in test_years]
    train_dataset, test_dataset = dataset_wheat.split_on_years(
        (train_years, test_years)
    )

    # Model
    model_cfg = remove_keys(cfg.model, "framework")
    model_cfg = remove_keys(model_cfg, "_search_")

    model = instantiate(model_cfg)
    fit_info = model.fit(train_dataset, val_dataset=test_dataset)
    test_preds, pred_info = model.predict(test_dataset)
    assert test_preds.shape[0] == len(test_dataset)

    data_path = os.path.join(PATH_DATA_DIR, "features", "maize", "US")
    train_csv = os.path.join(data_path, "grain_maize_US_train.csv")
    if not os.path.exists(train_csv):
        pytest.skip(f"Legacy US feature CSV not available at {train_csv}")
    train_df = pd.read_csv(train_csv, index_col=[KEY_LOC, KEY_YEAR])
    train_yields = cast(pd.DataFrame, train_df[[KEY_TARGET]].copy())
    feature_cols = [c for c in train_df.columns if c != KEY_TARGET]
    train_features = cast(pd.DataFrame, train_df.loc[:, feature_cols].copy())
    dataset_cv = PandasDataset(cfg=None, y=train_yields, x=train_features)

    # Test dataset
    test_csv = os.path.join(data_path, "grain_maize_US_train.csv")
    test_df = pd.read_csv(test_csv, index_col=[KEY_LOC, KEY_YEAR])
    test_yields = cast(pd.DataFrame, test_df[[KEY_TARGET]].copy())
    test_features = cast(pd.DataFrame, test_df.loc[:, feature_cols].copy())
    test_dataset = PandasDataset(cfg=None, y=test_yields, x=test_features)

    model = instantiate(model_cfg)
    fit_info = model.fit(dataset_cv, val_dataset=test_dataset)
    test_preds, pred_info = model.predict(test_dataset)
    assert test_preds.shape[0] == len(test_dataset)

    # TODO: Need alternative to hardcoding expected metrics.
    targets = test_dataset.targets
    evaluation_result = evaluate_predictions(targets, test_preds, cfg.evaluation)
    expected_values = {
        "normalized_rmse": [10.0, 20.0],
        "mape": [0.10, 0.20],
    }
    for metric, expected_value in expected_values.items():
        assert (
            metric in evaluation_result
        ), f"Metric '{metric}' not found in evaluation result"
        assert (
            round(evaluation_result[metric], 2) >= expected_value[0]
            and round(evaluation_result[metric], 2) <= expected_value[1]
        ), f"Value of metric '{metric}' does not match expected value"


@pytest.mark.skip(reason="Residual models (RidgeRes) were removed in v2.0 Hydra refactor")
def test_sklearn_res_model():
    pass


@pytest.fixture
def cfg_nn():
    with initialize(version_base=None, config_path="../../cybench/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset/crop=wheat",
                "dataset.country=NL",
                "dataset.framework=torch",
                "dataset.target.filter_samples=null",
                "model=cnn_lf",
                "experiment.device=cpu"
            ],
        )
    return cfg



def test_nn_model(cfg_nn):
    test_cfg = copy.deepcopy(cfg_nn)
    dataset_wheat = DataFactory(test_cfg.dataset).build()
    model_nn_cfg = adjust_model_cfg_to_dataset(test_cfg.model, dataset_wheat)

    even_years = {x for x in dataset_wheat.years if x % 2 == 0}
    odd_years = dataset_wheat.years - even_years
    train_dataset, test_dataset = dataset_wheat.split_on_years(years_split=(even_years, odd_years))
    model_nn_cfg = remove_keys(test_cfg.model, "_search_")
    # model_nn_cfg = remove_keys(model_nn_cfg, "framework")
    model = instantiate(model_nn_cfg)


    fit_info = model.fit(train_dataset, val_dataset=test_dataset)
    test_preds, pred_info = model.predict(test_dataset)
    assert test_preds.shape[0] == len(test_dataset)

    evaluation_result = evaluate_predictions(
        y_true=test_dataset.targets,
        y_pred=test_preds,
        cfg=test_cfg.evaluation,
    )
    min_expected_values = {
        "normalized_rmse": 0,
        "mape": 0.00,
    }
    for metric, expected_value in min_expected_values.items():
        assert metric in evaluation_result, f"Metric '{metric}' not found in evaluation result"
        assert evaluation_result[metric] >= expected_value, (
            f"Value of metric '{metric}' does not match expected value"
        )
        assert not np.isnan(evaluation_result[metric]), f"Value of metric '{metric}' is NaN"
