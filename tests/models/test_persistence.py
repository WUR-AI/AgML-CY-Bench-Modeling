from pathlib import Path

import numpy as np
import pandas as pd

from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR
from cybench.datasets.dataset import PandasDataset
from cybench.models.naive_models import AverageYieldModel
from cybench.models.persistence import pickle_path, save_pickle, load_pickle
from cybench.models.sklearn_models import Ridge
from cybench.models.trend_models import TrendModel


def test_pickle_path_resolution():
    assert pickle_path("output/run/", "ridge") == Path("output/run/ridge.pkl")
    assert pickle_path("output/run/ridge.pkl", "ridge") == Path(
        "output/run/ridge.pkl"
    )


def test_ridge_save_load_directory(tmp_path):
    rng = np.random.default_rng(0)
    n = 20
    index = pd.MultiIndex.from_product(
        [["loc-a"], range(2000, 2020)],
        names=[KEY_LOC, KEY_YEAR],
    )[:n]
    X = pd.DataFrame(rng.normal(size=(n, 3)), index=index)
    y = pd.DataFrame({KEY_TARGET: X.sum(axis=1).to_numpy()}, index=index)
    dataset = PandasDataset(cfg=None, y=y, x=X)

    model = Ridge()
    model.fit(dataset)
    model.save(str(tmp_path))

    loaded = Ridge.load(str(tmp_path))
    np.testing.assert_allclose(
        loaded.predict(dataset)[0],
        model.predict(dataset)[0],
    )


def test_average_yield_save_load_file_path(tmp_path):
    dummy_data = [
        ["US-01-001", 2000, 5.0],
        ["US-01-001", 2001, 5.5],
        ["US-01-002", 2000, 7.0],
        ["US-01-002", 2001, 7.5],
    ]
    yield_df = pd.DataFrame(
        dummy_data, columns=pd.Index([KEY_LOC, KEY_YEAR, KEY_TARGET])
    ).set_index([KEY_LOC, KEY_YEAR])

    model = AverageYieldModel()
    model.fit(PandasDataset(cfg=None, y=yield_df, x=pd.DataFrame(index=yield_df.index)))

    model_file = tmp_path / "average.pkl"
    model.save(str(model_file))
    loaded = AverageYieldModel.load(str(model_file))
    np.testing.assert_allclose(
        loaded._averages.to_numpy(),
        model._averages.to_numpy(),
    )


def test_trend_save_load_directory(tmp_path):
    dummy_data = [
        ["US-01-001", 2000, 4.1],
        ["US-01-001", 2001, 4.2],
        ["US-01-001", 2002, 4.3],
        ["US-01-001", 2003, 4.4],
        ["US-01-001", 2004, 4.5],
        ["US-01-001", 2005, 4.6],
    ]
    yield_df = pd.DataFrame(
        dummy_data, columns=pd.Index([KEY_LOC, KEY_YEAR, KEY_TARGET])
    ).set_index([KEY_LOC, KEY_YEAR])
    dataset = PandasDataset(cfg=None, y=yield_df, x=pd.DataFrame(index=yield_df.index))

    model = TrendModel()
    model.fit(dataset)
    model.save(str(tmp_path / "artifacts"))

    loaded = TrendModel.load(str(tmp_path / "artifacts"))
    assert loaded._train_df is not None
    pd.testing.assert_frame_equal(loaded._train_df, model._train_df)
