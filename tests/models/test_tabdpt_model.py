import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tabdpt")

from hydra import compose, initialize
from hydra.utils import instantiate

from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR
from cybench.datasets.data_factory import DataFactory
from cybench.datasets.dataset import PandasDataset
from cybench.evaluation.eval import evaluate_predictions
from cybench.models.persistence import pickle_path
from cybench.models.tabular_foundation_model import TabDPTModel, _pad_train_rows
from cybench.util.config_utils import remove_keys


def _synthetic_dataset(n: int = 40, seed: int = 0) -> tuple[PandasDataset, PandasDataset]:
    rng = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product(
        [["loc-a"], range(2000, 2040)],
        names=[KEY_LOC, KEY_YEAR],
    )[:n]
    X = pd.DataFrame(rng.normal(size=(n, 4)), index=index)
    y = pd.DataFrame(
        {KEY_TARGET: X.sum(axis=1).to_numpy() + rng.normal(scale=0.1, size=n)},
        index=index,
    )
    train = PandasDataset(cfg=None, y=y.iloc[:30], x=X.iloc[:30])
    test = PandasDataset(cfg=None, y=y.iloc[30:], x=X.iloc[30:])
    return train, test


def test_pad_train_rows_reaches_minimum():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(6, dtype=float)
    X_pad, y_pad = _pad_train_rows(X, y, 10)
    assert len(X_pad) == 10
    assert len(y_pad) == 10


def test_tabdpt_pickle_path_resolution():
    assert pickle_path("models/", "tabdpt") == Path("models/tabdpt.pkl")
    assert pickle_path("models/tabdpt.pkl", "tabdpt") == Path("models/tabdpt.pkl")


def test_tabdpt_fit_predict_synthetic():
    train, test = _synthetic_dataset()
    model = TabDPTModel(device="cpu", n_ensembles=2, min_train_samples=100)
    model.fit(train)
    preds, _ = model.predict(test)
    assert preds.shape == (len(test),)
    assert np.isfinite(preds).all()
    assert model._context_size == 100


def test_tabdpt_save_load_directory(tmp_path):
    train, test = _synthetic_dataset()
    model = TabDPTModel(device="cpu", n_ensembles=2, min_train_samples=100)
    model.fit(train)
    expected, _ = model.predict(test)

    save_dir = tmp_path / "run"
    model.save(str(save_dir))
    assert (save_dir / "tabdpt.pkl").is_file()

    loaded = TabDPTModel.load(str(save_dir))
    np.testing.assert_allclose(loaded.predict(test)[0], expected)


def test_tabdpt_integration_with_data_factory():
    with initialize(version_base=None, config_path="../../cybench/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset/crop=wheat",
                "dataset.country=NL",
                "dataset.framework=pandas",
                "dataset.target.filter_samples=null",
                "model=tabdpt",
                "experiment.device=cpu",
                "model.device=cpu",
                "model.n_ensembles=2",
                "model.min_train_samples=100",
            ],
        )

    test_cfg = copy.deepcopy(cfg)
    dataset = DataFactory(test_cfg.dataset).build()

    even_years = {year for year in dataset.years if year % 2 == 0}
    odd_years = dataset.years - even_years
    train_dataset, test_dataset = dataset.split_on_years(
        years_split=(even_years, odd_years)
    )

    model_cfg = remove_keys(test_cfg.model, "_search_")
    model = instantiate(model_cfg)
    assert isinstance(model, TabDPTModel)

    model.fit(train_dataset)
    test_preds, _ = model.predict(test_dataset)
    assert test_preds.shape[0] == len(test_dataset)

    evaluation_result = evaluate_predictions(
        y_true=test_dataset.targets,
        y_pred=test_preds,
        cfg=test_cfg.evaluation,
    )
    for metric in ("normalized_rmse", "mape"):
        assert metric in evaluation_result
        assert not np.isnan(evaluation_result[metric])
