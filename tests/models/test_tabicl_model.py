import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tabicl")

from hydra import compose, initialize
from hydra.utils import instantiate

from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR
from cybench.datasets.data_factory import DataFactory
from cybench.datasets.dataset import PandasDataset
from cybench.evaluation.eval import evaluate_predictions
from cybench.models.persistence import pickle_path
from cybench.models.tabular_foundation_model import TabICLModel
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


def test_tabicl_pickle_path_resolution():
    assert pickle_path("models/", "tabicl") == Path("models/tabicl.pkl")
    assert pickle_path("models/tabicl.pkl", "tabicl") == Path("models/tabicl.pkl")


def test_tabicl_fit_predict_synthetic():
    train, test = _synthetic_dataset()
    model = TabICLModel(device="cpu", n_estimators=1, random_state=0)
    model.fit(train)
    preds, _ = model.predict(test)
    assert preds.shape == (len(test),)
    assert np.isfinite(preds).all()


def test_tabicl_save_load_directory(tmp_path):
    train, test = _synthetic_dataset()
    model = TabICLModel(device="cpu", n_estimators=1, random_state=0)
    model.fit(train)

    save_dir = tmp_path / "run"
    model.save(str(save_dir))
    assert (save_dir / "tabicl.pkl").is_file()

    loaded = TabICLModel.load(str(save_dir))
    np.testing.assert_allclose(loaded.predict(test)[0], model.predict(test)[0])


def test_tabicl_integration_with_data_factory():
    with initialize(version_base=None, config_path="../../cybench/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset/crop=wheat",
                "dataset.country=NL",
                "dataset.framework=pandas",
                "dataset.target.filter_samples=null",
                "model=tabicl",
                "experiment.device=cpu",
                "model.device=cpu",
                "model.n_estimators=1",
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
    assert isinstance(model, TabICLModel)

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
