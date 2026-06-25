import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tabpfn")

from hydra import compose, initialize
from hydra.utils import instantiate

from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR
from cybench.datasets.data_factory import DataFactory
from cybench.datasets.dataset import PandasDataset
from cybench.evaluation.eval import evaluate_predictions
from cybench.models.persistence import pickle_path
from cybench.models.tabpfn_model import (
    TabPFNModel,
    _is_cuda_oom_error,
    _predict_in_batches,
    _subsample_indices,
)
from cybench.util.config_utils import remove_keys


def test_is_cuda_oom_error_matches_common_variants():
    class OutOfMemoryError(Exception):
        pass

    assert _is_cuda_oom_error(OutOfMemoryError("CUDA out of memory"))
    assert _is_cuda_oom_error(RuntimeError("CUDA error: out of memory"))
    assert _is_cuda_oom_error(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED"))
    assert _is_cuda_oom_error(
        RuntimeError("CUDA error: no kernel image is available for execution on the device")
    )
    assert not _is_cuda_oom_error(ValueError("shape mismatch"))


def test_quantile_subsample_includes_extremes():
    rng = np.random.default_rng(0)
    y = np.linspace(1.0, 10.0, 200)
    idx = _subsample_indices(y, 20, rng, method="quantile", n_bins=10)
    assert y[idx].min() < 2.0
    assert y[idx].max() > 9.0

    bulk = np.concatenate(
        [np.full(980, 8.0), np.linspace(1.0, 2.0, 10), np.linspace(18.0, 20.0, 10)]
    )
    random_idx = _subsample_indices(bulk, 50, np.random.default_rng(0), method="random")
    assert bulk[random_idx].min() == 8.0


def test_tabpfn_sklearn_preprocess_mode():
    rng = np.random.default_rng(2)
    n = 30
    index = pd.MultiIndex.from_product(
        [["loc-a"], range(2000, 2030)],
        names=[KEY_LOC, KEY_YEAR],
    )[:n]
    X = pd.DataFrame(rng.normal(size=(n, 4)), index=index)
    y = pd.DataFrame({KEY_TARGET: X.sum(axis=1).to_numpy()}, index=index)
    dataset = PandasDataset(cfg=None, y=y, x=X)

    model = TabPFNModel(
        device="cpu",
        n_estimators=1,
        random_state=0,
        preprocess="sklearn",
    )
    assert model.preprocessor is not None
    model.fit(dataset)
    preds, _ = model.predict(dataset)
    assert preds.shape == (n,)
    assert np.isfinite(preds).all()


def test_tabpfn_pickle_path_resolution():
    assert pickle_path("models/", "tabpfn") == Path("models/tabpfn.pkl")
    assert pickle_path("models/tabpfn.pkl", "tabpfn") == Path("models/tabpfn.pkl")


def test_tabpfn_save_load_directory(tmp_path):
    rng = np.random.default_rng(0)
    n = 30
    index = pd.MultiIndex.from_product(
        [["loc-a"], range(2000, 2030)],
        names=[KEY_LOC, KEY_YEAR],
    )[:n]
    X = pd.DataFrame(rng.normal(size=(n, 4)), index=index)
    y = pd.DataFrame({KEY_TARGET: X.sum(axis=1).to_numpy()}, index=index)
    dataset = PandasDataset(cfg=None, y=y, x=X)

    model = TabPFNModel(device="cpu", n_estimators=1, random_state=0)
    model.fit(dataset)

    save_dir = tmp_path / "run"
    model.save(str(save_dir))
    assert (save_dir / "tabpfn.pkl").is_file()

    loaded = TabPFNModel.load(str(save_dir))
    preds, _ = loaded.predict(dataset)
    orig_preds, _ = model.predict(dataset)
    np.testing.assert_allclose(preds, orig_preds)


def test_tabpfn_save_load_file_path(tmp_path):
    rng = np.random.default_rng(1)
    n = 20
    index = pd.MultiIndex.from_product(
        [["loc-b"], range(2000, 2020)],
        names=[KEY_LOC, KEY_YEAR],
    )[:n]
    X = pd.DataFrame(rng.normal(size=(n, 3)), index=index)
    y = pd.DataFrame({KEY_TARGET: X.sum(axis=1).to_numpy()}, index=index)
    dataset = PandasDataset(cfg=None, y=y, x=X)

    model = TabPFNModel(device="cpu", n_estimators=1, random_state=0)
    model.fit(dataset)

    model_file = tmp_path / "custom.pkl"
    model.save(str(model_file))
    loaded = TabPFNModel.load(str(model_file))
    np.testing.assert_allclose(
        loaded.predict(dataset)[0],
        model.predict(dataset)[0],
    )


def test_tabpfn_predict_in_batches():
    class DummyModel:
        def predict(self, X):
            return np.full(len(X), len(X), dtype=float)

    X = np.zeros((10, 3))
    out = _predict_in_batches(DummyModel(), X, batch_size=4)
    assert out.shape == (10,)
    np.testing.assert_array_equal(out, [4, 4, 4, 4, 4, 4, 4, 4, 2, 2])


def test_tabpfn_fit_predict_synthetic_verbose():
    rng = np.random.default_rng(0)
    n = 40
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

    model = TabPFNModel(device="cpu", n_estimators=1, random_state=0, verbose=True)
    model.fit(train)
    preds, _ = model.predict(test)
    assert preds.shape == (len(test),)


def test_tabpfn_fit_predict_synthetic():
    rng = np.random.default_rng(0)
    n = 80
    index = pd.MultiIndex.from_product(
        [["loc-a", "loc-b"], range(2000, 2040)],
        names=[KEY_LOC, KEY_YEAR],
    )[:n]
    X = pd.DataFrame(rng.normal(size=(n, 6)), index=index)
    y = pd.DataFrame(
        {KEY_TARGET: X.sum(axis=1).to_numpy() + rng.normal(scale=0.1, size=n)},
        index=index,
    )

    train_idx = index[:60]
    test_idx = index[60:]
    train = PandasDataset(cfg=None, y=y.loc[train_idx], x=X.loc[train_idx])
    test = PandasDataset(cfg=None, y=y.loc[test_idx], x=X.loc[test_idx])

    model = TabPFNModel(device="cpu", n_estimators=2, random_state=0)
    model.fit(train)
    preds, _ = model.predict(test)
    assert preds.shape == (len(test),)
    assert np.isfinite(preds).all()


def test_tabpfn_integration_with_data_factory():
    with initialize(version_base=None, config_path="../../cybench/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset/crop=wheat",
                "dataset.country=NL",
                "dataset.framework=pandas",
                "dataset.target.filter_samples=null",
                "model=tabpfn",
                "experiment.device=cpu",
                "model.device=cpu",
                "model.n_estimators=2",
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
    assert isinstance(model, TabPFNModel)

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
