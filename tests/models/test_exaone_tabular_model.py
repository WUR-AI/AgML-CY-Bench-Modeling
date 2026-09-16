import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("exaonetabular")

from hydra import compose, initialize
from hydra.utils import instantiate

from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR
from cybench.datasets.data_factory import DataFactory
from cybench.datasets.dataset import PandasDataset
from cybench.evaluation.eval import evaluate_predictions
from cybench.models.persistence import pickle_path
from cybench.models.tabular_foundation_model import EXAONETabularModel
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


def _maybe_skip_pretrained_failure(exc: BaseException) -> None:
    msg = str(exc).lower()
    markers = (
        "401",
        "403",
        "gated",
        "huggingface",
        "hf_token",
        "offline",
        "connection",
        "timed out",
        "timeout",
        "safetensors",
        "checkpoint",
        "not found",
    )
    if any(marker in msg for marker in markers):
        pytest.skip(f"EXAONE Tabular weights unavailable: {exc}")
    raise exc


def test_exaone_tabular_pickle_path_resolution():
    assert pickle_path("models/", "exaone_tabular") == Path("models/exaone_tabular.pkl")
    assert pickle_path("models/exaone_tabular.pkl", "exaone_tabular") == Path(
        "models/exaone_tabular.pkl"
    )


def test_exaone_tabular_fit_predict_synthetic():
    train, test = _synthetic_dataset()
    model = EXAONETabularModel(
        device="cpu",
        ensemble_count=1,
        random_state=0,
        compute_dtype="float32",
    )
    try:
        model.fit(train)
        preds, _ = model.predict(test)
    except Exception as exc:
        _maybe_skip_pretrained_failure(exc)
    assert preds.shape == (len(test),)
    assert np.isfinite(preds).all()


def test_exaone_tabular_save_load_directory(tmp_path):
    train, test = _synthetic_dataset()
    model = EXAONETabularModel(
        device="cpu",
        ensemble_count=1,
        random_state=0,
        compute_dtype="float32",
    )
    try:
        model.fit(train)
        expected, _ = model.predict(test)
        save_dir = tmp_path / "run"
        model.save(str(save_dir))
        loaded = EXAONETabularModel.load(str(save_dir))
        got, _ = loaded.predict(test)
    except Exception as exc:
        _maybe_skip_pretrained_failure(exc)
    assert (save_dir / "exaone_tabular.pkl").is_file()
    np.testing.assert_allclose(got, expected)


def test_exaone_tabular_rejects_nonfinite_targets():
    train, _ = _synthetic_dataset()
    train.y.iloc[0, 0] = np.nan
    model = EXAONETabularModel(device="cpu", ensemble_count=1, random_state=0)
    with pytest.raises(ValueError, match="finite targets"):
        model.fit(train)


def test_exaone_tabular_integration_with_data_factory():
    with initialize(version_base=None, config_path="../../cybench/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset/crop=wheat",
                "dataset.country=NL",
                "dataset.framework=pandas",
                "dataset.target.filter_samples=null",
                "model=exaone_tabular",
                "experiment.device=cpu",
                "model.device=cpu",
                "model.ensemble_count=1",
                "model.compute_dtype=float32",
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
    assert isinstance(model, EXAONETabularModel)

    try:
        model.fit(train_dataset)
        test_preds, _ = model.predict(test_dataset)
    except Exception as exc:
        _maybe_skip_pretrained_failure(exc)
    assert test_preds.shape[0] == len(test_dataset)

    evaluation_result = evaluate_predictions(
        y_true=test_dataset.targets,
        y_pred=test_preds,
        cfg=test_cfg.evaluation,
    )
    for metric in ("normalized_rmse", "mape"):
        assert metric in evaluation_result
        assert not np.isnan(evaluation_result[metric])
