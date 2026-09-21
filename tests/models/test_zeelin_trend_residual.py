from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize
from hydra.utils import instantiate

from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR
from cybench.datasets.dataset import PandasDataset
from cybench.models.persistence import pickle_path
from cybench.models.zeelin_trend_residual import ZeeLinTrendResidualEnsemble
from cybench.util.config_utils import remove_keys


def _dataset_from_rows(rows: list[list[object]], n_features: int = 3, seed: int = 0) -> PandasDataset:
    yield_df = pd.DataFrame(rows, columns=pd.Index([KEY_LOC, KEY_YEAR, KEY_TARGET])).set_index(
        [KEY_LOC, KEY_YEAR]
    )
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        rng.normal(size=(len(yield_df), n_features)),
        index=yield_df.index,
        columns=[f"f{i}" for i in range(n_features)],
    )
    return PandasDataset(cfg=None, y=yield_df, x=x)


def _smooth_loyo_split(test_year: int = 2005) -> tuple[PandasDataset, PandasDataset]:
    rows = []
    for loc, intercept in (("CN-01", 4.0), ("CN-02", 6.0)):
        for year in range(2000, 2011):
            rows.append([loc, year, intercept + 0.2 * (year - 2000)])
    full = _dataset_from_rows(rows)
    train_years = [year for year in sorted(full.years) if year != test_year]
    return full.split_on_years((train_years, [test_year]))


def _tiny_model(**kwargs) -> ZeeLinTrendResidualEnsemble:
    params = {
        "n_estimators": 8,
        "n_jobs": 1,
        "random_state": 0,
    }
    params.update(kwargs)
    return ZeeLinTrendResidualEnsemble(**params)


def test_nearest_decade_uses_both_sides_and_excludes_test_year():
    train_years = np.array([y for y in range(2003, 2023) if y != 2012])
    selected = ZeeLinTrendResidualEnsemble._select_decade_years(train_years, 2012, 10)
    assert 2012 not in selected
    assert set(selected) == {2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016, 2017}


def test_nearest_decade_at_series_start_uses_later_years():
    train_years = np.array(range(2004, 2023))
    selected = ZeeLinTrendResidualEnsemble._select_decade_years(train_years, 2003, 10)
    assert 2003 not in selected
    assert set(selected) == set(range(2004, 2014))


def test_local_decade_interpolates_smooth_interior_year():
    train, test = _smooth_loyo_split(2005)
    model = _tiny_model(trend_blend=1.0)
    model.fit(train)
    preds, info = model.predict(test)
    np.testing.assert_allclose(preds, test.targets, atol=1e-6)
    np.testing.assert_allclose(info["local_decade_trend"], test.targets, atol=1e-6)


def test_held_out_targets_are_not_used():
    train, test = _smooth_loyo_split(2005)
    model = _tiny_model()
    model.fit(train)
    expected, _ = model.predict(test)

    poisoned = PandasDataset(
        cfg=None,
        y=test.y.copy() + 100.0,
        x=test.x.copy(),
    )
    got, _ = model.predict(poisoned)
    np.testing.assert_allclose(got, expected)


def test_training_location_mean_is_leave_one_sample_out():
    train, _ = _smooth_loyo_split(2005)
    model = _tiny_model()
    frame = model._frame(train)
    matrix = model._design_matrix(frame, training=True)
    loc = frame[KEY_LOC].iloc[0]
    year = frame[KEY_YEAR].iloc[0]
    expected = frame.loc[
        (frame[KEY_LOC] == loc) & (frame[KEY_YEAR] != year), KEY_TARGET
    ].mean()
    assert matrix["loc_y_mean"].iloc[0] == expected
    assert not np.isclose(matrix["loc_y_mean"].iloc[0], frame[KEY_TARGET].iloc[0])


def test_fit_predict_synthetic_and_info_keys():
    train, test = _smooth_loyo_split(2005)
    model = _tiny_model()
    model.fit(train)
    preds, info = model.predict(test)
    assert preds.shape == (len(test),)
    assert np.isfinite(preds).all()
    assert set(info) >= {
        "official_trend",
        "local_decade_trend",
        "ml_prediction",
        "residual_components",
    }
    assert set(info["residual_components"]) == {"ridge", "extra", "rf", "xgb"}


def test_save_load_directory(tmp_path):
    train, test = _smooth_loyo_split(2005)
    model = _tiny_model()
    model.fit(train)
    expected, _ = model.predict(test)

    save_dir = tmp_path / "run"
    model.save(str(save_dir))
    assert pickle_path(str(save_dir), "zeelin_trend_residual") == Path(
        save_dir / "zeelin_trend_residual.pkl"
    )
    loaded = ZeeLinTrendResidualEnsemble.load(str(save_dir))
    np.testing.assert_allclose(loaded.predict(test)[0], expected)


def test_hydra_config_composes_and_instantiates():
    with initialize(version_base=None, config_path="../../cybench/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=zeelin_trend_residual",
                "model.n_estimators=8",
                "model.n_jobs=1",
            ],
        )
    assert cfg.model.name == "zeelin_trend_residual"
    assert cfg.model.framework == "pandas"
    assert cfg.model.trend_blend == 0.70
    model = instantiate(remove_keys(cfg.model, "_search_"))
    assert isinstance(model, ZeeLinTrendResidualEnsemble)
    assert model.extra_trees_n_estimators == 8
    assert model.rf_n_estimators == 8
    assert model.xgb_n_estimators == 8
