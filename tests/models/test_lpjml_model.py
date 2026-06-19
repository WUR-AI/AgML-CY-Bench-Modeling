import pandas as pd
import pytest

from cybench.datasets.dataset import PandasDataset
from cybench.models.lpjml_model import (
    LpjmlBiasCorrectedModel,
    LPJML_COL_IRRIGATED,
    LPJML_COL_RAINFED,
    bias_correct_lpj_yield,
    load_lpjml_yields,
    _LocationCalibration,
)
from cybench.config import KEY_LOC, KEY_YEAR, KEY_TARGET


def _write_lpjml_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_bias_correct_lpj_yield_rescales():
    cal = _LocationCalibration(
        mean_obs=10.0, std_obs=2.0, mean_mod=5.0, std_mod=1.0, n_years=5
    )
    assert bias_correct_lpj_yield(6.0, cal) == pytest.approx(12.0)


def test_lpjml_model_fit_predict(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    country_dir = data_dir / "maize" / "NL"
    country_dir.mkdir(parents=True)

    _write_lpjml_csv(
        country_dir / "lpjml_maize_NL.csv",
        {
            "crop_name": ["maize"] * 6,
            "adm_id": ["NL11", "NL11", "NL11", "NL12", "NL12", "NL12"],
            "date": [20010101, 20020101, 20030101, 20010101, 20020101, 20030101],
            LPJML_COL_RAINFED: [4.0, 5.0, 6.0, 7.0, 9.0, 7.0],
            LPJML_COL_IRRIGATED: [4.1, 5.1, 6.1, 7.1, 9.1, 7.1],
        },
    )

    train_index = pd.MultiIndex.from_tuples(
        [("NL11", 2001), ("NL11", 2002), ("NL12", 2001), ("NL12", 2002)],
        names=[KEY_LOC, KEY_YEAR],
    )
    train_y = pd.DataFrame({KEY_TARGET: [10.0, 12.0, 20.0, 18.0]}, index=train_index)
    cfg = {"crop": {"name": "maize"}, "country": "NL"}
    train_ds = PandasDataset(cfg=cfg, y=train_y, x=pd.DataFrame(index=train_index))

    model = LpjmlBiasCorrectedModel(
        data_dir=str(data_dir), variant="rainfed", min_location_years=2
    )
    model.fit(train_ds)

    test_index = pd.MultiIndex.from_tuples([("NL11", 2003), ("NL12", 2003)], names=[KEY_LOC, KEY_YEAR])
    test_ds = PandasDataset(
        cfg=cfg,
        y=pd.DataFrame({KEY_TARGET: [0.0, 0.0]}, index=test_index),
        x=pd.DataFrame(index=test_index),
    )
    preds, _ = model.predict(test_ds)

    assert preds[0] == pytest.approx(14.0)
    assert preds[1] == pytest.approx(18.0)


def test_load_lpjml_yields_two_columns(tmp_path):
    country_dir = tmp_path / "wheat" / "DE"
    country_dir.mkdir(parents=True)
    _write_lpjml_csv(
        country_dir / "lpjml_wheat_DE.csv",
        {
            "crop_name": ["wheat", "wheat"],
            "adm_id": ["DE111", "DE111"],
            "date": [20010101, 20020101],
            LPJML_COL_RAINFED: [5.5, pd.NA],
            LPJML_COL_IRRIGATED: [5.0, 6.0],
        },
    )

    assert load_lpjml_yields("wheat", "DE", data_dir=tmp_path, variant="rainfed")[
        ("DE111", 2001)
    ] == pytest.approx(5.5)
    assert load_lpjml_yields("wheat", "DE", data_dir=tmp_path, variant="irrigated")[
        ("DE111", 2002)
    ] == pytest.approx(6.0)


def test_lpjml_global_fallback_for_tiny_std_mod(tmp_path):
    data_dir = tmp_path / "data"
    country_dir = data_dir / "maize" / "US"
    country_dir.mkdir(parents=True)

    _write_lpjml_csv(
        country_dir / "lpjml_maize_US.csv",
        {
            "crop_name": ["maize"] * 9,
            "adm_id": ["US-01"] * 5 + ["US-02"] * 4,
            "date": [
                20010101,
                20020101,
                20030101,
                20040101,
                20050101,
                20010101,
                20020101,
                20030101,
                20040101,
            ],
            LPJML_COL_RAINFED: [4.0, 5.0, 6.0, 7.0, 8.0, 0.01, 0.02, 0.03, 0.05],
            LPJML_COL_IRRIGATED: [4.0, 5.0, 6.0, 7.0, 8.0, 0.01, 0.02, 0.03, 0.05],
        },
    )

    train_index = pd.MultiIndex.from_tuples(
        [
            ("US-01", 2001),
            ("US-01", 2002),
            ("US-01", 2003),
            ("US-01", 2004),
            ("US-01", 2005),
            ("US-02", 2001),
            ("US-02", 2002),
            ("US-02", 2003),
        ],
        names=[KEY_LOC, KEY_YEAR],
    )
    train_y = pd.DataFrame(
        {KEY_TARGET: [10.0, 12.0, 14.0, 16.0, 18.0, 8.0, 9.0, 10.0]},
        index=train_index,
    )
    cfg = {"crop": {"name": "maize"}, "country": "US"}
    train_ds = PandasDataset(cfg=cfg, y=train_y, x=pd.DataFrame(index=train_index))

    model = LpjmlBiasCorrectedModel(data_dir=str(data_dir), variant="rainfed")
    model.fit(train_ds)

    global_obs = train_y[KEY_TARGET]
    global_mod = pd.Series([4.0, 5.0, 6.0, 7.0, 8.0, 0.01, 0.02, 0.03])
    mean_obs = float(global_obs.mean())
    std_obs = float(global_obs.std(ddof=0))
    mean_mod = float(global_mod.mean())
    std_mod = float(global_mod.std(ddof=0))
    lpj_test = 0.05
    expected = mean_obs + (std_obs / std_mod) * (lpj_test - mean_mod)

    test_index = pd.MultiIndex.from_tuples([("US-02", 2004)], names=[KEY_LOC, KEY_YEAR])
    test_ds = PandasDataset(
        cfg=cfg,
        y=pd.DataFrame({KEY_TARGET: [0.0]}, index=test_index),
        x=pd.DataFrame(index=test_index),
    )
    preds, _ = model.predict(test_ds)

    assert preds[0] == pytest.approx(expected)
    assert preds[0] < 25.0
