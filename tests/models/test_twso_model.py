import numpy as np
import pandas as pd
import pytest

from cybench.datasets.dataset import PandasDataset
from cybench.models.lpjml_model import bias_correct_lpj_yield
from cybench.models.twso_model import (
    TwsoBiasCorrectedModel,
    TwsoNotApplicableError,
    TWSO_COL,
    load_twso_yields,
    twso_screening_viable,
)
from cybench.config import KEY_LOC, KEY_YEAR, KEY_TARGET


def _write_twso_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_crop_calendar(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _season_setup(tmp_path, crop="maize", country="NL", locs=None):
    country_dir = tmp_path / crop / country
    country_dir.mkdir(parents=True)
    locs = locs or ["NL11", "NL12"]
    _write_crop_calendar(
        country_dir / f"crop_calendar_{crop}_{country}.csv",
        {
            KEY_LOC: locs,
            "sos": [100] * len(locs),
            "eos": [280] * len(locs),
        },
    )
    return country_dir


def test_load_twso_yields_max_per_season(tmp_path):
    country_dir = _season_setup(tmp_path)
    _write_twso_csv(
        country_dir / "twso_maize_NL.csv",
        {
            "crop_name": ["maize"] * 8,
            KEY_LOC: ["NL11"] * 4 + ["NL11"] * 4,
            "date": [20010301, 20010401, 20010501, 20010601, 20020301, 20020401, 20020501, 20020601],
            TWSO_COL: [1000.0, 3000.0, 2000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0],
        },
    )

    series = load_twso_yields("maize", "NL", data_dir=tmp_path, max_days_before_eos=999)
    assert series[("NL11", 2001)] == pytest.approx(4.0)
    assert series[("NL11", 2002)] == pytest.approx(8.0)


def test_twso_model_fit_predict(tmp_path):
    country_dir = _season_setup(tmp_path)
    _write_twso_csv(
        country_dir / "twso_maize_NL.csv",
        {
            "crop_name": ["maize"] * 15,
            KEY_LOC: ["NL11"] * 3 + ["NL11"] * 3 + ["NL12"] * 3 + ["NL12"] * 3 + ["NL11"] * 2 + ["NL12"] * 1,
            "date": [
                20010301,
                20010401,
                20010501,
                20020301,
                20020401,
                20020501,
                20010301,
                20010401,
                20010501,
                20020301,
                20020401,
                20020501,
                20030301,
                20030401,
                20030301,
            ],
            TWSO_COL: [
                4000.0,
                5000.0,
                6000.0,
                4000.0,
                5000.0,
                6000.0,
                7000.0,
                9000.0,
                7000.0,
                7000.0,
                9000.0,
                7000.0,
                4000.0,
                7000.0,
                9000.0,
            ],
        },
    )

    train_index = pd.MultiIndex.from_tuples(
        [("NL11", 2001), ("NL11", 2002), ("NL12", 2001), ("NL12", 2002)],
        names=[KEY_LOC, KEY_YEAR],
    )
    train_y = pd.DataFrame({KEY_TARGET: [10.0, 12.0, 20.0, 18.0]}, index=train_index)
    cfg = {"crop": {"name": "maize"}, "country": "NL"}
    train_ds = PandasDataset(cfg=cfg, y=train_y, x=pd.DataFrame(index=train_index))

    model = TwsoBiasCorrectedModel(
        data_dir=str(tmp_path), min_location_years=2, max_days_before_eos=999
    )
    model.fit(train_ds)

    test_index = pd.MultiIndex.from_tuples([("NL11", 2003), ("NL12", 2003)], names=[KEY_LOC, KEY_YEAR])
    test_ds = PandasDataset(
        cfg=cfg,
        y=pd.DataFrame({KEY_TARGET: [0.0, 0.0]}, index=test_index),
        x=pd.DataFrame(index=test_index),
    )
    preds, _ = model.predict(test_ds)

    assert model._twso_yields is not None
    for i, (loc, year) in enumerate([("NL11", 2003), ("NL12", 2003)]):
        cal = model._effective_calibration(model._calibration.get(str(loc)))
        expected = bias_correct_lpj_yield(float(model._twso_yields[(loc, year)]), cal)
        assert preds[i] == pytest.approx(expected)


def test_twso_global_fallback_for_tiny_std_mod(tmp_path):
    country_dir = _season_setup(
        tmp_path, crop="maize", country="US", locs=["US-01", "US-02"]
    )
    _write_twso_csv(
        country_dir / "twso_maize_US.csv",
        {
            "crop_name": ["maize"] * 30,
            KEY_LOC: (
                ["US-01"] * 5
                + ["US-02"] * 3
                + ["US-01"] * 5
                + ["US-02"] * 3
                + ["US-01"] * 5
                + ["US-02"] * 3
                + ["US-01"] * 5
                + ["US-02"] * 1
            ),
            "date": [
                20010301,
                20010401,
                20010501,
                20010601,
                20010701,
                20010301,
                20010401,
                20010501,
                20020301,
                20020401,
                20020501,
                20020601,
                20020701,
                20020301,
                20020401,
                20020501,
                20030301,
                20030401,
                20030501,
                20030601,
                20030701,
                20030301,
                20030401,
                20030501,
                20040301,
                20040401,
                20040501,
                20040601,
                20040701,
                20040301,
            ],
            TWSO_COL: [
                4000.0,
                5000.0,
                6000.0,
                7000.0,
                8000.0,
                10.0,
                20.0,
                30.0,
                4000.0,
                5000.0,
                6000.0,
                7000.0,
                8000.0,
                10.0,
                20.0,
                30.0,
                4000.0,
                5000.0,
                6000.0,
                7000.0,
                8000.0,
                10.0,
                20.0,
                30.0,
                4000.0,
                5000.0,
                6000.0,
                7000.0,
                8000.0,
                50.0,
            ],
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

    model = TwsoBiasCorrectedModel(data_dir=str(tmp_path), max_days_before_eos=999)
    model.fit(train_ds)

    assert model._global_calibration is not None
    twso_test = float(model._twso_yields[("US-02", 2004)])
    expected = bias_correct_lpj_yield(twso_test, model._global_calibration)

    test_index = pd.MultiIndex.from_tuples([("US-02", 2004)], names=[KEY_LOC, KEY_YEAR])
    test_ds = PandasDataset(
        cfg=cfg,
        y=pd.DataFrame({KEY_TARGET: [0.0]}, index=test_index),
        x=pd.DataFrame(index=test_index),
    )
    preds, _ = model.predict(test_ds)

    assert preds[0] == pytest.approx(expected)
    assert preds[0] < 25.0


def test_load_twso_yields_skips_truncated_season(tmp_path):
    """Loc-years whose last TWSO is far before EOS are returned as NaN."""
    country_dir = _season_setup(tmp_path)
    # Only early-season obs; EOS for NL test calendar is ~day 280 (October).
    _write_twso_csv(
        country_dir / "twso_maize_NL.csv",
        {
            "crop_name": ["maize"] * 4,
            KEY_LOC: ["NL11"] * 4,
            "date": [20010301, 20010401, 20010501, 20010601],
            TWSO_COL: [1000.0, 3000.0, 2000.0, 4000.0],
        },
    )

    strict = load_twso_yields("maize", "NL", data_dir=tmp_path, max_days_before_eos=14)
    assert pd.isna(strict[("NL11", 2001)])

    relaxed = load_twso_yields("maize", "NL", data_dir=tmp_path, max_days_before_eos=999)
    assert relaxed[("NL11", 2001)] == pytest.approx(4.0)


def test_twso_model_predict_nan_for_truncated_season(tmp_path):
    country_dir = _season_setup(tmp_path, locs=["NL11", "NL12"])
    # NL12 includes late-season obs (complete); NL11 only early-season (truncated).
    _write_twso_csv(
        country_dir / "twso_maize_NL.csv",
        {
            "crop_name": ["maize"] * 10,
            KEY_LOC: ["NL12"] * 5 + ["NL11"] * 5,
            "date": [
                20010301,
                20010401,
                20010501,
                20010601,
                20011005,
                20030301,
                20030401,
                20030501,
                20030601,
                20030701,
            ],
            TWSO_COL: [4000.0, 5000.0, 6000.0, 7000.0, 8000.0] * 2,
        },
    )

    train_index = pd.MultiIndex.from_tuples(
        [("NL12", 2001), ("NL12", 2002)], names=[KEY_LOC, KEY_YEAR]
    )
    train_y = pd.DataFrame({KEY_TARGET: [10.0, 12.0]}, index=train_index)
    cfg = {"crop": {"name": "maize"}, "country": "NL"}
    train_ds = PandasDataset(cfg=cfg, y=train_y, x=pd.DataFrame(index=train_index))

    model = TwsoBiasCorrectedModel(
        data_dir=str(tmp_path), min_location_years=1, max_days_before_eos=14
    )
    model.fit(train_ds)

    test_index = pd.MultiIndex.from_tuples([("NL11", 2003)], names=[KEY_LOC, KEY_YEAR])
    test_ds = PandasDataset(
        cfg=cfg,
        y=pd.DataFrame({KEY_TARGET: [0.0]}, index=test_index),
        x=pd.DataFrame(index=test_index),
    )
    preds, _ = model.predict(test_ds)
    assert np.isnan(preds[0])


def test_twso_fit_raises_not_applicable_without_overlap(tmp_path):
    country_dir = _season_setup(tmp_path, locs=["NL11"])
    _write_twso_csv(
        country_dir / "twso_maize_NL.csv",
        {
            "crop_name": ["maize"] * 3,
            KEY_LOC: ["NL99"] * 3,
            "date": [20010301, 20010401, 20010501],
            TWSO_COL: [4000.0, 5000.0, 6000.0],
        },
    )

    train_index = pd.MultiIndex.from_tuples([("NL11", 2001)], names=[KEY_LOC, KEY_YEAR])
    train_y = pd.DataFrame({KEY_TARGET: [10.0]}, index=train_index)
    cfg = {"crop": {"name": "maize"}, "country": "NL"}
    train_ds = PandasDataset(cfg=cfg, y=train_y, x=pd.DataFrame(index=train_index))

    model = TwsoBiasCorrectedModel(data_dir=str(tmp_path), max_days_before_eos=999)
    with pytest.raises(TwsoNotApplicableError, match="No overlapping TWSO"):
        model.fit(train_ds)


def test_twso_screening_viable_detects_missing_overlap(tmp_path):
    country_dir = _season_setup(tmp_path, locs=["NL11"])
    years = list(range(2000, 2012))
    yield_rows = {
        "crop_name": [],
        "country_code": [],
        KEY_LOC: [],
        "harvest_year": [],
        KEY_TARGET: [],
    }
    for year in years:
        yield_rows["crop_name"].append("maize")
        yield_rows["country_code"].append("NL")
        yield_rows[KEY_LOC].append("NL11")
        yield_rows["harvest_year"].append(year)
        yield_rows[KEY_TARGET].append(10.0)
    pd.DataFrame(yield_rows).to_csv(
        country_dir / "yield_maize_NL.csv", index=False
    )
    _write_twso_csv(
        country_dir / "twso_maize_NL.csv",
        {
            "crop_name": ["maize"] * 3,
            KEY_LOC: ["NL99"] * 3,
            "date": [20010301, 20010401, 20010501],
            TWSO_COL: [4000.0, 5000.0, 6000.0],
        },
    )

    ok, reason = twso_screening_viable("maize", "NL", data_dir=tmp_path)
    assert not ok
    assert "no overlapping" in reason.lower()
