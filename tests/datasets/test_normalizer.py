"""Tests for feature normalization."""

import pandas as pd

from cybench.config import KEY_LOC, KEY_YEAR
from cybench.datasets.normalizer import Normalizer
from omegaconf import OmegaConf


def _standard_normalizer() -> Normalizer:
    cfg = OmegaConf.create(
        {
            "name": "fit",
            "features": {
                "tavg": {"type": "standard", "params": None},
            },
        }
    )
    return Normalizer(cfg)


def test_fit_normalize_respects_fit_years():
    """Stats must come from fit_years only; all rows are still transformed."""
    index = pd.MultiIndex.from_product(
        [["r1"], [2010, 2011, 2012], pd.date_range("2010-01-01", periods=2)],
        names=[KEY_LOC, KEY_YEAR, "date"],
    )
    df = pd.DataFrame(
        {
            "tavg": [10.0, 10.0, 20.0, 20.0, 30.0, 30.0],
        },
        index=index,
    )
    dfs = {"meteo": df}
    normalizer = _standard_normalizer()
    normalizer.fit_normalize(dfs, fit_years=[2010, 2011])

    params = normalizer.feature_cfg["tavg"]["params"]
    assert params["mean"] == 15.0
    assert params["std"] > 0

    # 2012 rows normalized with train+val stats (not their own mean=30)
    z_2012 = df.loc[(slice(None), 2012, slice(None)), "tavg"].iloc[0]
    assert z_2012 > 1.0


def test_fit_normalize_single_sample_standard_does_not_nan():
    """Fallback when n<2: std=0 so apply yields zeros, not NaN."""
    normalizer = Normalizer(
        OmegaConf.create(
            {
                "name": "fit",
                "features": {"awc": {"type": "standard", "params": None}},
            }
        )
    )
    aligned = pd.DataFrame({"awc": [13.0]}, index=pd.Index(["r1"], name=KEY_LOC))
    dfs = {"non_temporal": aligned}
    normalizer.fit_normalize(dfs)
    assert not dfs["non_temporal"]["awc"].isna().any()
    assert dfs["non_temporal"]["awc"].iloc[0] == 0.0


def test_fit_normalize_uses_full_non_temporal_when_aligned_degenerate():
    """Fallback: widen to all regions only when the aligned slice cannot be fit."""
    full = pd.DataFrame(
        {"awc": [10.0, 20.0, 30.0]},
        index=pd.Index(["r1", "r2", "r3"], name=KEY_LOC),
    )
    aligned = full.loc[["r1"]]
    dfs = {"non_temporal": aligned.copy()}
    normalizer = Normalizer(
        OmegaConf.create(
            {
                "name": "fit",
                "features": {"awc": {"type": "standard", "params": None}},
            }
        )
    )
    normalizer.fit_normalize(dfs, non_temporal_fit_df=full)
    params = normalizer.feature_cfg["awc"]["params"]
    assert params["mean"] == 20.0
    assert params["std"] > 0
    assert not dfs["non_temporal"]["awc"].isna().any()


def test_fit_normalize_prefers_aligned_non_temporal_when_sufficient():
    """No widening when yield-aligned regions already give stable stats."""
    full = pd.DataFrame(
        {"awc": [10.0, 20.0, 30.0]},
        index=pd.Index(["r1", "r2", "r3"], name=KEY_LOC),
    )
    aligned = full.loc[["r1", "r2"]]
    dfs = {"non_temporal": aligned.copy()}
    normalizer = Normalizer(
        OmegaConf.create(
            {
                "name": "fit",
                "features": {"awc": {"type": "standard", "params": None}},
            }
        )
    )
    normalizer.fit_normalize(dfs, non_temporal_fit_df=full)
    params = normalizer.feature_cfg["awc"]["params"]
    assert params["mean"] == 15.0
    assert params["std"] > 0
