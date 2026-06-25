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
