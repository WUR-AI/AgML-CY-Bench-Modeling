import numpy as np
import pandas as pd

from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR
from cybench.evaluation.aggregated_metrics import (
    calc_nrmse,
    calc_r_r2,
    compute_report_metrics,
    get_metrics_dict,
)


def _make_df() -> pd.DataFrame:
    rows = []
    for loc in ("A", "B", "C"):
        for year in (2019, 2020, 2021):
            y = 5.0 + hash((loc, year)) % 3
            rows.append({KEY_LOC: loc, KEY_YEAR: year, KEY_TARGET: y, "model": y + 0.1})
    return pd.DataFrame(rows)


def test_calc_r_r2_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    r, r2 = calc_r_r2(y, y)
    assert r == 1.0
    assert r2 == 1.0


def test_calc_nrmse_zero_error():
    y = np.array([2.0, 4.0, 6.0])
    assert calc_nrmse(y, y) == 0.0


def test_get_metrics_dict_has_anomaly_columns():
    df = _make_df()
    out = get_metrics_dict(df, KEY_TARGET, "model")
    assert set(out) == {"r", "r2", "nrmse", "r_res", "r2_res"}


def test_compute_report_metrics_views():
    df = _make_df()
    out = compute_report_metrics(df, KEY_TARGET, "model")
    assert out["n_regions"] == 3
    assert out["n_years"] == 3
    assert out["n_samples"] == 9
    assert "region_year" in out
    assert "spatial" in out
    assert "temporal" in out


def test_compute_report_metrics_ignores_nan_predictions():
    df = _make_df()
    df.loc[0, "model"] = np.nan
    out = compute_report_metrics(df, KEY_TARGET, "model")
    assert out["n_samples"] == 8
    assert np.isfinite(out["region_year"]["r"])
