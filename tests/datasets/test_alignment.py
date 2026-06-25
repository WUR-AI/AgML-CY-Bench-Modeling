import pandas as pd

from cybench.config import KEY_LOC, KEY_YEAR
from cybench.datasets.alignment import interpolate_time_series_data, make_aligned_tensors


def test_interpolate_time_series_drops_aggregation_meta_columns():
    idx = pd.MultiIndex.from_tuples(
        [
            ("r1", 2020, pd.Timestamp("2020-06-01")),
            ("r1", 2020, pd.Timestamp("2020-06-08")),
        ],
        names=[KEY_LOC, KEY_YEAR, "date"],
    )
    meteo = pd.DataFrame(
        {
            "tmin_min": [1.0, 2.0],
            "end_of_sequence_date": [pd.Timestamp("2020-08-01")] * 2,
        },
        index=idx,
    )
    fpar = pd.DataFrame(
        {
            "fpar_mean": [0.5, 0.6],
            "end_of_sequence_date": [pd.Timestamp("2020-08-01")] * 2,
        },
        index=idx,
    )

    df_ts = interpolate_time_series_data({"meteo": meteo, "fpar": fpar})

    assert "end_of_sequence_date" not in df_ts.columns
    assert list(df_ts.columns) == ["tmin_min", "fpar_mean"]


def test_make_aligned_tensors_stacks_weekly_numeric_features():
    loc, year = "r1", 2020
    dates = pd.date_range("2020-06-01", periods=3, freq="7D")
    idx = pd.MultiIndex.from_tuples(
        [(loc, year, d) for d in dates],
        names=[KEY_LOC, KEY_YEAR, "date"],
    )
    df_ts = pd.DataFrame(
        {"tmin_min": [1.0, 2.0, 3.0], "fpar_mean": [0.1, 0.2, 0.3]},
        index=idx,
    )
    df_y = pd.DataFrame(
        {"yield": [5.0]},
        index=pd.MultiIndex.from_tuples([(loc, year)], names=[KEY_LOC, KEY_YEAR]),
    )
    df_non_temporal = pd.DataFrame({"feat": [1.0]}, index=pd.Index([loc], name=KEY_LOC))

    (y, x_context, x_ts), column_names, doy_ts = make_aligned_tensors(
        df_y, df_non_temporal, df_ts
    )

    assert y.shape == (1, 1)
    assert x_context.shape == (1, 2)
    assert x_ts.shape == (1, 3, 2)
    assert doy_ts.shape == (1, 3)
    assert column_names[2] == ["tmin_min", "fpar_mean"]
    assert not x_ts.isnan().any()


def test_make_aligned_tensors_harmonizes_unequal_weekly_lengths():
    dates_short = pd.date_range("2020-06-01", periods=2, freq="7D")
    dates_long = pd.date_range("2020-05-01", periods=4, freq="7D")
    idx_short = pd.MultiIndex.from_tuples(
        [("r1", 2020, d) for d in dates_short],
        names=[KEY_LOC, KEY_YEAR, "date"],
    )
    idx_long = pd.MultiIndex.from_tuples(
        [("r2", 2020, d) for d in dates_long],
        names=[KEY_LOC, KEY_YEAR, "date"],
    )
    df_ts = pd.concat(
        [
            pd.DataFrame({"tmin_min": [1.0, 2.0], "fpar_mean": [0.1, 0.2]}, index=idx_short),
            pd.DataFrame(
                {"tmin_min": [3.0, 4.0, 5.0, 6.0], "fpar_mean": [0.3, 0.4, 0.5, 0.6]},
                index=idx_long,
            ),
        ]
    )
    df_y = pd.DataFrame(
        {"yield": [5.0, 6.0]},
        index=pd.MultiIndex.from_tuples(
            [("r1", 2020), ("r2", 2020)],
            names=[KEY_LOC, KEY_YEAR],
        ),
    )
    df_non_temporal = pd.DataFrame(
        {"feat": [1.0, 2.0]},
        index=pd.Index(["r1", "r2"], name=KEY_LOC),
    )

    (_, _, x_ts), _, doy_ts = make_aligned_tensors(df_y, df_non_temporal, df_ts)

    assert x_ts.shape == (2, 2, 2)
    assert doy_ts.shape == (2, 2)
