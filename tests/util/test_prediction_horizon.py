from cybench.util.prediction_horizon import parse_run_name_suffix, prediction_horizon_tag
import pytest


def test_prediction_horizon_tag_known_values():
    assert prediction_horizon_tag("eos") == "eos"
    assert prediction_horizon_tag("middle-of-season") == "mid_season"
    assert prediction_horizon_tag("mid-season") == "mid_season"
    assert prediction_horizon_tag("eos-60") == "eos_60"


def test_parse_run_name_suffix():
    assert parse_run_name_suffix("eos_20260615_120738") == ("eos", "20260615_120738")
    assert parse_run_name_suffix("mid_season_20260615_120738") == (
        "mid_season",
        "20260615_120738",
    )


def test_parse_run_name_suffix_rejects_timestamp_only():
    with pytest.raises(ValueError, match="Invalid run suffix"):
        parse_run_name_suffix("20260615_120738")
