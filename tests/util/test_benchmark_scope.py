"""Tests for benchmark evaluation scope exclusions."""

from __future__ import annotations

from cybench.util.benchmark_scope import (
    benchmark_evaluation_exclusion_reason,
    is_benchmark_evaluation_crop_country,
)


def test_is_benchmark_evaluation_crop_country_keeps_rows_when_yield_missing():
    assert is_benchmark_evaluation_crop_country("maize", "XX", data_dir="/nonexistent/path")


def test_is_benchmark_evaluation_crop_country_uses_year_rule():
    full_years = set(range(2000, 2025))
    short_years = set(range(2019, 2025))  # only 3 screening test years

    assert is_benchmark_evaluation_crop_country("maize", "DE", years=full_years)
    assert not is_benchmark_evaluation_crop_country("maize", "MW", years=short_years)
    assert not is_benchmark_evaluation_crop_country(
        "wheat", "IE", years=set(range(2020, 2025))
    )


def test_benchmark_evaluation_exclusion_reason():
    reason = benchmark_evaluation_exclusion_reason(
        "maize", "MW", years=set(range(2019, 2025))
    )
    assert reason is not None
    assert "3 screening test years" in reason
