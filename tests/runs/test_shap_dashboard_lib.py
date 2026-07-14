"""Tests for SHAP dashboard payload helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from cybench.runs.analysis.shap_dashboard_lib import (
    build_shap_dashboard_payload,
    resolve_shap_input_dir,
)


def _write_summary(
    path: Path,
    *,
    crop: str,
    country: str,
    model: str,
    horizon: str = "eos",
    features: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        OmegaConf.create(
            {
                "crop": crop,
                "country": country,
                "model": model,
                "horizon": horizon,
                "n_origins": 1,
                "origins": [
                    {
                        "test_years": [2020],
                        "explainer": "TreeExplainer",
                        "features": features,
                    }
                ],
            }
        ),
        path,
    )


def test_resolve_shap_input_dir_auto(tmp_path: Path):
    summary_rows = [
        {"crop": "maize", "country": "NL", "horizon": "eos", "model": "random_forest", "dataset": "maize_nl"}
    ]
    shap_root = tmp_path / "shap_importance" / "maize_NL_eos"
    shap_root.mkdir(parents=True)
    resolved = resolve_shap_input_dir(
        shap_dir=None,
        output_root=tmp_path,
        summary_rows=summary_rows,
    )
    assert resolved == shap_root


def test_build_shap_dashboard_payload(tmp_path: Path):
    shap_dir = tmp_path / "shap_importance" / "maize_NL_eos"
    _write_summary(
        shap_dir / "maize_NL" / "random_forest" / "shap_summary.yaml",
        crop="maize",
        country="NL",
        model="random_forest",
        features=[
            {"name": "prec_sum_8", "mean_abs_shap": 0.5, "rank": 1},
            {"name": "gdd_sum_4", "mean_abs_shap": 0.2, "rank": 2},
        ],
    )
    summary_rows = [
        {
            "crop": "maize",
            "country": "NL",
            "horizon": "eos",
            "model": "random_forest",
            "dataset": "maize_nl",
        }
    ]
    payload = build_shap_dashboard_payload(shap_dir, summary_rows)
    assert payload["available"] is True
    entry = payload["by_key"]["maize_nl|||random_forest"]
    assert entry["explainer"] == "TreeExplainer"
    assert entry["top_features"][0]["name"] == "prec_sum_8"
    assert entry["meta_groups"]


def test_build_shap_dashboard_payload_missing_returns_empty(tmp_path: Path):
    payload = build_shap_dashboard_payload(
        tmp_path / "missing",
        [{"crop": "maize", "country": "NL", "horizon": "eos", "model": "x", "dataset": "maize_nl"}],
    )
    assert payload["available"] is False
    assert payload["by_key"] == {}
