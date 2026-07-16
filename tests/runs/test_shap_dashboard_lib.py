"""Tests for SHAP dashboard payload helpers."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from cybench.runs.analysis.shap_dashboard_lib import (
    build_shap_dashboard_payload,
    resolve_shap_input_dir,
    resolve_shap_input_dirs,
)


def _write_summary(
    path: Path,
    *,
    crop: str,
    country: str,
    model: str,
    horizon: str = "eos",
    features: list[dict[str, object]],
    explainer: str = "TreeSHAP",
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
                        "explainer": explainer,
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


def test_resolve_shap_input_dirs_all_crops(tmp_path: Path):
    """Multi-crop countries must discover every crop batch, not only the first row."""
    summary_rows = [
        {
            "crop": "maize",
            "country": "AR",
            "horizon": "eos",
            "model": "transformer_lf",
            "dataset": "maize_AR",
        },
        {
            "crop": "wheat",
            "country": "AR",
            "horizon": "eos",
            "model": "transformer_lf",
            "dataset": "wheat_AR",
        },
    ]
    maize = tmp_path / "shap_importance" / "maize_AR_eos"
    wheat = tmp_path / "shap_importance" / "wheat_AR_eos"
    maize.mkdir(parents=True)
    wheat.mkdir(parents=True)
    resolved = resolve_shap_input_dirs(
        shap_dir=None,
        output_root=tmp_path,
        summary_rows=summary_rows,
    )
    assert resolved == [maize, wheat]


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
    assert entry["explainer"] == "TreeSHAP"
    assert entry["top_features"][0]["name"] == "prec_sum_8"
    assert entry["meta_groups"]


def test_build_shap_dashboard_payload_multi_crop(tmp_path: Path):
    maize_dir = tmp_path / "shap_importance" / "maize_AR_eos"
    wheat_dir = tmp_path / "shap_importance" / "wheat_AR_eos"
    _write_summary(
        maize_dir / "maize_AR" / "transformer_lf" / "shap_summary.yaml",
        crop="maize",
        country="AR",
        model="transformer_lf",
        explainer="GradientSHAP",
        features=[{"name": "ctx:awc", "mean_abs_shap": 0.4, "rank": 1}],
    )
    _write_summary(
        wheat_dir / "wheat_AR" / "transformer_lf" / "shap_summary.yaml",
        crop="wheat",
        country="AR",
        model="transformer_lf",
        explainer="GradientSHAP",
        features=[{"name": "ctx:bulk_density", "mean_abs_shap": 0.3, "rank": 1}],
    )
    summary_rows = [
        {
            "crop": "maize",
            "country": "AR",
            "horizon": "eos",
            "model": "transformer_lf",
            "dataset": "maize_AR",
        },
        {
            "crop": "wheat",
            "country": "AR",
            "horizon": "eos",
            "model": "transformer_lf",
            "dataset": "wheat_AR",
        },
    ]
    payload = build_shap_dashboard_payload([maize_dir, wheat_dir], summary_rows)
    assert payload["available"] is True
    assert set(payload["by_key"]) == {
        "maize_AR|||transformer_lf",
        "wheat_AR|||transformer_lf",
    }
    assert payload["by_key"]["wheat_AR|||transformer_lf"]["top_features"][0]["name"] == (
        "ctx:bulk_density"
    )


def test_build_shap_dashboard_payload_missing_returns_empty(tmp_path: Path):
    payload = build_shap_dashboard_payload(
        tmp_path / "missing",
        [{"crop": "maize", "country": "NL", "horizon": "eos", "model": "x", "dataset": "maize_nl"}],
    )
    assert payload["available"] is False
    assert payload["by_key"] == {}
