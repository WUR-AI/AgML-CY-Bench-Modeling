from pathlib import Path

import pandas as pd

from cybench.runs.analysis.export_wide_predictions import (
    CollectBundle,
    build_wide_table,
    discover_collect_bundles,
    export_bundle,
    normalize_prediction_frame,
)


def _write_year_csv(
    preds_dir: Path,
    *,
    dataset: str,
    year: int,
    rows: list[tuple[str, float, float]],
    model_col: str,
) -> None:
    preds_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "adm_id": [r[0] for r in rows],
            "year": [year] * len(rows),
            "yield": [r[1] for r in rows],
            model_col: [r[2] for r in rows],
        }
    )
    df.to_csv(preds_dir / f"{dataset}_h10_year_{year}.csv", index=False)


def test_normalize_prediction_frame():
    raw = pd.DataFrame(
        {
            "adm_id": ["NL-01"],
            "year": [2019],
            "yield": [10.0],
            "Ridge": [9.5],
        }
    )
    out = normalize_prediction_frame(
        raw, model_slug="ridge", model_col="Ridge", dataset="maize_NL"
    )
    assert list(out.columns) == ["adm_id", "year", "crop", "yield", "ridge"]
    assert out.iloc[0]["crop"] == "maize"
    assert out.iloc[0]["ridge"] == 9.5


def test_build_wide_table_from_collect_preds(tmp_path: Path):
    collect_dir = tmp_path / "paper_walk_forward_nl_eos_v2"
    collect_dir.mkdir()
    preds_root = collect_dir / "preds"

    _write_year_csv(
        preds_root / "ridge_eos",
        dataset="maize_NL",
        year=2019,
        rows=[("NL-01", 10.0, 9.5), ("NL-02", 11.0, 10.5)],
        model_col="Ridge",
    )
    _write_year_csv(
        preds_root / "average_eos",
        dataset="maize_NL",
        year=2019,
        rows=[("NL-01", 10.0, 8.0), ("NL-02", 11.0, 9.0)],
        model_col="AverageYieldModel",
    )

    summary_rows = [
        {
            "dataset": "maize_NL",
            "model": "ridge",
            "horizon": "eos",
            "model_col": "Ridge",
            "plot_seed": 42,
        },
        {
            "dataset": "maize_NL",
            "model": "average",
            "horizon": "eos",
            "model_col": "AverageYieldModel",
            "plot_seed": 42,
        },
    ]

    wide, meta = build_wide_table(summary_rows, collect_dir)
    assert len(wide) == 2
    assert set(wide.columns) == {"adm_id", "year", "crop", "yield", "ridge", "average"}
    assert wide.loc[wide["adm_id"] == "NL-01", "ridge"].iloc[0] == 9.5
    assert wide.loc[wide["adm_id"] == "NL-01", "average"].iloc[0] == 8.0
    assert meta["models"] == ["ridge", "average"]
    assert meta["seed"] == [42]


def test_export_bundle_writes_country_csv(tmp_path: Path):
    collect_dir = tmp_path / "paper_walk_forward_de_eos_v2"
    collect_dir.mkdir()
    preds_root = collect_dir / "preds"
    _write_year_csv(
        preds_root / "ridge_eos",
        dataset="maize_DE",
        year=2020,
        rows=[("DE-01", 8.0, 7.5)],
        model_col="Ridge",
    )
    pd.DataFrame(
        [
            {
                "crop": "maize",
                "country": "DE",
                "model": "ridge",
                "horizon": "eos",
                "model_col": "Ridge",
                "dataset": "maize_DE",
                "plot_seed": 42,
                "run_dir": "/tmp/unused",
            }
        ]
    ).to_csv(collect_dir / "walk_forward_summary.csv", index=False)

    bundle = CollectBundle(country="DE", horizon="eos", version=2, path=collect_dir)
    dest = tmp_path / "export"
    records = export_bundle(bundle, dest, split="country", source="collect", seed=None, crops=None)
    assert len(records) == 1
    out = dest / "de_eos_v2_preds.csv"
    assert out.is_file()
    df = pd.read_csv(out)
    assert list(df.columns) == ["adm_id", "year", "crop", "yield", "ridge"]
    assert df.iloc[0]["adm_id"] == "DE-01"


def test_discover_collect_bundles_filters(tmp_path: Path):
    for name in (
        "paper_walk_forward_de_eos_v2",
        "paper_walk_forward_de_mid_v2",
        "paper_walk_forward_fr_eos_v1",
    ):
        path = tmp_path / name
        path.mkdir()
        (path / "walk_forward_summary.csv").write_text("model\nridge\n", encoding="utf-8")

    found = discover_collect_bundles(
        tmp_path, version=2, countries={"DE"}, horizons={"eos"}
    )
    assert len(found) == 1
    assert found[0].country == "DE"
    assert found[0].horizon == "eos"
