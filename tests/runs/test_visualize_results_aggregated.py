"""Tests for aggregated evaluation plot export."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from cybench.runs.viz.visualize_results_aggregated import (
    process_dataset,
    save_panel_images,
)


def _synthetic_maize_df(*, n_regions: int = 200, n_years: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for year in range(2000, 2000 + n_years):
        for rid in range(n_regions):
            y = 5 + rng.uniform(-2, 5)
            rows.append(
                {
                    "adm_id": f"R{rid}",
                    "year": year,
                    "yield": y,
                    "Ridge": y + rng.normal(0, 1.2),
                }
            )
    return pd.DataFrame(rows)


def _band_has_ink(img: Image.Image, *, row_slice: slice | None = None, col_slice: slice | None = None) -> bool:
    arr = np.asarray(img.convert("RGB"))
    if row_slice is not None:
        arr = arr[row_slice, :, :]
    if col_slice is not None:
        arr = arr[:, col_slice, :]
    return bool((arr < 200).any())


def test_save_panel_images_scatter_includes_labels(tmp_path):
    """Regression: scatter PNG must include title and axis labels (not hexbin-only crop)."""
    df = _synthetic_maize_df()
    fig, _, panel_axes = process_dataset("maize_US", df, "Ridge", panels=("scatter",))
    paths = save_panel_images(fig, panel_axes, str(tmp_path), "maize_US")
    img = Image.open(paths["scatter"])
    w, h = img.size

    assert w >= 700 and h >= 700
    assert _band_has_ink(img, row_slice=slice(0, int(h * 0.12))), "missing title band"
    assert _band_has_ink(img, col_slice=slice(0, int(w * 0.14))), "missing y-axis labels"
    assert _band_has_ink(img, row_slice=slice(int(h * 0.82), h)), "missing x-axis labels"
