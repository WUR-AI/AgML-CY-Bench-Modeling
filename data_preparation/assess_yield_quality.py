"""CLI wrapper for yield quality assessment (Hydra)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from cybench.config import PATH_DATA_DIR
from cybench.datasets.yield_quality import (
    process_yield_quality_files,
    viz_flag_columns,
    yield_quality_settings_from_target,
)


def _visualize_module():
    """Load sibling visualize_yield_quality.py without requiring a package."""
    path = Path(__file__).with_name("visualize_yield_quality.py")
    module_name = "_yield_quality_viz"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load visualization module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _viz_config(cfg: DictConfig) -> DictConfig:
    viz = cfg.get("visualize")
    if viz is None:
        return OmegaConf.create({"enabled": False})
    return viz


@hydra.main(
    version_base=None,
    config_path="../cybench/conf/dataset",
    config_name="assess_yield_quality",
)
def main(cfg: DictConfig) -> None:
    settings = yield_quality_settings_from_target(cfg)
    directory = Path(cfg.directory or PATH_DATA_DIR)
    crops = list(cfg.crops)
    viz = _viz_config(cfg)

    print("Start crop yield data quality assessment...")
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    print(
        "Quality settings: "
        f"outlier_threshold={settings.outlier_threshold}, "
        f"polyfit_degree={settings.polyfit_degree}, "
        f"consecutive_threshold_factor={settings.consecutive_threshold_factor}, "
        f"consecutive_min_years={settings.consecutive_min_years}, "
        f"min_usable_year={settings.min_usable_year}"
    )
    process_yield_quality_files(directory, crops, settings=settings)

    if not viz.get("enabled", False):
        return

    flags = viz_flag_columns(cfg)
    countries = list(viz.countries) if viz.get("countries") is not None else None
    output_root = Path(viz.output_dir)

    print("\nGenerating yield quality visualizations...")
    viz_mod = _visualize_module()
    paths = viz_mod.run_yield_quality_visualizations(
        directory,
        crops,
        settings=settings,
        output_root=output_root,
        max_panels=int(viz.get("max_panels", 9)),
        flags=flags,
        countries=countries,
        only_if_flagged=bool(viz.get("only_if_flagged", True)),
    )
    print(f"\nWrote {len(paths)} visualization file(s) under {output_root.resolve()}")


if __name__ == "__main__":
    main()
