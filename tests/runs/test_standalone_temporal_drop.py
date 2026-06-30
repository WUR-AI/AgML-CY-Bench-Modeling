from hydra import compose, initialize
from omegaconf import OmegaConf

from cybench.runs.run_experiments import _maybe_drop_temporal_for_standalone_models


def _cfg_with_model(model: str):
    with initialize(version_base=None, config_path="../../cybench/conf"):
        return compose(
            config_name="config",
            overrides=[
                f"model={model}",
                "dataset/crop=wheat",
                "dataset.country=IN",
            ],
        )


def test_trend_model_drops_temporal_sources():
    cfg = _cfg_with_model("trend")
    assert cfg.dataset.temporal.sources
    _maybe_drop_temporal_for_standalone_models(cfg)
    assert not cfg.dataset.temporal.sources


def test_ridge_model_keeps_temporal_sources():
    cfg = _cfg_with_model("ridge")
    assert cfg.dataset.temporal.sources
    _maybe_drop_temporal_for_standalone_models(cfg)
    assert cfg.dataset.temporal.sources
