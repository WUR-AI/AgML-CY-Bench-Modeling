import pytest
from omegaconf import open_dict

from cybench.datasets.data_factory import DataFactory

_original_build = DataFactory.build


def _build_without_cache(self: DataFactory, *args, **kwargs):
    with open_dict(self.cfg):
        self.cfg.use_cache = False
    return _original_build(self, *args, **kwargs)


@pytest.fixture(autouse=True)
def disable_dataset_cache(monkeypatch):
    """Always build datasets fresh in tests; never read cybench/data/cache/."""
    monkeypatch.setattr(DataFactory, "build", _build_without_cache)
