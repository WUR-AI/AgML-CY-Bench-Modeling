import numpy as np
from hydra import compose, initialize

from cybench.models.tabular_foundation_model import (
    TabularFoundationModel,
    TabularRegressor,
    _exaone_from_pretrained_kwargs,
)


class _DeviceTrackingEstimator:
    def __init__(self, device: str):
        self.device = device
        self.fit_calls = 0
        self.predict_calls = 0

    def fit(self, X, y):
        self.fit_calls += 1

    def predict(self, X, *args, **kwargs):
        self.predict_calls += 1
        # Distinct values so we can tell GPU vs CPU estimators apart.
        value = 1.0 if self.device == "cpu" else 2.0
        return np.full(len(X), value, dtype=float)


class _StubTabularModel(TabularFoundationModel):
    def __init__(self, **kwargs):
        self._estimators: dict[str, _DeviceTrackingEstimator] = {}
        super().__init__(name="stub", **kwargs)

    @classmethod
    def _check_import(cls) -> None:
        return None

    def _make_estimator(self, device: str | None = None) -> TabularRegressor:
        resolved = device or self.device
        if resolved not in self._estimators:
            self._estimators[resolved] = _DeviceTrackingEstimator(resolved)
        return self._estimators[resolved]


def test_predict_on_cpu_uses_cpu_estimator_not_gpu():
    model = _StubTabularModel(device="cuda", predict_batch_size=2)
    model._train_X = np.zeros((4, 2), dtype=float)
    model._train_y = np.zeros(4, dtype=float)
    model.estimator = model._make_estimator(device="cuda")

    X = np.zeros((3, 2), dtype=float)
    preds = model._predict_on_cpu(X)

    cpu_estimator = model._estimators["cpu"]
    gpu_estimator = model._estimators["cuda"]
    assert cpu_estimator.fit_calls == 1
    assert cpu_estimator.predict_calls == 2  # batch_size=2 over 3 rows
    assert gpu_estimator.predict_calls == 0
    np.testing.assert_array_equal(preds, np.ones(3))


def test_exaone_from_pretrained_kwargs_cpu_forces_float32():
    kwargs = _exaone_from_pretrained_kwargs(
        device="cpu",
        random_state=7,
        estimator_kwargs={
            "ensemble_count": 4,
            "compute_dtype": "float16",
            "max_vram_bytes": 8 << 30,
            "n_estimators": 8,
        },
    )
    assert kwargs["device"] == "cpu"
    assert kwargs["compute_dtype"] == "float32"
    assert kwargs["seed"] == 7
    assert kwargs["ensemble_count"] == 4
    assert "max_vram_bytes" not in kwargs
    assert "n_estimators" not in kwargs


def test_exaone_from_pretrained_kwargs_cuda_defaults_and_rejects_float32():
    auto = _exaone_from_pretrained_kwargs(
        device="cuda",
        random_state=1,
        estimator_kwargs={"compute_dtype": "auto"},
    )
    assert auto["compute_dtype"] == "float16"
    assert auto["seed"] == 1

    forced = _exaone_from_pretrained_kwargs(
        device="cuda",
        random_state=1,
        estimator_kwargs={"compute_dtype": "float32", "seed": 99},
    )
    assert forced["compute_dtype"] == "float16"
    assert forced["seed"] == 99


def test_exaone_tabular_hydra_config_composes():
    with initialize(version_base=None, config_path="../../cybench/conf"):
        cfg = compose(config_name="config", overrides=["model=exaone_tabular"])
    assert cfg.model.name == "exaone_tabular"
    assert cfg.model.ensemble_count == 8
    assert (
        cfg.model._target_
        == "cybench.models.tabular_foundation_model.EXAONETabularModel"
    )
