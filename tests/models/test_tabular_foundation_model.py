import numpy as np

from cybench.models.tabular_foundation_model import TabularFoundationModel, TabularRegressor


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
