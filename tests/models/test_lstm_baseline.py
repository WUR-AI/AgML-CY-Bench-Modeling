import torch

from cybench.models.torch.architectures.lstm_baseline import LSTMBaseline


def test_lstm_baseline_last_pool_shape():
    model = LSTMBaseline(input_size=6, hidden_size=32, num_layers=2, pool="last")
    temporal = torch.randn(4, 20, 6)
    context = torch.randn(4, 0)
    doys = torch.randint(1, 366, (4, 20))

    out = model(context, temporal, doys)
    assert out.shape == (4,)


def test_lstm_baseline_mean_pool_shape():
    model = LSTMBaseline(input_size=6, hidden_size=16, num_layers=1, pool="mean")
    temporal = torch.randn(2, 10, 6)
    out = model(torch.zeros(2, 0), temporal, torch.zeros(2, 10, dtype=torch.int16))
    assert out.shape == (2,)


def test_lstm_baseline_ignores_context():
    model = LSTMBaseline(input_size=3, hidden_size=8, num_layers=1)
    temporal = torch.randn(1, 5, 3)
    out_a = model(torch.zeros(1, 4), temporal, torch.zeros(1, 5, dtype=torch.int16))
    out_b = model(torch.ones(1, 4), temporal, torch.zeros(1, 5, dtype=torch.int16))
    assert torch.allclose(out_a, out_b)
