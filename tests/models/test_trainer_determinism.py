"""Tests for reproducible TorchTrainer training."""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cybench.datasets.torch_dataset import TorchDataset
from cybench.models.torch.trainer import TorchTrainer
from cybench.util.config_utils import set_seed


class _TinyNet(nn.Module):
    def __init__(self, context_dim: int, temporal_dim: int):
        super().__init__()
        self.net = nn.Linear(context_dim + temporal_dim, 1)

    def forward(self, x_ctx, x_ts, _doy_ts):
        pooled = x_ts.mean(dim=1)
        return self.net(torch.cat([x_ctx, pooled], dim=-1)).squeeze(-1)


def _tiny_dataset(*, n: int = 48, seq_len: int = 6) -> TorchDataset:
    gen = torch.Generator().manual_seed(0)
    y = torch.randn(n, 1, generator=gen)
    x_c = torch.randn(n, 2, generator=gen)
    x_ts = torch.randn(n, seq_len, 3, generator=gen)
    doy = torch.randint(1, 100, (n, seq_len), dtype=torch.int16, generator=gen)
    indices = pd.DataFrame(
        {
            "adm_id": [f"A{i}" for i in range(n)],
            "year": list(range(2000, 2000 + n)),
        }
    )
    return TorchDataset(
        (y, x_c, x_ts),
        doy,
        (["yield"], ["c1", "c2"], ["f1", "f2", "f3"]),
        indices,
    )


def _fit_once(seed: int = 42) -> np.ndarray:
    set_seed(seed)
    dataset = _tiny_dataset()
    trainer = TorchTrainer(
        name="tiny",
        torch_model=_TinyNet(context_dim=2, temporal_dim=3),
        seed=seed,
        device="cpu",
        epochs=4,
        verbose=False,
        dataloader=partial(DataLoader, batch_size=16),
    )
    trainer.fit(dataset)
    preds, _ = trainer.predict(dataset)
    return np.asarray(preds, dtype=float)


def test_torch_trainer_fit_is_reproducible_for_same_seed():
    first = _fit_once(seed=42)
    second = _fit_once(seed=42)
    assert np.allclose(first, second)


def test_torch_trainer_dataloader_uses_seeded_generator():
    dataset = _tiny_dataset(n=32)
    trainer = TorchTrainer(
        name="tiny",
        torch_model=_TinyNet(context_dim=2, temporal_dim=3),
        seed=7,
        device="cpu",
        epochs=1,
        verbose=False,
        dataloader=partial(DataLoader, batch_size=8),
    )
    loader_a = trainer._create_dataloader(dataset, augment=False, shuffle=True)
    loader_b = trainer._create_dataloader(dataset, augment=False, shuffle=True)
    batches_a = [int(batch[0].sum().item()) for batch in loader_a]
    batches_b = [int(batch[0].sum().item()) for batch in loader_b]
    assert batches_a == batches_b
