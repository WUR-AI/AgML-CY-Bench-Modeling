"""Tests for early stopping epoch tracking."""

import torch
import torch.nn as nn

from cybench.models.torch.utils.early_stopping import EarlyStopping


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))


def test_early_stopping_tracks_best_epoch():
    es = EarlyStopping(patience=3, min_delta=0.0)
    model = _TinyModel()

    es(1.0, model, epoch=1)
    es(0.8, model, epoch=2)
    es(0.9, model, epoch=3)

    assert es.best_epoch == 2
    assert not es.early_stop
