"""
Trainer module inspired by Hugging Face's `Trainer` abstraction.

This implementation follows the design ideas of the Hugging Face
Transformers Trainer (see:
https://huggingface.co/docs/transformers/main_classes/trainer),
but is adapted to the CY-Bench codebase and a PyTorch regression setup.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from cybench.datasets.torch_dataset import TorchDataset
from cybench.models.model import BaseModel


class TorchTrainer(BaseModel):
    """
    High-level training and inference wrapper around a PyTorch model.

    Parameters
    ----------
    torch_model:
        The underlying PyTorch model (e.g. LateFusionNetwork) with
        forward(context, temporal) -> (B,) or (B, 1).
    optimizer:
        Optimizer instance, Hydra partial function, or None (defaults to AdamW).
        If a partial function, it will be instantiated with model.parameters().
    scheduler:
        LR scheduler instance, Hydra partial function, or None. If a partial
        function, it will be instantiated with the optimizer. When provided,
        scheduler.step() is called after each training step.
    loss_fn:
        Loss function used for training. Defaults to MSELoss for regression.
    device:
        Device string, e.g. "cuda", "cuda:0", or "cpu".
    dataloader_config:
        Dict specifying DataLoader parameters. Expected keys (all optional):
            - batch_size: int
            - num_workers: int
            - pin_memory: bool
            - drop_last: bool
    epochs:
        Default number of training epochs, can be overridden in `fit(...)`.
    max_grad_norm:
        If not None, gradients are clipped to this norm before optimizer.step().
    verbose:
        Decide whether to show progress bar or not.
    """

    def __init__(
        self,
        name: str,
        torch_model: nn.Module,
        optimizer = None,  # Could be Optimizer OR partial function OR None
        scheduler = None,  # Could be Scheduler OR partial function OR None
        loss_fn: Optional[nn.Module] = None,
        device: Optional[str] = None,
        dataloader_config: Optional[Dict[str, Any]] = None,
        augmentation_config: Optional[Dict[str, Any]] = None,
        epochs: int = 100,
        max_grad_norm: Optional[float] = None,
        verbose: bool = False,
    ):
        self.name = name
        self.model = torch_model
        self.loss_fn = loss_fn or nn.MSELoss()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        # Handle optimizer (could be None, partial, or instantiated)
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        elif not isinstance(optimizer, torch.optim.Optimizer):
            # It's a partial function - call it
            self.optimizer = optimizer(params=self.model.parameters())
        else:
            # Already an optimizer
            self.optimizer = optimizer

        # Handle scheduler (could be None, partial, or instantiated)
        if scheduler is None:
            self.scheduler = None
        elif not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            # It's a partial function - call it
            self.scheduler = scheduler(optimizer=self.optimizer)
        else:
            # Already a scheduler
            self.scheduler = scheduler

        self.dataloader_config = dataloader_config or {}
        self.augmentation_config = augmentation_config or {}
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        """
        Create a DataLoader from a TorchDataset using dataloader_config.

        Args:
            dataset: TorchDataset to wrap.
            shuffle: Whether to shuffle the data.

        Returns:
            Configured DataLoader instance.
        """
        dataloader = self.dataloader_config

        return dataloader(dataset=dataset)

    # ------------------------------------------------------------------
    # BaseModel API
    # ------------------------------------------------------------------

    def fit(self, dataset: TorchDataset, **fit_params) -> Tuple[TorchTrainer, Dict[str, Any]]:
        """
        Fit or train the model.

        Args:
            dataset: Training TorchDataset.
            **fit_params: Additional parameters. Supported:
                - epochs: int, number of training epochs (overrides self.epochs)
                - val_dataset: optional TorchDataset for validation
                - val_every_n_epochs: int, validate every N epochs (default: 5)

        Returns:
            A tuple containing the fitted model and a dict with training history.
        """
        epochs = fit_params.get("epochs", self.epochs)
        val_dataset = fit_params.get("val_dataset", None)
        val_every_n_epochs = fit_params.get("val_every_n_epochs", 5)

        train_loader = self._create_dataloader(dataset, shuffle=True)
        val_loader = (
            self._create_dataloader(val_dataset, shuffle=False)
            if val_dataset is not None
            else None
        )

        history = {"train_loss": [], "val_loss": []}

        pbar = None
        if self.verbose:
            print(f"Starting training for {epochs} epochs...")
            total_batches = epochs * len(train_loader)  # Total iterations across all epochs
            pbar = tqdm(total=total_batches, desc=f"{self.__class__.__name__}")

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                y, x_ctx, x_ts = batch
                y = y.to(self.device)
                x_ctx = x_ctx.to(self.device, non_blocking=True)
                x_ts = x_ts.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                pred = self.model(x_ctx, x_ts)
                if pred.ndim > 1:
                    pred = pred.squeeze(-1)

                loss = self.loss_fn(pred, y.squeeze(-1))
                loss.backward()

                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if pbar is not None:
                    pbar.update(1)

            avg_loss = total_loss / num_batches
            history["train_loss"].append(avg_loss)

            # Validate every N epochs
            if val_loader is not None and (epoch + 1) % val_every_n_epochs == 0:
                self.model.eval()
                val_loss = self._evaluate_loss(val_loader)
                self.model.train()
                history["val_loss"].append(val_loss)
                print(f"Validation loss: {val_loss} after epoch {epoch+1}")
            else:
                history["val_loss"].append(None)

            if self.scheduler is not None:
                self.scheduler.step()

        if pbar is not None:
            pbar.close()
        return self, history

    def _evaluate_loss(self, dataloader: DataLoader) -> float:
        """
        Evaluate model loss on a dataloader.

        Args:
            dataloader: DataLoader containing validation/test batches.

        Returns:
            Average loss across all batches.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                y, x_ctx, x_ts = batch
                y = y.to(self.device)
                x_ctx = x_ctx.to(self.device, non_blocking=True)
                x_ts = x_ts.to(self.device)

                pred = self.model(x_ctx, x_ts)
                if pred.ndim > 1:
                    pred = pred.squeeze(-1)

                total_loss += self.loss_fn(pred, y.squeeze(-1)).item()
                num_batches += 1
        return total_loss / num_batches

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, dataset: TorchDataset, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run fitted model on dataset.

        Args:
            dataset: TorchDataset to predict on.
            **kwargs: Additional parameters.
        Returns:
            A tuple containing a np.ndarray of predictions and a dict with
            additional information.
        """
        dataloader = self._create_dataloader(dataset, shuffle=False)

        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch in dataloader:
                y, x_ctx, x_ts = batch

                x_ctx = x_ctx.to(self.device, non_blocking=True)
                x_ts = x_ts.to(self.device)

                pred = self.model(x_ctx, x_ts)
                if pred.ndim > 1:
                    pred = pred.squeeze(-1)

                preds.append(pred.cpu())

        preds = torch.cat(preds).numpy()

        info = {} # TODO add whatever's interesting

        return preds, info

    def predict_items(self, X, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError # TODO: evaluate whether this methode is necessary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """
        Save model, optimizer, and training state to disk.

        Args:
            path: File path that will be used to save the model.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "dataloader_config": self.dataloader_config,
            "epochs": self.epochs,
            "max_grad_norm": self.max_grad_norm,
        }
        torch.save(checkpoint, os.path.join(path, self.name + ".pt"))

    @classmethod
    def load(cls, model_path: str, model: nn.Module, optimizer: torch.optim.Optimizer, **kwargs):
        """Load a saved checkpoint into a new Trainer instance."""
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(model_path, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Create new trainer instance
        trainer = cls(
            model=model,
            optimizer=optimizer,
            device=device,
            dataloader_config=ckpt.get("dataloader_config", {}),
            epochs=ckpt.get("epochs", 10),
            max_grad_norm=ckpt.get("max_grad_norm"),
        )

        if ckpt.get("scheduler_state_dict") and trainer.scheduler:
            trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        return trainer