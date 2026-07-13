"""
Trainer module inspired by Hugging Face's `Trainer` abstraction.

This implementation follows the design ideas of the Hugging Face
Transformers Trainer (see:
https://huggingface.co/docs/transformers/main_classes/trainer),
but is adapted to the CY-Bench codebase and a PyTorch regression setup.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import nullcontext
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from cybench.datasets.torch_dataset import TorchDataset
from cybench.models.model import BaseModel
from cybench.models.torch.utils.augmentation import create_collate_fn, AugmentationComposer
from cybench.models.torch.utils.early_stopping import EarlyStopping
from cybench.util.config_utils import deterministic_torch_training, set_seed

# init logger
log = logging.getLogger(__name__)

class TorchTrainer(BaseModel):
    """
    High-level training and inference wrapper around a PyTorch model.

    This trainer supports both training from scratch and transfer learning from
    pretrained checkpoints. It provides flexible fine-tuning strategies including
    linear probing, feature extraction, and selective layer freezing.

    Parameters
    ----------
    name : str
        Identifier for the trainer/model (used for logging and saving).
    torch_model : nn.Module, optional
        The underlying PyTorch model (e.g., ContextConditionalNetwork) with
        forward(context, temporal, doy) -> (B,) or (B, 1).
        Required if not using pretrained_from.
    pretrained_from : DictConfig, optional
        Configuration for loading a pretrained model. Required if torch_model is None.
        Should contain:
            - run_path: str, path to the pretrained run directory
            - split: str, optional, specific split subfolder to load from
            - non_transferred_modules: List[str], modules to keep randomly initialized
            - freeze_modules: List[str], modules to freeze during training
            - use_lora: bool, whether to apply LoRA adapters (not yet implemented)
    test_years : List[int], optional
        Test years used to locate the correct checkpoint when loading pretrained models.
        Required when using pretrained_from.
    optimizer : torch.optim.Optimizer or callable, optional
        Optimizer instance, Hydra partial function, or None (defaults to AdamW).
        If a partial function, it will be instantiated with model.parameters().
    scheduler : torch.optim.lr_scheduler.LRScheduler or callable, optional
        LR scheduler instance, Hydra partial function, or None. If a partial
        function, it will be instantiated with the optimizer. When provided,
        scheduler.step() is called after each epoch.
    loss_fn : nn.Module, optional
        Loss function used for training. Defaults to MSELoss for regression.
    device : str, optional
        Device string, e.g. "cuda", "cuda:0", or "cpu". Auto-detects if None.
    dataloader : callable, optional
        A partial of a DataLoader including all parameters except the dataset itself.
        Should be created using functools.partial or Hydra's _partial_: true.
    augmentation : AugmentationComposer, optional
        Augmentation strategy to apply during training. Applied via custom collate_fn.
    epochs : int, default=100
        Default number of training epochs. Can be overridden in fit().
    max_grad_norm : float, optional, default=1.0
        Maximum gradient norm for clipping. If None, no clipping is applied.
        Helps prevent exploding gradients during training.
    seed : int, default=42
        Random seed for reproducibility. Used for checkpoint loading and initialization.
    verbose : bool, default=False
        If True, displays progress bar during training and additional logging.
    """

    def __init__(
        self,
        name: str,
        torch_model: Optional[nn.Module] = None,
        pretrained_from: Optional[DictConfig] = None,
        test_years: Optional[List[int]] = None,
        optimizer = None,  # Could be Optimizer OR partial function OR None
        scheduler = None,  # Could be Scheduler OR partial function OR None
        loss_fn: Optional[nn.Module] = None,
        device: Optional[str] = None,
        preload_to_device: Optional[bool] = False,
        dataloader: Callable[..., DataLoader[Any]] | None = None,
        augmentation: AugmentationComposer | None = None,
        epochs: int = 100,
        early_stopping: Optional[EarlyStopping] = None,
        early_stopping_monitor: str = "val",
        max_grad_norm: Optional[float] = 1,
        seed: int = 42,
        verbose: bool = False,
        **kwargs
    ):
        self.name = name
        self.seed = seed
        self.verbose = verbose
        self.loss_fn = loss_fn or nn.MSELoss()
        if torch_model is None:
            raise ValueError("torch_model must be provided")
        self.model: nn.Module = torch_model

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif str(device).startswith("cuda") and not torch.cuda.is_available():
            log.warning(
                "CUDA requested (device=%s) but no NVIDIA GPU/driver found; using CPU.",
                device,
            )
            device = "cpu"
        self.device = torch.device(device)
        self.preload_to_device = preload_to_device
        self.model.to(self.device)
        if self.verbose:
            log.info(
                "Initialized %s | %.2fM parameters on %s",
                self.name,
                sum(p.numel() for p in self.model.parameters()) * 1e-6,
                self.device,
            )

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

        self.dataloader = dataloader or DataLoader
        self.augmentation = augmentation
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.early_stopping_monitor = early_stopping_monitor
        self.max_grad_norm = max_grad_norm

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dataloader_generator(self) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return generator

    def _create_dataloader(
        self, dataset: TorchDataset, augment: bool, shuffle: bool
    ) -> DataLoader[Any]:
        """
        Create a DataLoader from a TorchDataset using dataloader.

        Args:
            dataset: TorchDataset to wrap.
            augment: whether to augment the dataset.
            shuffle: Whether to shuffle the data.

        Returns:
            Configured DataLoader instance.
        """
        dataloader = self.dataloader
        loader_kwargs: dict[str, Any] = {"dataset": dataset, "shuffle": shuffle}
        if shuffle:
            loader_kwargs["generator"] = self._dataloader_generator()

        if self.augmentation is not None and augment:
            collate_fn = create_collate_fn(
                augmentation=self.augmentation,
                context_columns=dataset.x_context_columns,
            )
            loader_kwargs["collate_fn"] = collate_fn
        return dataloader(**loader_kwargs)

    # ------------------------------------------------------------------
    # BaseModel API
    # ------------------------------------------------------------------

    def fit(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, dataset: TorchDataset, **fit_params
    ) -> tuple[Any, Dict[str, Any]]:
        """
        Fit or train the model.

        Args:
            dataset: Training TorchDataset.
            **fit_params: Additional parameters. Supported:
                - epochs: int, number of training epochs (overrides self.epochs)
                - val_dataset: optional TorchDataset for validation
                - val_every_n_epochs: int, validate every N epochs (default: 1)
                - early_stopping_monitor: "val" or "train" (default: self.early_stopping_monitor)
                - epoch_log_interval: log train/val loss every N epochs when verbose=False

        Returns:
            A tuple containing the fitted model and a dict with training history.
        """
        set_seed(self.seed)

        epochs = fit_params.get("epochs", self.epochs)
        val_dataset = fit_params.get("val_dataset", None)
        val_every_n_epochs = fit_params.get("val_every_n_epochs", 1)
        early_stopping_monitor = fit_params.get(
            "early_stopping_monitor", self.early_stopping_monitor
        )
        epoch_log_interval = fit_params.get("epoch_log_interval")

        if self.early_stopping is not None:
            self.early_stopping.reset()

        if self.preload_to_device:
            dataset = dataset.to(self.device)
            if val_dataset is not None:
                val_dataset = val_dataset.to(self.device)

        train_loader = self._create_dataloader(dataset, augment=True, shuffle=True)
        val_loader = (
            self._create_dataloader(val_dataset, augment=False, shuffle=False)
            if val_dataset is not None
            else None
        )

        history = cast(Dict[str, Any], {"train_loss": [], "val_loss": []})
        epochs_run = 0

        if self.verbose:
            log.info(f"Starting training for {epochs} epochs...")
        total_batches = epochs * len(train_loader)
        pbar = tqdm(range(total_batches), desc=self.__class__.__name__) if self.verbose else None

        with deterministic_torch_training():
            self.model.train()
            # TODO delete time tracking. Only for debugging
            tt = 0
            start_training = time.time()
            for epoch in range(epochs):
                epochs_run = epoch + 1
                total_loss = 0.0
                num_batches = 0

                for batch in train_loader:
                    y, x_ctx, x_ts, doy_ts = batch
                    if (not self.preload_to_device) and (self.device.type != "cpu"):
                        y = y.to(self.device, non_blocking=True)
                        x_ctx = x_ctx.to(self.device, non_blocking=True)
                        x_ts = x_ts.to(self.device, non_blocking=True)
                        doy_ts = doy_ts.to(self.device, non_blocking=True)

                    self.optimizer.zero_grad(set_to_none=True)
                    start = time.time()
                    pred = self.model(x_ctx, x_ts, doy_ts)
                    # DEBUG Model:
                    #print(self.model.state_dict()["regression_head.net.3.weight"][0, 0])
                    #print(self.model.context_encoder(x_ctx[0]))
                    #print(self.model.temporal_encoder(x_ts[0]))

                    if pred.ndim > 1:
                        pred = pred.squeeze(-1)

                    loss = self.loss_fn(pred, y.squeeze(-1))
                    loss.backward()
                    tt += time.time() - start
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

                val_loss = None
                # Validate every N epochs
                if val_loader is not None and (epoch + 1) % val_every_n_epochs == 0:
                    val_loss = self._evaluate_loss(val_loader)
                    history["val_loss"].append(val_loss)
                else:
                    history["val_loss"].append(None)

                if self.early_stopping is not None:
                    monitor_loss = None
                    if early_stopping_monitor == "val" and val_loss is not None:
                        monitor_loss = val_loss
                    elif early_stopping_monitor == "train":
                        monitor_loss = avg_loss
                    if monitor_loss is not None:
                        self.early_stopping(monitor_loss, self.model, epoch + 1)
                        if self.early_stopping.early_stop:
                            log.info("Early stopping triggered.")
                            if self.verbose:
                                print(f"Early stopping triggered: after epoch {epoch + 1}")
                            break

                # Step Scheduler (ReduceLROnPlateau requires val loss)
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        # Only step if we have a valid metric this epoch
                        if val_loss is not None:
                            self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                if self.verbose:
                    lr = self.optimizer.param_groups[0]['lr']
                    msg = f"Epoch {epoch + 1:4d}/{epochs} | train {avg_loss:.4f}"
                    if val_loss is not None:
                        msg += f" | val {val_loss:.4f}"
                    msg += f" | lr {lr:.2e}"
                    tqdm.write(msg)
                elif epoch_log_interval and (
                    (epoch + 1) % int(epoch_log_interval) == 0
                    or epoch + 1 == epochs
                    or (self.early_stopping is not None and self.early_stopping.early_stop)
                ):
                    msg = f"Epoch {epoch + 1}/{epochs} | train {avg_loss:.4f}"
                    if val_loss is not None:
                        msg += f" | val {val_loss:.4f}"
                    log.info(msg)

            log.debug("Forward and backward pass took", np.round(tt / (time.time() - start_training) * 100), "% of training time.")
            if pbar is not None:
                pbar.close()

            # Restore Best Weights (Critical Step)
            if self.early_stopping is not None and self.early_stopping.best_model_state is not None:
                log.info(f"Restoring best model weights (Loss: {self.early_stopping.best_loss:.3f})")
                self.model.load_state_dict(self.early_stopping.best_model_state)

        if self.early_stopping is not None and self.early_stopping.best_epoch is not None:
            history["best_epoch"] = self.early_stopping.best_epoch
        else:
            history["best_epoch"] = epochs_run
        history["epochs_run"] = epochs_run
        return self, history

    @torch.no_grad()
    def _evaluate_loss(self, dataloader: DataLoader[Any]) -> float:
        """
        Evaluate model loss on a dataloader.

        Args:
            dataloader: DataLoader containing validation/test batches.

        Returns:
            Average loss across all batches.
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            y, x_ctx, x_ts, doy_ts = batch
            if (not self.preload_to_device) and (self.device.type != "cpu"):
                y = y.to(self.device, non_blocking=True)
                x_ctx = x_ctx.to(self.device, non_blocking=True)
                x_ts = x_ts.to(self.device, non_blocking=True)
                doy_ts = doy_ts.to(self.device, non_blocking=True)

            pred = self.model(x_ctx, x_ts, doy_ts)
            if pred.ndim > 1:
                pred = pred.squeeze(-1)

            loss = self.loss_fn(pred, y.squeeze(-1))

            # --- FIX STARTS HERE ---
            # Get the actual size of this specific batch
            current_batch_size = y.size(0)

            # 1. "Undo" the mean reduction to get the total sum of errors for this batch
            total_loss += loss.item() * current_batch_size

            # 2. Track the total number of samples seen
            total_samples += current_batch_size

        self.model.train()
        # Calculate the true average over all samples
        return total_loss / total_samples

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        dataset: TorchDataset,
        **kwargs,
    ) -> Tuple[npt.NDArray[Any], Dict[str, Any]]:
        """
        Run fitted model on dataset.

        Args:
            dataset: TorchDataset to predict on.
            **kwargs: Additional parameters. Supported:
                - inspector: ModelInspector instance. When provided, hooks are
                  registered once around the entire dataloader loop and the
                  concatenated results (one tensor per layer per key, spanning
                  all batches) are returned under ``info["inspect_results"]``.

        Returns:
            A tuple of:
                - preds: np.ndarray of shape (N,) with model predictions.
                - info: dict. Contains ``"inspect_results"`` when an inspector
                  is supplied: ``Dict[layer_name, Dict[key, Tensor(N, ...)]]``.
        """
        if self.preload_to_device:
            dataset = dataset.to(self.device)
        dataloader = self._create_dataloader(dataset, augment=False, shuffle=False)

        inspector = kwargs.get("inspector", None)
        info: Dict[str, Any] = {}
        preds = []

        self.model.eval()

        # Wrap the entire loop in inspector.session() so hooks are registered
        # once, buffers accumulate across all batches, and hooks are cleaned up
        # when the loop finishes — regardless of exceptions.
        session_ctx = inspector.session() if inspector is not None else nullcontext()

        with torch.no_grad(), session_ctx:
            for batch in dataloader:
                _, x_ctx, x_ts, doy_ts = batch

                if (not self.preload_to_device) and (self.device.type != "cpu"):
                    x_ctx = x_ctx.to(self.device, non_blocking=True)
                    x_ts = x_ts.to(self.device, non_blocking=True)
                    doy_ts = doy_ts.to(self.device, non_blocking=True)

                pred = self.model(x_ctx, x_ts, doy_ts)

                if pred.ndim > 1:
                    pred = pred.squeeze(-1)

                preds.append(pred.cpu())

        preds = cast(npt.NDArray[Any], torch.cat(preds).numpy())

        if inspector is not None:
            # results() cats all per-batch tensors → one tensor per layer/key
            info["inspect_results"] = inspector.results()

        return preds, info

    def predict_items(self, X, **kwargs) -> Tuple[npt.NDArray[Any], Dict[str, Any]]:
        raise NotImplementedError # TODO: evaluate whether this methode is necessary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, path: str, compress: bool = True, seed: Optional[int] = None
    ) -> None:
        """
        Save model, optimizer, and training state to disk.

        Args:
            path: File path that will be used to save the model.
            compress: Whether to compress the model or not, using half-precision (16bit)
            seed: add seed to file name in case of multiple repetitions.
        """
        model_state_dict = self.model.state_dict()
        if compress:
            # Half-precision
            model_state_dict = {k: v.half() for k, v in model_state_dict.items()}


        checkpoint = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epochs": self.epochs,
            "max_grad_norm": self.max_grad_norm,
        }
        if seed is not None:
            torch.save(checkpoint, os.path.join(path, self.name + f"_{seed}.pt"))
        else:
            torch.save(checkpoint, os.path.join(path, self.name + ".pt"))

    @classmethod
    def load(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        model_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        **kwargs: Any,
    ) -> "TorchTrainer":
        """Load a saved checkpoint into a new Trainer instance."""
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        trainer = cls(
            name=kwargs.get("name", "loaded"),
            torch_model=model,
            optimizer=optimizer,
            device=device,
            dataloader=kwargs.get("dataloader"),
            epochs=ckpt.get("epochs", 10),
            max_grad_norm=ckpt.get("max_grad_norm"),
        )

        if ckpt.get("scheduler_state_dict") and trainer.scheduler:
            trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        return trainer