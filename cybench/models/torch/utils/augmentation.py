import torch
import numpy as np
from typing import List, Tuple, Any, Optional, Union, Dict

from omegaconf import DictConfig


class TimeSeriesClipping:
    """
    Clips the time series data from start and end on batched tensors.
    Each sample in the batch gets independently clipped.

    The number of time steps to remove from the start is drawn from Uniform(0, max_start).
    The number of time steps to remove from the end is drawn from Uniform(0, max_end).

    Note: This returns the minimum sequence length across the batch to maintain
    consistent tensor shapes.
    """

    def __init__(self, start: int, end: int):
        self.max_start = start
        self.max_end = end

    def __call__(self,
                 batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: Tuple of batched tensors (y, x_ctx, x_ts, doy_ts)
                   x_ts shape: (batch_size, seq_len, features)
                   doy_ts shape: (batch_size, seq_len)

        Returns:
            Tuple of augmented batched tensors
        """
        y, x_ctx, x_ts, doy_ts = batch
        batch_size, seq_len, _ = x_ts.shape

        # Draw random clipping amounts for each sample in batch
        clip_start = np.random.randint(0, self.max_start + 1, size=batch_size)
        clip_end = np.random.randint(0, self.max_end + 1, size=batch_size)

        # Calculate the minimum resulting sequence length
        min_seq_len = seq_len - clip_start.max() - clip_end.max()

        # Safety check
        if min_seq_len <= 0:
            return batch  # Return original if clipping would eliminate sequence

        # For efficiency, clip all sequences to the minimum length
        # This keeps tensors rectangular without padding
        start_idx = clip_start.max()
        end_idx = seq_len - clip_end.max()

        new_x_ts = x_ts[:, start_idx:end_idx, :]
        new_doy_ts = doy_ts[:, start_idx:end_idx]

        return y, x_ctx, new_x_ts, new_doy_ts


class GaussianNoiseTimeSeries:
    """
    Adds Gaussian (Normal) noise to the time series data.
    Operates efficiently on the entire batch at once.
    """

    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self,
                 batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: Tuple of batched tensors (y, x_ctx, x_ts, doy_ts)
                   x_ts shape: (batch_size, seq_len, features)

        Returns:
            Tuple of augmented batched tensors with Gaussian noise added to time series
        """
        y, x_ctx, x_ts, doy_ts = batch

        # Generate noise for entire batch at once
        noise = torch.randn_like(x_ts, device=x_ts.device) * self.std

        return y, x_ctx, x_ts + noise, doy_ts


class YearUniformNoise:
    """
    Adds Uniform noise to the 'year' feature located within the context tensor.
    Each sample in the batch gets independent noise.
    """

    def __init__(self, noise_range: float):
        """
        Args:
            noise_range: The bound for the uniform noise. Noise is drawn from U(-noise_range/2, noise_range/2).
        """
        self.noise_range = noise_range

    def __call__(self,
                 batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: Tuple of batched tensors (y, x_ctx, x_ts, doy_ts)
                   x_ctx shape: (batch_size, context_features)
            **kwargs: Should contain 'context_columns' to locate the year feature

        Returns:
            Tuple of augmented batched tensors with noise added to year feature
        """
        y, x_ctx, x_ts, doy_ts = batch

        # Get year index from context columns
        year_index = None
        if "context_columns" in kwargs:
            year_indices = np.where(kwargs["context_columns"] == "year")[0]
            if len(year_indices) == 0: return batch
            year_index = year_indices[0]

            batch_size = x_ctx.shape[0]

            # Draw noise for entire batch at once: U(-noise_range, noise_range)
            noise = (torch.rand(batch_size, device=x_ctx.device) * self.noise_range) - self.noise_range / 2

            # Clone context to avoid in-place modification
            x_ctx_aug = x_ctx.clone()
            x_ctx_aug[:, year_index] += noise

            return y, x_ctx_aug, x_ts, doy_ts
        else:
            return batch


class AugmentationComposer:
    """
    Composes multiple augmentations.
    Now supports both List (positional) and Dict (named) configurations.
    """

    def __init__(self, augmentations: Union[List[Any], Dict[str, Any], DictConfig]):
        # Handle Dictionary (extract values)
        if isinstance(augmentations, (dict, DictConfig)):
             # Sorting by key ensures deterministic order if that matters,
             # otherwise .values() is usually insertion-ordered in modern Python.
            self.augmentations = list(augmentations.values())
        # Handle List
        elif isinstance(augmentations, list):
            self.augmentations = augmentations
        else:
            raise TypeError(f"Augmentations must be List or Dict, got {type(augmentations)}")

    def __call__(self,
                 batch_list: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
                 context_columns: Optional[np.ndarray] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate function that stacks samples and applies augmentations.

        Args:
            batch_list: List of samples from __getitem__, each sample is
                       (y, x_context, x_ts, doy_ts)
            context_columns: Column names for context features

        Returns:
            Tuple of augmented batched tensors (y, x_ctx, x_ts, doy_ts)
        """
        # Stack tensors into batch
        y = torch.stack([s[0] for s in batch_list])
        x_ctx = torch.stack([s[1] for s in batch_list])
        x_ts = torch.stack([s[2] for s in batch_list])
        doy = torch.stack([s[3] for s in batch_list])

        batch = (y, x_ctx, x_ts, doy)

        # Apply augmentations sequentially
        for aug in self.augmentations:
            batch = aug(batch, context_columns=context_columns)

        return batch


def create_collate_fn(augmentation: Optional[AugmentationComposer],
                      context_columns: Optional[np.ndarray] = None):
    """
    Factory function to create collate_fn with augmentation.

    Args:
        augmentation: AugmentationComposer instance or None
        context_columns: Column names for context features

    Returns:
        Collate function for DataLoader, or None to use default collate
    """
    if augmentation is None:
        return None  # Use default collate

    return lambda batch: augmentation(batch, context_columns=context_columns)