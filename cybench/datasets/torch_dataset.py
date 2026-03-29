from typing import Tuple, List, Union

import numpy as np
import pandas as pd
import torch.utils.data
from cybench.datasets.normalizer import Normalizer

from cybench.datasets.dataset import BaseDataset


class TorchDataset(BaseDataset, torch.utils.data.Dataset):
    def __init__(
            self,
            aligned_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            doy_tensor: torch.Tensor,
            column_names: Tuple[list, list, list],
            indices: pd.DataFrame,
            normalizer: Normalizer = None,
    ):
        """
        PyTorch Dataset wrapper for compatibility with torch DataLoader objects.
        Implements splitting by year.

        :param aligned_tensors: Triplet of (target, context, time_series) tensors
        :param doy_tensor: Tensor of (samples x ts_length) listing day of the year for each time-point in a sample
        :param column_names: Triplet of column names for the three aligned tensors
        :param indices: DataFrame with at least 'adm_id' and 'year' columns
        """
        self.y, self.x_context, self.x_ts = aligned_tensors
        self.doy = doy_tensor
        self.y_columns, self.x_context_columns, self.x_ts_columns = column_names
        self.indices = indices
        self.normalizer = normalizer

        self.target = self.y[:, self.y_columns == "yield"]

        # Validate that indices has required columns
        if 'year' not in self.indices.columns:
            raise ValueError("indices DataFrame must contain 'year' column")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.y)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        :param index: Index of the sample to retrieve
        :return: Tuple of (target, context, time_series) tensors for the given index
        """
        return self.target[index], self.x_context[index], self.x_ts[index], self.doy[index]

    def to(self, device):
        self.x_ts = self.x_ts.to(device, non_blocking=False)
        self.x_context = self.x_context.to(device, non_blocking=False)
        self.target = self.target.to(device, non_blocking=False)
        self.doy = self.doy.to(device, non_blocking=False)
        return self

    def process(self, process_cfg):
        if process_cfg.name == "select_context":
            assert (process_cfg.drop is None) ^ (process_cfg.keep is None), "When selecting contexts, either keep or drop must be specified."
            if process_cfg.keep is not None:
                assert set(self.x_context_columns) >= set(process_cfg.keep), f"Your selected context features that are not in the dataset: {set(process_cfg.keep) - set(self.x_context_columns)}"
                keep_ix = [feature in process_cfg.keep for feature in self.x_context_columns]
            if process_cfg.drop is not None:
                assert set(self.x_context_columns) >= set(process_cfg.drop), f"Your selected context features that are not in the dataset: {set(process_cfg.keep) - set(self.x_context_columns)}"
                keep_ix = [feature not in process_cfg.drop for feature in self.x_context_columns]
            # update dataset
            self.x_context_columns = self.x_context_columns[keep_ix]
            self.x_context = self.x_context[:, keep_ix]
        elif process_cfg.name == "clip_time_series":
            T = self.x_ts.shape[1]
            assert process_cfg.start + process_cfg.end < T, f"Number of cutted days {process_cfg.start + process_cfg.end} must be smaller than available days {T}"
            self.x_ts = self.x_ts[:, process_cfg.start:]
            self.doy = self.doy[:, process_cfg.start:]
            if process_cfg.end > 0:
                self.x_ts = self.x_ts[:, :-(process_cfg.end)]
                self.doy = self.doy[:, :-(process_cfg.end)]
            print(self.x_ts.shape)
        else:
            raise NotImplementedError(f"Dataset processing {process_cfg.name} not implemented yet.")


    def split_on_years(
            self, years_split: Tuple[list, list]
    ) -> Tuple['TorchDataset', 'TorchDataset']:
        """
        Create two new datasets based on the provided split in years.
        !!NOTE!!: This is a memory intensive operation, because its making two subsets by copying the original data.
        Future implementations might want to rely on more memory-efficiency, in case that becomes a bottleneck.

        :param years_split: Tuple of two lists, e.g., ([2012, 2014], [2015, 2017])
        :return: Tuple of two TorchDataset instances
        """
        years_set1, years_set2 = years_split

        # Create boolean masks for each split
        mask1 = self.indices['year'].isin(years_set1)
        mask2 = self.indices['year'].isin(years_set2)

        # Get integer indices for each subset
        indices1 = mask1[mask1].index.tolist()
        indices2 = mask2[mask2].index.tolist()

        # Create new datasets with sliced tensors
        dataset1 = TorchDataset(
            aligned_tensors=(
                self.y[indices1],
                self.x_context[indices1],
                self.x_ts[indices1]
            ),
            doy_tensor=self.doy[indices1],
            column_names=(
                self.y_columns,
                self.x_context_columns,
                self.x_ts_columns
            ),
            indices=self.indices.iloc[indices1].reset_index(drop=True),
            normalizer=self.normalizer,
        )
        dataset2 = TorchDataset(
            aligned_tensors=(
                self.y[indices2],
                self.x_context[indices2],
                self.x_ts[indices2]
            ),
            doy_tensor=self.doy[indices2],
            column_names=(
                self.y_columns,
                self.x_context_columns,
                self.x_ts_columns
            ),
            indices=self.indices.iloc[indices2].reset_index(drop=True),
            normalizer=self.normalizer,
        )
        return dataset1, dataset2

    def _subset_by_indices(self, indices: Union[List[int], np.ndarray]) -> 'TorchDataset':
        """
        Create a subset of the dataset based on provided indices.
        This is the core method used by all subset generation methods.

        :param indices: List or array of integer indices to include in subset
        :return: New TorchDataset instance with selected samples
        """
        # Convert to numpy array and ensure sorted for better memory access
        indices = np.array(indices)
        indices = np.sort(indices)

        # Validate indices
        if len(indices) == 0:
            raise ValueError("Cannot create subset with empty indices")
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise ValueError(f"Indices must be in range [0, {len(self)})")

        # Create new dataset with subset of data
        subset = TorchDataset(
            aligned_tensors=(
                self.y[indices],
                self.x_context[indices],
                self.x_ts[indices]
            ),
            doy_tensor=self.doy[indices],
            column_names=(
                self.y_columns,
                self.x_context_columns,
                self.x_ts_columns
            ),
            indices=self.indices.iloc[indices].reset_index(drop=True),
            normalizer=self.normalizer,
        )

        return subset

    def random_subset(self, n_samples: int, seed: int) -> 'TorchDataset':
        """
        Create a random subset of the dataset.

        :param n_samples: Number of samples to include in the subset
        :param seed: Random seed for reproducibility
        :return: New TorchDataset instance with randomly selected samples
        """
        # Ensure n_samples doesn't exceed dataset size
        n_samples = min(n_samples, len(self))

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Generate random indices without replacement
        random_indices = np.random.choice(len(self), size=n_samples, replace=False)

        return self._subset_by_indices(random_indices)

    def location_subset(self, location_id: Union[int, str]) -> 'TorchDataset':
        """
        Create a subset containing all samples from a specific location.

        :param location_id: The location ID (adm_id) to filter by
        :return: New TorchDataset instance with samples from specified location
        """
        # Find all indices matching the location_id
        location_mask = self.indices['adm_id'] == location_id
        location_indices = self.indices[location_mask].index.tolist()

        if len(location_indices) == 0:
            raise ValueError(f"No samples found for location_id: {location_id}")

        return self._subset_by_indices(location_indices)

    def k_nearest_subset(
        self,
        reference_idx: int,
        k: int,
        loc_columns: list = ['loc_x', 'loc_y','loc_z', 'prec_clmt', 'tavg_clmt', 'ssm_clmt', 'dd_clmt'],
    ) -> 'TorchDataset':
        """
        Create a subset containing the k nearest samples to a reference sample
        based on geographical and climatic distance

        :param reference_idx: Index of the reference sample
        :param k: Number of nearest neighbors to include
        :param loc_columns: List of columns that indicate similar climatic zones
        :return: New TorchDataset instance with k nearest samples
        """
        loc_columns_ix = [col in loc_columns for col in self.x_context_columns]
        ref = self.x_context[reference_idx, loc_columns_ix]

        # Calculate geographical and climatic distance
        distances = (self.x_context[:, loc_columns_ix] - ref) ** 2
        # artificially remove the reference ix from the list of closest kneighbors
        if distances.device.type == "cuda":
            distances = distances.cpu()
        distances = distances.mean(axis=1)
        distances[reference_idx] = 1

        nearest_indices = np.argpartition(distances, k)[:k]
        nearest_indices = nearest_indices[np.argsort(distances[nearest_indices])]

        return self._subset_by_indices(nearest_indices)

    @property
    def raw_ts_data(self) -> np.ndarray:
        """
        Get the original (denormalized) time series data.
        """
        return self.normalizer.denormalize(self.x_ts, self.x_ts_columns)

    @property
    def years(self) -> set:
        """
        Obtain a set containing all years occurring in the dataset
        """
        return set(self.indices.year.unique())

    @property
    def location_ids(self) -> set:
        """
        Obtain a set containing all location ids occurring in the dataset
        """
        return set(self.indices.adm_id.unique())

    @property
    def targets(self) -> np.ndarray[tuple[int], np.dtype[np.number]]:
        """
        Obtain a numpy array of targets or labels
        """
        return self.target.detach().cpu().numpy().reshape(-1)


