"""
Provides a utility for generating xarray Datasets with random data.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from xarray_binfile.read.file_metadata import ReadSpecs, ReadSpecsGetterProtocol


@dataclass(frozen=True)
class DatasetGenerator:
    """
    Generates xarray Datasets with random data based on metadata.

    Attributes:
        read_specs_getter: A callable that generates read specifications for binary files.
        random_generator: A random number generator for creating random data.
    """

    read_specs_getter: ReadSpecsGetterProtocol
    random_generator = np.random.Generator(np.random.PCG64(1234))

    def _get_numpy_array(self, metadata: ReadSpecs) -> np.ndarray:
        """
        Generates a random NumPy array based on metadata.

        Args:
            metadata: Metadata describing the array.

        Returns:
            A random NumPy array.
        """
        return self.random_generator.random(size=metadata.shape, dtype=metadata.dtype)

    def _get_xarray_array(self, metadata: ReadSpecs) -> xr.DataArray:
        """
        Generates a random xarray DataArray based on metadata.

        Args:
            metadata: Metadata describing the array.

        Returns:
            A random xarray DataArray.
        """
        return xr.DataArray(data=self._get_numpy_array(metadata), coords=metadata.coords, attrs=metadata.attrs)

    def _get_dataset(self, metadata: ReadSpecs) -> xr.Dataset:
        """
        Generates a random xarray Dataset based on metadata.

        Args:
            metadata: Metadata describing the dataset.

        Returns:
            A random xarray Dataset.
        """
        return self._get_xarray_array(metadata).to_dataset(name=metadata.name)

    def __call__(self, iter_filepath: Iterator[Path]) -> xr.Dataset:
        """
        Generates a merged xarray Dataset from multiple file paths.

        Args:
            iter_filepath: An iterator over file paths.

        Returns:
            A merged xarray Dataset.
        """
        metadata = map(self.read_specs_getter, iter_filepath)
        datasets = map(self._get_dataset, metadata)
        return xr.merge(datasets)
