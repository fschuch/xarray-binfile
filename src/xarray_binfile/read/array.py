"""
Defines a backend array for reading binary files in Xarray.
"""

import numpy as np
import xarray as xr
from xarray.backends import BackendArray
from xarray.core import indexing

from xarray_binfile.read.file_metadata import ReadSpecs


def _is_coord_sliced(size: int, slice_spec: slice) -> bool:
    """
    Check if a coordinate is sliced.
    Args:
        size: Size of the coordinate.
        slice_spec: Slice specification.
    Returns:
        True if the coordinate is sliced, False otherwise.
    """
    return any(
        (
            (slice_spec.start or 0) != 0,
            (slice_spec.stop or size) != size,
            (slice_spec.step or 1) != 1,
        ),
    )


class BinaryEngineBackendArray(BackendArray):
    """
    Backend array for reading binary files in Xarray.

    Attributes:
        metadata: Metadata describing the binary file.
        dtype: Data type of the array.
        shape: Shape of the array.
    """

    def __init__(self, metadata: ReadSpecs):
        """
        Initializes the backend array.

        Args:
            metadata: Metadata describing the binary file.
        """
        self.metadata = metadata

        # Attributes required by BackendArray
        self.dtype = self.metadata.dtype
        self.shape = self.metadata.shape

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        """
        Retrieves data from the array using explicit indexing.

        Args:
            key: Indexing key specifying the data to retrieve.

        Returns:
            The retrieved data.
        """
        return indexing.explicit_indexing_adapter(
            key=key,
            shape=self.metadata.shape,
            indexing_support=indexing.IndexingSupport.VECTORIZED,
            raw_indexing_method=self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple[slice, ...]) -> np.typing.ArrayLike:
        """
        Performs raw indexing on the binary file.

        Args:
            key: Tuple of slices specifying the indices to read.

        Returns:
            The data read from the binary file.
        """
        with open(self.metadata.filepath, "rb") as file:
            return self._read_binary_at_slices(file, key)

    def _is_sliced(self, key: tuple[slice, ...]) -> bool:
        """
        Checks if the key is a slice of the original array.

        Args:
            key: Tuple of slices specifying the indices to check.

        Returns:
            True if the key is a slice, False otherwise.
        """

        return any(_is_coord_sliced(s, k) for k, s in zip(key, self.metadata.shape, strict=True))

    def _wrap_numpy_fromfile(self, file) -> np.typing.NDArray:
        """
        Reads the entire binary file into a NumPy array.

        Args:
            file: The binary file to read.

        Returns:
            The data read from the file.
        """
        return np.fromfile(file, dtype=self.metadata.dtype, count=np.prod(self.metadata.shape)).reshape(
            self.metadata.shape
        )

    def _wrap_numpy_memmap(self, file, key: tuple[slice, ...]) -> np.typing.NDArray:
        """
        Reads a portion of the binary file using memory mapping.

        Args:
            file: The binary file to read.
            key: Tuple of slices specifying the indices to read.

        Returns:
            The data read from the file.
        """
        memory_map = np.memmap(file, dtype=self.metadata.dtype, mode="r", shape=self.metadata.shape, order="C")
        return np.asarray(memory_map[key])  # ensure we actually read the data

    def _read_binary_at_slices(self, file, key: tuple[slice, ...]) -> np.typing.NDArray:
        """
        Reads a binary file at specific locations based on the key tuple of slices.

        Args:
            file: The binary file to read.
            key: Tuple of slices specifying the indices to read.

        Returns:
            The array data read from the file at the specified slices.
        """
        if self._is_sliced(key):
            return self._wrap_numpy_memmap(file, key)
        return self._wrap_numpy_fromfile(file)

    def get_xarray_dataset(self) -> xr.Dataset:
        """
        Converts the backend array to an Xarray Dataset.

        Returns:
            The Xarray Dataset representation of the backend array.
        """
        return xr.Dataset(
            data_vars={self.metadata.name: (self.metadata.dims, indexing.LazilyIndexedArray(self))},
            coords=self.metadata.coords,
            attrs=self.metadata.attrs,
        )
