"""
Defines a backend array for reading binary files in Xarray.
"""

import numpy as np
import xarray as xr
from xarray.backends import BackendArray
from xarray.core import indexing

from xarray_binfile.read.file_metadata import ReadSpecs


def _is_coord_sliced(size: int, slice_spec: slice | int) -> bool:
    """
    Check if a coordinate is sliced.

    A slice counts as not sliced only when it selects every element of the
    axis, in order, however it is spelled: ``slice(None)``, ``slice(0, 1000)``
    and ``slice(-size, None)`` all do, while an empty ``slice(0, 0)`` does not
    (https://github.com/fschuch/xarray-binfile/issues/62). Two ``range``
    objects compare equal when they hold the same elements, so the check
    needs no special cases and builds no list.

    Args:
        size: Size of the coordinate.
        slice_spec: Slice specification or integer index.

    Returns:
        True if the coordinate is sliced, False otherwise.
    """
    if not isinstance(slice_spec, slice):
        return True
    return range(size)[slice_spec] != range(size)


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

        Raises:
            ValueError: If the file size does not match the size expected
                from the metadata shape and dtype.
            FileNotFoundError: If the binary file does not exist.
        """
        self.metadata = metadata

        # Attributes required by BackendArray
        self.dtype = np.dtype(self.metadata.dtype)
        self.shape = self.metadata.shape

        self._validate_file_size()

    def _validate_file_size(self) -> None:
        """
        Ensure the file size matches the metadata shape and dtype.

        Raw binary files carry no metadata, so a size mismatch is the earliest
        possible signal that the declared shape, dtype, or byte order is wrong.
        Failing fast here avoids confusing downstream errors or silently
        misread data.

        Raises:
            ValueError: If the file size does not match the expected size.
        """
        expected_bytes = int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize
        actual_bytes = self.metadata.filepath.stat().st_size
        if actual_bytes != expected_bytes:
            error_message = (
                f"Size mismatch for {self.metadata.filepath}: file has "
                f"{actual_bytes} bytes, but the read specs expect "
                f"{expected_bytes} bytes (shape {self.shape}, dtype {self.dtype}). "
                "Check the shape, dtype, and byte order declared by the read specs getter."
            )
            raise ValueError(error_message)

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
            indexing_support=indexing.IndexingSupport.BASIC,
            raw_indexing_method=self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple[slice | int, ...]) -> np.typing.ArrayLike:
        """
        Performs raw indexing on the binary file.

        Args:
            key: Tuple of slices or integers specifying the indices to read.

        Returns:
            The data read from the binary file.
        """
        with open(self.metadata.filepath, "rb") as file:
            return self._read_binary_at_slices(file, key)

    def _is_sliced(self, key: tuple[slice | int, ...]) -> bool:
        """
        Checks if the key is a slice of the original array.

        Args:
            key: Tuple of slices or integers specifying the indices to check.

        Returns:
            True if the key is a slice, False otherwise.
        """

        return any(
            _is_coord_sliced(s, k)
            for k, s in zip(key, self.metadata.shape, strict=True)
        )

    def _wrap_numpy_fromfile(self, file) -> np.typing.NDArray:
        """
        Reads the entire binary file into a NumPy array.

        Args:
            file: The binary file to read.

        Returns:
            The data read from the file.
        """
        return np.fromfile(
            file, dtype=self.metadata.dtype, count=np.prod(self.metadata.shape)
        ).reshape(self.metadata.shape, order=self.metadata.order)

    def _wrap_numpy_memmap(
        self, file, key: tuple[slice | int, ...]
    ) -> np.typing.NDArray:
        """
        Reads a portion of the binary file using memory mapping.

        Only the selected region is read, and it is copied out of the mapping,
        so the returned array owns its memory: it does not change if the file
        is rewritten, and it keeps neither the mapping nor a file descriptor
        alive (https://github.com/fschuch/xarray-binfile/issues/61).

        Args:
            file: The binary file to read.
            key: Tuple of slices or integers specifying the indices to read.

        Returns:
            The data read from the file, as a new array.
        """
        memory_map = np.memmap(
            file,
            dtype=self.metadata.dtype,
            mode="r",
            shape=self.metadata.shape,
            order=self.metadata.order,
        )
        # np.array copies; np.asarray would return a view of the mapped file
        return np.array(memory_map[key])

    def _read_binary_at_slices(
        self, file, key: tuple[slice | int, ...]
    ) -> np.typing.NDArray:
        """
        Reads a binary file at specific locations based on the key tuple of slices.

        Args:
            file: The binary file to read.
            key: Tuple of slices or integers specifying the indices to read.

        Returns:
            The array data read from the file at the specified slices.
        """
        if self._is_sliced(key):
            return self._wrap_numpy_memmap(file, key)
        # The key selects everything, so indexing returns a view of the array
        # just read, at no cost. It still applies the key, so a key wrongly
        # taken as whole yields a slower read instead of the wrong data.
        return self._wrap_numpy_fromfile(file)[key]

    def get_xarray_dataset(self) -> xr.Dataset:
        """
        Converts the backend array to an Xarray Dataset.

        Returns:
            The Xarray Dataset representation of the backend array.
        """
        dataset = xr.Dataset(
            data_vars={
                self.metadata.name: (
                    self.metadata.dims,
                    indexing.LazilyIndexedArray(self),
                )
            },
            coords=self.metadata.coords,
            attrs=self.metadata.attrs,
        )
        for coord, attrs in (self.metadata.coord_attrs or {}).items():
            if coord in dataset.coords:
                dataset[coord].attrs.update(attrs)
        return dataset
