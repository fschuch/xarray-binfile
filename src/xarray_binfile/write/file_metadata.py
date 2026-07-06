"""
Defines metadata structures and protocols for writing binary files.
"""

from collections.abc import Iterator
from typing import NamedTuple, Protocol

import xarray as xr

from xarray_binfile.typing import DTypeLike


class WriteSpecs(NamedTuple):
    """
    Immutable write instruction for one output binary file.

    Each item defines both the output filename and the exact DataArray slice to
    serialize into that file.

    Attributes:
        filename: The name of the binary file.
        sub_array: The portion of the DataArray to be written.
        dtype: Optional on-disk data type. When set, the sub-array is cast to
            this dtype (including byte order, for example ``"<f4"``) right
            before serialization. When ``None``, the in-memory dtype and
            native byte order are written as-is, so make sure they match what
            your read specs getter declares.
    """

    filename: str
    sub_array: xr.DataArray
    dtype: DTypeLike | None = None


class WriteSpecsGetterProtocol(Protocol):
    """
    Structural protocol for write spec getter implementations.

    Any callable matching
    ``(data_array: xr.DataArray) -> Iterator[WriteSpecs]`` can be used as a
    write specs getter. No inheritance is required.

    Typical responsibilities:
        - Define filename conventions.
        - Split a DataArray into one or more per-file slices.
        - Ensure each slice is transposed to the expected on-disk dimension order.
        - Keep each per-file slice small enough to fit in memory, since every
          ``sub_array`` is fully materialized (Dask compute) before its file
          is written in a single pass.
    """

    def __call__(self, data_array: xr.DataArray) -> Iterator[WriteSpecs]:
        """
        Generate write instructions for a DataArray.

        Args:
            data_array: The data array for which to generate write specifications.

        Returns:
            An iterator over per-file write instructions.
        """
        ...
