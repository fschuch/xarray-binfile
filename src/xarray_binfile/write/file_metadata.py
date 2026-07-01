"""
Defines metadata structures and protocols for writing binary files.
"""

from collections.abc import Iterator
from typing import NamedTuple, Protocol

import xarray as xr


class WriteSpecs(NamedTuple):
    """
    Immutable write instruction for one output binary file.

    Each item defines both the output filename and the exact DataArray slice to
    serialize into that file.

    Attributes:
        filename: The name of the binary file.
        sub_array: The portion of the DataArray to be written.
    """

    filename: str
    sub_array: xr.DataArray


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
