"""
Defines metadata structures and protocols for writing binary files.
"""

from collections.abc import Iterator
from typing import NamedTuple, Protocol

import xarray as xr


class WriteSpecs(NamedTuple):
    """
    Metadata for writing a portion of a DataArray to a binary file.

    Attributes
    ----------
    filename : str
        The name of the binary file.
    sub_array : xr.DataArray
        The portion of the DataArray to be written.
    """

    filename: str
    sub_array: xr.DataArray


class WriteSpecsGetterProtocol(Protocol):
    """
    Protocol for generating write specifications for a DataArray.
    """

    def __call__(self, data_array: xr.DataArray) -> Iterator[WriteSpecs]:
        """
        Generate write specifications for a DataArray.

        Parameters
        ----------
        data_array : xr.DataArray
            The data array for which to generate write specifications.

        Returns
        -------
        Iterator[WriteSpecs]
            An iterator over the write specifications.
        """
        ...
