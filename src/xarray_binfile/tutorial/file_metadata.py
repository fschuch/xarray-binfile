"""
Defines utilities for generating file metadata for reading and writing binary files.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from xarray import DataArray

from xarray_binfile.read.file_metadata import ReadSpecs
from xarray_binfile.typing import ArrayLike, DTypeLike
from xarray_binfile.write.file_metadata import WriteSpecs


@dataclass(frozen=True)
class FileSpecsGetter:
    """
    Generates file metadata for reading and writing binary files.

    Attributes:
        base_coords: Base coordinates for the data.
        dtype: Data type of the binary file. Defaults to np.float64.
        filename_template: Template for generating filenames.
        filename_regex: Regular expression for parsing filenames.
    """

    base_coords: dict[str, ArrayLike]
    dtype: DTypeLike = np.float64
    filename_template: str = "{name}-{digits:04}.bin"
    filename_regex: re.Pattern = re.compile(r"(?P<name>\w+)-(?P<digits>\d{4})\.bin")

    def reader(self, path: Path) -> ReadSpecs:
        """
        Generate read specifications for a binary file.

        Args:
            path: Path to the binary file.

        Returns:
            ReadSpecs: Metadata for reading the binary file.

        Raises:
            ValueError: If the filename does not match the expected pattern.
        """
        match = self.filename_regex.match(path.name)
        if not match:
            error_message = f"Invalid filename: {path.name}"
            raise ValueError(error_message)

        name, digits = match.groups()
        time = np.array([int(digits)], dtype=np.int64)

        return ReadSpecs(
            filepath=path.resolve(),
            dtype=self.dtype,
            coords=self.base_coords | {"time": time},
            name=name,
        )

    def writer(self, data_array: DataArray) -> Iterator[WriteSpecs]:
        """
        Generate write specifications for a DataArray.

        Args:
            data_array: The data array to generate write specifications for.

        Returns:
            An iterator over write specifications.
        """
        for time in data_array.coords["time"]:
            yield WriteSpecs(
                filename=self.filename_template.format(name=data_array.name, digits=int(time)),
                sub_array=data_array.sel(time=time).transpose(*self._base_dims, missing_dims="raise"),
            )

    @cached_property
    def _base_dims(self) -> tuple[str, ...]:
        """
        Get the base dimensions from the coordinates.

        Returns:
            A tuple of base dimension names.
        """
        return tuple(self.base_coords.keys())
