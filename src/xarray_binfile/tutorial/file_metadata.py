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
    Reference implementation of both read and write spec getter protocols.

    This helper is intended for tutorials, tests, and simple projects with a
    filename convention that encodes variable name and step index.

    It is intentionally narrow: the default regex expects time-indexed files
    (for example ``ux-0001.bin``). For mixed layouts that include static files
    (for example ``epsi.bin``), implement a custom getter that still follows
    the same read/write protocol contracts.

    It implements:
        - ``reader(path) -> ReadSpecs``
        - ``writer(data_array) -> Iterator[WriteSpecs]``

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
        Build ``ReadSpecs`` from a file path.

        The implementation parses ``path.name`` using ``filename_regex`` and
        appends a single-value ``time`` coordinate based on the extracted step
        number.

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
        Yield ``WriteSpecs`` for one DataArray.

        The default behavior writes one file per time coordinate value using
        ``filename_template`` and transposes each slice to ``base_coords``
        dimension order before serialization.

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
        Return the canonical base dimension order.

        Returns:
            A tuple of base dimension names.
        """
        return tuple(self.base_coords.keys())
