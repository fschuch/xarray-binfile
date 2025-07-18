"""
Defines metadata structures and protocols for reading binary files.
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Protocol

from xarray_binfile.typing import AttributesLike, CoordsLike, DTypeLike


@dataclass(frozen=True)
class ReadSpecs:
    """
    Metadata for reading a binary file.

    Attributes:
        filepath: Path to the binary file.
        dtype: Data type of the binary file.
        coords: Coordinates of the data in the binary file.
        name: Name of the dataset or variable.
        attrs: Additional attributes for the dataset or variable.
    """

    filepath: Path
    dtype: DTypeLike
    coords: CoordsLike
    name: str
    attrs: AttributesLike | None = None

    @cached_property
    def shape(self) -> tuple[int, ...]:
        """
        Gets the shape of the data based on the coordinates.

        Returns:
            Shape of the data.
        """
        return tuple(len(i) for i in self.coords.values())

    @cached_property
    def dims(self) -> tuple[str, ...]:
        """
        Gets the dimension names of the data.

        Returns:
            Dimension names.
        """
        return tuple(self.coords.keys())


class ReadSpecsGetterProtocol(Protocol):
    """
    Protocol for generating read specifications for a binary file.
    """

    def __call__(self, path: Path) -> ReadSpecs:
        """
        Generates read specifications for a binary file.

        Args:
            path: Path to the binary file.

        Returns:
            Metadata for reading the binary file.
        """
        ...
