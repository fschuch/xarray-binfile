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
    Immutable metadata contract for interpreting one binary file.

    A raw binary file does not carry enough semantic information to be decoded
    safely by itself. This object provides that missing context so the backend
    can expose the data as a labeled xarray variable.

    Notes:
        - Dimension order is inferred from ``coords`` key order.
        - ``shape`` is inferred from coordinate lengths.
        - ``dtype`` should be explicit about byte order when portability matters
          (for example ``"<f4"`` for little-endian float32).

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
    Structural protocol for read spec getter implementations.

    Any callable matching ``(path: Path) -> ReadSpecs`` can be used as a read
    specs getter. No inheritance is required.

    Typical responsibilities:
        - Parse filename conventions.
        - Resolve variable identity (for example variable name, step index).
        - Provide explicit dtype, coordinates, and optional attrs.
    """

    def __call__(self, path: Path) -> ReadSpecs:
        """
        Generate read specifications for a binary file.

        Args:
            path: Path to the binary file.

        Returns:
            The metadata required by the binary backend to decode ``path``.
        """
        ...
