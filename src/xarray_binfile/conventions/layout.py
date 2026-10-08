"""
On-disk array layouts shared by the shipped conventions.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import xarray as xr

from xarray_binfile.typing import ArrayLike, AttributesLike, DTypeLike, MemoryOrder


class LayoutMismatchError(ValueError):
    """Raised when a DataArray does not fit the on-disk layout of a convention."""


@dataclass(frozen=True)
class Layout:
    """
    Dimension order, coordinate values and dtype of the arrays stored on disk.

    Raw binary files carry none of this information, so a layout is the shared
    contract between the read and write sides of a convention: readers attach
    these coordinates to every file they decode, and writers refuse arrays that
    do not match them instead of guessing.

    Attributes:
        coords: Mapping of dimension name to coordinate values, in the order
            the dimensions are declared. Which of them varies fastest on disk
            is set by ``order``.
        dtype: On-disk data type. Be explicit about byte order when files move
            between machines (for example ``"<f4"``).
        order: Memory layout of each file. ``"C"`` (default) means the last
            dimension in ``coords`` varies fastest, as NumPy's ``tofile`` and
            C programs write. ``"F"`` means the first dimension varies
            fastest, as Fortran programs (2DECOMP&FFT, Xcompact3d) write.
            ``Layout({"x": x, "y": y, "z": z}, order="F")`` describes the
            same bytes as ``Layout({"z": z, "y": y, "x": x})``; pick the one
            that keeps your dimension names in their natural order.
        coord_attrs: Optional attributes attached to each coordinate on read,
            keyed by dimension name (for example ``{"x": {"units": "m"}}``).
    """

    coords: Mapping[str, ArrayLike]
    dtype: DTypeLike = np.float64
    order: MemoryOrder = "C"
    coord_attrs: Mapping[str, AttributesLike] | None = None

    @cached_property
    def dims(self) -> tuple[str, ...]:
        """
        Dimension names in on-disk order.

        Returns:
            The dimension names.
        """
        return tuple(self.coords.keys())

    @cached_property
    def shape(self) -> tuple[int, ...]:
        """
        Array shape derived from the coordinate lengths.

        Returns:
            The shape of one array on disk.
        """
        return tuple(np.size(values) for values in self.coords.values())

    def validate(
        self,
        data_array: xr.DataArray,
        *,
        extra_dims: Iterable[str] = (),
        check_coords: bool = True,
    ) -> None:
        """
        Check that ``data_array`` can be written with this layout.

        Args:
            data_array: The array about to be written.
            extra_dims: Dimensions the array may have on top of the layout
                dimensions, typically the dimension that is split across files.
            check_coords: Also require the coordinate values to be equal, not
                only the dimension names and sizes.

        Raises:
            LayoutMismatchError: If the dimensions, sizes or coordinate values
                of ``data_array`` differ from the layout.
        """
        expected = set(self.dims) | set(extra_dims)
        actual = set(data_array.dims)
        if expected != actual:
            error_message = (
                f"Dimension mismatch: the array has dims {tuple(data_array.dims)} "
                f"but the layout expects {self.dims} plus {tuple(extra_dims)}."
            )
            raise LayoutMismatchError(error_message)
        for dim, expected_values in self.coords.items():
            expected_size = np.size(expected_values)
            if data_array.sizes[dim] != expected_size:
                error_message = (
                    f"Size mismatch on dimension {dim!r}: the array has "
                    f"{data_array.sizes[dim]} elements but the layout expects "
                    f"{expected_size}."
                )
                raise LayoutMismatchError(error_message)
            if check_coords and dim in data_array.coords:
                actual_values = data_array.coords[dim].values
                if not np.array_equal(actual_values, np.asarray(expected_values)):
                    error_message = (
                        f"Coordinate mismatch on dimension {dim!r}: the array "
                        "coordinates differ from the layout coordinates. Pass "
                        "check_coords=False to skip this check."
                    )
                    raise LayoutMismatchError(error_message)

    def transpose(self, data_array: xr.DataArray) -> xr.DataArray:
        """
        Reorder ``data_array`` to the on-disk dimension order.

        Args:
            data_array: An array holding exactly the layout dimensions.

        Returns:
            The transposed array.
        """
        return data_array.transpose(*self.dims, missing_dims="raise")
