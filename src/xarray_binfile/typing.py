"""Module for type hints used in the binary_engine_xarray package."""

import typing

import numpy.typing

# TODO: Type aliases could look better on the docs https://github.com/sphinx-doc/sphinx/issues/10785#issuecomment-1897551241

ArrayLike = numpy.typing.ArrayLike
"""Type alias for an array-like object. Basically anything accepted by `numpy.array`."""

DTypeLike = numpy.typing.DTypeLike
"""Type alias for a data type. Basically anything accepted by `numpy.dtype`."""

AttributesLike: typing.TypeAlias = typing.Mapping[typing.Any, typing.Any]
"""Type alias for attributes of a dataset or array. Basically anything accepted by `xarray.Dataset.attrs`."""

CoordsLike: typing.TypeAlias = typing.Mapping[str, numpy.typing.ArrayLike]
"""Type alias for coordinates of a dataset or variable. Basically anything accepted by `xarray.Dataset.coords`."""
