"""Type hints shared across the xarray_binfile package."""

import typing

import numpy.typing

# TODO: Type aliases could look better on the docs https://github.com/sphinx-doc/sphinx/issues/10785#issuecomment-1897551241

ArrayLike = numpy.typing.ArrayLike
DTypeLike = numpy.typing.DTypeLike
AttributesLike: typing.TypeAlias = typing.Mapping[typing.Any, typing.Any]
CoordsLike: typing.TypeAlias = typing.Mapping[str, numpy.typing.ArrayLike]
# Memory layout of an array on disk: "C" (last axis varies fastest) or "F"
# (first axis varies fastest).
MemoryOrder: typing.TypeAlias = typing.Literal["C", "F"]
