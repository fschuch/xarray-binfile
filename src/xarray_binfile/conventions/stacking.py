"""
Declare dimensions that live in variable names rather than in the files.

Solvers often store a vector field as one file per component (``ux``, ``uy``,
``uz``) or a set of scalar fractions as numbered variables (``phi1``,
``phi2``). A :class:`VariableStack` describes that encoding once: the
dimension it stands for, the name template, and the values it may take.
Conventions use it to :meth:`~VariableStack.split` an array into per-file
slices on write, and to :meth:`~VariableStack.stack` the variables back into
one array on read.
"""

import string
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from xarray_binfile.conventions.filename_pattern import FilenamePattern


def _template_fields(template: str) -> set[str]:
    """
    Return the field names declared by a ``str.format`` template.

    Args:
        template: The template to inspect.

    Returns:
        The set of field names.
    """
    return {
        name for _, name, _, _ in string.Formatter().parse(template) if name is not None
    }


@dataclass(frozen=True)
class VariableStack:
    """
    A dimension encoded in variable names instead of in the files.

    Examples:
        >>> import numpy as np
        >>> velocity = VariableStack("i", "{name}{i}", values=("x", "y", "z"))
        >>> scalars = VariableStack("n", "{name}{n:d}", values=range(1, 10))
        >>> u = xr.DataArray(
        ...     np.zeros((3, 2)), coords={"i": ["x", "y", "z"], "x": [0, 1]}, name="u"
        ... )
        >>> [piece.name for piece in velocity.split(u)]
        ['ux', 'uy', 'uz']
        >>> ds = xr.Dataset({name: ("x", np.zeros(2)) for name in ("ux", "uy", "pp")})
        >>> stacked = velocity.stack(ds)
        >>> sorted(stacked.data_vars), stacked["i"].values.tolist()
        (['pp', 'u'], ['x', 'y'])

    Attributes:
        dim: Name of the stacked dimension (``"i"``).
        template: ``str.format`` template building a variable name from
            ``{name}`` and ``{dim}`` (``"{name}{i}"``, ``"{name}{n:d}"``). It
            is parsed with :class:`FilenamePattern`, so integer values can use
            zero-fill (``"{name}{n:02d}"``).
        values: The values the dimension may take, in the order the stacked
            coordinate should follow (``("x", "y", "z")``, ``range(1, 10)``).
            Splitting refuses coordinate values outside this set, so every
            file written can be stacked back. Required because templates
            without a separator match almost any name (``pp`` fits
            ``"{name}{i}"`` as ``p`` + ``p``).
        names: Optional base names to restrict stacking to (``("u",)``).
            When ``None``, every group of matching variables is stacked.
    """

    dim: str
    template: str
    values: Sequence[Any]
    names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        """
        Validate the template and freeze ``values``.

        Raises:
            ValueError: If ``template`` does not declare exactly ``{name}``
                and ``{dim}``, or if ``values`` is empty.
        """
        if _template_fields(self.template) != {"name", self.dim}:
            error_message = (
                f"Template {self.template!r} must declare exactly the fields "
                f"{{'name', {self.dim!r}}}."
            )
            raise ValueError(error_message)
        values = tuple(self.values)
        if not values:
            error_message = f"VariableStack for {self.dim!r} needs at least one value."
            raise ValueError(error_message)
        object.__setattr__(self, "values", values)
        if self.names is not None:
            object.__setattr__(self, "names", tuple(self.names))

    @property
    def pattern(self) -> FilenamePattern:
        """
        The template as a :class:`FilenamePattern`, for parsing variable names.

        Returns:
            The pattern.
        """
        return FilenamePattern(self.template)

    def split(self, data_array: xr.DataArray) -> Iterator[xr.DataArray]:
        """
        Yield one named slice per value along ``dim``.

        Arrays without ``dim`` are yielded unchanged. The slice name is built
        from ``data_array.name`` and the coordinate value (or the position,
        when ``dim`` has no coordinate).

        Args:
            data_array: A named array.

        Yields:
            The slices, with ``dim`` dropped.

        Raises:
            ValueError: If the array has no name, or if a coordinate value is
                not listed in ``values``.
        """
        if self.dim not in data_array.dims:
            yield data_array
            return
        if data_array.name is None:
            error_message = (
                f"Cannot split the unnamed array along {self.dim!r}: the template "
                f"{self.template!r} needs a name."
            )
            raise ValueError(error_message)
        coordinate = (
            data_array[self.dim].values
            if self.dim in data_array.coords
            else np.arange(data_array.sizes[self.dim])
        )
        values = [value.item() if hasattr(value, "item") else value for value in coordinate]
        unknown = [value for value in values if value not in self.values]
        if unknown:
            error_message = (
                f"Coordinate value(s) {unknown} on {self.dim!r} are not listed in "
                f"the VariableStack values {list(self.values)}; files written for "
                "them could not be stacked back."
            )
            raise ValueError(error_message)
        for index, value in enumerate(values):
            piece = data_array.isel({self.dim: index}, drop=True)
            yield piece.rename(
                self.template.format(name=data_array.name, **{self.dim: value})
            )

    def stack(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Combine the variables encoding ``dim`` into one array per base name.

        Variables whose name parses with ``template`` and whose decoded value
        is in ``values`` are grouped by base name and concatenated along a new
        dimension ``dim``, whose coordinate holds the values found, in the
        order of ``values``. Everything else is left untouched. The operation
        is lazy for Dask-backed data.

        Args:
            dataset: The dataset holding the split variables.

        Returns:
            A new dataset where each group is replaced by its stacked array.
        """
        pattern = self.pattern
        groups: dict[str, dict[Any, str]] = {}
        for variable in map(str, dataset.data_vars):
            if not pattern.matches(variable):
                continue
            fields = pattern.parse(variable)
            name, value = str(fields["name"]), fields[self.dim]
            if value not in self.values:
                continue
            if self.names is not None and name not in self.names:
                continue
            groups.setdefault(name, {})[value] = variable

        result = dataset
        for name, members in groups.items():
            found = [value for value in self.values if value in members]
            stacked = xr.concat(
                [dataset[members[value]] for value in found],
                dim=xr.DataArray(found, dims=self.dim, name=self.dim),
            )
            result = result.drop_vars(list(members.values())).assign({name: stacked})
        return result


def split_variables(
    data_array: xr.DataArray, stacks: Sequence[VariableStack]
) -> Iterator[xr.DataArray]:
    """
    Split an array along every stacked dimension it carries, in order.

    Args:
        data_array: A named array.
        stacks: The stacks to apply; the first one produces the outer loop.

    Yields:
        Named slices free of every stacked dimension.
    """
    if not stacks:
        yield data_array
        return
    first, rest = stacks[0], stacks[1:]
    for piece in first.split(data_array):
        yield from split_variables(piece, rest)


def stack_variables(dataset: xr.Dataset, stacks: Sequence[VariableStack]) -> xr.Dataset:
    """
    Stack a dataset along every declared dimension, undoing :func:`split_variables`.

    Stacks are applied in reverse order, so the innermost split is rebuilt first.

    Args:
        dataset: The dataset holding the split variables.
        stacks: The stacks that were used to split.

    Returns:
        The stacked dataset.
    """
    for stack in reversed(stacks):
        dataset = stack.stack(dataset)
    return dataset
