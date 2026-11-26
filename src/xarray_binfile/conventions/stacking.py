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

import re
import string
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import xarray as xr


def coordinate_or_index(data_array: xr.DataArray, dim: str) -> np.ndarray:
    """
    The values along ``dim``: its coordinate, or positions when it has none.

    Args:
        data_array: The array.
        dim: A dimension of the array.

    Returns:
        The coordinate values, or ``arange(size)``.
    """
    if dim in data_array.coords:
        return np.asarray(data_array[dim].values)
    return np.arange(data_array.sizes[dim])


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
            ``{name}`` and ``{dim}`` (``"{name}{i}"``, ``"{name}{n:d}"``).
            Format specifications apply both ways, so integer values can use
            zero-fill (``"{name}{n:02d}"``). Without it, longer values are
            tried first, so ``phi11`` reads as ``phi`` + ``11``; for a base
            name that ends in a digit itself (``phi1`` + ``1``), list it in
            ``names`` or use zero-fill.
        values: The values the dimension may take, in the order the stacked
            coordinate should follow (``("x", "y", "z")``, ``range(1, 10)``).
            Splitting refuses coordinate values outside this set, so every
            file written can be stacked back. Required because templates
            without a separator match almost any name (``pp`` fits
            ``"{name}{i}"`` as ``p`` + ``p``).
        names: Optional base names to restrict stacking to (``{"u"}``), any
            collection (not a bare string), kept as a ``frozenset``. When
            ``None``, every group of at least ``min_components`` matching
            variables is stacked.
        attrs: Optional attributes attached to the ``dim`` coordinate when
            stacking (for example ``{"long_name": "velocity component"}``).
        min_components: How many components a group needs before it is
            stacked. Defaults to 1 when ``names`` lists the fields explicitly
            and to 2 otherwise, so that a lone ``vorticity`` is not read as
            ``vorticit`` + ``y`` when guessing.
    """

    dim: str
    template: str
    values: Sequence[Any]
    names: Collection[str] | None = None
    attrs: Mapping[str, Any] | None = None
    min_components: int | None = None

    def __post_init__(self) -> None:
        """
        Validate the template and freeze ``values``.

        Raises:
            ValueError: If ``template`` does not declare exactly ``{name}``
                and ``{dim}``, or if ``values`` is empty.
            TypeError: If ``names`` is a bare string.
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
        if isinstance(self.names, str):
            error_message = (
                f"names must be a collection of names, not the string {self.names!r}."
            )
            raise TypeError(error_message)
        if self.names is not None:
            object.__setattr__(self, "names", frozenset(self.names))
        if self.min_components is None:
            object.__setattr__(
                self, "min_components", 1 if self.names is not None else 2
            )

    @cached_property
    def _matchers(self) -> tuple[tuple[Any, re.Pattern[str]], ...]:
        """
        One regular expression per allowed value, with the value spelled out.

        ``{name}`` and ``{dim}`` are usually adjacent (``"{name}{i}"``), so a
        generic pattern cannot tell where the name ends; matching the value
        literally (``vort`` + ``x``) removes the ambiguity. Longer values come
        first, so that without zero padding ``phi11`` is tried as ``phi`` +
        ``11`` before ``phi1`` + ``1``.

        Returns:
            Pairs of value and compiled pattern capturing ``name``.
        """
        parsed = list(string.Formatter().parse(self.template))
        matchers = []
        for value in self.values:
            parts = []
            spelled = ""
            for literal, field_name, spec, _ in parsed:
                parts.append(re.escape(literal))
                if field_name == "name":
                    parts.append(r"(?P<name>\w+)")
                elif field_name == self.dim:
                    spelled = format(value, spec or "")
                    parts.append(re.escape(spelled))
            matchers.append((len(spelled), value, re.compile("".join(parts))))
        matchers.sort(key=lambda matcher: -matcher[0])
        return tuple((value, pattern) for _, value, pattern in matchers)

    def _bases(self, variable: str) -> Iterator[tuple[Any, str]]:
        """
        Yield every accepted reading of ``variable``, longest value first.

        Args:
            variable: A variable name.

        Yields:
            Pairs of value and base name that ``names`` accepts.
        """
        for value, matcher in self._matchers:
            match = matcher.fullmatch(variable)
            if match is None:
                continue
            name = match.group("name")
            if self.names is None or name in self.names:
                yield value, name

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
        values = coordinate_or_index(data_array, self.dim).tolist()
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

        Raises:
            ValueError: If a group's base name is already a variable of the
                dataset, which would silently overwrite it.
        """
        return self._stack(dataset)

    def _stack(
        self, dataset: xr.Dataset, prefer: Callable[[str], bool] | None = None
    ) -> xr.Dataset:
        """
        :meth:`stack`, preferring the readings whose base name ``prefer`` accepts.

        Args:
            dataset: The dataset holding the split variables.
            prefer: Optional test on a base name. A variable read more than
                one way (``phi11`` as ``phi`` + ``11`` or ``phi1`` + ``1``)
                takes the first reading it accepts, or the longest value
                when it accepts none.

        Returns:
            A new dataset where each group is replaced by its stacked array.

        Raises:
            ValueError: If a group's base name is already a variable of the
                dataset, which would silently overwrite it.
        """
        groups = self._groups(dataset, prefer)
        to_drop: list[str] = []
        stacked: dict[str, xr.DataArray] = {}
        for name, members in groups.items():
            if len(members) < (self.min_components or 0):
                continue
            if name in dataset.variables:
                error_message = (
                    f"Cannot stack {sorted(members.values())} into {name!r}: a variable "
                    f"{name!r} already exists in the dataset. Rename it, restrict "
                    "``names``, or open with stack=False."
                )
                raise ValueError(error_message)
            stacked[name] = self._concat(dataset, members)
            to_drop.extend(members.values())
        return dataset.drop_vars(to_drop).assign(stacked)

    def _groups(
        self, dataset: xr.Dataset, prefer: Callable[[str], bool] | None = None
    ) -> dict[str, dict[Any, str]]:
        """
        Group the variables encoding ``dim`` by base name.

        Args:
            dataset: The dataset holding the split variables.
            prefer: Optional test on a base name, see :meth:`_stack`.

        Returns:
            ``{base name: {value: variable name}}`` for the accepted names.
        """
        groups: dict[str, dict[Any, str]] = {}
        for variable in map(str, dataset.data_vars):
            readings = list(self._bases(variable))
            if not readings:
                continue
            preferred = (r for r in readings if prefer is not None and prefer(r[1]))
            value, name = next(preferred, readings[0])
            groups.setdefault(name, {})[value] = variable
        return groups

    def _concat(self, dataset: xr.Dataset, members: Mapping[Any, str]) -> xr.DataArray:
        """
        Concatenate one group along a new ``dim`` coordinate, in ``values`` order.

        Args:
            dataset: The dataset holding the split variables.
            members: ``{value: variable name}`` for the group.

        Returns:
            The stacked array.
        """
        found = [value for value in self.values if value in members]
        return xr.concat(
            [dataset[members[value]] for value in found],
            dim=xr.DataArray(
                found, dims=self.dim, name=self.dim, attrs=dict(self.attrs or {})
            ),
        )


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


def stacked_names(variable: str, stacks: Sequence[VariableStack]) -> set[str]:
    """
    Every name ``variable`` could end up under once the stacks are applied.

    Stacks are undone in reverse order, as :func:`stack_variables` does, and
    every accepted reading is kept, so ``uy2`` gives ``{"uy2", "uy", "u"}``
    with a velocity stack followed by a numbered one.

    Args:
        variable: A variable name as found on disk.
        stacks: The stacks that were used to split.

    Returns:
        ``variable`` and the base names it may decode into.
    """
    names = {variable}
    for stack in reversed(stacks):
        names |= {base for name in names for _, base in stack._bases(name)}
    return names


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


def stack_listed(
    dataset: xr.Dataset, stacks: Sequence[VariableStack], names: Collection[str]
) -> xr.Dataset:
    """
    :func:`stack_variables`, preferring readings that lead to one of ``names``.

    A convention restricted to ``names=("phi1",)`` writes ``phi11`` for
    ``phi1`` at ``n=1``. Read on its own, ``phi11`` is ``phi`` + ``11``; here
    ``phi1`` + ``1`` wins because ``phi1`` is listed. A base name counts when
    it is listed or when the stacks still to apply turn it into a listed
    name (``ux`` for ``u``). Variables with no such reading are stacked as
    :func:`stack_variables` would.

    Args:
        dataset: The dataset holding the split variables.
        stacks: The stacks that were used to split.
        names: The names the variables are expected to end up under.

    Returns:
        The stacked dataset.
    """
    for index in reversed(range(len(stacks))):
        outer = stacks[:index]

        def prefer(base: str, outer: Sequence[VariableStack] = outer) -> bool:
            return any(name in names for name in stacked_names(base, outer))

        dataset = stacks[index]._stack(dataset, prefer)
    return dataset
