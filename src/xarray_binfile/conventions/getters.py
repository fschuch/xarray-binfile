"""
Ready-to-use conventions for common raw-binary file layouts.

Each class pairs a :class:`FilenamePattern` with a :class:`Layout` and exposes
``reader`` and ``writer`` methods that implement the read and write spec getter
protocols. They are intentionally small: when a project follows a different
convention, copy the closest class and change only what differs.
"""

import math
import os
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import xarray as xr

from xarray_binfile.conventions.filename_pattern import FilenamePattern
from xarray_binfile.conventions.layout import Layout, LayoutMismatchError
from xarray_binfile.conventions.stacking import (
    VariableStack,
    split_variables,
    stack_variables,
)
from xarray_binfile.read.file_metadata import ReadSpecs
from xarray_binfile.typing import DTypeLike
from xarray_binfile.write.file_metadata import WriteSpecs

_STEP_TOLERANCE = 1e-6


def _require_name(data_array: xr.DataArray) -> str:
    """
    Return the array name or fail with a clear message.

    Args:
        data_array: The array about to be written.

    Returns:
        The array name.

    Raises:
        ValueError: If the array has no name.
    """
    if data_array.name is None:
        error_message = (
            "The DataArray has no name, but the filename pattern needs one. "
            "Set data_array.name or use data_array.rename(...) before writing."
        )
        raise ValueError(error_message)
    return str(data_array.name)


def _require_unique(filenames: list[str]) -> None:
    """
    Fail if two slices would be written to the same file.

    Args:
        filenames: The planned output filenames.

    Raises:
        ValueError: If any filename appears more than once.
    """
    duplicated = sorted(name for name, count in Counter(filenames).items() if count > 1)
    if duplicated:
        error_message = (
            f"Several slices map to the same file(s) {duplicated}; the later ones "
            "would silently overwrite the earlier ones. Check for duplicated "
            "coordinate values or increase the precision of the filename pattern."
        )
        raise ValueError(error_message)


def open_files(
    paths: Iterable[Path], reader: Any, **open_mfdataset_kwargs: Any
) -> xr.Dataset:
    """
    Open binary files as one lazy dataset with the ``binfile`` engine.

    Thin wrapper over :func:`xarray.open_mfdataset` that fills in the engine
    and the read specs getter, so conventions can offer an ``open`` method.

    Args:
        paths: The files to open.
        reader: The read specs getter to decode them.
        **open_mfdataset_kwargs: Forwarded to :func:`xarray.open_mfdataset`,
            for example ``chunks`` or ``parallel``.

    Returns:
        The lazily opened dataset.

    Raises:
        FileNotFoundError: If ``paths`` is empty.
    """
    paths = sorted(paths)
    if not paths:
        error_message = "No file matches the convention; nothing to open."
        raise FileNotFoundError(error_message)
    return xr.open_mfdataset(
        paths, engine="binfile", read_specs_getter=reader, **open_mfdataset_kwargs
    )


@dataclass(frozen=True)
class _ConventionBase:
    """
    Shared plumbing for the shipped conventions.

    Attributes:
        layout: Dimension order, coordinates and dtype of each file.
        pattern: Filename pattern, as a :class:`FilenamePattern` or template string.
        check_coords: Whether :meth:`writer` also requires coordinate values to
            match ``layout`` (dimensions and sizes are always checked).
        stacks: Dimensions that are not stored in the files but encoded in
            the variable names, as :class:`VariableStack` declarations. For
            example ``VariableStack("i", "{name}{i}", values=("x", "y", "z"))``
            writes a velocity array ``u`` as ``ux``, ``uy`` and ``uz``, and
            ``VariableStack("n", "{name}{n:d}", values=range(1, 10))`` writes
            ``phi`` as ``phi1``, ``phi2``, ... On write, arrays are split in
            the order given, before the file split; on read, :meth:`stack`
            (called by :meth:`open` by default) rebuilds the stacked arrays.
        name_of: Optional hook returning the name to write an array under,
            in place of ``data_array.name``; for example
            ``lambda da: da.attrs["file_name"]``.
        names: Optional variable names the convention accepts. When set,
            :meth:`reader` and :meth:`files` ignore files whose parsed name
            is not listed, and :meth:`writer` refuses other arrays. Use it
            when the pattern alone is too permissive, for example a bare
            ``"{name}"`` template (no extension) that would otherwise match
            ``README`` or ``Makefile`` in the same folder.
    """

    layout: Layout
    pattern: FilenamePattern | str
    check_coords: bool = True
    stacks: Sequence[VariableStack] = ()
    name_of: Callable[[xr.DataArray], str] | None = None
    names: Sequence[str] | None = None

    _required_fields: ClassVar[frozenset[str]] = frozenset({"name"})

    def __post_init__(self) -> None:
        """
        Normalize ``pattern`` and check that it declares the required fields.

        Raises:
            ValueError: If the pattern misses a required field.
        """
        if isinstance(self.pattern, str):
            object.__setattr__(self, "pattern", FilenamePattern(self.pattern))
        assert isinstance(self.pattern, FilenamePattern)
        object.__setattr__(self, "stacks", tuple(self.stacks))
        if self.names is not None:
            object.__setattr__(self, "names", tuple(self.names))
        missing = self._required_fields - set(self.pattern.fields)
        if missing:
            error_message = (
                f"Pattern {self.pattern.template!r} must declare the field(s) "
                f"{sorted(missing)} for {type(self).__name__}."
            )
            raise ValueError(error_message)

    @property
    def _pattern(self) -> FilenamePattern:
        assert isinstance(self.pattern, FilenamePattern)
        return self.pattern

    def _parse(self, path: Path) -> dict[str, Any]:
        """
        Parse a filename and check the variable name against ``names``.

        Args:
            path: Path to the binary file.

        Returns:
            The parsed fields.

        Raises:
            ValueError: If the filename does not follow ``pattern`` or its
                name is not listed in ``names``.
        """
        fields = self._pattern.parse(path.name)
        if self.names is not None and fields["name"] not in self.names:
            error_message = (
                f"Variable {fields['name']!r} of {path.name} is not among the "
                f"names accepted by this convention: {list(self.names)}."
            )
            raise ValueError(error_message)
        return fields

    def _accepts(self, path: Path) -> bool:
        """
        Tell whether a file follows the convention, without reading it.

        Args:
            path: The file to test.

        Returns:
            True if the name matches ``pattern`` and, when set, ``names``.
        """
        try:
            self._parse(path)
        except ValueError:
            return False
        return True

    def _named(self, data_array: xr.DataArray) -> xr.DataArray:
        """
        Apply the ``name_of`` hook, returning an array that carries its output name.

        Args:
            data_array: The array about to be written.

        Returns:
            The same array, renamed when ``name_of`` is set.

        Raises:
            ValueError: If the array ends up without a name.
        """
        if self.name_of is not None:
            data_array = data_array.rename(self.name_of(data_array))
        name = _require_name(data_array)
        if self.names is not None and name.rpartition("/")[2] not in self.names:
            error_message = (
                f"Variable {name!r} is not among the names accepted by this "
                f"convention: {list(self.names)}."
            )
            raise LayoutMismatchError(error_message)
        return data_array

    def split(self, data_array: xr.DataArray) -> Iterator[xr.DataArray]:
        """
        Split ``data_array`` along every declared stacked dimension it carries.

        Args:
            data_array: A named array.

        Yields:
            One named array per combination of stacked values, in the order
            of ``stacks`` and of their values.
        """
        yield from split_variables(data_array, self.stacks)

    def stack(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Rebuild the stacked arrays declared in ``stacks`` from their split variables.

        Args:
            dataset: A dataset as returned by the ``binfile`` engine.

        Returns:
            The dataset with ``ux``, ``uy``, ``uz`` replaced by ``u`` and so on.
        """
        return stack_variables(dataset, self.stacks)

    def files(self, directory: str | os.PathLike[str]) -> list[Path]:
        """
        List the files in ``directory`` that follow the convention.

        Args:
            directory: The folder to scan (not recursively).

        Returns:
            The matching paths, sorted.
        """
        folder = Path(directory)
        return sorted(
            path for path in folder.iterdir() if path.is_file() and self._accepts(path)
        )

    def open(
        self,
        directory: str | os.PathLike[str],
        *,
        variables: Iterable[str] | None = None,
        stack: bool = True,
        **open_mfdataset_kwargs: Any,
    ) -> xr.Dataset:
        """
        Open every file in ``directory`` that follows the convention as one lazy dataset.

        Args:
            directory: The folder holding the files (not scanned recursively).
            variables: Optional names to keep, as found on disk (``"ux"``,
                not ``"u"``); other files are not opened.
            stack: Whether to rebuild the arrays declared in ``stacks``.
            **open_mfdataset_kwargs: Forwarded to :func:`xarray.open_mfdataset`,
                for example ``chunks={"time": 1}`` or ``parallel=True``.

        Returns:
            The lazily opened dataset, combined by coordinates.

        Raises:
            FileNotFoundError: If no file matches.
        """
        paths = self.files(directory)
        if variables is not None:
            wanted = set(variables)
            paths = [path for path in paths if self._parse(path)["name"] in wanted]
        dataset = open_files(paths, self.reader, **open_mfdataset_kwargs)
        return self.stack(dataset) if stack else dataset

    def reader(self, path: Path) -> ReadSpecs:  # no cov
        raise NotImplementedError

    def writer(self, data_array: xr.DataArray) -> Iterator[WriteSpecs]:  # no cov
        raise NotImplementedError


@dataclass(frozen=True)
class StaticFiles(_ConventionBase):
    """
    One file per variable, with no dimension split across files.

    Typical for fields that do not change in time, such as a mask or a
    geometry (``epsi.bin``). Reading attaches exactly the layout coordinates;
    writing produces a single file per DataArray.

    Examples:
        >>> import numpy as np
        >>> convention = StaticFiles(
        ...     layout=Layout({"x": np.arange(4), "y": np.arange(3)}, dtype="<f4"),
        ...     pattern="{name}.bin",
        ... )
        >>> specs = convention.reader(Path("epsi.bin"))
        >>> specs.name, specs.dims
        ('epsi', ('x', 'y'))

    Attributes:
        layout: Dimension order, coordinates and dtype of each file.
        pattern: Filename pattern declaring ``{name}``. Defaults to ``"{name}.bin"``.
        check_coords: Whether :meth:`writer` also requires coordinate values to
            match ``layout``.
        stacks: Dimensions encoded in the variable names, as :class:`VariableStack` declarations.
        name_of: Optional hook returning the name to write an array under.
        names: Optional variable names the convention accepts.
    """

    pattern: FilenamePattern | str = "{name}.bin"

    def reader(self, path: Path) -> ReadSpecs:
        """
        Build read specs for one static file.

        Args:
            path: Path to the binary file.

        Returns:
            The metadata required to decode ``path``.

        Raises:
            ValueError: If the filename does not follow ``pattern`` or its
                name is not listed in ``names``.
        """
        fields = self._parse(path)
        return ReadSpecs(
            filepath=path.resolve(),
            dtype=self.layout.dtype,
            coords=dict(self.layout.coords),
            name=str(fields["name"]),
            order=self.layout.order,
            coord_attrs=self.layout.coord_attrs,
        )

    def writer(self, data_array: xr.DataArray) -> Iterator[WriteSpecs]:
        """
        Yield the write specs for ``data_array``, one per stacked value.

        Args:
            data_array: The array to write, holding exactly the layout
                dimensions plus any stacked dimension.

        Yields:
            One write spec per output file.

        Raises:
            LayoutMismatchError: If the array does not fit ``layout``.
            ValueError: If the array has no name.
        """
        for piece in self.split(self._named(data_array)):
            yield from self._write_one(piece)

    def _write_one(self, data_array: xr.DataArray) -> Iterator[WriteSpecs]:
        name = _require_name(data_array)
        self.layout.validate(data_array, check_coords=self.check_coords)
        yield WriteSpecs(
            filename=self._pattern.format(name=name),
            sub_array=self.layout.transpose(data_array),
            dtype=self.layout.dtype,
            order=self.layout.order,
        )


@dataclass(frozen=True)
class _SplitAlongDimension(_ConventionBase):
    """
    Base for conventions writing one file per value of ``time_dim``.

    Attributes:
        layout: Dimension order, coordinates and dtype of each file.
        pattern: Filename pattern.
        check_coords: Whether :meth:`writer` also requires coordinate values to
            match ``layout``.
        stacks: Dimensions encoded in the variable names, as :class:`VariableStack` declarations.
        name_of: Optional hook returning the name to write an array under.
        names: Optional variable names the convention accepts.
        time_dim: Name of the dimension split across files.
        time_dtype: Optional dtype for the ``time_dim`` coordinate on read,
            for example ``np.float32`` to match single-precision data. When
            ``None`` the natural type is kept (``int64`` steps, ``float64``
            times).
    """

    time_dim: str = "time"
    time_dtype: DTypeLike | None = None

    def _time_from_fields(self, fields: Mapping[str, Any]) -> Any:
        raise NotImplementedError  # no cov

    def _fields_from_time(self, value: Any) -> dict[str, Any]:
        raise NotImplementedError  # no cov

    def reader(self, path: Path) -> ReadSpecs:
        """
        Build read specs for one file of the sequence.

        The layout coordinates are extended with a single-value ``time_dim``
        coordinate decoded from the filename.

        Args:
            path: Path to the binary file.

        Returns:
            The metadata required to decode ``path``.

        Raises:
            ValueError: If the filename does not follow ``pattern`` or its
                name is not listed in ``names``.
        """
        fields = self._parse(path)
        time = np.atleast_1d(
            np.asarray(self._time_from_fields(fields), dtype=self.time_dtype)
        )
        return ReadSpecs(
            filepath=path.resolve(),
            dtype=self.layout.dtype,
            coords=dict(self.layout.coords) | {self.time_dim: time},
            name=str(fields["name"]),
            order=self.layout.order,
            coord_attrs=self.layout.coord_attrs,
        )

    def writer(self, data_array: xr.DataArray) -> Iterator[WriteSpecs]:
        """
        Yield one write spec per value along ``time_dim`` (and per stacked value).

        All filenames of a slice are computed before its first spec is
        yielded, so collisions are reported before any file is written.

        Args:
            data_array: The array to write, holding the layout dimensions plus
                ``time_dim`` and any stacked dimension.

        Yields:
            One write spec per output file, in on-disk dimension order.

        Raises:
            LayoutMismatchError: If the array does not fit ``layout``.
            ValueError: If the array has no name, if a coordinate value or the
                name cannot be encoded in a filename the reader can parse, or
                if two values map to one file.
        """
        for piece in self.split(self._named(data_array)):
            yield from self._write_one(piece)

    def _write_one(self, data_array: xr.DataArray) -> Iterator[WriteSpecs]:
        name = _require_name(data_array)
        self.layout.validate(
            data_array, extra_dims=(self.time_dim,), check_coords=self.check_coords
        )
        values = (
            data_array[self.time_dim].values
            if self.time_dim in data_array.coords
            else np.arange(data_array.sizes[self.time_dim])
        )
        filenames = [
            self._pattern.format(name=name, **self._fields_from_time(value))
            for value in values
        ]
        _require_unique(filenames)
        for index, filename in enumerate(filenames):
            yield WriteSpecs(
                filename=filename,
                sub_array=self.layout.transpose(
                    data_array.isel({self.time_dim: index}, drop=True)
                ),
                dtype=self.layout.dtype,
                order=self.layout.order,
            )


@dataclass(frozen=True)
class StepIndexedFiles(_SplitAlongDimension):
    """
    One file per snapshot, numbered with an integer step counter.

    This is the most common solver convention, for example ``ux-0001.bin``,
    ``ux.0002`` or ``ux0003.dat``. The ``time`` coordinate is derived from the
    step: it is the step itself by default, or ``step * time_step`` when the
    interval between snapshots is known. Writing inverts that mapping and
    refuses coordinate values that do not land on an integer step.

    Examples:
        >>> import numpy as np
        >>> convention = StepIndexedFiles(
        ...     layout=Layout({"x": np.arange(4), "y": np.arange(3)}),
        ...     pattern="{name}-{step:04d}.bin",
        ...     time_step=0.5,
        ... )
        >>> specs = convention.reader(Path("ux-0007.bin"))
        >>> specs.name, specs.coords["time"]
        ('ux', array([3.5]))
        >>> da = xr.DataArray(
        ...     np.zeros((4, 3, 2)),
        ...     coords={"x": np.arange(4), "y": np.arange(3), "time": [1.0, 1.5]},
        ...     name="ux",
        ... )
        >>> [spec.filename for spec in convention.writer(da)]
        ['ux-0002.bin', 'ux-0003.bin']

    Attributes:
        layout: Dimension order, coordinates and dtype of each file.
        pattern: Filename pattern declaring ``{name}`` and ``{step}``.
            Defaults to ``"{name}-{step:04d}.bin"``.
        check_coords: Whether :meth:`writer` also requires coordinate values to
            match ``layout``.
        stacks: Dimensions encoded in the variable names, as :class:`VariableStack` declarations.
        name_of: Optional hook returning the name to write an array under.
        names: Optional variable names the convention accepts.
        time_dim: Name of the dimension split across files. Defaults to ``"time"``.
        time_dtype: Optional dtype for the ``time`` coordinate on read.
        time_step: Interval between consecutive steps. When ``None`` the
            ``time`` coordinate is the integer step itself.
    """

    pattern: FilenamePattern | str = "{name}-{step:04d}.bin"
    time_step: float | None = None

    _required_fields: ClassVar[frozenset[str]] = frozenset({"name", "step"})

    def _time_from_fields(self, fields: Mapping[str, Any]) -> Any:
        step = np.int64(fields["step"])  # explicit width, stable across platforms
        if self.time_step is None:
            return step
        return step * self.time_step

    def _fields_from_time(self, value: Any) -> dict[str, Any]:
        step = float(value) if self.time_step is None else float(value) / self.time_step
        rounded = round(step)
        if not math.isclose(step, rounded, rel_tol=0.0, abs_tol=_STEP_TOLERANCE):
            hint = (
                "Set time_step to the interval between snapshots"
                if self.time_step is None
                else f"Check that time_step={self.time_step} matches the data"
            )
            error_message = (
                f"Coordinate value {value} on {self.time_dim!r} does not "
                f"correspond to an integer step (got {step}). {hint}, or use "
                "TimeStampedFiles to encode the value itself in the filename."
            )
            raise ValueError(error_message)
        return {"step": int(rounded)}


@dataclass(frozen=True)
class TimeStampedFiles(_SplitAlongDimension):
    """
    One file per snapshot, named with the physical time of the snapshot.

    Some solvers write ``ux-0.250.bin`` instead of a step counter. The ``time``
    coordinate is the float parsed from the filename, so what is on disk is
    the single source of truth. Writing formats each coordinate value with the
    pattern precision and refuses arrays whose values collide at that precision.

    Examples:
        >>> import numpy as np
        >>> convention = TimeStampedFiles(
        ...     layout=Layout({"x": np.arange(4)}),
        ...     pattern="{name}-{time:.3f}.bin",
        ... )
        >>> convention.reader(Path("ux-0.250.bin")).coords["time"]
        array([0.25])

    Attributes:
        layout: Dimension order, coordinates and dtype of each file.
        pattern: Filename pattern declaring ``{name}`` and ``{time}``.
            Defaults to ``"{name}-{time:.3f}.bin"``.
        check_coords: Whether :meth:`writer` also requires coordinate values to
            match ``layout``.
        stacks: Dimensions encoded in the variable names, as :class:`VariableStack` declarations.
        name_of: Optional hook returning the name to write an array under.
        names: Optional variable names the convention accepts.
        time_dim: Name of the dimension split across files. Defaults to ``"time"``.
        time_dtype: Optional dtype for the ``time`` coordinate on read.
    """

    pattern: FilenamePattern | str = "{name}-{time:.3f}.bin"

    _required_fields: ClassVar[frozenset[str]] = frozenset({"name", "time"})

    def _time_from_fields(self, fields: Mapping[str, Any]) -> Any:
        return float(fields["time"])

    def _fields_from_time(self, value: Any) -> dict[str, Any]:
        return {"time": float(value)}
