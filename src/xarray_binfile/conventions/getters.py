"""
Ready-to-use conventions for common raw-binary file layouts.

Each class pairs a :class:`FilenamePattern` with a :class:`Layout` and exposes
``reader`` and ``writer`` methods that implement the read and write spec getter
protocols. They are intentionally small: when a project follows a different
convention, copy the closest class and change only what differs.
"""

import math
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import xarray as xr

from xarray_binfile.conventions.filename_pattern import FilenamePattern
from xarray_binfile.conventions.layout import Layout
from xarray_binfile.read.file_metadata import ReadSpecs
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


@dataclass(frozen=True)
class _ConventionBase:
    """
    Shared plumbing for the shipped conventions.

    Attributes:
        layout: Dimension order, coordinates and dtype of each file.
        pattern: Filename pattern, as a :class:`FilenamePattern` or template string.
        check_coords: Whether :meth:`writer` also requires coordinate values to
            match ``layout`` (dimensions and sizes are always checked).
    """

    layout: Layout
    pattern: FilenamePattern | str
    check_coords: bool = True

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
            ValueError: If the filename does not follow ``pattern``.
        """
        fields = self._pattern.parse(path.name)
        return ReadSpecs(
            filepath=path.resolve(),
            dtype=self.layout.dtype,
            coords=dict(self.layout.coords),
            name=str(fields["name"]),
        )

    def writer(self, data_array: xr.DataArray) -> Iterator[WriteSpecs]:
        """
        Yield the single write spec for ``data_array``.

        Args:
            data_array: The array to write, holding exactly the layout dimensions.

        Yields:
            One write spec.

        Raises:
            LayoutMismatchError: If the array does not fit ``layout``.
            ValueError: If the array has no name.
        """
        name = _require_name(data_array)
        self.layout.validate(data_array, check_coords=self.check_coords)
        yield WriteSpecs(
            filename=self._pattern.format(name=name),
            sub_array=self.layout.transpose(data_array),
            dtype=self.layout.dtype,
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
        time_dim: Name of the dimension split across files.
    """

    time_dim: str = "time"

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
            ValueError: If the filename does not follow ``pattern``.
        """
        fields = self._pattern.parse(path.name)
        time = np.atleast_1d(self._time_from_fields(fields))
        return ReadSpecs(
            filepath=path.resolve(),
            dtype=self.layout.dtype,
            coords=dict(self.layout.coords) | {self.time_dim: time},
            name=str(fields["name"]),
        )

    def writer(self, data_array: xr.DataArray) -> Iterator[WriteSpecs]:
        """
        Yield one write spec per value along ``time_dim``.

        All filenames are computed before the first spec is yielded, so
        collisions are reported before any file is written.

        Args:
            data_array: The array to write, holding the layout dimensions plus
                ``time_dim``.

        Yields:
            One write spec per value of ``time_dim``, in on-disk dimension order.

        Raises:
            LayoutMismatchError: If the array does not fit ``layout``.
            ValueError: If the array has no name, if a coordinate value or the
                name cannot be encoded in a filename the reader can parse, or
                if two values map to one file.
        """
        name = _require_name(data_array)
        self.layout.validate(
            data_array, extra_dims=(self.time_dim,), check_coords=self.check_coords
        )
        values = data_array[self.time_dim].values
        filenames = [
            self._pattern.format(name=name, **self._fields_from_time(value))
            for value in values
        ]
        _require_unique(filenames)
        for index, filename in enumerate(filenames):
            yield WriteSpecs(
                filename=filename,
                sub_array=self.layout.transpose(
                    data_array.isel({self.time_dim: index})
                ),
                dtype=self.layout.dtype,
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
        time_dim: Name of the dimension split across files. Defaults to ``"time"``.
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
        time_dim: Name of the dimension split across files. Defaults to ``"time"``.
    """

    pattern: FilenamePattern | str = "{name}-{time:.3f}.bin"

    _required_fields: ClassVar[frozenset[str]] = frozenset({"name", "time"})

    def _time_from_fields(self, fields: Mapping[str, Any]) -> Any:
        return float(fields["time"])

    def _fields_from_time(self, value: Any) -> dict[str, Any]:
        return {"time": float(value)}
