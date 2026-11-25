"""
The base class shared by every convention, with the defaults a composite relies on.

:class:`~xarray_binfile.conventions.protocol.ConventionProtocol` is the minimal
contract (``reader`` and ``writer``). :class:`Convention` builds the rest on top
of it: discovering files, decoding a variable name, stacking and opening a
folder. Shipped conventions and composites inherit it; any object that only has
``reader`` and ``writer`` is wrapped through :meth:`Convention.adapt` when it
joins a composite, so the defaults apply to it as well.
"""

import os
from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from xarray_binfile.conventions.protocol import ConventionProtocol
from xarray_binfile.read.file_metadata import ReadSpecs, ReadSpecsGetterProtocol
from xarray_binfile.write.file_metadata import WriteSpecs


def open_files(
    paths: Iterable[Path], reader: Any, **open_mfdataset_kwargs: Any
) -> xr.Dataset:
    """
    Open binary files as one lazy dataset with the ``binfile`` engine.

    Wrapper over :func:`xarray.open_mfdataset` that fills in the engine and
    the read specs getter. Before opening, it reads every file's specs and
    refuses files that cannot be combined into one variable (see
    :func:`_require_one_variable_per_name`); xarray only sees one file at a
    time, so it cannot tell.

    Args:
        paths: The files to open.
        reader: The read specs getter to decode them.
        **open_mfdataset_kwargs: Forwarded to :func:`xarray.open_mfdataset`,
            for example ``chunks`` or ``parallel``.

    Returns:
        The lazily opened dataset.

    Raises:
        FileNotFoundError: If ``paths`` is empty.
        ValueError: If one variable name comes from files with different
            dimensions, or from two files at the same coordinates.
    """
    paths = sorted(paths)
    if not paths:
        error_message = "No file matches the convention; nothing to open."
        raise FileNotFoundError(error_message)
    if not isinstance(reader, _SpecsCache):
        reader = _SpecsCache(paths, reader)
    _require_one_variable_per_name(reader.read_specs())
    return xr.open_mfdataset(
        paths, engine="binfile", read_specs_getter=reader, **open_mfdataset_kwargs
    )


def _require_one_variable_per_name(specs: Iterable[ReadSpecs]) -> None:
    """
    Refuse files that share a variable name but cannot be one variable.

    A :class:`~xarray_binfile.conventions.FolderConventions` reads
    ``xy_planes/ux-0001.bin`` and ``3d/ux-0001.bin`` both as ``ux``
    (https://github.com/fschuch/xarray-binfile/issues/68). With different
    dimensions, xarray fails with an error that does not name the clash; at
    the same coordinates, it silently keeps one of the files. Files of one
    name with the same dimensions at different coordinates, such as one run
    split in time across two folders, are still combined.

    Args:
        specs: The read specs of the files to open.

    Raises:
        ValueError: If one name comes from files with different dimensions,
            or from two files at the same coordinates.
    """
    by_name: defaultdict[str, list[ReadSpecs]] = defaultdict(list)
    for spec in specs:
        by_name[spec.name].append(spec)
    advice = "Give them distinct names, or open them separately."
    for name, group in by_name.items():
        by_dims = {frozenset(spec.dims): spec for spec in group}
        if len(by_dims) > 1:
            examples = "; ".join(
                f"{spec.filepath} {spec.dims}" for spec in by_dims.values()
            )
            error_message = (
                f"The variable {name!r} comes from files with different "
                f"dimensions: {examples}. {advice}"
            )
            raise ValueError(error_message)
        at: dict[Hashable, Path] = {}
        for spec in group:
            where = _coordinates_key(spec)
            if where in at:
                error_message = (
                    f"The files {at[where]} and {spec.filepath} both hold "
                    f"{name!r} at the same coordinates, so only one could be "
                    f"kept. {advice}"
                )
                raise ValueError(error_message)
            at[where] = spec.filepath


def _coordinates_key(spec: ReadSpecs) -> Hashable:
    """Every coordinate's values, comparable across files whatever their order."""
    return tuple(
        sorted(
            (dim, tuple(np.asarray(values).tolist()))
            for dim, values in spec.coords.items()
        )
    )


def _slab_chunks(specs: Iterable[ReadSpecs]) -> dict[str, str]:
    """
    Chunks that split the files into slabs along their slowest axis on disk.

    Since Dask 2024.12.0, Dask reads every chunk a selection touches in full,
    so with one chunk per file, reading a single value reads the whole file
    (https://github.com/fschuch/xarray-binfile/issues/63). Slabs along the
    slowest axis are contiguous in the file, and ``"auto"`` lets Dask size
    them by its ``array.chunk-size`` setting, so a file that fits in one chunk
    keeps one chunk. Only one ``chunks`` mapping applies to every file, so
    the largest file decides the axis; the dimensions not named stay whole,
    and files without that dimension are not split. Dask cannot size an
    ``"auto"`` chunk on an array that holds nothing, so when any file is
    empty, no file is split.

    Args:
        specs: The read specs of the files to open.

    Returns:
        ``{axis: "auto"}`` for the slowest axis longer than one element of the
        largest file, or ``{}`` (one chunk per file) when there is none or
        when any file is empty.
    """
    specs = list(specs)
    if any(0 in spec.shape for spec in specs):
        return {}
    largest = max(
        specs,
        key=lambda s: int(np.prod(s.shape)) * np.dtype(s.dtype).itemsize,
        default=None,
    )
    if largest is None:
        return {}
    sizes = dict(zip(largest.dims, largest.shape, strict=True))
    slow_to_fast = largest.dims if largest.order == "C" else largest.dims[::-1]
    axis = next((dim for dim in slow_to_fast if sizes[dim] > 1), None)
    return {} if axis is None else {axis: "auto"}


class _SpecsCache:
    """
    Each file's specs, read once, for planning the chunks and for opening it.

    xarray passes absolute paths, so the cache is keyed by them, and an
    unknown path falls back to ``reader``. A file the reader rejects keeps
    its error, raised again when xarray opens it, so ``errors=`` applies to
    it as usual.
    """

    def __init__(self, paths: Iterable[Path], reader: ReadSpecsGetterProtocol):
        self._reader = reader
        self._specs: dict[Path, ReadSpecs | Exception] = {}
        for path in paths:
            try:
                self._specs[self._key(path)] = reader(path)
            except Exception as error:  # noqa: BLE001
                self._specs[self._key(path)] = error

    @staticmethod
    def _key(path: Path) -> Path:
        return Path(os.path.abspath(path))

    def read_specs(self) -> list[ReadSpecs]:
        """The specs of the files the reader accepted."""
        return [s for s in self._specs.values() if not isinstance(s, Exception)]

    def __call__(self, path: Path) -> ReadSpecs:
        cached = self._specs.get(self._key(path))
        if cached is None:
            return self._reader(path)
        if isinstance(cached, Exception):
            raise cached
        return cached


def scan_files(directory: str | os.PathLike[str], accepts: Any) -> list[Path]:
    """
    List the regular files in ``directory`` that ``accepts`` approves of.

    The cheap name test runs before the ``is_file`` stat, so unrelated files
    (notes, XDMF indexes, backups) cost nothing more than a regex match.

    Args:
        directory: The folder to scan (not recursively).
        accepts: Predicate on the candidate path, usually a filename check.

    Returns:
        The accepted paths, sorted.
    """
    with os.scandir(directory) as entries:
        return sorted(
            Path(entry.path)
            for entry in entries
            if accepts(Path(entry.path)) and entry.is_file()
        )


class Convention:
    """
    Base class for conventions: ``reader`` and ``writer`` plus sensible defaults.

    Subclasses implement :meth:`reader` and :meth:`writer`. The defaults
    derive everything else from them, and can be overridden when a cheaper
    or smarter implementation exists (the shipped conventions decode names
    from the filename alone, for instance).
    """

    def reader(self, path: Path) -> ReadSpecs:  # no cov
        raise NotImplementedError

    def writer(self, data_array: xr.DataArray) -> Iterator[WriteSpecs]:  # no cov
        raise NotImplementedError

    def name_of_file(self, path: Path) -> str:
        """
        The variable name stored in ``path``, without reading it.

        Args:
            path: A binary file.

        Returns:
            The decoded variable name.

        Raises:
            ValueError: If the convention does not accept ``path``.
        """
        return self.reader(path).name

    def accepts(self, path: Path) -> bool:
        """
        Tell whether the convention can decode ``path``.

        Args:
            path: The candidate file.

        Returns:
            True if :meth:`name_of_file` succeeds.
        """
        try:
            self.name_of_file(path)
        except ValueError:
            return False
        return True

    def files(self, directory: str | os.PathLike[str]) -> list[Path]:
        """
        List the files in ``directory`` that the convention accepts.

        Args:
            directory: The folder to scan (not recursively).

        Returns:
            The matching paths, sorted.
        """
        return scan_files(directory, self.accepts)

    def stack(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Rebuild the arrays that the convention splits across variables.

        The default keeps the dataset unchanged.

        Args:
            dataset: A dataset as returned by the ``binfile`` engine.

        Returns:
            The dataset, stacked when the convention declares stacks.
        """
        return dataset

    def open(
        self,
        directory: str | os.PathLike[str],
        *,
        variables: Iterable[str] | None = None,
        stack: bool = True,
        **open_mfdataset_kwargs: Any,
    ) -> xr.Dataset:
        """
        Open every accepted file in ``directory`` as one lazy dataset.

        Since Dask 2024.12.0, Dask reads every chunk a selection touches in
        full, and holds it in memory while it is processed, so the chunks
        decide what a selection costs. Without ``chunks``, a file larger than
        Dask's ``array.chunk-size`` (128 MiB by default) is split into slabs
        along its slowest axis on disk, which are contiguous in the file; a
        smaller file stays one chunk. One ``chunks`` applies to every file, so
        the largest file picks the axis, and files without that dimension
        stay whole, whatever their size. When any file is empty, no file is
        split. Set ``chunks`` to choose for your access
        pattern: smaller chunks make points and planes cheaper to read, and
        larger ones mean fewer tasks when every value is needed.

        Args:
            directory: The folder holding the files.
            variables: Optional names to keep, as found on disk (``"ux"``, not
                ``"u"``); other files are not opened.
            stack: Whether to rebuild the stacked arrays.
            **open_mfdataset_kwargs: Forwarded to :func:`xarray.open_mfdataset`,
                for example ``chunks={"x": 16}`` or ``parallel=True``.

        Returns:
            The lazily opened dataset, combined by coordinates.

        Raises:
            FileNotFoundError: If no file matches.
            ValueError: If one variable name comes from files with different
                dimensions, or from two files at the same coordinates, which
                xarray could not combine or would silently reduce to one.
        """
        paths = self.files(directory)
        if variables is not None:
            wanted = set(variables)
            paths = [path for path in paths if self.name_of_file(path) in wanted]
        reader = _SpecsCache(paths, self.reader)
        if "chunks" not in open_mfdataset_kwargs:
            chunks = _slab_chunks(reader.read_specs())
            if chunks:
                open_mfdataset_kwargs["chunks"] = chunks
        dataset = open_files(paths, reader, **open_mfdataset_kwargs)
        return self.stack(dataset) if stack else dataset

    @classmethod
    def adapt(cls, convention: ConventionProtocol) -> "Convention":
        """
        Give any ``reader``/``writer`` object the defaults of this class.

        Args:
            convention: A :class:`Convention`, returned as is, or any object
                following :class:`ConventionProtocol`.

        Returns:
            A :class:`Convention`.
        """
        if isinstance(convention, Convention):
            return convention
        return _Adapted(convention)


class _Adapted(Convention):
    """A bare ``reader``/``writer`` object with the default behaviour on top."""

    def __init__(self, convention: ConventionProtocol) -> None:
        self._convention = convention

    def reader(self, path: Path) -> ReadSpecs:
        return self._convention.reader(path)

    def writer(self, data_array: xr.DataArray) -> Iterator[WriteSpecs]:
        return self._convention.writer(data_array)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._convention!r})"
