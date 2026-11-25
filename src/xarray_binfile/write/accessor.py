"""
Provides accessors for writing xarray Dataset and DataArray objects to binary files.
"""

import contextlib
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import xarray as xr

from xarray_binfile.write.file_metadata import WriteSpecs, WriteSpecsGetterProtocol

# Wraps the iterator of write specs, for example ``tqdm`` or
# ``functools.partial(tqdm, desc="ux")``.
Progress = Callable[[Iterable[WriteSpecs]], Iterable[WriteSpecs]]


@xr.register_dataset_accessor("binary_engine")
class BinaryEngineDataset:
    """
    An accessor with extra utilities for xarray.Dataset.
    """

    def __init__(self, data_set: xr.Dataset):
        """
        Initializes the BinaryEngineDataset accessor.

        Args:
            data_set: The dataset to attach the accessor to.
        """
        self._data_set = data_set

    def to_file(
        self,
        write_specs_getter: WriteSpecsGetterProtocol,
        directory: str | os.PathLike[str] | None = None,
        *,
        progress: Progress | None = None,
    ) -> None:
        """
        Writes the dataset to binary files.

        Every data variable is delegated to
        :meth:`BinaryEngineDataArray.to_file`, so the same eager, atomic,
        whole-file write semantics apply: each output file is fully
        materialized in memory (triggering a Dask compute for lazy variables),
        written in a single pass, and moved into place only once complete.
        See that method for guidance on sizing files and on alternatives when
        streaming or partial writes are needed.

        The write specifications of every variable are collected first, and
        the call is refused before any file is written when two of them map
        to the same file, for example a stacked ``u`` whose x component is
        ``ux-0000.bin`` next to a variable ``ux``, or ``U`` and ``u`` where
        the filesystem ignores case
        (https://github.com/fschuch/xarray-binfile/issues/67).

        The variables are then written one after the other, so the warning
        there about writing over the inputs also covers one variable computed
        from another (https://github.com/fschuch/xarray-binfile/issues/64).

        Args:
            write_specs_getter: A callable that generates write specifications for the data arrays.
            directory: The directory where the binary files will be written. Defaults to the current working directory.
            progress: Optional wrapper applied to the write specs of each
                variable, for example ``tqdm``, to report progress.

        Raises:
            FileNotFoundError: If a relative filename is used and ``directory`` does not exist.
            ValueError: If a relative filename escapes ``directory``, or if
                two write specifications map to the same file.
        """
        _directory = Path(directory) if directory is not None else Path.cwd()
        planned = [
            list(write_specs_getter(data_array))
            for data_array in self._data_set.data_vars.values()
        ]
        _require_distinct_files(
            _directory, [spec for specs in planned for spec in specs]
        )
        for specs in planned:
            _write_specs(_directory, specs, progress)


@xr.register_dataarray_accessor("binary_engine")
class BinaryEngineDataArray:
    """
    An accessor with extra utilities for xarray.DataArray.
    """

    def __init__(self, data_array: xr.DataArray):
        """
        Initializes the BinaryEngineDataArray accessor.

        Args:
            data_array: The data array to attach the accessor to.
        """
        self._data_array = data_array

    def to_file(
        self,
        write_specs_getter: WriteSpecsGetterProtocol,
        directory: str | os.PathLike[str] | None = None,
        *,
        progress: Progress | None = None,
    ) -> None:
        """
        Writes the data array to binary files.

        Writes are eager and whole-file only. For each write specification,
        the entire ``sub_array`` is loaded into memory (triggering a Dask
        compute for lazy data) and the target file is written in full, in a
        single pass. There is no partial, appending, or resuming write mode:
        re-writing a file always replaces its whole content instead of trying
        to guess or patch existing bytes, which avoids leaving files in a
        partially-updated, corrupted state.

        Writes are also atomic per file: the bytes are first serialized into
        a temporary file created next to the destination (so the final move
        stays on the same filesystem), and each file is moved to its final
        path with :func:`os.replace` only once it is complete. An interrupted
        write never leaves a truncated file at the destination, and the
        temporary file is removed automatically.

        .. warning::

            **Writing a lazy result over its own inputs can give wrong
            numbers, without an error.** Each file is computed separately,
            and each compute reads the lazy inputs again. When a file depends
            on more than its own input, such as an anomaly
            ``ux - ux.mean("time")``, and it is written into the folder it was
            read from, later files are computed from inputs already replaced
            (https://github.com/fschuch/xarray-binfile/issues/64). A result
            that only needs its own file, such as ``ux * 2``, is safe. To
            write over the inputs, either:

            - compute the result first, with ``result = result.compute()``,
              which costs the whole result in memory; or
            - write to another folder and move the files over the originals
              afterwards, which costs the whole result on disk.

        Plan the write specifications so that every individual output file
        fits comfortably in memory, for example by splitting the array into
        one file per time step. If you need streaming, incremental, or
        partial writes, prefer one of the other file formats supported by
        xarray, such as NetCDF or Zarr.

        Each file is written with the in-memory dtype and native byte order,
        unless the write specification sets ``dtype``, in which case the
        values are cast right before serialization. Bytes are laid out in
        the memory order declared by ``WriteSpecs.order``: ``"C"`` (last
        axis fastest, the default) or ``"F"`` (first axis fastest).

        A relative ``WriteSpecs.filename`` is resolved against ``directory``,
        which must already exist, and may contain sub-folders (for example
        ``"3d/ux-0001.bin"``), which are created on demand. It must stay
        inside ``directory``: paths escaping it through ``..`` are rejected
        (symbolic links inside ``directory`` are not followed for this check,
        so a linked sub-folder is fine). An absolute filename is used as is,
        which lets one call target several locations.

        Every write specification is collected before the first file is
        written, and the call is refused when two of them map to the same
        file, including paths that differ only in case where the filesystem
        ignores case (https://github.com/fschuch/xarray-binfile/issues/67).
        An error from ``write_specs_getter`` therefore leaves nothing
        written. The specs hold their ``sub_array`` until written, which is
        free for the shipped conventions (views of the array), but a custom
        getter that computes each ``sub_array`` keeps all of them in memory
        at once. Each call is independent: an existing file is replaced.

        Args:
            write_specs_getter: A callable that generates write specifications for the data array.
            directory: The base directory for relative filenames. Defaults to the current working directory.
            progress: Optional wrapper applied to the collected write specs
                before files are written, for example ``tqdm`` or
                ``functools.partial(tqdm, desc="ux")``, to report progress.
                It is iterated as the files are written.

        Raises:
            FileNotFoundError: If a relative filename is used and ``directory`` does not exist.
            ValueError: If a relative filename escapes ``directory``, or if
                two write specifications map to the same file.
        """
        _directory = Path(directory) if directory is not None else Path.cwd()
        specs = list(write_specs_getter(self._data_array))
        _require_distinct_files(_directory, specs)
        _write_specs(_directory, specs, progress)


def _write_specs(
    directory: Path, specs: Iterable[WriteSpecs], progress: Progress | None
) -> None:
    """
    Write every spec to its file, one after the other.

    Args:
        directory: Base directory for relative filenames.
        specs: The write specs, already checked for colliding files.
        progress: Optional wrapper applied to ``specs``.
    """
    if progress is not None:
        specs = progress(specs)
    for details in specs:
        final_file = _resolve_destination(directory, details.filename)
        new_type = (
            details.dtype if details.dtype is not None else details.sub_array.dtype
        )
        final_file.parent.mkdir(parents=True, exist_ok=True)
        # ``tofile`` always serializes in C order, so Fortran order is
        # obtained by flattening first. ``astype`` lays the data out in the
        # target order, so a cast and a reorder cost a single copy.
        values = details.sub_array.values.astype(
            new_type, order=details.order, copy=False
        )
        _write_atomically(final_file, values.ravel(order=details.order))


def _require_distinct_files(directory: Path, specs: list[WriteSpecs]) -> None:
    """
    Fail if two specs would be written to the same file.

    Paths are compared once made absolute, so a relative and an absolute
    filename for one file collide. Paths that differ only in case collide
    too where the filesystem ignores case (the default on macOS and Windows).

    Args:
        directory: Base directory for relative filenames.
        specs: Every write spec of one ``to_file`` call.

    Raises:
        ValueError: If two specs map to the same file.
    """
    paths = [
        os.path.abspath(_resolve_destination(directory, spec.filename))
        for spec in specs
    ]
    duplicated = sorted(path for path, count in Counter(paths).items() if count > 1)
    if duplicated:
        error_message = (
            f"Several write specs map to the same file(s) {duplicated}; the later "
            "ones would silently overwrite the earlier ones. Rename the variables "
            "or stacked components, or change the filename pattern."
        )
        raise ValueError(error_message)
    by_folded_case: defaultdict[str, list[str]] = defaultdict(list)
    for path in paths:
        by_folded_case[path.casefold()].append(path)
    clashes = [
        sorted(group)
        for group in by_folded_case.values()
        if len(group) > 1 and _ignores_case(Path(group[0]).parent)
    ]
    if clashes:
        error_message = (
            f"The files {clashes} differ only in case, and the filesystem there "
            "ignores case, so they would be the same file and the later ones "
            "would silently overwrite the earlier ones. Rename the variables."
        )
        raise ValueError(error_message)


def _ignores_case(folder: Path) -> bool:
    """
    Whether the filesystem holding ``folder`` ignores case in filenames.

    A temporary file is created in ``folder``, or in its nearest existing
    parent when it does not exist yet, and looked up with its case swapped.

    Args:
        folder: The folder files will be written to.

    Returns:
        True if a name and its case-swapped twin are the same file there.
    """
    existing = folder
    while not existing.exists():
        existing = existing.parent
    handle, name = tempfile.mkstemp(dir=existing, prefix=".binary_engine-CaseProbe-")
    os.close(handle)
    try:
        probe = Path(name)
        return probe.with_name(probe.name.swapcase()).exists()
    finally:
        os.unlink(name)


def _resolve_destination(directory: Path, filename: str | os.PathLike[str]) -> Path:
    """
    Resolve a write spec filename to its final destination.

    Args:
        directory: Base directory for relative filenames.
        filename: Absolute path, or path relative to ``directory``.

    Returns:
        The absolute destination path.

    Raises:
        FileNotFoundError: If ``filename`` is relative and ``directory`` does not exist.
        ValueError: If a relative ``filename`` escapes ``directory``.
    """
    path = Path(filename)
    if path.is_absolute():
        return path
    # Lexical normalisation only: ``..`` segments are collapsed without
    # following symbolic links, so a linked sub-folder inside ``directory``
    # is accepted while anything climbing above ``directory`` is rejected.
    normalized = Path(os.path.normpath(path))
    if normalized.parts[:1] == ("..",):
        error_message = (
            f"WriteSpecs.filename {str(filename)!r} escapes the output directory "
            f"{str(directory)!r}. Use an absolute path to write elsewhere."
        )
        raise ValueError(error_message)
    if not directory.is_dir():
        error_message = f"Output directory does not exist: {str(directory)!r}"
        raise FileNotFoundError(error_message)
    return directory / normalized


def _write_atomically(final_file: Path, values: np.ndarray) -> None:
    """
    Serialize ``values`` to a temporary sibling file and move it into place.

    Args:
        final_file: Destination path; its parent directory must exist.
        values: Array to serialize with :meth:`numpy.ndarray.tofile`.
    """
    handle, temporary_name = tempfile.mkstemp(
        dir=final_file.parent, prefix=f".{final_file.name}.", suffix=".binary_engine"
    )
    os.close(handle)
    try:
        values.tofile(temporary_name)
        os.replace(temporary_name, final_file)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
