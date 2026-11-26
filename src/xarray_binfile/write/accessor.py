"""
Provides accessors for writing xarray Dataset and DataArray objects to binary files.
"""

import contextlib
import os
import secrets
import tempfile
import unicodedata
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
        (https://github.com/fschuch/xarray-binfile/issues/67), or when a
        ``dtype`` would change the kind of a variable's data.

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
            TypeError: If a write specification's ``dtype`` would change the
                kind of its data.
        """
        _directory = Path(directory) if directory is not None else Path.cwd()
        planned = [
            list(write_specs_getter(data_array))
            for data_array in self._data_set.data_vars.values()
        ]
        _require_writable(_directory, [spec for specs in planned for spec in specs])
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
        values are cast right before serialization. That cast may narrow
        within a kind (``f8`` to ``f4``, ``i8`` to ``i4``, without a range
        check) or widen (``i4`` to ``f8``), but a cast that changes the kind
        (float to integer, which truncates and turns NaN into 0, or complex
        to float) is refused before any file is written. Bytes are laid out in
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
        at once. Each call is independent: an existing file is replaced. A
        symbolic link at the destination is followed, so the file it points
        to is replaced and the link stays. New files get the permissions the
        process umask allows, as with any other new file.

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
            TypeError: If a write specification's ``dtype`` would change the
                kind of its data.
        """
        _directory = Path(directory) if directory is not None else Path.cwd()
        specs = list(write_specs_getter(self._data_array))
        _require_writable(_directory, specs)
        _write_specs(_directory, specs, progress)


def _require_writable(directory: Path, specs: list[WriteSpecs]) -> None:
    """
    Check every spec of one ``to_file`` call before the first file is written.

    Args:
        directory: Base directory for relative filenames.
        specs: Every write spec of the call.

    Raises:
        ValueError: If two specs map to the same file.
        TypeError: If a cast would change the kind of the data.
    """
    _require_distinct_files(directory, specs)
    _require_same_kind_casts(specs)


def _require_same_kind_casts(specs: Iterable[WriteSpecs]) -> None:
    """
    Fail if a spec's ``dtype`` would change the kind of its data.

    Only the dtypes are compared, so a lazy ``sub_array`` is not computed.
    Casts within a kind (``f8`` to ``f4``, ``i8`` to ``i4``) or to a wider
    kind (``i4`` to ``f8``) are allowed, as ``casting="same_kind"`` does;
    float to integer, which truncates and turns NaN into 0, is refused.

    Args:
        specs: Every write spec of one ``to_file`` call.

    Raises:
        TypeError: If a cast would change the kind of the data.
    """
    for spec in specs:
        source = spec.sub_array.dtype
        if spec.dtype is None or np.can_cast(source, spec.dtype, "same_kind"):
            continue
        target = np.dtype(spec.dtype)
        error_message = (
            f"Writing {str(spec.filename)!r} would cast {source} to {target}, "
            "which can change the values (floats are truncated to integers and "
            "NaN becomes 0, complex numbers lose their imaginary part). Cast the "
            f"array first if that is intended, for example with "
            f".astype({target.str!r})."
        )
        raise TypeError(error_message)


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

    Each destination is compared as its folder, with symbolic links followed,
    plus its filename, so a relative and an absolute filename for one file
    collide, as do two paths through a linked folder. Names that differ only
    in case or in Unicode form (``café`` composed or with a combining accent)
    collide too where the filesystem ignores that difference, as macOS does
    for both and Windows for case.

    Args:
        directory: Base directory for relative filenames.
        specs: Every write spec of one ``to_file`` call.

    Raises:
        ValueError: If two specs map to the same file.
    """
    paths = [_destination_key(directory, spec) for spec in specs]
    duplicated = sorted(path for path, count in Counter(paths).items() if count > 1)
    if duplicated:
        error_message = (
            f"Several write specs map to the same file(s) {duplicated}; the later "
            "ones would silently overwrite the earlier ones. Rename the variables "
            "or stacked components, or change the filename pattern."
        )
        raise ValueError(error_message)
    by_folded_name: defaultdict[str, list[str]] = defaultdict(list)
    for path in paths:
        by_folded_name[_fold(path, case=True, form=True)].append(path)
    clashes = [
        sorted(group)
        for group in by_folded_name.values()
        if len(group) > 1 and _same_file_there(group)
    ]
    if clashes:
        error_message = (
            f"The files {clashes} differ only in case or Unicode form, and the "
            "filesystem there ignores that difference, so they would be the same "
            "file and the later ones would silently overwrite the earlier ones. "
            "Rename the variables."
        )
        raise ValueError(error_message)


def _destination_key(directory: Path, spec: WriteSpecs) -> str:
    """
    The destination of ``spec``, with symbolic links followed.

    Links are followed in its folders and at the file itself, as the write
    does, so a link and the file it points to are one destination.

    Args:
        directory: Base directory for relative filenames.
        spec: A write spec.

    Returns:
        The absolute path of the file, as a string.
    """
    return os.path.realpath(_resolve_destination(directory, spec.filename))


def _fold(path: str, *, case: bool, form: bool) -> str:
    """
    ``path`` with the differences a filesystem may ignore removed.

    Args:
        path: An absolute path.
        case: Whether to ignore case.
        form: Whether to ignore the Unicode normalization form.

    Returns:
        The folded path.
    """
    if form:
        path = unicodedata.normalize("NFC", path)
    if case:
        path = path.casefold()
    return path


def _same_file_there(paths: list[str]) -> bool:
    """
    Whether some of ``paths``, equal up to case and form, are one file.

    Args:
        paths: Distinct paths that fold to the same name.

    Returns:
        True if the filesystem holding them ignores the differences.
    """
    folder = Path(paths[0]).parent
    case = _ignores_case(folder)
    form = _ignores_normalization(folder)
    return len({_fold(path, case=case, form=form) for path in paths}) < len(paths)


def _ignores_case(folder: Path) -> bool:
    """
    Whether the filesystem holding ``folder`` ignores case in filenames.

    Args:
        folder: The folder files will be written to.

    Returns:
        True if a name and its case-swapped twin are the same file there.
    """
    return _probe(folder, str.swapcase)


def _ignores_normalization(folder: Path) -> bool:
    """
    Whether the filesystem holding ``folder`` ignores the Unicode form.

    Args:
        folder: The folder files will be written to.

    Returns:
        True if a name with a composed ``é`` and the same name with ``e``
        and a combining accent are the same file there.
    """
    return _probe(folder, lambda name: unicodedata.normalize("NFD", name))


def _probe(folder: Path, twin: Callable[[str], str]) -> bool:
    """
    Whether a temporary file can be found under another spelling of its name.

    The file is created in ``folder``, or in its nearest existing parent when
    it does not exist yet, and removed afterwards.

    Args:
        folder: The folder files will be written to.
        twin: Respells the file name, for example swapping its case.

    Returns:
        True if the respelled name finds the temporary file.
    """
    existing = folder
    while not existing.exists():
        existing = existing.parent
    # Mixed case and a composed accent, so both kinds of respelling change it.
    handle, name = tempfile.mkstemp(dir=existing, prefix=".binary_engine-Probe-\u00e9-")
    os.close(handle)
    try:
        probe = Path(name)
        return probe.with_name(twin(probe.name)).exists()
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

    A symbolic link at ``final_file`` is followed, so the file it points to
    is replaced and the link stays; a dangling link creates that file, and
    its folder when missing, as sub-folders of a filename are. The temporary
    file sits next to the real destination, so the move stays on one
    filesystem.

    Args:
        final_file: Destination path; its parent directory must exist.
        values: Array to serialize with :meth:`numpy.ndarray.tofile`.
    """
    destination = Path(os.path.realpath(final_file))
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = _create_temporary(destination)
    try:
        with os.fdopen(handle, "wb") as file:
            values.tofile(file)
        os.replace(temporary_name, destination)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _create_temporary(destination: Path) -> tuple[int, str]:
    """
    Create an empty, hidden file next to ``destination``, open for writing.

    Unlike :func:`tempfile.mkstemp`, which always uses mode 0600, the file
    is created with 0666 and the process umask applies, so the written file
    gets the same permissions as any other new file. The name is short and
    fixed in length (35 bytes), so a destination name close to the
    filesystem's limit still gets a temporary file. As ``mkstemp`` does, a
    name that is already taken (a file left behind by a crash, say) is
    retried with a new random one; any other error is raised at once.

    Args:
        destination: The file the temporary one will replace.

    Returns:
        The open file descriptor and the path of the temporary file.

    Raises:
        FileExistsError: If every name tried is taken.
    """
    for _ in range(_TEMPORARY_ATTEMPTS - 1):
        with contextlib.suppress(FileExistsError):
            return _open_new_temporary(destination.parent)
    return _open_new_temporary(destination.parent)  # raises if taken again


# 64 random bits per name make a clash all but impossible, so this bound only
# matters with a broken random source; mkstemp uses TMP_MAX (10000).
_TEMPORARY_ATTEMPTS = 100


def _open_new_temporary(folder: Path) -> tuple[int, str]:
    """
    Create a new temporary file with a random name in ``folder``.

    Args:
        folder: Where to create it.

    Returns:
        The open file descriptor and the path of the file.

    Raises:
        FileExistsError: If the name drawn is taken.
    """
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    name = folder / f".binary_engine-{secrets.token_hex(8)}.tmp"
    return os.open(name, flags, 0o666), str(name)
