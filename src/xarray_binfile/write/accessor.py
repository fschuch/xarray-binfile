"""
Provides accessors for writing xarray Dataset and DataArray objects to binary files.
"""

import contextlib
import os
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from xarray_binfile.write.file_metadata import WriteSpecsGetterProtocol


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

        Args:
            write_specs_getter: A callable that generates write specifications for the data arrays.
            directory: The directory where the binary files will be written. Defaults to the current working directory.
        """
        for data_array in self._data_set.data_vars.values():
            data_array.binary_engine.to_file(write_specs_getter, directory)


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

        Plan the write specifications so that every individual output file
        fits comfortably in memory, for example by splitting the array into
        one file per time step. If you need streaming, incremental, or
        partial writes, prefer one of the other file formats supported by
        xarray, such as NetCDF or Zarr.

        Each file is written with the in-memory dtype and native byte order,
        unless the write specification sets ``dtype``, in which case the
        values are cast right before serialization.

        A relative ``WriteSpecs.filename`` is resolved against ``directory``,
        which must already exist, and may contain sub-folders (for example
        ``"3d/ux-0001.bin"``), which are created on demand. It must stay
        inside ``directory``: paths escaping it through ``..`` are rejected
        (symbolic links inside ``directory`` are not followed for this check,
        so a linked sub-folder is fine). An absolute filename is used as is,
        which lets one call target several locations.

        Args:
            write_specs_getter: A callable that generates write specifications for the data array.
            directory: The base directory for relative filenames. Defaults to the current working directory.

        Raises:
            FileNotFoundError: If a relative filename is used and ``directory`` does not exist.
            ValueError: If a relative filename escapes ``directory``.
        """
        _directory = Path(directory) if directory is not None else Path.cwd()
        for details in write_specs_getter(self._data_array):
            final_file = _resolve_destination(_directory, details.filename)
            new_type = (
                details.dtype if details.dtype is not None else details.sub_array.dtype
            )
            final_file.parent.mkdir(parents=True, exist_ok=True)
            values = details.sub_array.values.astype(new_type, copy=False)
            _write_atomically(final_file, values)


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
