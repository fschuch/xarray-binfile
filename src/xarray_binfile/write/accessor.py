"""
Provides accessors for writing xarray Dataset and DataArray objects to binary files.
"""

import os
import tempfile
from pathlib import Path

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
        directory: Path | None = None,
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
        directory: Path | None = None,
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
        a unique temporary directory created inside the destination directory
        (so the final move stays on the same filesystem), and each file is
        moved to its final path with :func:`os.replace` only once it is
        complete. An interrupted write never leaves a truncated file at the
        destination, and the temporary directory is removed automatically.

        Plan the write specifications so that every individual output file
        fits comfortably in memory, for example by splitting the array into
        one file per time step. If you need streaming, incremental, or
        partial writes, prefer one of the other file formats supported by
        xarray, such as NetCDF or Zarr.

        Each file is written with the in-memory dtype and native byte order,
        unless the write specification sets ``dtype``, in which case the
        values are cast right before serialization.

        Args:
            write_specs_getter: A callable that generates write specifications for the data array.
            directory: The directory where the binary files will be written. Defaults to the current working directory.
        """
        _directory = directory or Path.cwd()
        with tempfile.TemporaryDirectory(
            dir=_directory, prefix=".binary_engine-"
        ) as temporary_directory:
            for details in write_specs_getter(self._data_array):
                new_type = (
                    details.dtype
                    if details.dtype is not None
                    else details.sub_array.dtype
                )
                temporary_file = Path(temporary_directory) / details.filename
                details.sub_array.values.astype(new_type).tofile(temporary_file)
                os.replace(temporary_file, _directory / details.filename)
