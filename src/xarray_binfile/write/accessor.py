"""
Provides accessors for writing xarray Dataset and DataArray objects to binary files.
"""

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

        Args:
            write_specs_getter: A callable that generates write specifications for the data array.
            directory: The directory where the binary files will be written. Defaults to the current working directory.
        """
        _directory = directory or Path.cwd()
        for details in write_specs_getter(self._data_array):
            details.sub_array.to_numpy().tofile(_directory / details.filename)
