"""
Backend for reading binary files in Xarray.

References:
    * https://docs.xarray.dev/en/latest/internals/how-to-add-new-backend.html
    * https://github.com/pydata/xarray/discussions/6406
"""

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from xarray import Dataset
from xarray.backends import BackendEntrypoint

from xarray_binfile.read.array import BinaryEngineBackendArray
from xarray_binfile.read.file_metadata import ReadSpecsGetterProtocol


class RawBinaryEntrypoint(BackendEntrypoint):
    """
    Backend entry point for reading binary files in Xarray.

    Attributes:
        open_dataset_parameters: Parameters accepted by the `open_dataset` method.
        description: Description of the backend.
        url: URL to the backend documentation.
    """

    open_dataset_parameters = ("filename_or_obj", "drop_variables, read_specs_getter")
    description = "Read and write raw binary files using the familiar interface from the Xarray library."
    url = "https://docs.fschuch.com/xarray-binfile/"

    def open_dataset(  # type: ignore[override]
        self,
        filename_or_obj: str | os.PathLike[Any],
        *,
        read_specs_getter: ReadSpecsGetterProtocol,
        drop_variables: str | Iterable[str] | None = None,
    ) -> Dataset:
        """
        Open a dataset from a binary file.

        Args:
            filename_or_obj: Path to the binary file or a file-like object.
            read_specs_getter: A callable that generates read specifications for the binary file.
            drop_variables: Variables to drop from the dataset. Defaults to None.

        Returns:
            The opened Xarray dataset.

        Raises:
            ValueError: If `filename_or_obj` is not a valid file path.
            ValueError: If there is an error reading the metadata from the file path.
        """
        try:
            file_path = Path(filename_or_obj)
        except TypeError as err:
            error_message = f"Expected a file path or file-like object, but got: {filename_or_obj!r}"
            raise ValueError(error_message) from err
        try:
            file_metadata = read_specs_getter(path=file_path)
        except Exception as err:
            error_message = f"Error reading metadata from {file_path}: {err}"
            raise ValueError(error_message) from err
        if (
            isinstance(drop_variables, str) and file_metadata.name == drop_variables
        ) or (
            isinstance(drop_variables, Iterable)
            and file_metadata.name in drop_variables
        ):
            return Dataset()
        return BinaryEngineBackendArray(metadata=file_metadata).get_xarray_dataset()
