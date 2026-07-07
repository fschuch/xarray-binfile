"""Read and write raw binary files through xarray."""

from xarray_binfile._version import __version__, __version_tuple__
from xarray_binfile.read.entrypoint import RawBinaryEntrypoint
from xarray_binfile.read.file_metadata import ReadSpecs, ReadSpecsGetterProtocol

# The write accessor classes are xarray plugins: importing their module
# registers ``.binary_engine`` on Dataset and DataArray, which is the only
# supported way to reach them. The read backend, in contrast, is registered
# automatically through the ``xarray.backends`` entry point.
from xarray_binfile.write import accessor as _accessor
from xarray_binfile.write.file_metadata import WriteSpecs, WriteSpecsGetterProtocol

__all__ = [
    "RawBinaryEntrypoint",
    "ReadSpecs",
    "ReadSpecsGetterProtocol",
    "WriteSpecs",
    "WriteSpecsGetterProtocol",
    "__version__",
    "__version_tuple__",
]
