"""
Protocol shared by the conventions shipped with xarray-binfile.
"""

from typing import Protocol

from xarray_binfile.read.file_metadata import ReadSpecsGetterProtocol
from xarray_binfile.write.file_metadata import WriteSpecsGetterProtocol


class ConventionProtocol(Protocol):
    """
    Structural protocol for a file convention.

    A convention bundles a read specs getter and a write specs getter that
    agree on the same filename pattern and on-disk layout, so files written by
    one can always be read back by the other. No inheritance is required: any
    object exposing ``reader`` and ``writer`` attributes with the right
    signatures is a convention, including the classes shipped in
    :mod:`xarray_binfile.conventions`.
    """

    @property
    def reader(self) -> ReadSpecsGetterProtocol:
        """
        Read specs getter for files that follow the convention.

        Returns:
            A callable matching :class:`ReadSpecsGetterProtocol`.
        """
        ...

    @property
    def writer(self) -> WriteSpecsGetterProtocol:
        """
        Write specs getter producing files that follow the convention.

        Returns:
            A callable matching :class:`WriteSpecsGetterProtocol`.
        """
        ...
