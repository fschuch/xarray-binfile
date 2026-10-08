"""
Ready-to-use file conventions built on the read and write spec protocols.

Start from the convention closest to your project's naming scheme, or copy one
and adapt it when none fits.
"""

from xarray_binfile.conventions.base import Convention
from xarray_binfile.conventions.filename_pattern import FilenamePattern
from xarray_binfile.conventions.folders import FolderConventions, PatternConventions
from xarray_binfile.conventions.getters import (
    StaticFiles,
    StepIndexedFiles,
    TimeStampedFiles,
)
from xarray_binfile.conventions.layout import Layout, LayoutMismatchError
from xarray_binfile.conventions.protocol import ConventionProtocol
from xarray_binfile.conventions.stacking import (
    VariableStack,
    split_variables,
    stack_variables,
)

__all__ = [
    "Convention",
    "ConventionProtocol",
    "FilenamePattern",
    "FolderConventions",
    "Layout",
    "LayoutMismatchError",
    "PatternConventions",
    "StaticFiles",
    "StepIndexedFiles",
    "TimeStampedFiles",
    "VariableStack",
    "split_variables",
    "stack_variables",
]
