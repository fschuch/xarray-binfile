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
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import xarray as xr

from xarray_binfile.conventions.protocol import ConventionProtocol
from xarray_binfile.read.file_metadata import ReadSpecs
from xarray_binfile.write.file_metadata import WriteSpecs


def open_files(
    paths: Iterable[Path], reader: Any, **open_mfdataset_kwargs: Any
) -> xr.Dataset:
    """
    Open binary files as one lazy dataset with the ``binfile`` engine.

    Thin wrapper over :func:`xarray.open_mfdataset` that fills in the engine
    and the read specs getter.

    Args:
        paths: The files to open.
        reader: The read specs getter to decode them.
        **open_mfdataset_kwargs: Forwarded to :func:`xarray.open_mfdataset`,
            for example ``chunks`` or ``parallel``.

    Returns:
        The lazily opened dataset.

    Raises:
        FileNotFoundError: If ``paths`` is empty.
    """
    paths = sorted(paths)
    if not paths:
        error_message = "No file matches the convention; nothing to open."
        raise FileNotFoundError(error_message)
    return xr.open_mfdataset(
        paths, engine="binfile", read_specs_getter=reader, **open_mfdataset_kwargs
    )


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

        Args:
            directory: The folder holding the files.
            variables: Optional names to keep, as found on disk (``"ux"``, not
                ``"u"``); other files are not opened.
            stack: Whether to rebuild the stacked arrays.
            **open_mfdataset_kwargs: Forwarded to :func:`xarray.open_mfdataset`,
                for example ``chunks={"time": 1}`` or ``parallel=True``.

        Returns:
            The lazily opened dataset, combined by coordinates.

        Raises:
            FileNotFoundError: If no file matches.
        """
        paths = self.files(directory)
        if variables is not None:
            wanted = set(variables)
            paths = [path for path in paths if self.name_of_file(path) in wanted]
        dataset = open_files(paths, self.reader, **open_mfdataset_kwargs)
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
