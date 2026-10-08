"""
Compose several conventions, by the folder their files live in or by filename.
"""

import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import Any

import xarray as xr

from xarray_binfile.conventions.getters import accepts_file, open_convention, scan_files
from xarray_binfile.conventions.layout import LayoutMismatchError
from xarray_binfile.conventions.protocol import ConventionProtocol
from xarray_binfile.read.file_metadata import ReadSpecs
from xarray_binfile.write.file_metadata import WriteSpecs


def _is_inside(parent_parts: tuple[str, ...], folder: Path) -> bool:
    """
    Check whether a parent path ends with ``folder``.

    The root folder (``"."`` or ``""``, no parts) matches every path.

    Args:
        parent_parts: Parts of the file's parent directory.
        folder: Registered folder, possibly nested.

    Returns:
        True if the last components of ``parent_parts`` equal ``folder``.
    """
    folder_parts = folder.parts
    if not folder_parts:
        return True
    return parent_parts[-len(folder_parts) :] == folder_parts


def _single_acceptor(
    candidates: Mapping[str, ConventionProtocol], data_array: xr.DataArray
) -> tuple[str, Iterator[WriteSpecs]]:
    """
    Offer ``data_array`` to every convention and require exactly one to accept it.

    Args:
        candidates: Conventions keyed by a label used in error messages.
        data_array: The array to write.

    Returns:
        The label of the accepting convention and its write specs.

    Raises:
        LayoutMismatchError: If no convention or more than one convention
            accepts the array.
    """
    accepted: list[tuple[str, Iterator[WriteSpecs]]] = []
    rejections: list[str] = []
    for label, convention in candidates.items():
        specs = iter(convention.writer(data_array))
        try:
            first = next(specs)
        except LayoutMismatchError as err:
            rejections.append(f"{label!r}: {err}")
        except StopIteration:
            accepted.append((label, iter(())))
        else:
            accepted.append((label, chain([first], specs)))
    if len(accepted) != 1:
        status = (
            "No convention accepts" if not accepted else "Several conventions accept"
        )
        error_message = (
            f"{status} the array {data_array.name!r} with dims "
            f"{tuple(data_array.dims)}. Accepted by: "
            f"{[label for label, _ in accepted]}. Rejected by: {rejections}."
        )
        raise LayoutMismatchError(error_message)
    return accepted[0]


def _stack_with(conventions: Iterable[Any], dataset: xr.Dataset) -> xr.Dataset:
    """
    Apply the ``stack`` method of every member convention that has one.

    Args:
        conventions: The member conventions.
        dataset: The dataset to stack.

    Returns:
        The stacked dataset.
    """
    for convention in conventions:
        if hasattr(convention, "stack"):
            dataset = convention.stack(dataset)
    return dataset


def _longest(
    folders: Iterable[str], predicate: Callable[[tuple[str, ...]], bool]
) -> str | None:
    """
    The most specific registered folder whose parts satisfy ``predicate``.

    Args:
        folders: Registered folder keys.
        predicate: Test on the folder's path parts.

    Returns:
        The longest matching folder, or ``None``.
    """
    matches = [
        (len(parts), folder)
        for folder in folders
        if predicate(parts := Path(folder).parts)
    ]
    return max(matches)[1] if matches else None


@dataclass(frozen=True)
class FolderConventions:
    """
    Dispatch to one convention per sub-folder.

    Projects often keep arrays of different shapes in different folders, for
    example ``xy_planes/ux-0001.bin`` next to ``3d/ux-0001.bin``. Reading
    picks the convention whose folder matches the end of the file's parent
    path, preferring the most specific (longest) folder when several match.
    The root of the dataset is registered as ``"."``; it matches any file not
    claimed by a more specific folder. Writing offers the array to every
    convention and requires exactly one of them to accept it, then prefixes
    the resulting filenames with that folder; a mismatch raises instead of
    guessing. An array whose name starts with a registered folder
    (``"geometry/epsi"``) is sent to that folder's convention directly, under
    the remaining name (``"epsi"``), which resolves the ambiguity between
    folders sharing one layout.

    Attributes:
        conventions: Mapping of folder (relative to the dataset root, possibly
            nested such as ``"snapshots/3d"``, or ``"."`` for the root) to
            the convention used inside it.
    """

    conventions: Mapping[str, ConventionProtocol]

    def _folder_of(self, path: Path) -> str:
        # Most specific (longest) folder wins, so "snapshots/3d" is preferred
        # over "3d" regardless of the order the conventions were registered in.
        parts = path.parent.parts
        folder = _longest(
            self.conventions,
            lambda folder_parts: _is_inside(parts, Path(*folder_parts)),
        )
        if folder is not None:
            return folder
        error_message = (
            f"No convention registered for the folder of {path}. Known folders: "
            f"{sorted(self.conventions)}."
        )
        raise ValueError(error_message)

    def _folder_for_name(self, name: PurePosixPath) -> str | None:
        """
        Find the registered folder a name is prefixed with.

        Args:
            name: The array name, with folder segments (``geometry/sub/epsi``).

        Returns:
            The longest registered folder the name starts with, or ``None``
            when there is none (the root does not count).
        """
        parts = name.parent.parts
        return _longest(
            self.conventions,
            lambda folder_parts: (
                bool(folder_parts) and parts[: len(folder_parts)] == folder_parts
            ),
        )

    def reader(self, path: Path) -> ReadSpecs:
        """
        Build read specs using the convention of the file's folder.

        Args:
            path: Path to the binary file, inside one of the known folders.

        Returns:
            The metadata required to decode ``path``.

        Raises:
            ValueError: If no convention is registered for the file's folder.
        """
        return self.conventions[self._folder_of(path)].reader(path)

    def writer(self, data_array: xr.DataArray) -> Iterator[WriteSpecs]:
        """
        Yield write specs from the single convention that accepts the array.

        Args:
            data_array: The array to write.

        Yields:
            The accepting convention's write specs, with filenames prefixed by
            its folder (always using ``/`` as separator).

        Raises:
            LayoutMismatchError: If no convention or more than one convention
                accepts the array.
        """
        name = PurePosixPath(str(data_array.name or ""))
        if folder := self._folder_for_name(name):
            relative = name.relative_to(PurePosixPath(folder)).as_posix()
            folder, specs = _single_acceptor(
                {folder: self.conventions[folder]}, data_array.rename(relative)
            )
        else:
            folder, specs = _single_acceptor(self.conventions, data_array)
        yield from self._prefixed(folder, specs)

    @staticmethod
    def _prefixed(folder: str, specs: Iterator[WriteSpecs]) -> Iterator[WriteSpecs]:
        for spec in specs:
            # POSIX separators on every platform keep filenames deterministic;
            # Windows accepts them when the accessor turns them into paths.
            yield spec._replace(
                filename=PurePosixPath(
                    *Path(folder).parts, *Path(spec.filename).parts
                ).as_posix()
            )

    def files(self, directory: str | os.PathLike[str]) -> list[Path]:
        """
        List the files of every registered folder inside ``directory``.

        Args:
            directory: The dataset root.

        Returns:
            The matching paths, sorted. Folders that do not exist are skipped.
        """
        root = Path(directory)
        found: set[Path] = set()
        for folder, convention in self.conventions.items():
            target = root / folder
            if target.is_dir():
                found.update(scan_files(target, partial(accepts_file, convention)))
        return sorted(found)

    def stack(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Rebuild the stacked arrays declared by the member conventions.

        Args:
            dataset: A dataset as returned by the ``binfile`` engine.

        Returns:
            The stacked dataset.
        """
        return _stack_with(self.conventions.values(), dataset)

    def open(
        self,
        directory: str | os.PathLike[str],
        *,
        variables: Iterable[str] | None = None,
        stack: bool = True,
        **open_mfdataset_kwargs: Any,
    ) -> xr.Dataset:
        """
        Open every file of every registered folder as one lazy dataset.

        Files are combined by coordinates, so the folders must hold variables
        that fit together in one dataset (for example the same mesh in each).
        Open folders separately when their layouts conflict.

        Args:
            directory: The dataset root.
            variables: Optional names to keep, as found on disk; other files
                are not opened.
            stack: Whether to rebuild the arrays declared by the members' stacks.
            **open_mfdataset_kwargs: Forwarded to :func:`xarray.open_mfdataset`.

        Returns:
            The lazily opened dataset.

        Raises:
            FileNotFoundError: If no file matches.
        """
        return open_convention(
            self, directory, variables=variables, stack=stack, **open_mfdataset_kwargs
        )


@dataclass(frozen=True)
class PatternConventions:
    """
    Dispatch to the first convention whose filename pattern matches.

    Use it when files following different conventions share one folder, for
    example ``ux-0001.bin`` (a :class:`StepIndexedFiles`) next to
    ``epsi.bin`` (a :class:`StaticFiles`). Reading tries the conventions in
    order and keeps the first whose ``reader`` accepts the filename, so list
    the most specific pattern first. Writing offers the array to every
    convention and requires exactly one of them to accept it.

    Attributes:
        conventions: The conventions to try, most specific first.
    """

    conventions: Sequence[ConventionProtocol]

    def reader(self, path: Path) -> ReadSpecs:
        """
        Build read specs with the first convention that accepts the filename.

        Args:
            path: Path to the binary file.

        Returns:
            The metadata required to decode ``path``.

        Raises:
            ValueError: If no convention accepts the filename.
        """
        errors: list[str] = []
        for convention in self.conventions:
            try:
                return convention.reader(path)
            except ValueError as err:
                errors.append(str(err))
        error_message = f"No convention accepts the file {path}: {errors}"
        raise ValueError(error_message)

    def writer(self, data_array: xr.DataArray) -> Iterator[WriteSpecs]:
        """
        Yield write specs from the single convention that accepts the array.

        Args:
            data_array: The array to write.

        Yields:
            The accepting convention's write specs.

        Raises:
            LayoutMismatchError: If no convention or more than one convention
                accepts the array.
        """
        candidates = {
            f"{index}:{type(convention).__name__}": convention
            for index, convention in enumerate(self.conventions)
        }
        _, specs = _single_acceptor(candidates, data_array)
        yield from specs

    def files(self, directory: str | os.PathLike[str]) -> list[Path]:
        """
        List the files in ``directory`` accepted by any of the conventions.

        Args:
            directory: The folder to scan (not recursively).

        Returns:
            The matching paths, sorted.
        """
        return scan_files(
            directory,
            lambda path: any(
                accepts_file(convention, path) for convention in self.conventions
            ),
        )

    def stack(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Rebuild the stacked arrays declared by the member conventions.

        Args:
            dataset: A dataset as returned by the ``binfile`` engine.

        Returns:
            The stacked dataset.
        """
        return _stack_with(self.conventions, dataset)

    def open(
        self,
        directory: str | os.PathLike[str],
        *,
        variables: Iterable[str] | None = None,
        stack: bool = True,
        **open_mfdataset_kwargs: Any,
    ) -> xr.Dataset:
        """
        Open every file in ``directory`` accepted by the conventions as one lazy dataset.

        Args:
            directory: The folder holding the files.
            variables: Optional names to keep, as found on disk; other files
                are not opened.
            stack: Whether to rebuild the arrays declared by the members' stacks.
            **open_mfdataset_kwargs: Forwarded to :func:`xarray.open_mfdataset`.

        Returns:
            The lazily opened dataset.

        Raises:
            FileNotFoundError: If no file matches.
        """
        return open_convention(
            self, directory, variables=variables, stack=stack, **open_mfdataset_kwargs
        )
