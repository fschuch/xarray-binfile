"""
Compose several conventions, by the folder their files live in or by filename.
"""

import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path, PurePosixPath

import xarray as xr

from xarray_binfile.conventions.base import Convention, scan_files
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

    A convention that accepts the array but has nothing to write (an empty
    split dimension) does not make the choice ambiguous: it is used only when
    no other convention produces files.

    Args:
        candidates: Conventions keyed by a label used in error messages.
        data_array: The array to write.

    Returns:
        The label of the accepting convention and its write specs.

    Raises:
        LayoutMismatchError: If no convention or more than one convention
            accepts the array.
    """
    producing: list[tuple[str, Iterator[WriteSpecs]]] = []
    empty: list[tuple[str, Iterator[WriteSpecs]]] = []
    rejections: list[str] = []
    for label, convention in candidates.items():
        specs = iter(convention.writer(data_array))
        try:
            first = next(specs)
        except LayoutMismatchError as err:
            rejections.append(f"{label!r}: {err}")
        except StopIteration:
            empty.append((label, iter(())))
        else:
            producing.append((label, chain([first], specs)))
    if len(producing) == 1:
        return producing[0]
    if not producing and empty:
        return empty[0]
    status = "No convention accepts" if not producing else "Several conventions accept"
    error_message = (
        f"{status} the array {data_array.name!r} with dims "
        f"{tuple(data_array.dims)}. Accepted by: "
        f"{[label for label, _ in producing]}. Rejected by: {rejections}."
    )
    raise LayoutMismatchError(error_message)


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
class FolderConventions(Convention):
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

    def __post_init__(self) -> None:
        """Give every member the :class:`Convention` defaults."""
        adapted = {
            folder: Convention.adapt(c) for folder, c in self.conventions.items()
        }
        object.__setattr__(self, "conventions", adapted)

    def _member(self, folder: str) -> Convention:
        return Convention.adapt(self.conventions[folder])

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
        return self._member(self._folder_of(path)).reader(path)

    def name_of_file(self, path: Path) -> str:
        """
        The variable name of ``path``, decoded by the convention of its folder.

        Args:
            path: Path to the binary file.

        Returns:
            The decoded name.

        Raises:
            ValueError: If no convention is registered for the file's folder,
                or if that convention rejects the file.
        """
        return self._member(self._folder_of(path)).name_of_file(path)

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
        name = PurePosixPath("" if data_array.name is None else str(data_array.name))
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
        for folder in self.conventions:
            target = root / folder
            if target.is_dir():
                found.update(scan_files(target, self._member(folder).accepts))
        return sorted(found)

    def stack(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Rebuild the stacked arrays declared by the member conventions.

        Args:
            dataset: A dataset as returned by the ``binfile`` engine.

        Returns:
            The stacked dataset.
        """
        for folder in self.conventions:
            dataset = self._member(folder).stack(dataset)
        return dataset


@dataclass(frozen=True)
class PatternConventions(Convention):
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

    def __post_init__(self) -> None:
        """Give every member the :class:`Convention` defaults."""
        members = tuple(Convention.adapt(c) for c in self.conventions)
        object.__setattr__(self, "conventions", members)

    @property
    def _members(self) -> tuple[Convention, ...]:
        return tuple(Convention.adapt(c) for c in self.conventions)

    def name_of_file(self, path: Path) -> str:
        """
        The variable name of ``path``, from the first member accepting it.

        Args:
            path: Path to the binary file.

        Returns:
            The decoded name.

        Raises:
            ValueError: If no member accepts the file.
        """
        errors: list[str] = []
        for member in self._members:
            try:
                return member.name_of_file(path)
            except ValueError as err:
                errors.append(str(err))
        error_message = f"No convention accepts the file {path}: {errors}"
        raise ValueError(error_message)

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
        members = self._members
        return scan_files(directory, lambda path: any(m.accepts(path) for m in members))

    def stack(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Rebuild the stacked arrays declared by the member conventions.

        Args:
            dataset: A dataset as returned by the ``binfile`` engine.

        Returns:
            The stacked dataset.
        """
        for member in self._members:
            dataset = member.stack(dataset)
        return dataset
