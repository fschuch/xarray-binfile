"""
Compose several conventions by the folder their files live in.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import chain
from pathlib import Path, PurePosixPath

import xarray as xr

from xarray_binfile.conventions.layout import LayoutMismatchError
from xarray_binfile.conventions.protocol import ConventionProtocol
from xarray_binfile.read.file_metadata import ReadSpecs
from xarray_binfile.write.file_metadata import WriteSpecs


@dataclass(frozen=True)
class FolderConventions:
    """
    Dispatch to one convention per sub-folder.

    Projects often keep arrays of different shapes in different folders, for
    example ``xy_planes/ux-0001.bin`` next to ``3d/ux-0001.bin``. Reading
    picks the convention whose folder matches the end of the file's parent
    path, preferring the most specific (longest) folder when several match. Writing offers the array to every convention and requires exactly
    one of them to accept it, then prefixes the resulting filenames with that
    folder; a mismatch raises instead of guessing.

    Attributes:
        conventions: Mapping of folder (relative to the dataset root, possibly
            nested such as ``"snapshots/3d"``) to the convention used inside it.
    """

    conventions: Mapping[str, ConventionProtocol]

    def _folder_of(self, path: Path) -> str:
        parts = path.parent.parts
        # Most specific (longest) folder wins, so "snapshots/3d" is preferred
        # over "3d" regardless of the order the conventions were registered in.
        matches = sorted(
            (
                folder
                for folder in self.conventions
                if (folder_parts := Path(folder).parts)
                and parts[-len(folder_parts) :] == folder_parts
            ),
            key=lambda folder: len(Path(folder).parts),
            reverse=True,
        )
        if matches:
            return matches[0]
        error_message = (
            f"No convention registered for the folder of {path}. Known folders: "
            f"{sorted(self.conventions)}."
        )
        raise ValueError(error_message)

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
        accepted: list[tuple[str, Iterator[WriteSpecs]]] = []
        rejections: list[str] = []
        for folder, convention in self.conventions.items():
            specs = iter(convention.writer(data_array))
            try:
                first = next(specs)
            except LayoutMismatchError as err:
                rejections.append(f"{folder!r}: {err}")
            except StopIteration:
                accepted.append((folder, iter(())))
            else:
                accepted.append((folder, chain([first], specs)))
        if len(accepted) != 1:
            status = (
                "No convention accepts"
                if not accepted
                else "Several conventions accept"
            )
            error_message = (
                f"{status} the array {data_array.name!r} with dims "
                f"{tuple(data_array.dims)}. Accepted by: "
                f"{[folder for folder, _ in accepted]}. Rejected by: {rejections}."
            )
            raise LayoutMismatchError(error_message)
        folder, specs = accepted[0]
        for spec in specs:
            # POSIX separators on every platform keep filenames deterministic;
            # Windows accepts them when the accessor turns them into paths.
            yield spec._replace(
                filename=PurePosixPath(
                    *Path(folder).parts, *Path(spec.filename).parts
                ).as_posix()
            )
