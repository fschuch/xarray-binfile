"""
How ``convention.open()`` chunks the files, and what each chunk costs to read.

From the pyOpenSci review, https://github.com/fschuch/xarray-binfile/issues/63:
Dask reads every chunk a selection touches in full. With one chunk per file,
reading one point from a time series read every file whole. ``open()`` now
splits files larger than Dask's ``array.chunk-size`` into slabs along the
slowest axis on disk, which are contiguous in the file; smaller files keep one
chunk each. Explicit ``chunks`` always win.

Before Dask 2024.12.0, Dask fused the selection into the read, so the backend
was asked only for the selected values; the read-extent tests check what each
supported Dask version does.
"""

import collections
import contextlib
import pathlib

import dask
import numpy as np
import pytest
import xarray as xr
from packaging.version import Version

from xarray_binfile.conventions import FolderConventions, Layout, StepIndexedFiles
from xarray_binfile.read.array import BinaryEngineBackendArray

# Dask fused the selection into the read until 2024.12.0.
DASK_FUSES_SELECTION = Version(dask.__version__) < Version("2024.12.0")

# 32 KiB per file: four 8 KiB slabs once Dask's chunk size is 8 KiB.
GRID = {"x": np.arange(32), "y": np.arange(16), "z": np.arange(16)}


def _convention(order="C") -> StepIndexedFiles:
    return StepIndexedFiles(Layout(GRID, dtype="<f4", order=order))


def _write(directory: pathlib.Path, convention, name="ux", steps=2) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    size = int(np.prod([len(c) for c in convention.layout.coords.values()]))
    for step in range(1, steps + 1):
        np.zeros(size, "<f4").tofile(
            directory / convention.pattern.format(name=name, step=step)
        )


@pytest.fixture
def small_chunks():
    with dask.config.set({"array.chunk-size": "8KiB"}):
        yield


@pytest.fixture
def keys_read(monkeypatch) -> list[tuple]:
    """Record every key Dask asks the backend to read."""
    keys: list[tuple] = []
    original = BinaryEngineBackendArray._raw_indexing_method

    def spy(self, key):
        keys.append(key)
        return original(self, key)

    monkeypatch.setattr(BinaryEngineBackendArray, "_raw_indexing_method", spy)
    return keys


def _extent(key_part: slice | int, size: int) -> int:
    return 1 if isinstance(key_part, int) else len(range(size)[key_part])


@pytest.mark.usefixtures("small_chunks")
@pytest.mark.parametrize(
    ("order", "expected"),
    [
        pytest.param("C", (8, 16, 16, 1), id="C-x-slowest"),
        pytest.param("F", (32, 16, 4, 1), id="F-z-slowest"),
    ],
)
def test_open_splits_large_files_along_the_slowest_axis(tmp_path, order, expected):
    convention = _convention(order)
    _write(tmp_path, convention)

    ds = convention.open(tmp_path)

    assert ds.ux.dims == ("x", "y", "z", "time")
    assert ds.ux.data.chunksize == expected


def test_open_keeps_one_chunk_per_file_when_it_fits(tmp_path):
    convention = _convention()
    _write(tmp_path, convention)

    ds = convention.open(tmp_path)

    assert ds.ux.data.chunksize == (32, 16, 16, 1)


@pytest.mark.usefixtures("small_chunks")
def test_explicit_chunks_win(tmp_path):
    convention = _convention()
    _write(tmp_path, convention)

    assert convention.open(tmp_path, chunks={"time": 1}).ux.data.chunksize == (
        32,
        16,
        16,
        1,
    )
    assert convention.open(tmp_path, chunks={"y": 4}).ux.data.chunksize == (
        32,
        4,
        16,
        1,
    )


@pytest.mark.usefixtures("small_chunks")
def test_a_probe_reads_one_slab_per_file(tmp_path, keys_read):
    convention = _convention()
    _write(tmp_path, convention)
    ds = convention.open(tmp_path)

    with dask.config.set(scheduler="synchronous"):
        ds.ux.isel(x=5, y=5, z=5).compute()

    assert len(keys_read) == 2  # one read per file
    # One slab, not 32 planes; older Dask asked for the point alone.
    expected = 1 if DASK_FUSES_SELECTION else 8
    assert all(_extent(key[0], 32) == expected for key in keys_read)


@pytest.mark.usefixtures("small_chunks")
def test_a_probe_reads_every_chunk_it_touches_in_full(tmp_path, keys_read):
    # Documents the behaviour behind #63: with one chunk per file, Dask asks
    # the backend for the whole file to get one value from it.
    convention = _convention()
    _write(tmp_path, convention)
    ds = convention.open(tmp_path, chunks={"time": 1})

    with dask.config.set(scheduler="synchronous"):
        ds.ux.isel(x=5, y=5, z=5).compute()

    assert len(keys_read) == 2
    assert all(
        _extent(part, size) == (1 if DASK_FUSES_SELECTION else size)
        for key in keys_read
        for part, size in zip(key, (32, 16, 16), strict=False)
    )


@pytest.mark.usefixtures("small_chunks")
def test_mixed_folders_split_along_the_largest_files_slowest_axis(tmp_path):
    # One ``chunks`` applies to every file, so the largest files decide: here
    # 3-D snapshots split along x, and the smaller planes keep one chunk each.
    snapshots = _convention()
    planes = StepIndexedFiles(Layout({"y": GRID["y"], "z": GRID["z"]}, dtype="<f4"))
    conventions = FolderConventions({"3d": snapshots, "planes": planes})
    _write(tmp_path / "3d", snapshots, name="ux")
    _write(tmp_path / "planes", planes, name="pp")

    ds = conventions.open(tmp_path)

    assert ds.ux.data.chunksize == (8, 16, 16, 1)
    assert ds.pp.data.chunksize == (16, 16, 1)


@pytest.mark.usefixtures("small_chunks")
def test_open_works_when_no_axis_can_be_split(tmp_path):
    convention = StepIndexedFiles(Layout({"x": [0.0]}, dtype="<f4"))
    _write(tmp_path, convention)

    ds = convention.open(tmp_path)

    assert ds.ux.data.chunksize == (1, 1)


ZERO_LENGTH_LAYOUTS = (
    pytest.param({"x": np.arange(0), "y": np.arange(4)}, id="slowest-empty"),
    pytest.param({"x": np.arange(8), "y": np.arange(0)}, id="fastest-empty"),
)


@pytest.mark.usefixtures("small_chunks")
@pytest.mark.parametrize("coords", ZERO_LENGTH_LAYOUTS)
def test_open_keeps_zero_length_files_in_one_chunk(tmp_path, coords):
    # Dask cannot size an "auto" chunk on an array that holds nothing.
    convention = StepIndexedFiles(Layout(coords, dtype="<f4"))
    _write(tmp_path, convention)

    ds = convention.open(tmp_path)

    shape = tuple(len(c) for c in coords.values())
    assert ds.ux.data.chunksize == (*shape, 1)
    assert ds.ux.values.shape == (*shape, 2)


@pytest.mark.usefixtures("small_chunks")
def test_open_does_not_split_when_any_file_is_empty(tmp_path):
    # The axis the largest files pick would reach the empty files too, so one
    # empty file keeps every file in one chunk, as before the default existed.
    snapshots = _convention()
    empty = StepIndexedFiles(Layout({"x": GRID["x"], "w": np.arange(0)}, dtype="<f4"))
    conventions = FolderConventions({"3d": snapshots, "empty": empty})
    _write(tmp_path / "3d", snapshots, name="ux")
    _write(tmp_path / "empty", empty, name="ee")

    ds = conventions.open(tmp_path)

    assert ds.ux.data.chunksize == (32, 16, 16, 1)
    assert ds.ee.values.shape == (32, 0, 2)


class _FailsOnStep2(StepIndexedFiles):
    """Accepts every ``.bin`` file, but cannot read the second step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: collections.Counter[str] = collections.Counter()

    def accepts(self, path):
        return path.suffix == ".bin"

    def reader(self, path):
        self.calls[path.name] += 1
        if "0002" in path.name:
            raise OSError("unreadable header")
        return super().reader(path)


needs_per_file_errors = pytest.mark.skipif(
    Version(xr.__version__) < Version("2025.8.0"),
    reason="xarray handles errors per file from 2025.08.0",
)


@needs_per_file_errors
@pytest.mark.usefixtures("small_chunks")
@pytest.mark.parametrize("errors", ["ignore", "warn"])
def test_open_lets_xarray_skip_a_file_the_reader_rejects(tmp_path, errors):
    convention = _FailsOnStep2(Layout(GRID, dtype="<f4"))
    _write(tmp_path, convention, steps=3)

    with pytest.warns() if errors == "warn" else contextlib.nullcontext():
        ds = convention.open(tmp_path, errors=errors)

    assert ds.sizes["time"] == 2
    assert ds.ux.data.chunksize == (8, 16, 16, 1)  # planned from the good files
    assert set(convention.calls.values()) == {1}  # each file read once


@needs_per_file_errors
@pytest.mark.usefixtures("small_chunks")
def test_open_still_raises_a_reader_failure_by_default(tmp_path):
    convention = _FailsOnStep2(Layout(GRID, dtype="<f4"))
    _write(tmp_path, convention, steps=3)

    # xarray names the file in a ValueError from 2025.08.0 on.
    with pytest.raises((OSError, ValueError), match="unreadable header"):
        convention.open(tmp_path)
