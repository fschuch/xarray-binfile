"""
How ``convention.open()`` chunks the files, and what each chunk costs to read.

From the pyOpenSci review, https://github.com/fschuch/xarray-binfile/issues/63:
Dask reads every chunk a selection touches in full. With one chunk per file,
reading one point from a time series read every file whole. ``open()`` now
splits files larger than Dask's ``array.chunk-size`` into slabs along the
slowest axis on disk, which are contiguous in the file; smaller files keep one
chunk each. Explicit ``chunks`` always win.
"""

import pathlib

import dask
import numpy as np
import pytest

from xarray_binfile.conventions import FolderConventions, Layout, StepIndexedFiles
from xarray_binfile.read.array import BinaryEngineBackendArray

ISSUE_63 = pytest.mark.xfail(
    strict=True,
    reason="https://github.com/fschuch/xarray-binfile/issues/63: "
    "open() makes one chunk per file, so any selection reads whole files",
)

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


@ISSUE_63
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


@ISSUE_63
@pytest.mark.usefixtures("small_chunks")
def test_a_probe_reads_one_slab_per_file(tmp_path, keys_read):
    convention = _convention()
    _write(tmp_path, convention)
    ds = convention.open(tmp_path)

    with dask.config.set(scheduler="synchronous"):
        ds.ux.isel(x=5, y=5, z=5).values

    assert len(keys_read) == 2  # one read per file
    assert all(_extent(key[0], 32) == 8 for key in keys_read)  # one slab, not 32


@pytest.mark.usefixtures("small_chunks")
def test_a_probe_reads_every_chunk_it_touches_in_full(tmp_path, keys_read):
    # Documents the behaviour behind #63: with one chunk per file, Dask asks
    # the backend for the whole file to get one value from it.
    convention = _convention()
    _write(tmp_path, convention)
    ds = convention.open(tmp_path, chunks={"time": 1})

    with dask.config.set(scheduler="synchronous"):
        ds.ux.isel(x=5, y=5, z=5).values

    assert len(keys_read) == 2
    assert all(
        _extent(part, size) == size
        for key in keys_read
        for part, size in zip(key, (32, 16, 16), strict=False)
    )


@ISSUE_63
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
