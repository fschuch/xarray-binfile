"""
Loaded data must not stay tied to the file it was read from.

From the pyOpenSci review, https://github.com/fschuch/xarray-binfile/issues/61:
a sliced read used to return a view of the memory-mapped file, so a loaded
slice changed when the file was rewritten, kept the file locked on Windows,
and held one file descriptor per slice. With Dask, every chunk held one until
the chunks were joined, so loading a finely chunked dataset could hit the
open-file limit ("Too many open files" at 256 on macOS).
"""

import os
import pathlib

import dask
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from xarray_binfile import ReadSpecs

ISSUE_61 = pytest.mark.xfail(
    strict=True,
    reason="https://github.com/fschuch/xarray-binfile/issues/61: "
    "sliced reads return a view of the memory-mapped file",
)


def _reader(path: pathlib.Path) -> ReadSpecs:
    return ReadSpecs(path, "f8", {"x": range(3), "y": range(4)}, "u")


def _open(path: pathlib.Path, chunks=None) -> xr.Dataset:
    return xr.open_dataset(
        path, engine="binfile", read_specs_getter=_reader, chunks=chunks
    )


@pytest.mark.parametrize(
    "chunks",
    [
        pytest.param(None, id="lazy", marks=ISSUE_61),
        # Dask joins the chunks into a new array, so this path already copied.
        pytest.param({"x": 1}, id="dask"),
    ],
)
def test_loaded_slice_is_independent_of_the_file(tmp_path, chunks):
    path = tmp_path / "u.bin"
    np.arange(12.0).tofile(path)

    row = _open(path, chunks).u.isel(x=0).load()
    # Rewrite in place. While a mapping is alive this changes the loaded
    # values on Linux and macOS, and raises PermissionError on Windows.
    np.zeros(12).tofile(path)

    np.testing.assert_array_equal(row.values, [0.0, 1.0, 2.0, 3.0])


def _open_descriptors() -> int:
    return len(os.listdir("/dev/fd"))


NEEDS_DEV_FD = pytest.mark.skipif(
    not os.path.isdir("/dev/fd"),
    reason="counts open file descriptors through /dev/fd (Linux and macOS)",
)


@ISSUE_61
@NEEDS_DEV_FD
def test_loaded_slices_do_not_hold_file_descriptors(tmp_path):
    path = tmp_path / "u.bin"
    np.arange(12.0).tofile(path)

    before = _open_descriptors()
    rows = [_open(path).u.isel(x=0).load() for _ in range(20)]
    growth = _open_descriptors() - before

    # One descriptor per slice before the fix; a small slack absorbs
    # descriptors opened by unrelated library code.
    assert growth < 5, f"{growth} file descriptors held by {len(rows)} slices"


class _PeakDescriptors(Callback):
    """Dask callback recording the most file descriptors open after any task."""

    def __init__(self) -> None:
        super().__init__()
        self.peak = 0

    def _posttask(self, *args, **kwargs) -> None:
        self.peak = max(self.peak, _open_descriptors())


@ISSUE_61
@NEEDS_DEV_FD
def test_chunked_load_does_not_hold_one_descriptor_per_chunk(tmp_path):
    path = tmp_path / "u.bin"
    np.arange(256.0).tofile(path)
    coords = {"x": range(8), "y": range(8), "z": range(4)}
    dataset = xr.open_dataset(
        path,
        engine="binfile",
        read_specs_getter=lambda p: ReadSpecs(p, "f8", coords, "u"),
        chunks={"x": 2, "y": 2, "z": 1},
    )
    chunks = dataset.u.data.npartitions  # 64

    before = _open_descriptors()
    with dask.config.set(scheduler="synchronous"), _PeakDescriptors() as tracker:
        dataset.load()
    growth = tracker.peak - before

    # Dask keeps every chunk until it joins them, so the peak used to grow by
    # one descriptor per chunk.
    assert growth < 5, f"{growth} file descriptors open at once for {chunks} chunks"
