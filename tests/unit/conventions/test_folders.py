import pathlib

import numpy as np
import pytest
import xarray as xr

from xarray_binfile.conventions import (
    FolderConventions,
    Layout,
    LayoutMismatchError,
    StaticFiles,
    StepIndexedFiles,
)

X, Y, Z = np.arange(3), np.arange(2), np.arange(4)
CONVENTIONS = FolderConventions(
    {
        "xy_planes": StepIndexedFiles(Layout({"x": X, "y": Y})),
        "snapshots/3d": StepIndexedFiles(Layout({"x": X, "y": Y, "z": Z})),
        "static": StaticFiles(Layout({"x": X, "y": Y, "z": Z})),
    }
)


def test_reader_dispatches_on_parent_folder():
    plane = CONVENTIONS.reader(pathlib.Path("/case/xy_planes/ux-0001.bin"))
    volume = CONVENTIONS.reader(pathlib.Path("case/snapshots/3d/ux-0001.bin"))

    assert plane.dims == ("x", "y", "time")
    assert volume.dims == ("x", "y", "z", "time")


def test_reader_prefers_most_specific_folder_regardless_of_order():
    conventions = FolderConventions(
        {
            "3d": StepIndexedFiles(Layout({"x": X})),
            "snapshots/3d": StepIndexedFiles(Layout({"x": X, "y": Y, "z": Z})),
        }
    )

    specs = conventions.reader(pathlib.Path("case/snapshots/3d/ux-0001.bin"))

    assert specs.dims == ("x", "y", "z", "time")
    assert conventions.reader(pathlib.Path("case/3d/ux-0001.bin")).dims == ("x", "time")


def test_reader_rejects_unknown_folder():
    with pytest.raises(ValueError, match="No convention registered"):
        CONVENTIONS.reader(pathlib.Path("case/3d/ux-0001.bin"))


def test_writer_picks_the_single_matching_convention_and_prefixes_folder():
    array = xr.DataArray(
        np.zeros((2, 3, 2)), coords={"time": [0, 1], "x": X, "y": Y}, name="ux"
    )

    assert [s.filename for s in CONVENTIONS.writer(array)] == [
        "xy_planes/ux-0000.bin",
        "xy_planes/ux-0001.bin",
    ]


def test_writer_rejects_when_nothing_matches():
    array = xr.DataArray(np.zeros(3), coords={"x": X}, name="ux")

    with pytest.raises(LayoutMismatchError, match="No convention accepts"):
        next(CONVENTIONS.writer(array))


def test_writer_rejects_ambiguity():
    ambiguous = FolderConventions(
        {
            "a": StepIndexedFiles(Layout({"x": X})),
            "b": StepIndexedFiles(Layout({"x": X}), pattern="{name}.{step:d}"),
        }
    )
    array = xr.DataArray(np.zeros((1, 3)), coords={"time": [0], "x": X}, name="ux")

    with pytest.raises(LayoutMismatchError, match="Several conventions accept"):
        next(ambiguous.writer(array))


def test_writer_accepts_convention_that_yields_nothing():
    array = xr.DataArray(
        np.zeros((0, 3, 2)), coords={"time": [], "x": X, "y": Y}, name="ux"
    )

    assert list(CONVENTIONS.writer(array)) == []
