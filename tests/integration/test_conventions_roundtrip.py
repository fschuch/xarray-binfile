import pathlib
import warnings

import numpy as np
import pytest
import xarray as xr

import xarray_binfile  # noqa: F401  (registers the accessors)
from xarray_binfile.conventions import (
    FolderConventions,
    Layout,
    StaticFiles,
    StepIndexedFiles,
    TimeStampedFiles,
)
from xarray_binfile.tutorial import FileSpecsGetter

X = np.linspace(0.0, 1.0, 4, dtype=np.float32)
Y = np.linspace(-1.0, 1.0, 3, dtype=np.float32)
Z = np.linspace(0.0, 2.0, 5, dtype=np.float32)
RNG = np.random.default_rng(0)


def _deprecated_getter():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return FileSpecsGetter(base_coords={"x": X, "y": Y, "z": Z}, dtype="<f4")


def _dataset(time, dims_coords):
    coords = dict(dims_coords) | ({"time": time} if time is not None else {})
    shape = tuple(len(v) for v in coords.values())
    return xr.Dataset(
        {
            name: (tuple(coords), RNG.random(shape, dtype=np.float32))
            for name in ("ux", "uy")
        },
        coords=coords,
    )


@pytest.mark.parametrize(
    ("convention", "time"),
    [
        (StepIndexedFiles(Layout({"x": X, "y": Y, "z": Z}, dtype="<f4")), [0, 1, 2]),
        (
            StepIndexedFiles(
                Layout({"x": X, "y": Y, "z": Z}, dtype="<f4"), time_step=0.5
            ),
            [0.0, 0.5, 1.0],
        ),
        (
            TimeStampedFiles(Layout({"x": X, "y": Y, "z": Z}, dtype="<f4")),
            [0.125, 0.25, 1.5],
        ),
        (StaticFiles(Layout({"x": X, "y": Y, "z": Z}, dtype="<f4")), None),
        (_deprecated_getter(), [0, 1, 2]),
    ],
    ids=["step", "step-with-time_step", "time-stamped", "static", "deprecated"],
)
def test_roundtrip(tmp_path: pathlib.Path, convention, time):
    # Dims deliberately out of on-disk order to exercise the transpose.
    dataset = _dataset(time, {"z": Z, "x": X, "y": Y})

    dataset.binary_engine.to_file(convention.writer, tmp_path)
    roundtrip = xr.open_mfdataset(
        sorted(tmp_path.glob("*.bin")),
        engine="binfile",
        read_specs_getter=convention.reader,
    ).load()

    xr.testing.assert_identical(roundtrip, dataset.transpose(*roundtrip["ux"].dims))


def test_folder_conventions_roundtrip(tmp_path: pathlib.Path):
    conventions = FolderConventions(
        {
            "xy_planes": StepIndexedFiles(Layout({"x": X, "y": Y}, dtype="<f4")),
            "3d": StepIndexedFiles(Layout({"x": X, "y": Y, "z": Z}, dtype="<f4")),
            "static": StaticFiles(Layout({"x": X, "y": Y, "z": Z}, dtype="<f4")),
        }
    )
    planes = _dataset([0, 1], {"x": X, "y": Y})
    volumes = _dataset([0, 1], {"x": X, "y": Y, "z": Z}).rename(ux="vx", uy="vy")
    static = _dataset(None, {"x": X, "y": Y, "z": Z}).rename(ux="epsi", uy="mask")

    for dataset in (planes, volumes, static):
        dataset.binary_engine.to_file(conventions.writer, tmp_path)

    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.bin")) == [
        "3d/vx-0000.bin",
        "3d/vx-0001.bin",
        "3d/vy-0000.bin",
        "3d/vy-0001.bin",
        "static/epsi.bin",
        "static/mask.bin",
        "xy_planes/ux-0000.bin",
        "xy_planes/ux-0001.bin",
        "xy_planes/uy-0000.bin",
        "xy_planes/uy-0001.bin",
    ]
    roundtrip = xr.open_mfdataset(
        sorted(tmp_path.rglob("*.bin")),
        engine="binfile",
        read_specs_getter=conventions.reader,
    ).load()
    xr.testing.assert_identical(roundtrip, xr.merge([planes, volumes, static]))
