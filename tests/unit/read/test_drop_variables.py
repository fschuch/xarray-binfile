"""
``drop_variables`` on the backend.

From the pyOpenSci review, https://github.com/fschuch/xarray-binfile/issues/69:
a string went through the ``in`` check, so ``drop_variables="rhou"`` also
dropped ``rho``. And a dropped file came back as an empty Dataset, which
xarray cannot combine, so ``open_mfdataset`` failed as soon as two files were
dropped. A dropped file now keeps its coordinates, as a netCDF file keeps its
dimensions when one of its variables is dropped.
"""

import numpy as np
import pytest
import xarray as xr

from xarray_binfile.read.file_metadata import ReadSpecs


def _reader(path):
    name, _, step = path.stem.partition("-")
    return ReadSpecs(
        filepath=path,
        dtype="=f8",
        coords={"x": range(3), "time": [int(step)]},
        name=name,
        attrs={"case": "demo"},
    )


def _write(folder, files):
    for name, steps in files.items():
        for step in steps:
            np.full(3, float(step)).tofile(folder / f"{name}-{step}.bin")
    return sorted(folder.iterdir())


def _open(paths, **kwargs):
    return xr.open_mfdataset(
        paths, engine="binfile", read_specs_getter=_reader, join="outer", **kwargs
    )


def test_a_string_drops_that_name_only(tmp_path):
    (path,) = _write(tmp_path, {"rho": [0]})

    opened = xr.open_dataset(
        path, engine="binfile", read_specs_getter=_reader, drop_variables="rhou"
    )

    assert list(opened.data_vars) == ["rho"]


@pytest.mark.parametrize("drop", ["uy", ["uy"]])
def test_a_variable_spanning_files_is_dropped(tmp_path, drop):
    paths = _write(tmp_path, {"ux": [0, 1], "uy": [0, 1]})

    opened = _open(paths, drop_variables=drop)

    assert list(opened.data_vars) == ["ux"]
    assert opened["time"].values.tolist() == [0, 1]
    assert opened.attrs == {"case": "demo"}


def test_a_dropped_file_keeps_its_coordinates(tmp_path):
    paths = _write(tmp_path, {"ux": [0, 1], "uy": [0, 1, 2]})

    opened = _open(paths, drop_variables=["uy"])

    # As with netCDF; Convention.open skips dropped files instead.
    assert opened["time"].values.tolist() == [0, 1, 2]
    assert opened["ux"].sel(time=2).isnull().all()


def test_a_dropped_file_is_not_read(tmp_path):
    missing = tmp_path / "ux-0.bin"

    opened = xr.open_dataset(
        missing, engine="binfile", read_specs_getter=_reader, drop_variables=["ux"]
    )

    assert list(opened.data_vars) == []
