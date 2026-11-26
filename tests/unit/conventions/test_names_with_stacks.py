"""
A convention's ``names`` list the variables as they are stacked.

From the pyOpenSci review, https://github.com/fschuch/xarray-binfile/issues/69:
with a velocity stack and ``names=("u", "pp")``, the writer wrote ``ux``,
``uy``, ``uz`` and ``pp``, but ``open()`` returned only ``pp``, because the
reader checked the names on disk. Writing already checks the stacked name, so
reading now accepts a file whose name, or the stacked name it decodes into,
is listed.
"""

import pathlib

import numpy as np
import pytest
import xarray as xr

import xarray_binfile  # noqa: F401  (registers the accessors)
from xarray_binfile.conventions import Layout, StaticFiles, VariableStack

VELOCITY = VariableStack("i", "{name}{i}", values=("x", "y", "z"))
SCALARS = VariableStack("n", "{name}{n:d}", values=range(1, 10))
LAYOUT = Layout({"x": np.arange(3)})
X = np.arange(3)


def _velocity():
    return xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        coords={"i": ["x", "y", "z"], "x": X},
        name="u",
    )


def test_what_is_written_opens_back(tmp_path):
    convention = StaticFiles(LAYOUT, stacks=(VELOCITY,), names=("u", "pp"))
    dataset = xr.Dataset({"u": _velocity(), "pp": ("x", np.ones(3))}, coords={"x": X})

    dataset.binary_engine.to_file(convention.writer, tmp_path)
    opened = convention.open(tmp_path).load()

    assert sorted(opened.data_vars) == ["pp", "u"]
    xr.testing.assert_equal(opened["u"], dataset["u"])


def test_the_reader_keeps_the_name_on_disk():
    convention = StaticFiles(LAYOUT, stacks=(VELOCITY,), names=("u",))

    assert convention.reader(pathlib.Path("ux.bin")).name == "ux"
    assert convention.accepts(pathlib.Path("uz.bin"))


def test_nested_stacks_decode_to_the_listed_name():
    convention = StaticFiles(LAYOUT, stacks=(VELOCITY, SCALARS), names=("u",))

    assert convention.accepts(pathlib.Path("uy2.bin"))


@pytest.mark.parametrize(
    "filename",
    [
        "uw.bin",  # w is not a component
        "pp.bin",  # not listed
        "u1.bin",  # SCALARS only stacks phi
    ],
)
def test_other_files_are_still_rejected(filename):
    scalars = VariableStack("n", "{name}{n:d}", values=range(1, 10), names={"phi"})
    convention = StaticFiles(LAYOUT, stacks=(VELOCITY, scalars), names=("u",))

    assert not convention.accepts(pathlib.Path(filename))


@pytest.mark.xfail(strict=True, reason="PR #93 review: stacking ignores names")
@pytest.mark.parametrize("with_velocity", [False, True], ids=["numbers", "nested"])
def test_a_listed_name_ending_in_a_digit_opens_back(tmp_path, with_velocity):
    numbers = VariableStack("n", "{name}{n:d}", values=range(1, 13))
    stacks = (numbers, VELOCITY) if with_velocity else (numbers,)
    convention = StaticFiles(LAYOUT, stacks=stacks, names=("phi1",))
    phi1 = xr.DataArray(
        np.arange(6.0).reshape(2, 3), coords={"n": [1, 2], "x": X}, name="phi1"
    )
    if with_velocity:
        phi1 = phi1.expand_dims(i=["x", "y"]).copy()

    phi1.to_dataset().binary_engine.to_file(convention.writer, tmp_path)
    opened = convention.open(tmp_path).load()

    assert list(opened.data_vars) == ["phi1"]
    xr.testing.assert_equal(opened["phi1"].transpose(*phi1.dims), phi1)


def test_names_on_disk_still_read(tmp_path):
    convention = StaticFiles(LAYOUT, stacks=(VELOCITY,), names=("ux", "uy", "uz"))
    for index, component in enumerate("xyz"):
        np.full(3, float(index)).tofile(tmp_path / f"u{component}.bin")

    assert list(convention.open(tmp_path).data_vars) == ["u"]
