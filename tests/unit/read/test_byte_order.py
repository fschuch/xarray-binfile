"""
Arrays come back in the machine's native byte order.

From the pyOpenSci review, https://github.com/fschuch/xarray-binfile/issues/69:
a big-endian file opened on its own stayed big-endian in memory (``>f8``), and
``to_dataframe()["u"].mean()`` then failed in pandas. xarray's netCDF and zarr
backends return native arrays, and so does this one now. The byte order on
disk is still declared by the read specs.
"""

import sys

import numpy as np
import pytest
import xarray as xr

from xarray_binfile.read.file_metadata import ReadSpecs

FOREIGN = ">f8" if sys.byteorder == "little" else "<f8"
VALUES = np.arange(12.0).reshape(3, 4)


@pytest.fixture
def foreign(tmp_path):
    """A float64 file in the byte order this machine does not use."""
    path = tmp_path / "u.bin"
    VALUES.astype(FOREIGN).tofile(path)
    return path


def _reader(path):
    return ReadSpecs(
        filepath=path, dtype=FOREIGN, coords={"x": range(3), "y": range(4)}, name="u"
    )


def _open(path, **kwargs):
    return xr.open_dataset(path, engine="binfile", read_specs_getter=_reader, **kwargs)


def test_the_variable_is_native_before_it_is_loaded(foreign):
    dataset = _open(foreign)

    assert dataset["u"].dtype == np.dtype("float64")
    assert dataset["u"].dtype.isnative


@pytest.mark.parametrize(
    "select",
    [lambda u: u, lambda u: u.isel(x=slice(1, 3))],
    ids=["whole", "part"],  # one read of the file, and the memory-mapped path
)
def test_loaded_values_are_native_and_correct(foreign, select):
    loaded = select(_open(foreign)["u"]).values

    assert loaded.dtype.isnative
    np.testing.assert_array_equal(loaded, select(xr.DataArray(VALUES, dims=("x", "y"))))


def test_a_chunked_variable_is_native(foreign):
    pytest.importorskip("dask")
    u = _open(foreign, chunks={})["u"]

    assert u.dtype.isnative
    assert u.compute().dtype.isnative


def test_the_reviewers_mean_through_pandas_works(foreign):
    frame = _open(foreign).to_dataframe()

    assert frame["u"].mean() == VALUES.mean()


def test_a_native_file_reads_as_before(tmp_path):
    path = tmp_path / "u.bin"
    VALUES.tofile(path)
    specs = ReadSpecs(
        filepath=path, dtype="=f8", coords={"x": range(3), "y": range(4)}, name="u"
    )

    u = xr.open_dataset(path, engine="binfile", read_specs_getter=lambda _: specs)["u"]

    assert u.dtype == np.dtype("float64")
    np.testing.assert_array_equal(u.values, VALUES)
