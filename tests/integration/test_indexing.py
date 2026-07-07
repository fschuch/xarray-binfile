import pathlib

import numpy as np
import pytest
import xarray as xr

from xarray_binfile.read.file_metadata import ReadSpecs

SHAPE = (4, 5, 6)
COORDS = {"x": np.arange(4), "y": np.arange(5), "z": np.arange(6)}


def reader(path: pathlib.Path) -> ReadSpecs:
    return ReadSpecs(filepath=path, dtype=np.float64, coords=COORDS, name="ux")


@pytest.fixture
def reference(tmp_path) -> np.ndarray:
    data = np.arange(np.prod(SHAPE), dtype=np.float64).reshape(SHAPE)
    data.tofile(tmp_path / "ux.bin")
    return data


@pytest.fixture(params=[None, {"x": 2}], ids=["eager", "chunked"])
def ux(request, tmp_path, reference) -> xr.DataArray:
    dataset = xr.open_dataset(
        tmp_path / "ux.bin",
        engine="binfile",
        read_specs_getter=reader,
        chunks=request.param,
    )
    return dataset["ux"]


def test_isel_with_integer(ux, reference):
    np.testing.assert_array_equal(ux.isel(x=1).values, reference[1])


def test_isel_with_list(ux, reference):
    np.testing.assert_array_equal(ux.isel(x=[0, 2]).values, reference[[0, 2]])


def test_isel_with_negative_step(ux, reference):
    np.testing.assert_array_equal(
        ux.isel(x=slice(None, None, -1)).values, reference[::-1]
    )


def test_sel_nearest_scalar(ux, reference):
    np.testing.assert_array_equal(ux.sel(x=1.2, method="nearest").values, reference[1])


def test_pointwise_vectorized_indexing(ux, reference):
    points = xr.DataArray([0, 1, 2], dims="points")
    np.testing.assert_array_equal(
        ux.isel(x=points, y=points, z=points).values,
        reference[[0, 1, 2], [0, 1, 2], [0, 1, 2]],
    )


def test_full_load(ux, reference):
    np.testing.assert_array_equal(ux.values, reference)
