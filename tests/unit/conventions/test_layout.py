import numpy as np
import pytest
import xarray as xr

from xarray_binfile.conventions import Layout, LayoutMismatchError


@pytest.fixture
def layout():
    return Layout({"x": np.arange(3), "y": np.linspace(0, 1, 4)}, dtype="<f4")


def test_dims_and_shape(layout):
    assert layout.dims == ("x", "y")
    assert layout.shape == (3, 4)


def test_validate_accepts_matching_array_in_any_order(layout):
    array = xr.DataArray(
        np.zeros((4, 2, 3)),
        coords={"y": np.linspace(0, 1, 4), "time": [0, 1], "x": np.arange(3)},
        dims=("y", "time", "x"),
    )

    layout.validate(array, extra_dims=("time",))
    assert layout.transpose(array.isel(time=0)).dims == ("x", "y")


def test_validate_rejects_missing_dimension(layout):
    array = xr.DataArray(np.zeros(3), coords={"x": np.arange(3)})

    with pytest.raises(LayoutMismatchError, match="Dimension mismatch"):
        layout.validate(array)


def test_validate_rejects_unexpected_dimension(layout):
    array = xr.DataArray(
        np.zeros((3, 4, 2)),
        coords={"x": np.arange(3), "y": np.linspace(0, 1, 4), "z": [0, 1]},
    )

    with pytest.raises(LayoutMismatchError, match="Dimension mismatch"):
        layout.validate(array)


def test_validate_rejects_size_mismatch(layout):
    array = xr.DataArray(
        np.zeros((3, 5)), coords={"x": np.arange(3), "y": np.linspace(0, 1, 5)}
    )

    with pytest.raises(LayoutMismatchError, match="Size mismatch on dimension 'y'"):
        layout.validate(array)


def test_validate_rejects_coordinate_values_by_default(layout):
    array = xr.DataArray(
        np.zeros((3, 4)), coords={"x": np.arange(3) + 10, "y": np.linspace(0, 1, 4)}
    )

    with pytest.raises(LayoutMismatchError, match="Coordinate mismatch on dimension 'x'"):
        layout.validate(array)
    layout.validate(array, check_coords=False)


def test_validate_accepts_dimension_without_coordinate(layout):
    array = xr.DataArray(np.zeros((3, 4)), dims=("x", "y"))

    layout.validate(array)
