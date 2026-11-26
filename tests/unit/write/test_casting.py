"""
The cast to the write spec's ``dtype`` must not lose values silently.

From the pyOpenSci review, https://github.com/fschuch/xarray-binfile/issues/69:
floats written through an int32 layout were truncated with no warning (1.9
became 1), and NaN became 0. ``to_file`` now refuses a cast that changes the
kind of the data, before writing any file. Casts within a kind (``f8`` to
``f4``, ``i8`` to ``i4``) are still allowed, as ``casting="same_kind"`` does.
"""

import numpy as np
import pytest
import xarray as xr

from xarray_binfile.write import WriteSpecs


def _through(dtype):
    def writer(array: xr.DataArray):
        yield WriteSpecs(filename="first.bin", sub_array=array.astype("<f8"))
        yield WriteSpecs(filename="ux.bin", sub_array=array, dtype=dtype)

    return writer


def _written(folder) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([1.9, 2.5, -0.7], "<i4"),  # the reviewer's case
        ([np.nan, 1.0, 2.0], "<i4"),  # NaN would become 0
        ([1 + 1j, 2, 3], "<f8"),  # the imaginary part would be dropped
    ],
    ids=["float-to-int", "nan-to-int", "complex-to-float"],
)
def test_a_cast_that_changes_the_kind_is_refused(tmp_path, values, dtype):
    array = xr.DataArray(np.array(values), dims=("x",), name="ux")

    writer = _through(dtype)

    with pytest.raises(TypeError, match="ux.bin") as error:
        array.binary_engine.to_file(writer, tmp_path)
    assert str(array.dtype) in str(error.value)
    assert _written(tmp_path) == []  # refused before the first file


def test_a_lazy_array_is_refused_without_being_computed(tmp_path):
    dask = pytest.importorskip("dask.array")

    def never(block):  # no cover - only runs if the array is computed
        raise AssertionError("the array was computed")

    lazy = dask.ones(2, chunks=1).map_blocks(never, dtype="<f8")
    array = xr.DataArray(lazy, dims=("x",))

    writer = _through("<i4")

    with pytest.raises(TypeError, match="ux.bin"):
        array.binary_engine.to_file(writer, tmp_path)
    assert _written(tmp_path) == []


def test_a_dataset_is_refused_before_its_first_variable_is_written(tmp_path):
    dataset = xr.Dataset(
        {"a": ("x", np.array([1, 2], "<i4")), "b": ("x", np.array([1.9, 2.5]))}
    )

    def writer(array: xr.DataArray):
        yield WriteSpecs(filename=f"{array.name}.bin", sub_array=array, dtype="<i4")

    with pytest.raises(TypeError, match="b.bin"):
        dataset.binary_engine.to_file(writer, tmp_path)
    assert _written(tmp_path) == []


@pytest.mark.parametrize(
    ("source", "dtype"),
    [("<f8", "<f4"), ("<f8", ">f8"), ("<i8", "<i4"), ("<i4", "<f8"), ("|b1", "<i4")],
)
def test_a_cast_within_a_kind_or_to_a_wider_one_is_allowed(tmp_path, source, dtype):
    array = xr.DataArray(np.array([1, 0, 1], dtype=source), dims=("x",), name="ux")

    array.binary_engine.to_file(_through(dtype), tmp_path)

    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "ux.bin", dtype=dtype), [1, 0, 1]
    )
