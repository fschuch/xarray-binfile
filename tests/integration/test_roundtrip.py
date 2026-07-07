import pathlib
import tempfile

import numpy as np
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

import xarray_binfile  # noqa: F401  (registers the accessors)
from xarray_binfile.read import ReadSpecs
from xarray_binfile.write import WriteSpecs

DTYPES = st.sampled_from(["<f4", ">f4", "<f8", ">f8", "<i4", ">i8"])
SHAPES = npst.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=5)


@settings(max_examples=25, deadline=None)
@given(data=st.data(), dtype=DTYPES, shape=SHAPES)
def test_write_read_roundtrip_is_exact(data, dtype, shape):
    values = data.draw(
        npst.arrays(
            dtype=dtype,
            shape=shape,
            elements={"allow_nan": False, "allow_infinity": False},
        )
    )
    coords = {f"dim_{i}": np.arange(size) for i, size in enumerate(shape)}
    data_array = xr.DataArray(values, coords=coords, name="ux")

    def writer(array: xr.DataArray):
        yield WriteSpecs(filename="ux.bin", sub_array=array, dtype=dtype)

    def reader(path: pathlib.Path) -> ReadSpecs:
        return ReadSpecs(filepath=path, dtype=dtype, coords=coords, name="ux")

    with tempfile.TemporaryDirectory() as tmp_dir:
        directory = pathlib.Path(tmp_dir)
        data_array.binary_engine.to_file(writer, directory)
        roundtrip = xr.open_dataset(
            directory / "ux.bin", engine="binfile", read_specs_getter=reader
        ).load()

    np.testing.assert_array_equal(roundtrip["ux"].values, values)
