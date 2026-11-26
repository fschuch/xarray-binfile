"""
Where ``ReadSpecs.attrs`` end up.

From the pyOpenSci review, https://github.com/fschuch/xarray-binfile/issues/69:
the docstring said "dataset or variable". They are Dataset attributes, and
across files xarray keeps one file's, chosen by how it orders and combines the
files: along a dimension it sorts them by coordinate, so it is not always the
first path. Per-variable attributes are tracked in
https://github.com/fschuch/xarray-binfile/issues/91; until then they can be
set after loading.
"""

import numpy as np
import pytest
import xarray as xr

from xarray_binfile.read.file_metadata import ReadSpecs


def _reader(path):
    return ReadSpecs(
        filepath=path,
        dtype="<f8",
        coords={"x": range(2)},
        name=path.stem,
        attrs={"source": path.stem},
    )


@pytest.mark.parametrize("names", [("a", "b"), ("b", "a")])
def test_attrs_are_dataset_attributes_from_one_file(tmp_path, names):
    paths = [tmp_path / f"{name}.bin" for name in names]
    for path in paths:
        np.zeros(2).tofile(path)

    dataset = xr.open_mfdataset(paths, engine="binfile", read_specs_getter=_reader)

    assert dataset.attrs in ({"source": "a"}, {"source": "b"})
    assert dataset["a"].attrs == {}
    assert dataset["b"].attrs == {}
