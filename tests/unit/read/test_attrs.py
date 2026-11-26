"""
Where ``ReadSpecs.attrs`` end up.

From the pyOpenSci review, https://github.com/fschuch/xarray-binfile/issues/69:
the docstring said "dataset or variable". They are Dataset attributes, and
across files xarray keeps the first file's. Per-variable attributes are
tracked in https://github.com/fschuch/xarray-binfile/issues/91; until then
they can be set after loading.
"""

import numpy as np
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


def test_attrs_are_dataset_attributes_and_the_first_file_wins(tmp_path):
    paths = [tmp_path / "a.bin", tmp_path / "b.bin"]
    for path in paths:
        np.zeros(2).tofile(path)

    dataset = xr.open_mfdataset(paths, engine="binfile", read_specs_getter=_reader)

    assert dataset.attrs == {"source": "a"}
    assert dataset["a"].attrs == {}
    assert dataset["b"].attrs == {}
