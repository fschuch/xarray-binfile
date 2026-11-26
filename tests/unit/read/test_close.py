"""
Datasets from the backend must close.

From the pyOpenSci review, https://github.com/fschuch/xarray-binfile/issues/69:
``with convention.open(...) as ds:`` raised ``TypeError: 'NoneType' object is
not callable`` when the block ended, and so did ``ds.close()``, unless the
convention had stacks. xarray calls the closer of every Dataset the backend
returns, and the backend left it unset.
"""

import numpy as np
import pytest
import xarray as xr

from xarray_binfile.conventions import Layout, StepIndexedFiles
from xarray_binfile.read.file_metadata import ReadSpecs


def _reader(path):
    return ReadSpecs(filepath=path, dtype="<f4", coords={"x": range(3)}, name="ux")


@pytest.fixture
def ux(tmp_path):
    path = tmp_path / "ux.bin"
    np.zeros(3, "<f4").tofile(path)
    return path


def test_open_mfdataset_on_backend_files_closes(ux):
    dataset = xr.open_mfdataset([ux], engine="binfile", read_specs_getter=_reader)

    dataset.close()


@pytest.mark.parametrize("open_kwargs", [{}, {"chunks": {}}], ids=["default", "chunks"])
def test_an_opened_convention_closes_at_the_end_of_a_with_block(tmp_path, open_kwargs):
    convention = StepIndexedFiles(Layout({"x": np.arange(3)}, dtype="<f4"))
    for step in (1, 2):
        np.full(3, step, "<f4").tofile(tmp_path / f"ux-{step:04d}.bin")

    with convention.open(tmp_path, **open_kwargs) as dataset:
        assert dataset.ux.shape == (3, 2)
