"""
An empty selection must load as empty, not as the whole file.

From the pyOpenSci review, https://github.com/fschuch/xarray-binfile/issues/62:
a slice with a stop of 0 was taken for "up to the end", so the whole file was
read. Lazily, the shape said empty while the values were the full array. With
Dask, which asks for ``slice(0, 0)`` whenever an empty window starts on a chunk
boundary, reductions silently covered everything, or the chunks could not be
joined at all.
"""

import pathlib

import numpy as np
import pytest
import xarray as xr

from xarray_binfile import ReadSpecs
from xarray_binfile.conventions import Layout, StaticFiles

VALUES = np.arange(12.0).reshape(3, 4)


def _reader(path: pathlib.Path) -> ReadSpecs:
    return ReadSpecs(path, "f8", {"x": [0.0, 1.0, 2.0], "y": range(4)}, "u")


@pytest.fixture
def path(tmp_path) -> pathlib.Path:
    path = tmp_path / "u.bin"
    VALUES.tofile(path)
    return path


SELECTIONS = [
    pytest.param(lambda u: u.sel(x=slice(-2, -1)), id="below-grid"),
    pytest.param(lambda u: u.head(x=0), id="head-0"),
    pytest.param(lambda u: u[:0], id="getitem-0"),
    pytest.param(lambda u: u.sel(x=slice(5, 6)), id="above-grid"),
]


@pytest.mark.parametrize("select", SELECTIONS)
def test_empty_selection_loads_empty(path, select):
    ds = xr.open_dataset(path, engine="binfile", read_specs_getter=_reader)

    loaded = select(ds.u).load()

    assert loaded.shape == (0, 4)
    assert loaded.values.shape == (0, 4)


# With Dask chunks the window above the grid also reaches the backend as
# ``slice(0, 0)``, because the end of the array is a chunk boundary.
@pytest.mark.parametrize("select", SELECTIONS)
def test_empty_selection_loads_empty_with_dask_chunks(path, select):
    ds = xr.open_dataset(
        path, engine="binfile", read_specs_getter=_reader, chunks={"x": 1}
    )

    loaded = select(ds.u).load()

    assert loaded.shape == (0, 4)
    assert loaded.values.shape == (0, 4)


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_mean_of_an_empty_window_from_a_convention_is_nan(tmp_path):
    convention = StaticFiles(Layout({"x": np.arange(3.0), "y": np.arange(4.0)}))
    xr.Dataset({"u": (("x", "y"), VALUES)}).binary_engine.to_file(
        convention.writer, tmp_path
    )
    ds = convention.open(tmp_path)

    mean = ds.u.sel(x=slice(5, 6)).mean().values

    assert np.isnan(mean)
