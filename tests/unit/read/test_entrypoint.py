import pathlib

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xarray_binfile.read.entrypoint import RawBinaryEntrypoint
from xarray_binfile.read.file_metadata import ReadSpecs, ReadSpecsGetterProtocol


def make_reader(name="ux", dtype=np.float32, coords=None):
    coords = coords or {"x": range(2), "y": range(3)}

    def reader(path: pathlib.Path) -> ReadSpecs:
        return ReadSpecs(filepath=path, dtype=dtype, coords=coords, name=name)

    return reader


def write_file(path: pathlib.Path, shape=(2, 3), dtype=np.float32) -> None:
    np.zeros(shape, dtype=dtype).tofile(path)


def test_open_dataset_exposes_variable(tmp_path):
    write_file(tmp_path / "ux.bin")
    dataset = RawBinaryEntrypoint().open_dataset(
        tmp_path / "ux.bin", read_specs_getter=make_reader()
    )
    assert list(dataset.data_vars) == ["ux"]


@pytest.mark.parametrize("drop_variables", ["ux", ["ux"], ("uy", "ux")])
def test_open_dataset_drops_matching_variable(tmp_path, drop_variables):
    dataset = RawBinaryEntrypoint().open_dataset(
        tmp_path / "ux.bin",
        read_specs_getter=make_reader(),
        drop_variables=drop_variables,
    )
    assert len(dataset.data_vars) == 0


def test_open_dataset_keeps_non_matching_variable(tmp_path):
    write_file(tmp_path / "ux.bin")
    dataset = RawBinaryEntrypoint().open_dataset(
        tmp_path / "ux.bin",
        read_specs_getter=make_reader(),
        drop_variables=["zz"],
    )
    assert list(dataset.data_vars) == ["ux"]


def test_open_dataset_invalid_path_raises_value_error():
    with pytest.raises(ValueError, match="Expected a file path"):
        RawBinaryEntrypoint().open_dataset(123, read_specs_getter=make_reader())


def test_open_dataset_wraps_metadata_errors(tmp_path):
    def failing_reader(path: pathlib.Path) -> ReadSpecs:
        error_message = "cannot parse"
        raise RuntimeError(error_message)

    with pytest.raises(ValueError, match="Error reading metadata"):
        RawBinaryEntrypoint().open_dataset(
            tmp_path / "ux.bin", read_specs_getter=failing_reader
        )


def test_open_dataset_calls_getter_positionally(tmp_path):
    # https://github.com/fschuch/xarray-binfile/issues/71 and #73: any
    # single-argument callable is a valid getter, whatever its parameter name.
    write_file(tmp_path / "ux.bin")

    def custom_reader(filepath: pathlib.Path) -> ReadSpecs:
        return ReadSpecs(filepath, np.float32, {"x": range(2), "y": range(3)}, "ux")

    getter: ReadSpecsGetterProtocol = custom_reader  # checked by mypy
    dataset = xr.open_dataset(
        tmp_path / "ux.bin", engine="binfile", read_specs_getter=getter
    )

    assert list(dataset.data_vars) == ["ux"]


def test_read_specs_shape_does_not_compute_lazy_coordinates(tmp_path):
    # ReadSpecs.shape reads the length of a Dask coordinate without computing it.
    def forbid_compute(*args, **kwargs):  # pragma: no cover - runs only on regression
        error_message = "ReadSpecs.shape computed a lazy coordinate"
        raise AssertionError(error_message)

    lazy = da.arange(5, chunks=2)
    specs = ReadSpecs(tmp_path / "u.bin", np.float32, {"x": lazy}, "u")
    with dask.config.set(scheduler=forbid_compute):
        assert specs.shape == (5,)
