import numpy as np
import pytest
import xarray as xr

import xarray_binfile  # noqa: F401  (registers the accessors)
from xarray_binfile.write import WriteSpecs


@pytest.fixture(params=["numpy", "dask"])
def data_array(request) -> xr.DataArray:
    array = xr.DataArray(
        np.arange(6, dtype=np.float64).reshape(2, 3),
        coords={"x": np.arange(2), "y": np.arange(3)},
        name="ux",
    )
    if request.param == "dask":
        array = array.chunk({"x": 1})
    return array


def whole_array_writer(data_array: xr.DataArray):
    yield WriteSpecs(filename=f"{data_array.name}.bin", sub_array=data_array)


def casting_writer(data_array: xr.DataArray):
    yield WriteSpecs(
        filename=f"{data_array.name}.bin", sub_array=data_array, dtype="<f4"
    )


def test_to_file_writes_in_memory_dtype(tmp_path, data_array):
    data_array.binary_engine.to_file(whole_array_writer, tmp_path)

    written = np.fromfile(tmp_path / "ux.bin", dtype=np.float64).reshape(2, 3)
    np.testing.assert_array_equal(written, data_array.values)


def test_to_file_casts_to_requested_dtype(tmp_path, data_array):
    data_array.binary_engine.to_file(casting_writer, tmp_path)

    written = np.fromfile(tmp_path / "ux.bin", dtype="<f4").reshape(2, 3)
    assert (tmp_path / "ux.bin").stat().st_size == 6 * 4
    np.testing.assert_array_equal(written, data_array.values.astype("<f4"))


def test_to_file_defaults_to_current_directory(tmp_path, data_array, monkeypatch):
    monkeypatch.chdir(tmp_path)

    data_array.binary_engine.to_file(whole_array_writer)

    assert (tmp_path / "ux.bin").exists()


def test_to_file_leaves_no_temporary_artifacts(tmp_path, data_array):
    data_array.binary_engine.to_file(whole_array_writer, tmp_path)

    leftovers = [path.name for path in tmp_path.iterdir() if path.name != "ux.bin"]
    assert leftovers == []


def test_to_file_overwrites_existing_file_completely(tmp_path, data_array):
    (tmp_path / "ux.bin").write_bytes(b"\x00" * 1024)

    data_array.binary_engine.to_file(whole_array_writer, tmp_path)

    assert (tmp_path / "ux.bin").stat().st_size == 6 * 8
    written = np.fromfile(tmp_path / "ux.bin", dtype=np.float64).reshape(2, 3)
    np.testing.assert_array_equal(written, data_array.values)


def test_to_file_keeps_completed_files_and_cleans_up_on_failure(tmp_path, data_array):
    def failing_writer(array: xr.DataArray):
        yield WriteSpecs(filename="first.bin", sub_array=array)
        error_message = "boom"
        raise RuntimeError(error_message)

    with pytest.raises(RuntimeError, match="boom"):
        data_array.binary_engine.to_file(failing_writer, tmp_path)

    # the file completed before the failure was atomically moved into place,
    # and no temporary directory or partial file is left behind
    assert [path.name for path in tmp_path.iterdir()] == ["first.bin"]
    written = np.fromfile(tmp_path / "first.bin", dtype=np.float64).reshape(2, 3)
    np.testing.assert_array_equal(written, data_array.values)


def test_dataset_to_file_writes_every_variable(tmp_path, data_array):
    dataset = xr.Dataset({"ux": data_array, "uy": data_array + 1.0})

    dataset.binary_engine.to_file(whole_array_writer, tmp_path)

    assert sorted(path.name for path in tmp_path.glob("*.bin")) == [
        "ux.bin",
        "uy.bin",
    ]
    written = np.fromfile(tmp_path / "uy.bin", dtype=np.float64).reshape(2, 3)
    np.testing.assert_array_equal(written, data_array.values + 1.0)
