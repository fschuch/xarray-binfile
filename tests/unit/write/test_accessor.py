import os

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


def test_to_file_creates_sub_folders(tmp_path, data_array):
    def nested(array):
        yield WriteSpecs(filename="snapshots/3d/ux.bin", sub_array=array)

    data_array.binary_engine.to_file(nested, tmp_path)

    assert (tmp_path / "snapshots" / "3d" / "ux.bin").stat().st_size == 6 * 8
    assert sorted(p.name for p in tmp_path.rglob("*")) == ["3d", "snapshots", "ux.bin"]


def test_to_file_accepts_absolute_paths_outside_directory(tmp_path, data_array):
    elsewhere = tmp_path / "run_B" / "3d"

    def absolute(array):
        yield WriteSpecs(filename=str(elsewhere / "ux.bin"), sub_array=array)

    data_array.binary_engine.to_file(absolute, tmp_path / "run_A")

    assert (elsewhere / "ux.bin").stat().st_size == 6 * 8
    assert not (tmp_path / "run_A").exists()
    assert [p.name for p in elsewhere.iterdir()] == ["ux.bin"]


@pytest.mark.parametrize("filename", ["../ux.bin", "sub/../../ux.bin"])
def test_to_file_rejects_relative_paths_escaping_directory(
    tmp_path, data_array, filename
):
    def escaping(array):
        yield WriteSpecs(filename=filename, sub_array=array)

    with pytest.raises(ValueError, match="escapes the output directory"):
        data_array.binary_engine.to_file(escaping, tmp_path / "out")

    assert not (tmp_path / "ux.bin").exists()


def test_to_file_requires_existing_directory_for_relative_paths(tmp_path, data_array):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        data_array.binary_engine.to_file(whole_array_writer, tmp_path / "missing")

    assert not (tmp_path / "missing").exists()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unsupported")
def test_to_file_follows_symlinked_sub_folder_inside_directory(tmp_path, data_array):
    target = tmp_path / "big_disk"
    target.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    (out / "3d").symlink_to(target, target_is_directory=True)

    def linked(array):
        yield WriteSpecs(filename="3d/ux.bin", sub_array=array)

    data_array.binary_engine.to_file(linked, out)

    assert (target / "ux.bin").stat().st_size == 6 * 8


def test_to_file_does_not_copy_when_dtype_is_unchanged(tmp_path, monkeypatch):
    array = xr.DataArray(np.arange(6, dtype=np.float64), dims=("x",), name="ux")
    written = []

    def capture(final_file, values):
        written.append(values)

    monkeypatch.setattr("xarray_binfile.write.accessor._write_atomically", capture)

    array.binary_engine.to_file(whole_array_writer, tmp_path)
    array.binary_engine.to_file(casting_writer, tmp_path)

    assert np.shares_memory(written[0], array.values)  # same dtype: no copy
    assert not np.shares_memory(written[1], array.values)  # cast: new array
    assert written[1].dtype == np.dtype("<f4")


def test_to_file_allows_dotdot_that_stays_inside_directory(tmp_path, data_array):
    def inside(array):
        yield WriteSpecs(filename="sub/../ux.bin", sub_array=array)

    data_array.binary_engine.to_file(inside, tmp_path)

    assert (tmp_path / "ux.bin").exists()


def test_to_file_cleans_temporary_file_when_move_fails(
    tmp_path, data_array, monkeypatch
):
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("xarray_binfile.write.accessor.os.replace", boom)

    with pytest.raises(OSError, match="disk full"):
        data_array.binary_engine.to_file(whole_array_writer, tmp_path)

    assert list(tmp_path.iterdir()) == []


def fortran_writer(data_array: xr.DataArray):
    yield WriteSpecs(filename=f"{data_array.name}.bin", sub_array=data_array, order="F")


def test_to_file_writes_fortran_order(tmp_path, data_array):
    data_array.binary_engine.to_file(fortran_writer, tmp_path)

    written = np.fromfile(tmp_path / "ux.bin", dtype=np.float64)
    np.testing.assert_array_equal(written, data_array.values.ravel(order="F"))
    np.testing.assert_array_equal(
        written.reshape(data_array.shape, order="F"), data_array.values
    )


def test_to_file_reports_progress(tmp_path, data_array):
    seen = []

    def progress(specs):
        for spec in specs:
            seen.append(spec.filename)
            yield spec

    data_array.binary_engine.to_file(whole_array_writer, tmp_path, progress=progress)
    data_array.to_dataset().binary_engine.to_file(
        whole_array_writer, tmp_path, progress=progress
    )

    assert seen == ["ux.bin", "ux.bin"]
    assert (tmp_path / "ux.bin").exists()
