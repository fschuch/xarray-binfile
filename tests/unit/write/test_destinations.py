"""
How ``to_file`` creates its files.

From the pyOpenSci review, https://github.com/fschuch/xarray-binfile/issues/69:
on Linux and macOS every file was written with mode 0600 whatever the umask,
and a symbolic link at the destination was replaced by a regular file, leaving
the file it pointed to unchanged. Files now get the permissions the umask
allows, and a linked destination is written through: the file it points to
is updated and the link stays.
"""

import os
import stat
import sys

import numpy as np
import pytest
import xarray as xr

from xarray_binfile.write import WriteSpecs


def _ux(value: float = 1.0) -> xr.DataArray:
    return xr.DataArray(np.full(3, value), dims=("x",), name="ux")


def _to(*filenames):
    def writer(array: xr.DataArray):
        for factor, filename in enumerate(filenames, start=1):
            yield WriteSpecs(filename=filename, sub_array=array * factor)

    return writer


def _link(link, target):
    try:
        os.symlink(target, link)
    except OSError:  # no cover - Windows without symlink rights
        pytest.skip("symbolic links are not available")


@pytest.fixture
def umask():
    """Set a known umask for the test and restore the previous one."""
    previous = os.umask(0o027)
    yield 0o027
    os.umask(previous)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_a_new_file_gets_the_permissions_the_umask_allows(tmp_path, umask):
    _ux().binary_engine.to_file(_to("ux.bin"), tmp_path)

    mode = stat.S_IMODE((tmp_path / "ux.bin").stat().st_mode)
    assert mode == 0o666 & ~umask


def test_a_linked_destination_is_written_through(tmp_path):
    target = tmp_path / "elsewhere" / "ux-real.bin"
    target.parent.mkdir()
    target.write_bytes(b"old")
    out = tmp_path / "out"
    out.mkdir()
    _link(out / "ux.bin", target)

    _ux(2.0).binary_engine.to_file(_to("ux.bin"), out)

    assert (out / "ux.bin").is_symlink()
    np.testing.assert_array_equal(np.fromfile(target), [2.0, 2.0, 2.0])


def test_a_dangling_link_creates_the_file_it_points_to(tmp_path):
    target = tmp_path / "ux-real.bin"
    out = tmp_path / "out"
    out.mkdir()
    _link(out / "ux.bin", target)

    _ux(3.0).binary_engine.to_file(_to("ux.bin"), out)

    assert (out / "ux.bin").is_symlink()
    np.testing.assert_array_equal(np.fromfile(target), [3.0, 3.0, 3.0])


def test_a_dangling_link_into_a_missing_folder_creates_that_folder(tmp_path):
    target = tmp_path / "new" / "ux-real.bin"
    out = tmp_path / "out"
    out.mkdir()
    _link(out / "ux.bin", target)

    _ux(5.0).binary_engine.to_file(_to("ux.bin"), out)

    assert (out / "ux.bin").is_symlink()
    np.testing.assert_array_equal(np.fromfile(target), [5.0, 5.0, 5.0])


@pytest.mark.skipif(
    sys.platform == "win32", reason="the whole path exceeds MAX_PATH by default"
)
def test_a_long_filename_that_fits_the_filesystem_is_written(tmp_path):
    # 229 bytes, under the usual 255-byte limit for one path component. On
    # main the temporary name for it was 253 bytes, so it was written.
    name = "x" * 225 + ".bin"

    _ux(6.0).binary_engine.to_file(_to(name), tmp_path)

    assert [path.name for path in tmp_path.iterdir()] == [name]


def test_a_link_and_the_file_it_points_to_are_one_file(tmp_path):
    (tmp_path / "b.bin").write_bytes(b"")
    _link(tmp_path / "a.bin", tmp_path / "b.bin")

    array, writer = _ux(), _to("a.bin", "b.bin")

    with pytest.raises(ValueError, match="same file"):
        array.binary_engine.to_file(writer, tmp_path)
    assert (tmp_path / "b.bin").read_bytes() == b""


def test_an_existing_regular_file_is_still_replaced(tmp_path):
    (tmp_path / "ux.bin").write_bytes(b"old")

    _ux(4.0).binary_engine.to_file(_to("ux.bin"), tmp_path)

    assert not (tmp_path / "ux.bin").is_symlink()
    np.testing.assert_array_equal(np.fromfile(tmp_path / "ux.bin"), [4.0, 4.0, 4.0])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ux.bin"]


RETRY = pytest.mark.xfail(strict=True, reason="a name clash is not retried (#90)")
TAKEN = "0" * 16


def _tokens(monkeypatch, *tokens):
    """Make the temporary names use ``tokens``, then repeat the last one."""
    drawn = []

    def token_hex(nbytes):
        drawn.append(tokens[min(len(drawn), len(tokens) - 1)])
        return drawn[-1]

    monkeypatch.setattr("xarray_binfile.write.accessor.secrets.token_hex", token_hex)
    return drawn


def _leftover(folder):
    """A temporary file left behind, for example by a crash."""
    path = folder / f".binary_engine-{TAKEN}.tmp"
    path.write_bytes(b"leftover")
    return path


@RETRY
def test_a_temporary_name_already_taken_is_retried_with_a_new_one(
    tmp_path, monkeypatch
):
    leftover = _leftover(tmp_path)
    drawn = _tokens(monkeypatch, TAKEN, "1" * 16)

    _ux(7.0).binary_engine.to_file(_to("ux.bin"), tmp_path)

    assert drawn == [TAKEN, "1" * 16]
    np.testing.assert_array_equal(np.fromfile(tmp_path / "ux.bin"), [7.0, 7.0, 7.0])
    assert leftover.read_bytes() == b"leftover"  # never touched


@RETRY
def test_a_temporary_name_always_taken_fails_after_a_bounded_number_of_tries(
    tmp_path, monkeypatch
):
    leftover = _leftover(tmp_path)
    drawn = _tokens(monkeypatch, TAKEN)
    array, writer = _ux(), _to("ux.bin")

    with pytest.raises(FileExistsError):
        array.binary_engine.to_file(writer, tmp_path)
    assert 1 < len(drawn) <= 1000
    assert leftover.read_bytes() == b"leftover"
    assert not (tmp_path / "ux.bin").exists()


def test_other_errors_creating_the_temporary_file_are_not_retried(
    tmp_path, monkeypatch
):
    drawn = _tokens(monkeypatch, "1" * 16, "2" * 16)
    real_open = os.open

    def denied(path, *args, **kwargs):
        # ``os.open`` is the global function: fail only our temporary file.
        if ".binary_engine-" not in str(path):
            return real_open(path, *args, **kwargs)  # no cover - other callers
        raise PermissionError("denied")

    monkeypatch.setattr("xarray_binfile.write.accessor.os.open", denied)
    array, writer = _ux(), _to("ux.bin")

    with pytest.raises(PermissionError, match="denied"):
        array.binary_engine.to_file(writer, tmp_path)
    assert drawn == ["1" * 16]
