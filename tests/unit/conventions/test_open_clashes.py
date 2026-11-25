"""
``open()`` must not combine files that cannot be one variable.

From the pyOpenSci review, https://github.com/fschuch/xarray-binfile/issues/68:
a :class:`FolderConventions` reads ``xy_planes/ux-0001.bin`` and
``3d/ux-0001.bin`` both as ``ux``. With different shapes, xarray failed with an
error that did not mention the clash; with the same layout and steps, one
folder's values were silently dropped. ``open()`` now refuses both before
opening anything.
"""

import numpy as np
import pytest

from xarray_binfile.conventions import FolderConventions, Layout, StepIndexedFiles

X, Y, Z = np.arange(4), np.arange(3), np.arange(2)
PLANES = StepIndexedFiles(Layout({"x": X, "y": Y}, dtype="<f4"))
VOLUMES = StepIndexedFiles(Layout({"x": X, "y": Y, "z": Z}, dtype="<f4"))

CLASH = pytest.mark.xfail(strict=True, reason="#68")


def _write(path, shape, value=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.full(shape, value, "<f4").tofile(path)


@CLASH
def test_the_reviewers_planes_and_volumes_are_refused(tmp_path):
    _write(tmp_path / "xy_planes" / "ux-0001.bin", (4, 3))
    _write(tmp_path / "3d" / "ux-0001.bin", (4, 3, 2))
    conventions = FolderConventions({"xy_planes": PLANES, "3d": VOLUMES})

    with pytest.raises(ValueError, match="'ux'.*different dimensions") as error:
        conventions.open(tmp_path)
    assert "xy_planes" in str(error.value)
    assert "3d" in str(error.value)


@CLASH
@pytest.mark.parametrize("open_kwargs", [{}, {"chunks": {}}], ids=["default", "chunks"])
def test_two_folders_at_the_same_step_are_refused(tmp_path, open_kwargs):
    for folder, value in (("bottom", 1.0), ("top", 2.0)):
        for step in (0, 1):
            _write(tmp_path / folder / f"ux-{step:04d}.bin", (4, 3), value)
    conventions = FolderConventions({"bottom": PLANES, "top": PLANES})

    with pytest.raises(ValueError, match="'ux'.*same coordinates") as error:
        conventions.open(tmp_path, **open_kwargs)
    assert "bottom" in str(error.value)
    assert "top" in str(error.value)


@CLASH
def test_two_files_in_one_folder_at_the_same_step_are_refused(tmp_path):
    # A zero-fill width is a minimum, so both names read as step 1.
    _write(tmp_path / "ux-0001.bin", (4, 3), 1.0)
    _write(tmp_path / "ux-00001.bin", (4, 3), 2.0)

    with pytest.raises(ValueError, match="'ux'.*same coordinates"):
        PLANES.open(tmp_path)


def test_one_name_split_in_time_across_folders_is_opened(tmp_path):
    for folder, steps in (("early", (0, 1)), ("late", (2, 3))):
        for step in steps:
            _write(tmp_path / folder / f"ux-{step:04d}.bin", (4, 3), step)
    conventions = FolderConventions({"early": PLANES, "late": PLANES})

    ux = conventions.open(tmp_path)["ux"]

    np.testing.assert_array_equal(ux.time, [0, 1, 2, 3])
    np.testing.assert_array_equal(ux.isel(x=0, y=0), [0, 1, 2, 3])
