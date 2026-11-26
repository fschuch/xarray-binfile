"""
``open()`` selects files by ``variables`` and ``drop_variables``.

From the pyOpenSci review, https://github.com/fschuch/xarray-binfile/issues/69:
``drop_variables`` failed in ``open()`` as soon as two files were dropped.
``open()`` now skips dropped files before opening, as ``variables`` does, so
their coordinates do not reach the dataset either. Both take names as found
on disk (``ux``) or as stacked (``u``), as the convention's ``names`` do.
"""

import numpy as np
import pytest

from xarray_binfile.conventions import (
    Convention,
    FolderConventions,
    Layout,
    StaticFiles,
    StepIndexedFiles,
    VariableStack,
)

VELOCITY = VariableStack("i", "{name}{i}", values=("x", "y", "z"))
LAYOUT = Layout({"x": np.arange(3)})


def _write(folder, *filenames):
    folder.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        np.zeros(3).tofile(folder / filename)


@pytest.mark.parametrize("drop", ["rhou", ["rhou"]])
def test_drop_variables_skips_that_name_only(tmp_path, drop):
    _write(tmp_path, "rho.bin", "rhou.bin", "e.bin")

    opened = StaticFiles(LAYOUT).open(tmp_path, drop_variables=drop)

    assert sorted(opened.data_vars) == ["e", "rho"]


@pytest.mark.parametrize("keep", ["rhou", ["rhou"]])
def test_variables_keeps_that_name_only(tmp_path, keep):
    _write(tmp_path, "rho.bin", "rhou.bin", "u.bin")

    opened = StaticFiles(LAYOUT).open(tmp_path, variables=keep)

    # A bare string is one name, not the set of its characters.
    assert list(opened.data_vars) == ["rhou"]


def test_drop_variables_skips_the_files_of_a_variable(tmp_path):
    _write(tmp_path, "ux-0000.bin", "ux-0001.bin", "uy-0000.bin", "uy-0001.bin")
    _write(tmp_path, "uy-0002.bin")

    opened = StepIndexedFiles(LAYOUT).open(tmp_path, drop_variables=["uy"])

    assert list(opened.data_vars) == ["ux"]
    assert opened["time"].values.tolist() == [0, 1]


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ({"variables": ["u"]}, ["u"]),
        ({"drop_variables": ["u"]}, ["pp"]),
        ({"drop_variables": "u"}, ["pp"]),
    ],
)
def test_stacked_names_select_their_components(tmp_path, selection, expected):
    _write(tmp_path, "ux.bin", "uy.bin", "uz.bin", "pp.bin")
    convention = StaticFiles(LAYOUT, stacks=(VELOCITY,))

    assert sorted(convention.open(tmp_path, **selection).data_vars) == expected


def test_folders_select_by_stacked_names_too(tmp_path):
    _write(tmp_path / "flow", "ux.bin", "uy.bin")
    _write(tmp_path / "mesh", "dx.bin", "dy.bin")
    conventions = FolderConventions(
        {"flow": StaticFiles(LAYOUT, stacks=(VELOCITY,)), "mesh": StaticFiles(LAYOUT)}
    )

    assert list(conventions.open(tmp_path, variables=["u"]).data_vars) == ["u"]


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ({"variables": ["ux", "uy", "pp"]}, ["pp", "u"]),
        ({"variables": ["ux"]}, ["ux"]),
    ],
)
def test_names_on_disk_still_select(tmp_path, selection, expected):
    _write(tmp_path, "ux.bin", "uy.bin", "uz.bin", "pp.bin")
    convention = StaticFiles(LAYOUT, stacks=(VELOCITY,))

    assert sorted(convention.open(tmp_path, **selection).data_vars) == expected


def test_a_bare_reader_selects_by_the_name_on_disk(tmp_path):
    _write(tmp_path, "ux.bin", "uy.bin")

    class Bare:
        def reader(self, path):
            return StaticFiles(LAYOUT).reader(path)

        def writer(self, data_array):  # no cov
            raise NotImplementedError

    opened = Convention.adapt(Bare()).open(tmp_path, drop_variables="uy")

    assert list(opened.data_vars) == ["ux"]
