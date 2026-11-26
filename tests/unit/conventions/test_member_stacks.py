"""
Each member convention stacks only the variables it read.

From the pyOpenSci review, https://github.com/fschuch/xarray-binfile/issues/69:
``FolderConventions`` applied every member's stacks to the whole dataset, so
with a velocity stack on one folder, ``dx`` and ``dy`` from another came back
as one variable ``d``. ``PatternConventions`` did the same between members
sharing a folder. ``open()`` knows which member claimed each file, and now
lets each member stack only those variables.
"""

import numpy as np
import pytest

from xarray_binfile.conventions import (
    FolderConventions,
    Layout,
    PatternConventions,
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


@pytest.mark.xfail(strict=True, reason="#69: stacks apply across members")
def test_folder_stacks_stay_in_their_folder(tmp_path):
    _write(tmp_path / "flow", "ux.bin", "uy.bin")
    _write(tmp_path / "mesh", "dx.bin", "dy.bin")
    conventions = FolderConventions(
        {"flow": StaticFiles(LAYOUT, stacks=(VELOCITY,)), "mesh": StaticFiles(LAYOUT)}
    )

    opened = conventions.open(tmp_path)

    assert sorted(opened.data_vars) == ["dx", "dy", "u"]
    assert opened["u"]["i"].values.tolist() == ["x", "y"]


@pytest.mark.xfail(strict=True, reason="#69: stacks apply across members")
def test_pattern_stacks_stay_with_their_member(tmp_path):
    _write(tmp_path, "ux-0000.bin", "uy-0000.bin", "dx.bin", "dy.bin")
    conventions = PatternConventions(
        [StepIndexedFiles(LAYOUT, stacks=(VELOCITY,)), StaticFiles(LAYOUT)]
    )

    assert sorted(conventions.open(tmp_path).data_vars) == ["dx", "dy", "u"]


@pytest.mark.xfail(strict=True, reason="#69: stacks apply across members")
def test_nested_members_keep_their_own_stacks(tmp_path):
    _write(tmp_path, "ux-0000.bin", "uy-0000.bin", "dx.bin", "dy.bin")
    _write(tmp_path / "planes", "vx.bin", "vy.bin")
    conventions = FolderConventions(
        {
            ".": PatternConventions(
                [StepIndexedFiles(LAYOUT, stacks=(VELOCITY,)), StaticFiles(LAYOUT)]
            ),
            "planes": StaticFiles(LAYOUT, stacks=(VELOCITY,)),
        }
    )

    assert sorted(conventions.open(tmp_path).data_vars) == ["dx", "dy", "u", "v"]


def test_a_stacked_name_taken_by_another_member_is_refused(tmp_path):
    _write(tmp_path / "flow", "ux.bin", "uy.bin")
    _write(tmp_path / "mesh", "u.bin")
    conventions = FolderConventions(
        {"flow": StaticFiles(LAYOUT, stacks=(VELOCITY,)), "mesh": StaticFiles(LAYOUT)}
    )

    with pytest.raises(ValueError, match="'u' already exists"):
        conventions.open(tmp_path)


def test_without_stacking_nothing_changes(tmp_path):
    _write(tmp_path / "flow", "ux.bin", "uy.bin")
    _write(tmp_path / "mesh", "dx.bin", "dy.bin")
    conventions = FolderConventions(
        {"flow": StaticFiles(LAYOUT, stacks=(VELOCITY,)), "mesh": StaticFiles(LAYOUT)}
    )

    opened = conventions.open(tmp_path, stack=False)

    assert sorted(opened.data_vars) == ["dx", "dy", "ux", "uy"]
