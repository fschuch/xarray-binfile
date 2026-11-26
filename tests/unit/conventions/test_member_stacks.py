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


def test_folder_stacks_stay_in_their_folder(tmp_path):
    _write(tmp_path / "flow", "ux.bin", "uy.bin")
    _write(tmp_path / "mesh", "dx.bin", "dy.bin")
    conventions = FolderConventions(
        {"flow": StaticFiles(LAYOUT, stacks=(VELOCITY,)), "mesh": StaticFiles(LAYOUT)}
    )

    opened = conventions.open(tmp_path)

    assert sorted(opened.data_vars) == ["dx", "dy", "u"]
    assert opened["u"]["i"].values.tolist() == ["x", "y"]


def test_pattern_stacks_stay_with_their_member(tmp_path):
    _write(tmp_path, "ux-0000.bin", "uy-0000.bin", "dx.bin", "dy.bin")
    conventions = PatternConventions(
        [StepIndexedFiles(LAYOUT, stacks=(VELOCITY,)), StaticFiles(LAYOUT)]
    )

    assert sorted(conventions.open(tmp_path).data_vars) == ["dx", "dy", "u"]


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


def test_a_variable_spanning_folders_is_stacked_once(tmp_path):
    _write(tmp_path / "early", "ux-0000.bin", "uy-0000.bin")
    _write(tmp_path / "late", "ux-0001.bin", "uy-0001.bin")
    member = StepIndexedFiles(LAYOUT, stacks=(VELOCITY,))
    conventions = FolderConventions({"early": member, "late": member})

    opened = conventions.open(tmp_path)

    assert list(opened.data_vars) == ["u"]
    assert dict(opened["u"].sizes) == {"i": 2, "x": 3, "time": 2}


@pytest.mark.parametrize("shared", [True, False], ids=["one member", "equal members"])
def test_components_spread_over_folders_are_stacked_together(tmp_path, shared):
    _write(tmp_path / "early", "ux-0000.bin", "uy-0000.bin")
    _write(tmp_path / "late", "ux-0001.bin", "uz-0001.bin")
    early = StepIndexedFiles(LAYOUT, stacks=(VELOCITY,))
    late = early if shared else StepIndexedFiles(LAYOUT, stacks=(VELOCITY,))
    conventions = FolderConventions({"early": early, "late": late})

    opened = conventions.open(tmp_path)

    assert list(opened.data_vars) == ["u"]
    assert opened["i"].values.tolist() == ["x", "y", "z"]


def test_members_with_different_components_keep_them_all(tmp_path):
    _write(tmp_path / "flow", "ux.bin", "uy.bin")
    _write(tmp_path / "wake", "vx.bin", "vy.bin", "vz.bin")
    conventions = FolderConventions(
        {
            "flow": StaticFiles(LAYOUT, stacks=(VELOCITY,)),
            "wake": StaticFiles(LAYOUT, stacks=(VELOCITY,)),
        }
    )

    opened = conventions.open(tmp_path).load()

    assert opened["i"].values.tolist() == ["x", "y", "z"]
    assert opened["v"].notnull().all()
    assert opened["u"].sel(i="z").isnull().all()


@pytest.mark.parametrize(
    "conventions",
    [
        FolderConventions(
            {
                "flow": StaticFiles(LAYOUT, stacks=(VELOCITY,)),
                "mesh": StaticFiles(LAYOUT),
            }
        ),
        PatternConventions(
            [StaticFiles(LAYOUT, stacks=(VELOCITY,)), StaticFiles(LAYOUT)]
        ),
    ],
    ids=["folders", "patterns"],
)
def test_stack_by_hand_applies_every_members_stacks(tmp_path, conventions):
    _write(tmp_path / "flow", "ux.bin", "uy.bin")
    _write(tmp_path / "mesh", "dx.bin", "dy.bin")
    plain = StaticFiles(LAYOUT)
    opened = plain.open(tmp_path / "flow").merge(plain.open(tmp_path / "mesh"))

    # A dataset does not record where its variables came from.
    assert sorted(conventions.stack(opened).data_vars) == ["d", "u"]
