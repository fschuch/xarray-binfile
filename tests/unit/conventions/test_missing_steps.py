"""
``open()`` refuses variables that miss steps the others have.

From the pyOpenSci review, https://github.com/fschuch/xarray-binfile/issues/69:
when a variable missed a snapshot, ``open()`` filled it with NaN, so an int32
variable became float64, xarray warned that its default join would change,
and writing the dataset back created the missing file. ``open()`` now names
the gap, and ``join="outer"`` opts in to filling it.
"""

import warnings

import numpy as np
import pytest

from xarray_binfile.conventions import (
    Layout,
    PatternConventions,
    StaticFiles,
    StepIndexedFiles,
)

LAYOUT = Layout({"x": np.arange(3)}, dtype="int32")


def _write(folder, files):
    for name, steps in files.items():
        for step in steps:
            np.arange(3, dtype="int32").tofile(folder / f"{name}-{step:04d}.bin")


def test_a_missing_step_is_refused_by_name(tmp_path):
    _write(tmp_path, {"a": [0, 1, 2], "b": [0, 2]})

    with pytest.raises(ValueError, match=r"'b' has no file at time=1") as info:
        StepIndexedFiles(LAYOUT).open(tmp_path)

    assert "join='outer'" in str(info.value)


def test_join_outer_opts_in_to_filling_without_a_warning(tmp_path):
    _write(tmp_path, {"a": [0, 1], "b": [0]})

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        opened = StepIndexedFiles(LAYOUT).open(tmp_path, join="outer").load()

    assert opened["a"].dtype == np.int32
    assert opened["b"].dtype == np.float64
    assert opened["b"].sel(time=1).isnull().all()


def test_variables_with_the_same_steps_open(tmp_path):
    _write(tmp_path, {"a": [0, 1], "b": [0, 1]})

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        opened = StepIndexedFiles(LAYOUT).open(tmp_path)

    assert opened["b"].dtype == np.int32


def test_variables_with_other_dimensions_are_not_compared(tmp_path):
    _write(tmp_path, {"a": [0, 1]})
    np.arange(3, dtype="int32").tofile(tmp_path / "mask.bin")
    conventions = PatternConventions([StepIndexedFiles(LAYOUT), StaticFiles(LAYOUT)])

    assert sorted(conventions.open(tmp_path).data_vars) == ["a", "mask"]
