"""
A single-precision time coordinate must write back to the steps it came from.

From the pyOpenSci review, https://github.com/fschuch/xarray-binfile/issues/65:
``StepIndexedFiles`` recovers the step as ``value / time_step`` and required it
to land within 1e-6 of an integer, tighter than float32 can hold. With
``time_step=0.1``, step 42 was the first refused, and most steps after it. The
tolerance now scales with the precision of the coordinate's own dtype.
"""

import pathlib

import numpy as np
import pytest
import xarray as xr

from xarray_binfile.conventions import Layout, StepIndexedFiles

LAYOUT = Layout({"x": np.arange(3)})
STEPS = np.arange(1, 1001)


def _array(time) -> xr.DataArray:
    return xr.DataArray(
        np.zeros((len(time), 3)), coords={"time": time, "x": np.arange(3)}, name="ux"
    )


def _steps_written(convention: StepIndexedFiles, time) -> list[int]:
    return [
        int(spec.filename[3:7]) for spec in convention.writer(_array(np.asarray(time)))
    ]


@pytest.mark.xfail(strict=True, reason="#65: 1e-6 is tighter than float32")
def test_the_reviewers_float32_step_writes_back():
    convention = StepIndexedFiles(LAYOUT, time_step=0.1, time_dtype=np.float32)
    time = convention.reader(pathlib.Path("ux-0042.bin")).coords["time"]

    assert time.dtype == np.float32
    assert _steps_written(convention, time) == [42]


TIME_STEPS = (0.1, 0.01, 0.001, 0.3)


@pytest.mark.xfail(strict=True, reason="#65: 1e-6 is tighter than float32")
@pytest.mark.parametrize("time_step", TIME_STEPS)
def test_float32_times_read_from_files_write_back_to_their_steps(time_step):
    convention = StepIndexedFiles(LAYOUT, time_step=time_step, time_dtype=np.float32)
    # What the reader produces for each step.
    time = np.asarray(STEPS * time_step, dtype=np.float32)

    assert _steps_written(convention, time) == STEPS.tolist()


@pytest.mark.xfail(strict=True, reason="#65: 1e-6 is tighter than float32")
@pytest.mark.parametrize("time_step", TIME_STEPS)
def test_float32_times_built_in_single_precision_write_back_to_their_steps(
    time_step,
):
    convention = StepIndexedFiles(LAYOUT, time_step=time_step)
    # Built by the user in float32, which rounds once more than the reader.
    time = STEPS.astype(np.float32) * np.float32(time_step)

    assert _steps_written(convention, time) == STEPS.tolist()


@pytest.mark.parametrize(
    "time",
    [
        pytest.param(np.float32(42.5 * 0.1), id="half-a-step"),
        pytest.param(np.float32(42.1 * 0.1), id="a-tenth-of-a-step"),
    ],
)
def test_float32_times_between_steps_are_still_refused(time):
    convention = StepIndexedFiles(LAYOUT, time_step=0.1)

    with pytest.raises(ValueError, match="does not correspond to an integer step"):
        _steps_written(convention, [time])


def test_float32_times_more_than_a_few_ulps_away_are_refused():
    convention = StepIndexedFiles(LAYOUT, time_step=0.1)
    exact = np.float32(4.2)
    off = exact + 16 * np.spacing(exact)

    with pytest.raises(ValueError, match="does not correspond to an integer step"):
        _steps_written(convention, [off])


def test_float64_tolerance_is_unchanged():
    convention = StepIndexedFiles(LAYOUT, time_step=0.1)

    assert _steps_written(convention, [42 * 0.1]) == [42]
    with pytest.raises(ValueError, match="does not correspond to an integer step"):
        _steps_written(convention, [42 * 0.1 + 1e-5])
