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
        int(spec.filename.removeprefix("ux-").removesuffix(".bin"))
        for spec in convention.writer(_array(np.asarray(time)))
    ]


def test_the_reviewers_float32_step_writes_back():
    convention = StepIndexedFiles(LAYOUT, time_step=0.1, time_dtype=np.float32)
    time = convention.reader(pathlib.Path("ux-0042.bin")).coords["time"]

    assert time.dtype == np.float32
    assert _steps_written(convention, time) == [42]


TIME_STEPS = (0.1, 0.01, 0.001, 0.3)


@pytest.mark.parametrize("time_step", TIME_STEPS)
def test_float32_times_read_from_files_write_back_to_their_steps(time_step):
    convention = StepIndexedFiles(LAYOUT, time_step=time_step, time_dtype=np.float32)
    # What the reader produces for each step.
    time = np.asarray(STEPS * time_step, dtype=np.float32)

    assert _steps_written(convention, time) == STEPS.tolist()


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
    off = exact + np.float32(16) * np.spacing(exact)
    assert off.dtype == np.float32  # numpy 1.x promotes a Python int to float64

    with pytest.raises(ValueError, match="does not correspond to an integer step"):
        _steps_written(convention, [off])


def test_float64_tolerance_is_unchanged():
    convention = StepIndexedFiles(LAYOUT, time_step=0.1)

    assert _steps_written(convention, [42 * 0.1]) == [42]
    with pytest.raises(ValueError, match="does not correspond to an integer step"):
        _steps_written(convention, [42 * 0.1 + 1e-5])


WIDE = "{name}-{step:08d}.bin"


@pytest.mark.parametrize("time_step", (0.1, 0.001, 1.0))
def test_float32_times_near_a_million_steps_write_back_to_their_steps(time_step):
    convention = StepIndexedFiles(
        LAYOUT, pattern=WIDE, time_step=time_step, time_dtype=np.float32
    )
    steps = np.arange(1_000_000, 1_000_050)
    time = np.asarray(steps * time_step, dtype=np.float32)

    assert _steps_written(convention, time) == steps.tolist()


@pytest.mark.parametrize(
    "step",
    [
        pytest.param(3_000_000, id="past-half-a-step"),
        pytest.param(10_485_762, id="last-correct-before-collision"),
        pytest.param(10_485_763, id="written-as-10485762"),
    ],
)
def test_float32_times_too_coarse_to_identify_their_step_are_refused(step):
    convention = StepIndexedFiles(
        LAYOUT, pattern=WIDE, time_step=0.1, time_dtype=np.float32
    )
    time = convention.reader(pathlib.Path(f"ux-{step:08d}.bin")).coords["time"]

    with pytest.raises(ValueError, match="float64"):
        _steps_written(convention, time)
