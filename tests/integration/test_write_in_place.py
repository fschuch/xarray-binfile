"""
Writing a result over its own inputs is safe once it is computed.

From the pyOpenSci review, https://github.com/fschuch/xarray-binfile/issues/64:
``to_file`` computes one file at a time, and each compute reads the lazy
inputs again. A result that depends on more than its own file, written into
the folder it came from, is then computed from inputs already replaced. The
docs advise computing the result first; these tests keep that advice true.
"""

import pathlib

import numpy as np
import pytest

from xarray_binfile.conventions import Layout, StepIndexedFiles

STEPS = (1.0, 2.0, 3.0, 4.0)


def _write_steps(directory: pathlib.Path, convention: StepIndexedFiles) -> None:
    for name in ("ux", "uy"):
        for step, value in enumerate(STEPS, start=1):
            np.full(2, value).tofile(
                directory / convention.pattern.format(name=name, step=step)
            )


@pytest.fixture
def convention() -> StepIndexedFiles:
    return StepIndexedFiles(Layout({"x": np.arange(2)}))


@pytest.mark.parametrize("compute", ["compute", "load", "persist"])
def test_a_computed_anomaly_can_be_written_over_its_inputs(
    tmp_path, convention, compute
):
    _write_steps(tmp_path, convention)
    ux = convention.open(tmp_path).ux
    anomaly = getattr(ux - ux.mean("time"), compute)()

    anomaly.binary_engine.to_file(convention.writer, tmp_path)

    np.testing.assert_array_equal(
        convention.open(tmp_path).ux.isel(x=0).values, [-1.5, -0.5, 0.5, 1.5]
    )


def test_a_computed_dataset_can_be_written_over_its_inputs(tmp_path, convention):
    # Variables are written one after the other, so ``uy`` depends on the
    # ``ux`` that is replaced first.
    _write_steps(tmp_path, convention)
    ds = convention.open(tmp_path)
    result = ds.assign(ux=ds.ux - ds.ux.mean("time"), uy=ds.uy + ds.ux).compute()

    result.binary_engine.to_file(convention.writer, tmp_path)

    written = convention.open(tmp_path).isel(x=0)
    np.testing.assert_array_equal(written.ux.values, [-1.5, -0.5, 0.5, 1.5])
    np.testing.assert_array_equal(written.uy.values, [2.0, 4.0, 6.0, 8.0])
