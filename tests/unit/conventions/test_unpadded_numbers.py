"""
Numbered variables beyond nine, without zero padding.

From the pyOpenSci review, https://github.com/fschuch/xarray-binfile/issues/69:
with ``"{name}{n:d}"`` and values 1 to 12, ``phi11`` was taken as ``phi1`` +
``1``. With ``names={"phi"}`` that reading was rejected and ``phi`` + ``11``
was never tried, so ``phi11`` and ``phi12`` stayed apart. Without ``names``
they were grouped as ``phi1`` and clashed with the variable ``phi1``.
Longer values are now tried first, and a value whose base name ``names``
rejects no longer stops the search.
"""

import numpy as np
import pytest
import xarray as xr

from xarray_binfile.conventions import VariableStack

NUMBERS = range(1, 13)


@pytest.fixture
def dataset():
    return xr.Dataset({f"phi{n}": ("x", np.full(2, float(n))) for n in NUMBERS})


@pytest.mark.xfail(strict=True, reason="#69: phi11 is only tried as phi1 + 1")
@pytest.mark.parametrize("names", [None, {"phi"}], ids=["guessing", "names"])
def test_all_twelve_are_stacked(dataset, names):
    stack = VariableStack("n", "{name}{n:d}", values=NUMBERS, names=names)

    stacked = stack.stack(dataset)

    assert list(stacked.data_vars) == ["phi"]
    assert stacked["n"].values.tolist() == list(NUMBERS)
    np.testing.assert_array_equal(stacked["phi"].sel(n=11), dataset["phi11"])


def test_names_can_still_pick_the_shorter_value():
    dataset = xr.Dataset({name: ("x", np.zeros(2)) for name in ("phi11", "phi12")})
    stack = VariableStack("n", "{name}{n:d}", values=NUMBERS, names={"phi1"})

    stacked = stack.stack(dataset)

    assert list(stacked.data_vars) == ["phi1"]
    assert stacked["n"].values.tolist() == [1, 2]
