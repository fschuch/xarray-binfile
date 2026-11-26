"""
Placeholders for the code blocks in the Markdown pages.

``pytest --markdown-docs`` runs every ``python`` block. The pages use
``dataset``, ``output_dir`` and ``case_dir`` as stand-ins for the reader's own
data; a block that needs one names it in its info string
(```` ```python continuation fixture:dataset ````), which is not rendered.
These fixtures give them small, real values in a temporary folder.
"""

import numpy as np
import pytest
import xarray as xr

X, Y, Z = np.arange(64), np.arange(32), np.arange(16)


@pytest.fixture
def output_dir(tmp_path):
    """An empty folder to write into."""
    path = tmp_path / "output"
    path.mkdir()
    return path


@pytest.fixture
def dataset():
    """A velocity ``u`` with components ``i`` over two steps."""
    shape = (X.size, Y.size, Z.size, 3, 2)
    return xr.Dataset(
        {"u": (("x", "y", "z", "i", "time"), np.zeros(shape, "<f4"))},
        coords={"x": X, "y": Y, "z": Z, "i": ["x", "y", "z"], "time": [1, 2]},
    )


@pytest.fixture
def case_dir(tmp_path):
    """The ``caseC`` tree drawn on the protocols page."""
    root = tmp_path / "caseC"
    for relative, shape in {
        "xy_planes/uxy-0001.bin": (X.size, Y.size),
        "3d/ux-0001.bin": (X.size, Y.size, Z.size),
        "static/epsi.bin": (X.size, Y.size, Z.size),
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.zeros(shape, "<f4").tofile(path)
    return root
