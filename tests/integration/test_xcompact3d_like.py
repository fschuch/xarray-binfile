"""
End-to-end scenario shaped like the Xcompact3d toolbox: Fortran-ordered
single-precision snapshots with stacked velocity and scalar fields in the
dataset root, a static geometry file in a sub-folder, and a physical time
derived from the step counter.
"""

import pathlib

import numpy as np
import xarray as xr

import xarray_binfile  # noqa: F401  (registers the accessors)
from xarray_binfile.conventions import (
    FolderConventions,
    Layout,
    PatternConventions,
    StaticFiles,
    StepIndexedFiles,
    VariableStack,
)

MESH = {
    "x": np.linspace(0.0, 1.0, 5, dtype=np.float32),
    "y": np.linspace(0.0, 2.0, 3, dtype=np.float32),
    "z": np.linspace(0.0, 0.5, 4, dtype=np.float32),
}
DT = 0.25
LAYOUT = Layout(
    MESH,
    dtype=np.float32,
    order="F",
    coord_attrs={"x": {"long_name": "Streamwise coordinate"}},
)
SNAPSHOTS = StepIndexedFiles(
    LAYOUT,
    pattern="{name}-{step:03d}.bin",
    time_dim="t",
    time_step=DT,
    time_dtype=np.float32,
    stacks=[
        VariableStack("i", "{name}{i}", values=("x", "y", "z")),
        VariableStack("n", "{name}{n:d}", values=range(1, 10), names=("phi",)),
    ],
    name_of=lambda da: da.attrs.get("file_name", da.name),
)
STATIC = StaticFiles(LAYOUT, name_of=lambda da: da.attrs.get("file_name", da.name))
CONVENTION = FolderConventions({".": PatternConventions([SNAPSHOTS, STATIC])})

RNG = np.random.default_rng(0)


def _field(*extra):
    coords = {**{k: v for k, v in extra}, **MESH}
    shape = tuple(len(v) for v in coords.values())
    return xr.DataArray(RNG.random(shape, dtype=np.float32), coords=coords)


def _case():
    t = np.arange(3, dtype=np.float32) * DT
    return xr.Dataset(
        {
            "u": _field(("i", ["x", "y", "z"]), ("t", t)).assign_attrs(file_name="u"),
            "phi": _field(("n", [1, 2]), ("t", t)).assign_attrs(file_name="phi"),
            "pp": _field(("t", t)).assign_attrs(file_name="pp"),
            "epsi": _field().assign_attrs(file_name="geometry/epsilon"),
        }
    )


def test_roundtrip(tmp_path: pathlib.Path):
    case = _case()

    case.binary_engine.to_file(CONVENTION.writer, tmp_path)

    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.bin")) == [
        "geometry/epsilon.bin",
        *[f"{n}-{s:03d}.bin" for n in ("phi1", "phi2", "pp", "ux", "uy", "uz") for s in range(3)],
    ]

    # Files are Fortran-ordered: x varies fastest, as the solver would write.
    on_disk = np.fromfile(tmp_path / "pp-001.bin", dtype=np.float32)
    np.testing.assert_array_equal(on_disk[:5], case["pp"].isel(t=1, y=0, z=0).values)

    opened = CONVENTION.open(tmp_path, chunks={"t": 1})

    assert sorted(opened.data_vars) == ["phi", "pp", "u"]
    assert opened["u"].dims == ("i", "x", "y", "z", "t")
    assert opened["t"].dtype == np.float32
    assert opened["x"].attrs == {"long_name": "Streamwise coordinate"}
    np.testing.assert_allclose(opened["t"], [0.0, 0.25, 0.5])

    for name in ("u", "phi", "pp"):
        expected = case[name].transpose(*opened[name].dims)
        xr.testing.assert_allclose(opened[name].load(), expected)

    epsilon = xr.open_dataset(
        tmp_path / "geometry" / "epsilon.bin",
        engine="binfile",
        read_specs_getter=CONVENTION.reader,
    )["epsilon"]
    np.testing.assert_array_equal(epsilon, case["epsi"])

    raw = CONVENTION.open(tmp_path, stack=False)
    assert sorted(raw.data_vars) == ["phi1", "phi2", "pp", "ux", "uy", "uz"]
