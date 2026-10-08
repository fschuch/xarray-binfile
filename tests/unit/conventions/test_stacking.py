import numpy as np
import pytest
import xarray as xr

from xarray_binfile.conventions import VariableStack, split_variables, stack_variables

VELOCITY = VariableStack("i", "{name}{i}", values=("x", "y", "z"))
SCALARS = VariableStack("n", "{name}{n:d}", values=range(1, 10))


@pytest.fixture
def dataset():
    return xr.Dataset(
        {
            name: (("x", "time"), np.full((3, 2), index, dtype=np.float32))
            for index, name in enumerate(("ux", "uy", "uz", "pp", "phi1", "phi2"))
        },
        coords={"x": np.arange(3), "time": [0, 1]},
    )


class TestStack:
    def test_stacks_velocity_components(self, dataset):
        stacked = VELOCITY.stack(dataset)

        assert sorted(stacked.data_vars) == ["phi1", "phi2", "pp", "u"]
        assert stacked["u"].dims == ("i", "x", "time")
        assert stacked["i"].values.tolist() == ["x", "y", "z"]
        np.testing.assert_array_equal(stacked["u"].sel(i="y"), dataset["uy"])

    def test_stacks_scalar_fractions_with_typed_values(self, dataset):
        stacked = SCALARS.stack(dataset)

        assert "phi" in stacked.data_vars
        assert stacked["n"].values.tolist() == [1, 2]
        assert "phi1" not in stacked.data_vars

    def test_values_order_defines_coordinate_order(self, dataset):
        stacked = VariableStack("i", "{name}{i}", values=("z", "x")).stack(dataset)

        assert stacked["i"].values.tolist() == ["z", "x"]
        assert "uy" in stacked.data_vars  # not listed, left alone

    def test_names_restricts_the_groups(self, dataset):
        stack = VariableStack("n", "{name}{n:d}", values=[1, 2], names=["other"])

        xr.testing.assert_identical(stack.stack(dataset), dataset)

    def test_is_lazy_for_dask_arrays(self, dataset):
        stacked = VELOCITY.stack(dataset.chunk({"time": 1}))

        assert stacked["u"].chunks is not None

    def test_stack_variables_applies_in_reverse_order(self, dataset):
        both = stack_variables(dataset, [VELOCITY, SCALARS])

        assert sorted(both.data_vars) == ["phi", "pp", "u"]


class TestSplit:
    def test_split_names_slices_from_coordinate(self):
        u = xr.DataArray(
            np.zeros((2, 3)), coords={"i": ["x", "z"], "x": np.arange(3)}, name="u"
        )
        pieces = list(VELOCITY.split(u))

        assert [p.name for p in pieces] == ["ux", "uz"]
        assert all(p.dims == ("x",) and "i" not in p.coords for p in pieces)

    def test_split_uses_position_without_coordinate(self):
        phi = xr.DataArray(np.zeros((2, 3)), dims=("n", "x"), name="phi")
        zero_based = VariableStack("n", "{name}{n:d}", values=range(10))

        assert [p.name for p in zero_based.split(phi)] == ["phi0", "phi1"]

    def test_split_leaves_arrays_without_the_dim_alone(self):
        pp = xr.DataArray(np.zeros(3), dims=("x",), name="pp")

        assert [p.name for p in VELOCITY.split(pp)] == ["pp"]

    def test_split_rejects_unlisted_values(self):
        u = xr.DataArray(
            np.zeros((1, 3)), coords={"i": ["w"]}, dims=("i", "x"), name="u"
        )

        with pytest.raises(ValueError, match="not listed"):
            list(VELOCITY.split(u))

    def test_split_rejects_unnamed_arrays(self):
        u = xr.DataArray(np.zeros((1, 3)), coords={"i": ["x"]}, dims=("i", "x"))

        with pytest.raises(ValueError, match="unnamed array"):
            list(VELOCITY.split(u))

    def test_split_variables_nests_in_order(self):
        both = xr.DataArray(
            np.zeros((2, 2, 3)),
            coords={"i": ["x", "y"], "n": [1, 2]},
            dims=("i", "n", "x"),
            name="u",
        )

        assert [p.name for p in split_variables(both, [VELOCITY, SCALARS])] == [
            "ux1",
            "ux2",
            "uy1",
            "uy2",
        ]


def test_roundtrip_with_a_convention(tmp_path):
    import xarray_binfile  # noqa: F401  (registers the accessors)
    from xarray_binfile.conventions import Layout, StepIndexedFiles

    convention = StepIndexedFiles(
        Layout({"x": np.arange(3)}, dtype="<f4"), stacks=[VELOCITY]
    )
    u = xr.DataArray(
        np.arange(12, dtype="<f4").reshape(2, 3, 2),
        coords={"i": ["x", "y"], "x": np.arange(3), "time": [0, 1]},
        name="u",
    )

    u.binary_engine.to_file(convention.writer, tmp_path)
    back = convention.open(tmp_path)

    xr.testing.assert_identical(back["u"].transpose(*u.dims).load(), u)


@pytest.mark.parametrize("template", ["{name}", "{name}{j}", "{name}{i}{n}"])
def test_rejects_templates_without_name_and_dim(template):
    with pytest.raises(ValueError, match="must declare exactly the fields"):
        VariableStack("i", template, values=["x"])


def test_rejects_empty_values():
    with pytest.raises(ValueError, match="at least one value"):
        VariableStack("i", "{name}{i}", values=[])


def test_stack_attaches_attrs_to_the_new_coordinate(dataset):
    stack = VariableStack(
        "i", "{name}{i}", values=("x", "y", "z"), attrs={"long_name": "component"}
    )

    stacked = stack.stack(dataset)

    assert stacked["i"].attrs == {"long_name": "component"}
    assert VELOCITY.stack(dataset)["i"].attrs == {}


class TestLongNames:
    """Adjacent string fields: the value must be matched literally, not lazily."""

    def test_stacks_multi_letter_base_names(self):
        ds = xr.Dataset(
            {
                name: ("x", np.zeros(2))
                for name in ("vortx", "vorty", "vortz", "vorticity")
            }
        )

        stacked = VariableStack("i", "{name}{i}", values=("x", "y", "z")).stack(ds)

        assert sorted(stacked.data_vars) == ["vort", "vorticity"]
        assert stacked["i"].values.tolist() == ["x", "y", "z"]

    def test_stacks_zero_filled_integers(self):
        ds = xr.Dataset(
            {name: ("x", np.zeros(2)) for name in ("phi01", "phi02", "phi1")}
        )

        stacked = VariableStack("n", "{name}{n:02d}", values=range(1, 10)).stack(ds)

        assert sorted(stacked.data_vars) == ["phi", "phi1"]
        assert stacked["n"].values.tolist() == [1, 2]

    def test_names_accept_any_collection_and_are_a_frozenset(self):
        stack = VariableStack("i", "{name}{i}", values=("x", "y"), names={"u", "vort"})

        assert stack.names == frozenset({"u", "vort"})
        assert VariableStack(
            "i", "{name}{i}", values=("x",), names=["u"]
        ).names == frozenset({"u"})


class TestReviewFixes:
    def test_stack_refuses_to_overwrite_an_existing_variable(self):
        ds = xr.Dataset({name: ("x", np.zeros(2)) for name in ("pp", "ppx", "ppy")})

        with pytest.raises(ValueError, match="'pp' already exists"):
            VariableStack("i", "{name}{i}", values=("x", "y", "z")).stack(ds)

    def test_names_must_be_a_collection_not_a_str(self):
        with pytest.raises(TypeError, match="collection of names"):
            VariableStack("i", "{name}{i}", values=("x",), names="vort")

    def test_min_components_is_explicit(self, dataset):
        single = dataset[["ux"]]
        guessing = VariableStack("i", "{name}{i}", values=("x", "y", "z"))
        eager = VariableStack(
            "i", "{name}{i}", values=("x", "y", "z"), min_components=1
        )

        assert guessing.min_components == 2
        assert VELOCITY.min_components == 2
        assert (
            VariableStack("i", "{name}{i}", values=("x",), names={"u"}).min_components
            == 1
        )
        assert sorted(guessing.stack(single).data_vars) == ["ux"]
        assert sorted(eager.stack(single).data_vars) == ["u"]
