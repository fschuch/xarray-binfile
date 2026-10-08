import pathlib

import numpy as np
import pytest
import xarray as xr

from xarray_binfile.conventions import (
    FilenamePattern,
    Layout,
    LayoutMismatchError,
    StaticFiles,
    StepIndexedFiles,
    TimeStampedFiles,
    VariableStack,
)

VELOCITY = VariableStack("i", "{name}{i}", values=("x", "y", "z"))
SCALARS = VariableStack("n", "{name}{n:d}", values=range(10))

LAYOUT = Layout({"x": np.arange(3), "y": np.arange(2)}, dtype="<f4")


def _array(time):
    return xr.DataArray(
        np.zeros((len(time), 2, 3), dtype="<f4"),
        coords={"time": time, "y": np.arange(2), "x": np.arange(3)},
        name="ux",
    )


class TestStepIndexedFiles:
    def test_reader_defaults_to_integer_time(self):
        specs = StepIndexedFiles(LAYOUT).reader(pathlib.Path("data/ux-0007.bin"))

        assert specs.name == "ux"
        assert specs.dims == ("x", "y", "time")
        assert specs.dtype == "<f4"
        assert specs.coords["time"].tolist() == [7]
        assert specs.filepath.is_absolute()

    def test_reader_scales_step_by_time_step(self):
        specs = StepIndexedFiles(LAYOUT, time_step=0.25).reader(
            pathlib.Path("ux-0003.bin")
        )

        np.testing.assert_allclose(specs.coords["time"], [0.75])

    def test_reader_accepts_pattern_object_and_wide_steps(self):
        convention = StepIndexedFiles(
            LAYOUT, pattern=FilenamePattern("{name}{step:03d}")
        )

        assert convention.reader(pathlib.Path("ux12345")).coords["time"].tolist() == [
            12345
        ]

    def test_reader_rejects_other_files(self):
        convention = StepIndexedFiles(LAYOUT)

        with pytest.raises(ValueError, match="does not match the pattern"):
            convention.reader(pathlib.Path("ux-0007.bin.bak"))

    def test_pattern_must_declare_step(self):
        with pytest.raises(ValueError, match=r"must declare the field\(s\) \['step'\]"):
            StepIndexedFiles(LAYOUT, pattern="{name}.bin")

    def test_writer_one_file_per_step_in_disk_order(self):
        specs = list(StepIndexedFiles(LAYOUT).writer(_array([0, 1, 12345])))

        assert [s.filename for s in specs] == [
            "ux-0000.bin",
            "ux-0001.bin",
            "ux-12345.bin",
        ]
        assert all(s.sub_array.dims == ("x", "y") for s in specs)
        assert all(s.dtype == "<f4" for s in specs)

    def test_writer_inverts_time_step(self):
        specs = StepIndexedFiles(LAYOUT, time_step=0.25).writer(
            _array([0.25, 0.75, 1.5])
        )

        assert [s.filename for s in specs] == [
            "ux-0001.bin",
            "ux-0003.bin",
            "ux-0006.bin",
        ]

    def test_writer_uses_position_when_time_has_no_coordinate(self):
        array = _array([0, 1]).drop_vars("time")

        assert [s.filename for s in StepIndexedFiles(LAYOUT).writer(array)] == [
            "ux-0000.bin",
            "ux-0001.bin",
        ]

    def test_writer_rejects_non_integer_steps_instead_of_truncating(self):
        specs = StepIndexedFiles(LAYOUT).writer(_array([0.25, 0.75]))

        with pytest.raises(ValueError, match="does not correspond to an integer step"):
            next(specs)

    def test_writer_rejects_wrong_time_step(self):
        specs = StepIndexedFiles(LAYOUT, time_step=0.3).writer(_array([0.25]))

        with pytest.raises(ValueError, match="time_step=0.3"):
            next(specs)

    def test_writer_rejects_duplicated_steps_before_yielding(self):
        specs = StepIndexedFiles(LAYOUT).writer(_array([1, 1]))

        with pytest.raises(ValueError, match="same file"):
            next(specs)

    def test_writer_rejects_name_the_reader_cannot_parse(self):
        specs = StepIndexedFiles(LAYOUT).writer(_array([0]).rename("u.x"))

        with pytest.raises(ValueError, match="cannot be parsed back"):
            next(specs)

    def test_private_class_attributes_are_not_constructor_arguments(self):
        import inspect

        for convention in (StepIndexedFiles, TimeStampedFiles, StaticFiles):
            assert "_required_fields" not in inspect.signature(convention).parameters
            assert "_required_fields" not in repr(convention(LAYOUT))

    def test_reader_time_coordinate_is_int64(self):
        specs = StepIndexedFiles(LAYOUT).reader(pathlib.Path("ux-0007.bin"))

        assert specs.coords["time"].dtype == np.int64

    def test_writer_rejects_unnamed_array(self):
        specs = StepIndexedFiles(LAYOUT).writer(_array([0]).rename(None))

        with pytest.raises(ValueError, match="has no name"):
            next(specs)

    def test_writer_rejects_layout_mismatch(self):
        specs = StepIndexedFiles(LAYOUT).writer(_array([0]).isel(y=0, drop=True))

        with pytest.raises(LayoutMismatchError):
            next(specs)

    def test_writer_can_skip_coordinate_values_check(self):
        array = _array([0]).assign_coords(x=np.arange(3) + 100)
        specs = StepIndexedFiles(LAYOUT).writer(array)

        with pytest.raises(LayoutMismatchError, match="Coordinate mismatch"):
            next(specs)
        assert (
            len(list(StepIndexedFiles(LAYOUT, check_coords=False).writer(array))) == 1
        )

    def test_custom_time_dim(self):
        array = _array([3]).rename(time="step")
        convention = StepIndexedFiles(LAYOUT, time_dim="step")

        assert [s.filename for s in convention.writer(array)] == ["ux-0003.bin"]
        assert convention.reader(pathlib.Path("ux-0003.bin")).dims == ("x", "y", "step")


class TestTimeStampedFiles:
    def test_reader_parses_float_time(self):
        specs = TimeStampedFiles(LAYOUT).reader(pathlib.Path("ux-0.250.bin"))

        assert specs.name == "ux"
        np.testing.assert_array_equal(specs.coords["time"], [0.25])

    def test_writer_formats_time_with_pattern_precision(self):
        specs = TimeStampedFiles(LAYOUT).writer(_array([0.25, 0.75, 1.5]))

        assert [s.filename for s in specs] == [
            "ux-0.250.bin",
            "ux-0.750.bin",
            "ux-1.500.bin",
        ]

    def test_writer_rejects_collisions_at_pattern_precision(self):
        convention = TimeStampedFiles(LAYOUT, pattern="{name}-{time:.1f}.bin")
        specs = convention.writer(_array([0.21, 0.24]))

        with pytest.raises(ValueError, match="same file"):
            next(specs)

    def test_pattern_must_declare_time(self):
        with pytest.raises(ValueError, match=r"must declare the field\(s\) \['time'\]"):
            TimeStampedFiles(LAYOUT, pattern="{name}-{step:04d}.bin")


class TestStaticFiles:
    def test_reader_has_no_time_dimension(self):
        specs = StaticFiles(LAYOUT).reader(pathlib.Path("epsi.bin"))

        assert specs.name == "epsi"
        assert specs.dims == ("x", "y")

    def test_writer_yields_single_file(self):
        array = _array([0]).isel(time=0, drop=True).rename("epsi")

        specs = list(StaticFiles(LAYOUT).writer(array))

        assert [s.filename for s in specs] == ["epsi.bin"]
        assert specs[0].sub_array.dims == ("x", "y")

    def test_writer_rejects_time_dependent_array(self):
        specs = StaticFiles(LAYOUT).writer(_array([0, 1]))

        with pytest.raises(LayoutMismatchError, match="Dimension mismatch"):
            next(specs)


class TestStacks:
    convention = StepIndexedFiles(LAYOUT, stacks=[VELOCITY, SCALARS])

    def test_writer_splits_velocity_components(self):
        u = xr.concat([_array([0, 1])] * 3, dim="i").assign_coords(i=["x", "y", "z"])
        u = u.rename("u")

        assert [s.filename for s in self.convention.writer(u)] == [
            "ux-0000.bin",
            "ux-0001.bin",
            "uy-0000.bin",
            "uy-0001.bin",
            "uz-0000.bin",
            "uz-0001.bin",
        ]

    def test_writer_splits_nested_dimensions_in_declaration_order(self):
        base = _array([0])
        phi = xr.concat([base, base], dim="n").assign_coords(n=[1, 2])
        both = xr.concat([phi, phi], dim="i").assign_coords(i=["x", "y"]).rename("u")

        assert [s.filename for s in self.convention.writer(both)] == [
            "ux1-0000.bin",
            "ux2-0000.bin",
            "uy1-0000.bin",
            "uy2-0000.bin",
        ]

    def test_writer_uses_position_when_split_dim_has_no_coordinate(self):
        u = xr.concat([_array([0])] * 2, dim="n").rename("phi")

        assert [s.filename for s in self.convention.writer(u)] == [
            "phi0-0000.bin",
            "phi1-0000.bin",
        ]

    def test_writer_without_split_dim_is_unchanged(self):
        assert [s.filename for s in self.convention.writer(_array([3]))] == [
            "ux-0003.bin"
        ]

    def test_slices_drop_the_split_coordinate(self):
        u = xr.concat([_array([0])] * 2, dim="i").assign_coords(i=["x", "y"])
        specs = list(self.convention.writer(u.rename("u")))

        assert all("i" not in s.sub_array.coords for s in specs)
        assert all(s.sub_array.dims == ("x", "y") for s in specs)

    def test_writer_refuses_values_outside_the_stack(self):
        u = xr.concat([_array([0])] * 2, dim="i").assign_coords(i=["x", "w"])

        with pytest.raises(ValueError, match=r"\['w'\] on 'i' are not listed"):
            list(self.convention.writer(u.rename("u")))

    def test_stacks_are_frozen_as_a_tuple(self):
        assert self.convention.stacks == (VELOCITY, SCALARS)

    def test_static_files_split_too(self):
        static = StaticFiles(LAYOUT, stacks=[VELOCITY])
        u = (
            xr.concat([_array([0]).isel(time=0, drop=True)] * 2, dim="i")
            .assign_coords(i=["x", "y"])
            .rename("u")
        )

        assert [s.filename for s in static.writer(u)] == ["ux.bin", "uy.bin"]


class TestNameOf:
    def test_writer_takes_name_from_hook(self):
        convention = StepIndexedFiles(LAYOUT, name_of=lambda da: da.attrs["file_name"])
        array = _array([0]).rename("vorticity").assign_attrs(file_name="w3")

        assert [s.filename for s in convention.writer(array)] == ["w3-0000.bin"]

    def test_hook_applies_before_split(self):
        convention = StepIndexedFiles(LAYOUT, stacks=[VELOCITY], name_of=lambda da: "u")
        u = xr.concat([_array([0])] * 2, dim="i").assign_coords(i=["x", "y"])

        assert [s.filename for s in convention.writer(u.rename("anything"))] == [
            "ux-0000.bin",
            "uy-0000.bin",
        ]


class TestTimeDtype:
    def test_reader_casts_time(self):
        convention = StepIndexedFiles(LAYOUT, time_step=0.5, time_dtype=np.float32)

        time = convention.reader(pathlib.Path("ux-0003.bin")).coords["time"]

        assert time.dtype == np.float32
        np.testing.assert_allclose(time, [1.5])

    def test_time_stamped_casts_too(self):
        convention = TimeStampedFiles(LAYOUT, time_dtype="<f4")

        assert convention.reader(pathlib.Path("ux-0.250.bin")).coords[
            "time"
        ].dtype == np.dtype("<f4")


class TestLayoutPassthrough:
    layout = Layout(
        {"x": np.arange(3), "y": np.arange(2)},
        dtype="<f4",
        order="F",
        coord_attrs={"x": {"units": "m"}},
    )

    def test_reader_forwards_order_and_coord_attrs(self):
        specs = StepIndexedFiles(self.layout).reader(pathlib.Path("ux-0001.bin"))

        assert specs.order == "F"
        assert specs.coord_attrs == {"x": {"units": "m"}}

    def test_writer_forwards_order(self):
        specs = list(StepIndexedFiles(self.layout).writer(_array([0])))
        static = list(
            StaticFiles(self.layout).writer(_array([0]).isel(time=0, drop=True))
        )

        assert specs[0].order == "F"
        assert static[0].order == "F"

    def test_defaults_are_c_order_without_attrs(self):
        specs = StepIndexedFiles(LAYOUT).reader(pathlib.Path("ux-0001.bin"))

        assert specs.order == "C"
        assert specs.coord_attrs is None


class TestFilesAndOpen:
    convention = StepIndexedFiles(LAYOUT)

    @pytest.fixture
    def directory(self, tmp_path):
        import xarray_binfile  # noqa: F401  (registers the accessors)

        for name in ("ux", "uy"):
            _array([0, 1]).rename(name).binary_engine.to_file(
                self.convention.writer, tmp_path
            )
        (tmp_path / "epsi.bin").write_bytes(b"\0" * 24)
        (tmp_path / "notes.txt").write_text("ignored")
        (tmp_path / "sub").mkdir()
        return tmp_path

    def test_files_lists_only_matching_regular_files(self, directory):
        assert [p.name for p in self.convention.files(directory)] == [
            "ux-0000.bin",
            "ux-0001.bin",
            "uy-0000.bin",
            "uy-0001.bin",
        ]

    def test_open_combines_all_files(self, directory):
        dataset = self.convention.open(directory)

        assert sorted(dataset.data_vars) == ["ux", "uy"]
        assert dataset["ux"].dims == ("x", "y", "time")
        assert dataset["time"].values.tolist() == [0, 1]

    def test_open_filters_variables_and_forwards_kwargs(self, directory):
        dataset = self.convention.open(directory, variables=["uy"], chunks={"time": 1})

        assert list(dataset.data_vars) == ["uy"]
        assert dataset["uy"].chunks is not None

    def test_open_raises_when_nothing_matches(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No file matches"):
            self.convention.open(tmp_path)

    def test_open_stacks_by_default(self, directory):
        convention = StepIndexedFiles(LAYOUT, stacks=[VELOCITY])

        stacked = convention.open(directory)
        raw = convention.open(directory, stack=False)

        assert sorted(stacked.data_vars) == ["u"]
        assert stacked["i"].values.tolist() == ["x", "y"]
        assert sorted(raw.data_vars) == ["ux", "uy"]


class TestNames:
    @pytest.fixture
    def directory(self, tmp_path):
        import xarray_binfile  # noqa: F401  (registers the accessors)

        static = StaticFiles(LAYOUT, pattern="{name}")
        for name in ("epsilon", "pp"):
            _array([0]).isel(time=0, drop=True).rename(name).binary_engine.to_file(
                static.writer, tmp_path
            )
        for extra in ("README", "Makefile", "snapshots.xdmf", "notes.txt", ".DS_Store"):
            (tmp_path / extra).write_bytes(b"\0" * 8)
        return tmp_path

    def test_bare_name_pattern_is_too_permissive_without_names(self, directory):
        static = StaticFiles(LAYOUT, pattern="{name}")

        assert [p.name for p in static.files(directory)] == [
            "Makefile",
            "README",
            "epsilon",
            "pp",
        ]

    def test_names_restrict_discovery_read_and_open(self, directory):
        static = StaticFiles(LAYOUT, pattern="{name}", names=("epsilon",))

        assert [p.name for p in static.files(directory)] == ["epsilon"]
        assert list(static.open(directory).data_vars) == ["epsilon"]
        with pytest.raises(ValueError, match="not among the names"):
            static.reader(directory / "README")

    def test_names_restrict_the_writer(self):
        static = StaticFiles(LAYOUT, names=("epsilon",))
        array = _array([0]).isel(time=0, drop=True)

        assert [s.filename for s in static.writer(array.rename("epsilon"))] == [
            "epsilon.bin"
        ]
        assert [
            s.filename for s in static.writer(array.rename("geometry/epsilon"))
        ] == ["geometry/epsilon.bin"]
        with pytest.raises(LayoutMismatchError, match="not among the names"):
            next(static.writer(array.rename("pp")))

    def test_names_work_for_time_series_too(self, tmp_path):
        convention = StepIndexedFiles(LAYOUT, names=("ux",))

        assert convention.reader(pathlib.Path("ux-0001.bin")).name == "ux"
        with pytest.raises(ValueError, match="not among the names"):
            convention.reader(pathlib.Path("uy-0001.bin"))
        with pytest.raises(LayoutMismatchError, match="not among the names"):
            next(convention.writer(_array([0]).rename("uy")))

    def test_names_let_pattern_conventions_skip_to_the_next_member(self, directory):
        from xarray_binfile.conventions import PatternConventions

        conventions = PatternConventions(
            [
                StaticFiles(LAYOUT, pattern="{name}", names=("epsilon",)),
                StaticFiles(LAYOUT, pattern="{name}", names=("pp",)),
            ]
        )

        assert [p.name for p in conventions.files(directory)] == ["epsilon", "pp"]
        assert conventions.reader(directory / "pp").name == "pp"
