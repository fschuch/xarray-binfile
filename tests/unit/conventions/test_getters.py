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
)

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
        with pytest.raises(ValueError, match="does not match the pattern"):
            StepIndexedFiles(LAYOUT).reader(pathlib.Path("ux-0007.bin.bak"))

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
        with pytest.raises(ValueError, match="does not correspond to an integer step"):
            list(StepIndexedFiles(LAYOUT).writer(_array([0.25, 0.75])))

    def test_writer_rejects_wrong_time_step(self):
        with pytest.raises(ValueError, match="time_step=0.3"):
            list(StepIndexedFiles(LAYOUT, time_step=0.3).writer(_array([0.25])))

    def test_writer_rejects_duplicated_steps_before_yielding(self):
        with pytest.raises(ValueError, match="same file"):
            next(StepIndexedFiles(LAYOUT).writer(_array([1, 1])))

    def test_writer_rejects_name_the_reader_cannot_parse(self):
        with pytest.raises(ValueError, match="cannot be parsed back"):
            next(StepIndexedFiles(LAYOUT).writer(_array([0]).rename("u.x")))

    def test_private_class_attributes_are_not_constructor_arguments(self):
        import inspect

        for convention in (StepIndexedFiles, TimeStampedFiles, StaticFiles):
            assert "_required_fields" not in inspect.signature(convention).parameters
            assert "_required_fields" not in repr(convention(LAYOUT))

    def test_reader_time_coordinate_is_int64(self):
        specs = StepIndexedFiles(LAYOUT).reader(pathlib.Path("ux-0007.bin"))

        assert specs.coords["time"].dtype == np.int64

    def test_writer_rejects_unnamed_array(self):
        with pytest.raises(ValueError, match="has no name"):
            next(StepIndexedFiles(LAYOUT).writer(_array([0]).rename(None)))

    def test_writer_rejects_layout_mismatch(self):
        array = _array([0]).isel(y=0, drop=True)

        with pytest.raises(LayoutMismatchError):
            next(StepIndexedFiles(LAYOUT).writer(array))

    def test_writer_can_skip_coordinate_values_check(self):
        array = _array([0]).assign_coords(x=np.arange(3) + 100)

        with pytest.raises(LayoutMismatchError, match="Coordinate mismatch"):
            next(StepIndexedFiles(LAYOUT).writer(array))
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
        with pytest.raises(ValueError, match="same file"):
            next(
                TimeStampedFiles(LAYOUT, pattern="{name}-{time:.1f}.bin").writer(
                    _array([0.21, 0.24])
                )
            )

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
        with pytest.raises(LayoutMismatchError, match="Dimension mismatch"):
            next(StaticFiles(LAYOUT).writer(_array([0, 1])))
