import pathlib

import numpy as np
import pytest
import xarray as xr

from xarray_binfile.conventions import (
    FolderConventions,
    Layout,
    LayoutMismatchError,
    PatternConventions,
    StaticFiles,
    StepIndexedFiles,
)

X, Y, Z = np.arange(3), np.arange(2), np.arange(4)
CONVENTIONS = FolderConventions(
    {
        "xy_planes": StepIndexedFiles(Layout({"x": X, "y": Y})),
        "snapshots/3d": StepIndexedFiles(Layout({"x": X, "y": Y, "z": Z})),
        "static": StaticFiles(Layout({"x": X, "y": Y, "z": Z})),
    }
)


def test_reader_dispatches_on_parent_folder():
    plane = CONVENTIONS.reader(pathlib.Path("/case/xy_planes/ux-0001.bin"))
    volume = CONVENTIONS.reader(pathlib.Path("case/snapshots/3d/ux-0001.bin"))

    assert plane.dims == ("x", "y", "time")
    assert volume.dims == ("x", "y", "z", "time")


def test_reader_prefers_most_specific_folder_regardless_of_order():
    conventions = FolderConventions(
        {
            "3d": StepIndexedFiles(Layout({"x": X})),
            "snapshots/3d": StepIndexedFiles(Layout({"x": X, "y": Y, "z": Z})),
        }
    )

    specs = conventions.reader(pathlib.Path("case/snapshots/3d/ux-0001.bin"))

    assert specs.dims == ("x", "y", "z", "time")
    assert conventions.reader(pathlib.Path("case/3d/ux-0001.bin")).dims == ("x", "time")


def test_reader_rejects_unknown_folder():
    with pytest.raises(ValueError, match="No convention registered"):
        CONVENTIONS.reader(pathlib.Path("case/3d/ux-0001.bin"))


def test_writer_picks_the_single_matching_convention_and_prefixes_folder():
    array = xr.DataArray(
        np.zeros((2, 3, 2)), coords={"time": [0, 1], "x": X, "y": Y}, name="ux"
    )

    assert [s.filename for s in CONVENTIONS.writer(array)] == [
        "xy_planes/ux-0000.bin",
        "xy_planes/ux-0001.bin",
    ]


def test_writer_rejects_when_nothing_matches():
    array = xr.DataArray(np.zeros(3), coords={"x": X}, name="ux")
    specs = CONVENTIONS.writer(array)

    with pytest.raises(LayoutMismatchError, match="No convention accepts"):
        next(specs)


def test_writer_rejects_ambiguity():
    ambiguous = FolderConventions(
        {
            "a": StepIndexedFiles(Layout({"x": X})),
            "b": StepIndexedFiles(Layout({"x": X}), pattern="{name}.{step:d}"),
        }
    )
    array = xr.DataArray(np.zeros((1, 3)), coords={"time": [0], "x": X}, name="ux")
    specs = ambiguous.writer(array)

    with pytest.raises(LayoutMismatchError, match="Several conventions accept"):
        next(specs)


def test_writer_accepts_convention_that_yields_nothing():
    array = xr.DataArray(
        np.zeros((0, 3, 2)), coords={"time": [], "x": X, "y": Y}, name="ux"
    )

    assert list(CONVENTIONS.writer(array)) == []


class TestRootFolder:
    conventions = FolderConventions(
        {
            ".": StepIndexedFiles(Layout({"x": X, "y": Y, "z": Z})),
            "xy_planes": StepIndexedFiles(Layout({"x": X, "y": Y})),
        }
    )

    def test_reader_falls_back_to_root(self):
        assert self.conventions.reader(pathlib.Path("case/ux-0001.bin")).dims == (
            "x",
            "y",
            "z",
            "time",
        )
        assert self.conventions.reader(
            pathlib.Path("case/xy_planes/ux-0001.bin")
        ).dims == ("x", "y", "time")

    def test_writer_does_not_prefix_root_files(self):
        array = xr.DataArray(
            np.zeros((1, 3, 2, 4)),
            coords={"time": [0], "x": X, "y": Y, "z": Z},
            name="ux",
        )

        assert [s.filename for s in self.conventions.writer(array)] == ["ux-0000.bin"]

    def test_empty_string_is_the_root_too(self):
        conventions = FolderConventions({"": StepIndexedFiles(Layout({"x": X}))})

        assert conventions.reader(pathlib.Path("ux-0001.bin")).dims == ("x", "time")


class TestFolderPrefixedNames:
    layout = Layout({"x": X, "y": Y, "z": Z})
    conventions = FolderConventions(
        {".": StaticFiles(layout), "geometry": StaticFiles(layout)}
    )
    array = xr.DataArray(np.zeros((3, 2, 4)), coords={"x": X, "y": Y, "z": Z})

    def test_name_with_folder_prefix_selects_that_folder(self):
        specs = list(self.conventions.writer(self.array.rename("geometry/epsi")))

        assert [s.filename for s in specs] == ["geometry/epsi.bin"]
        assert specs[0].sub_array.name == "epsi"

    def test_name_without_prefix_is_ambiguous_between_equal_layouts(self):
        with pytest.raises(LayoutMismatchError, match="Several conventions accept"):
            next(self.conventions.writer(self.array.rename("epsi")))

    def test_unknown_folder_prefix_falls_back_to_root(self):
        conventions = FolderConventions({".": StaticFiles(self.layout)})

        assert [
            s.filename for s in conventions.writer(self.array.rename("other/epsi"))
        ] == ["other/epsi.bin"]

    def test_nested_prefix_strips_the_registered_part_only(self):
        conventions = FolderConventions({"geometry": StaticFiles(self.layout)})

        assert [
            s.filename
            for s in conventions.writer(self.array.rename("geometry/sub/epsi"))
        ] == ["geometry/sub/epsi.bin"]


class TestFilesAndOpen:
    @pytest.fixture
    def case(self, tmp_path):
        import xarray_binfile  # noqa: F401  (registers the accessors)

        conventions = FolderConventions(
            {
                ".": StepIndexedFiles(Layout({"x": X, "y": Y, "z": Z})),
                "static": StaticFiles(Layout({"x": X, "y": Y, "z": Z})),
            }
        )
        ux = xr.DataArray(
            np.zeros((2, 3, 2, 4)),
            coords={"time": [0, 1], "x": X, "y": Y, "z": Z},
            name="ux",
        )
        epsi = xr.DataArray(np.ones((3, 2, 4)), coords={"x": X, "y": Y, "z": Z})
        ux.binary_engine.to_file(conventions.writer, tmp_path)
        epsi.rename("static/epsi").binary_engine.to_file(conventions.writer, tmp_path)
        (tmp_path / "missing").mkdir()
        return conventions, tmp_path

    def test_files_walks_registered_folders(self, case):
        conventions, root = case

        assert [p.relative_to(root).as_posix() for p in conventions.files(root)] == [
            "static/epsi.bin",
            "ux-0000.bin",
            "ux-0001.bin",
        ]

    def test_open_merges_folders(self, case):
        conventions, root = case
        dataset = conventions.open(root)

        assert sorted(dataset.data_vars) == ["epsi", "ux"]
        assert dataset["epsi"].dims == ("x", "y", "z")
        assert dataset["ux"].dims == ("x", "y", "z", "time")


class TestPatternConventions:
    layout = Layout({"x": X, "y": Y, "z": Z})
    conventions = PatternConventions([StepIndexedFiles(layout), StaticFiles(layout)])

    def test_reader_tries_in_order(self):
        step = self.conventions.reader(pathlib.Path("case/ux-0001.bin"))
        static = self.conventions.reader(pathlib.Path("case/epsi.bin"))

        assert step.dims == ("x", "y", "z", "time")
        assert static.dims == ("x", "y", "z")

    def test_reader_rejects_unknown_files(self):
        with pytest.raises(ValueError, match="No convention accepts the file"):
            self.conventions.reader(pathlib.Path("ux-0001.bak"))

    def test_writer_picks_by_layout(self):
        snapshot = xr.DataArray(
            np.zeros((1, 3, 2, 4)),
            coords={"time": [0], "x": X, "y": Y, "z": Z},
            name="ux",
        )
        static = xr.DataArray(np.zeros((3, 2, 4)), coords={"x": X, "y": Y, "z": Z})

        assert [s.filename for s in self.conventions.writer(snapshot)] == [
            "ux-0000.bin"
        ]
        assert [s.filename for s in self.conventions.writer(static.rename("e"))] == [
            "e.bin"
        ]

    def test_writer_rejects_ambiguity(self):
        ambiguous = PatternConventions(
            [StaticFiles(self.layout), StaticFiles(self.layout, pattern="{name}.dat")]
        )
        static = xr.DataArray(np.zeros((3, 2, 4)), coords={"x": X, "y": Y, "z": Z})

        with pytest.raises(LayoutMismatchError, match="Several conventions accept"):
            next(ambiguous.writer(static.rename("e")))

    def test_files_and_open_cover_both_patterns(self, tmp_path):
        import xarray_binfile  # noqa: F401  (registers the accessors)

        snapshot = xr.DataArray(
            np.zeros((1, 3, 2, 4)),
            coords={"time": [0], "x": X, "y": Y, "z": Z},
            name="ux",
        )
        static = xr.DataArray(
            np.ones((3, 2, 4)), coords={"x": X, "y": Y, "z": Z}, name="epsi"
        )
        snapshot.binary_engine.to_file(self.conventions.writer, tmp_path)
        static.binary_engine.to_file(self.conventions.writer, tmp_path)

        assert [p.name for p in self.conventions.files(tmp_path)] == [
            "epsi.bin",
            "ux-0000.bin",
        ]
        assert sorted(self.conventions.open(tmp_path).data_vars) == ["epsi", "ux"]


class TestUnrelatedFilesAreNeverOpened:
    layout = Layout({"x": X, "y": Y, "z": Z})

    @pytest.fixture
    def case(self, tmp_path):
        for name in ("ux-0000.bin", "epsi.bin"):
            (tmp_path / name).write_bytes(b"\0" * 8 * 24)
        for extra in (
            "snapshots.xdmf",
            "notes.txt",
            "input.i3d",
            ".DS_Store",
            ".ux-0000.bin.tmp123.binary_engine",
            "ux-0000.bin.bak",
        ):
            (tmp_path / extra).write_bytes(b"\0" * 8)
        return tmp_path

    def test_composed_conventions_skip_them(self, case):
        conventions = FolderConventions(
            {
                ".": PatternConventions(
                    [StepIndexedFiles(self.layout), StaticFiles(self.layout)]
                )
            }
        )

        assert [p.name for p in conventions.files(case)] == ["epsi.bin", "ux-0000.bin"]

    def test_custom_convention_without_files_is_asked_through_its_reader(self, case):
        class OnlyUx:
            def reader(self, path):
                return StepIndexedFiles(self.__class__.layout).reader(path)

            def writer(self, data_array):  # no cov
                raise NotImplementedError

        OnlyUx.layout = self.layout
        conventions = FolderConventions({".": OnlyUx()})

        assert [p.name for p in conventions.files(case)] == ["ux-0000.bin"]
