import pathlib
from functools import cached_property

import numpy as np
import pytest
import xarray as xr

from xarray_binfile.tutorial import DatasetGenerator, FileSpecsGetter
from xarray_binfile.write import BinaryEngineDataset  # noqa F401


class TestOpenDataset:
    file_specs_getter = FileSpecsGetter(
        base_coords={"x": np.arange(5), "y": np.arange(10), "z": np.arange(15)}
    )
    dataset_generator = DatasetGenerator(file_specs_getter.reader)

    @cached_property
    def dataset(self) -> xr.Dataset:
        filenames = (
            self.file_specs_getter.filename_template.format(name=n, digits=t)
            for n in ("ux", "uy")
            for t in range(5)
        )
        return self.dataset_generator(map(pathlib.Path, filenames))

    @pytest.fixture  # (scope="class")
    def write_files(self, tmpdir_factory) -> pathlib.Path:
        tmp_path = tmpdir_factory.mktemp("data")
        self.dataset.binary_engine.to_file(self.file_specs_getter.writer, tmp_path)
        return pathlib.Path(tmp_path)

    @pytest.mark.parametrize(
        "chunks",
        [
            {"x": 2, "y": 5, "z": 3, "time": 1},
            None,
        ],
    )
    def test_load_dataset__success(self, write_files, chunks):
        ds = xr.open_mfdataset(
            write_files.glob("*.bin"),
            engine="binfile",
            read_specs_getter=self.file_specs_getter.reader,
            chunks=chunks,
            parallel=True,
        ).load()
        xr.testing.assert_equal(ds, self.dataset)
