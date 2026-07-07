import pathlib

import numpy as np
import pytest

from xarray_binfile.tutorial import FileSpecsGetter


def test_reader_parses_name_and_time():
    getter = FileSpecsGetter(base_coords={"x": np.arange(3)})

    specs = getter.reader(pathlib.Path("ux-0007.bin"))

    assert specs.name == "ux"
    assert specs.coords["time"].tolist() == [7]


def test_reader_rejects_invalid_filename():
    getter = FileSpecsGetter(base_coords={"x": np.arange(3)})

    with pytest.raises(ValueError, match="Invalid filename"):
        getter.reader(pathlib.Path("not-a-valid-name.txt"))
