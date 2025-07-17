import pathlib

import numpy as np
import pytest

from xarray_binfile.read.array import BinaryEngineBackendArray, _is_coord_sliced
from xarray_binfile.read.file_metadata import ReadSpecs, ReadSpecsGetterProtocol
from xarray_binfile.typing import AttributesLike, CoordsLike, DTypeLike


def file_read_specs_getter_factory(
    coords: CoordsLike,
    dtype: DTypeLike = np.float64,
    attrs: AttributesLike | None = None,
) -> ReadSpecsGetterProtocol:
    def helper(path: pathlib.Path) -> ReadSpecs:
        return ReadSpecs(
            filepath=path,
            dtype=dtype,
            coords=coords,
            name=path.name.removesuffix(".bin"),
            attrs=attrs,
        )

    return helper


@pytest.mark.parametrize(
    ("size", "slice_spec", "expected"),
    [
        (100, slice(0, 100), False),
        (100, slice(0, 50), True),
        (100, slice(50, 100), True),
        (100, slice(0, 100, 2), True),
        (100, slice(1, 100), True),
        (100, slice(0, 99), True),
        (100, slice(None), False),
    ],
)
def test_is_coord_sliced(size: int, slice_spec: slice, *, expected: bool):
    actual = _is_coord_sliced(size, slice_spec)
    assert actual == expected


class TestArrayBenchmark:
    random_generator = np.random.Generator(np.random.PCG64(1234))

    @pytest.fixture
    def file_path(self, tmp_path):
        return tmp_path / "test.bin"

    @pytest.fixture
    def metadata(self, file_path):
        read_specs_getter = file_read_specs_getter_factory(
            coords={"x": range(1000), "y": range(100), "z": range(100)}
        )
        return read_specs_getter(file_path)

    @pytest.fixture
    def array(self, metadata):
        return BinaryEngineBackendArray(metadata)

    @pytest.fixture
    def write_array(self, file_path, metadata) -> np.ndarray:
        array = self.random_generator.random(size=metadata.shape, dtype=metadata.dtype)
        array.tofile(file_path)
        return array

    @pytest.mark.limit_memory("86 MB")
    def test_read_array__numpy_fromfile(self, file_path, array, write_array, benchmark):
        def helper():
            with open(file_path, "rb") as file:
                return array._wrap_numpy_fromfile(file)  # noqa: SLF001

        result = benchmark(helper)
        assert np.array_equal(result, write_array)

    @pytest.mark.limit_memory("86 MB")
    def test_read_array__numpy_memmap(self, file_path, array, write_array, benchmark):
        keys = tuple(slice(None) for _ in array.metadata.shape)

        def helper():
            with open(file_path, "rb") as file:
                return array._wrap_numpy_memmap(file, key=keys)  # noqa: SLF001

        result = benchmark(helper)
        assert np.array_equal(result, write_array)
