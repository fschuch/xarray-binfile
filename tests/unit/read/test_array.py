import pathlib

import numpy as np
import pytest
from xarray.core.indexing import BasicIndexer, OuterIndexer, VectorizedIndexer

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
        # non-slice keys (integers) must be treated as sliced, never assumed
        # to expose slice attributes such as ``.start``
        (100, 0, True),
        (100, 50, True),
    ],
)
def test_is_coord_sliced(size: int, slice_spec: slice | int, *, expected: bool):
    actual = _is_coord_sliced(size, slice_spec)
    assert actual == expected


def test_backend_array_dtype_is_normalized_to_numpy_dtype(tmp_path):
    file_path = tmp_path / "test.bin"
    np.zeros((2, 3), dtype=np.float32).tofile(file_path)
    metadata = file_read_specs_getter_factory(
        coords={"x": range(2), "y": range(3)},
        dtype=np.float32,
    )(file_path)

    array = BinaryEngineBackendArray(metadata)

    assert array.dtype == np.dtype(np.float32)
    assert array.dtype.itemsize == 4


@pytest.mark.parametrize("num_values", [5, 7], ids=["too_small", "too_large"])
def test_backend_array_rejects_file_size_mismatch(tmp_path, num_values):
    file_path = tmp_path / "test.bin"
    np.zeros(num_values, dtype=np.float32).tofile(file_path)
    metadata = file_read_specs_getter_factory(
        coords={"x": range(2), "y": range(3)},
        dtype=np.float32,
    )(file_path)

    with pytest.raises(ValueError, match="Size mismatch"):
        BinaryEngineBackendArray(metadata)


class TestBackendArrayIndexing:
    """
    Regression tests for indexing with non-slice keys.

    The backend once declared VECTORIZED indexing support while only handling
    slices, so any integer or array key coming out of ``isel``/``sel`` crashed
    with ``AttributeError: 'int' object has no attribute 'start'``. These
    tests exercise ``__getitem__`` with every explicit indexer flavor to make
    sure the declared indexing support matches what the implementation
    handles.
    """

    reference = np.arange(12, dtype=np.float64).reshape(3, 4)

    @pytest.fixture
    def array(self, tmp_path) -> BinaryEngineBackendArray:
        file_path = tmp_path / "test.bin"
        self.reference.tofile(file_path)
        metadata = file_read_specs_getter_factory(
            coords={"x": range(3), "y": range(4)},
        )(file_path)
        return BinaryEngineBackendArray(metadata)

    @pytest.mark.parametrize(
        "key",
        [
            (1, slice(None)),
            (slice(None), 2),
            (0, 3),
            (slice(1, 3), slice(None)),
            (slice(None), slice(None)),
        ],
        ids=["int_slice", "slice_int", "int_int", "partial_slices", "full_slices"],
    )
    def test_getitem_basic_indexer(self, array, key):
        actual = array[BasicIndexer(key)]
        np.testing.assert_array_equal(actual, self.reference[key])

    def test_getitem_outer_indexer_with_array(self, array):
        key = (np.array([0, 2]), slice(None))
        actual = array[OuterIndexer(key)]
        np.testing.assert_array_equal(actual, self.reference[[0, 2], :])

    def test_getitem_vectorized_indexer(self, array):
        key = (np.array([0, 1, 2]), np.array([0, 1, 2]))
        actual = array[VectorizedIndexer(key)]
        np.testing.assert_array_equal(actual, self.reference[[0, 1, 2], [0, 1, 2]])


def test_backend_array_rejects_missing_file(tmp_path):
    metadata = file_read_specs_getter_factory(
        coords={"x": range(2), "y": range(3)},
        dtype=np.float32,
    )(tmp_path / "missing.bin")

    with pytest.raises(FileNotFoundError):
        BinaryEngineBackendArray(metadata)


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
    def write_array(self, file_path, metadata) -> np.ndarray:
        array = self.random_generator.random(size=metadata.shape, dtype=metadata.dtype)
        array.tofile(file_path)
        return array

    @pytest.fixture
    def array(self, metadata, write_array):
        return BinaryEngineBackendArray(metadata)

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
