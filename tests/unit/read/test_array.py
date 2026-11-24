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
        (100, slice(None, None, 1), False),
        (100, slice(0, None), False),
        (100, slice(None, 100), False),
        # non-slice keys (integers) must be treated as sliced, never assumed
        # to expose slice attributes such as ``.start``
        (100, 0, True),
        (100, 50, True),
        # Empty selections. A stop of 0 must not fall back to "up to the end"
        # (https://github.com/fschuch/xarray-binfile/issues/62).
        pytest.param(100, slice(0, 0), True, id="empty-0-0"),
        pytest.param(100, slice(None, 0), True, id="empty-None-0"),
        pytest.param(100, slice(0, 0, 1), True, id="empty-from-dask"),
        (100, slice(100, 100), True),
        (100, slice(50, 10), True),
        (100, slice(None, -100), True),
        # Reversed: same elements, different order
        (100, slice(None, None, -1), True),
        (100, slice(99, None, -1), True),
        # The whole axis in another spelling: out-of-range bounds are clipped,
        # and negative bounds count from the end. Taking these as sliced would
        # not be wrong, only slower (memmap instead of one whole read).
        pytest.param(100, slice(0, 1000), False, id="stop-past-end"),
        pytest.param(100, slice(-100, None), False, id="neg-start"),
        pytest.param(100, slice(-1000, None), False, id="neg-start-past"),
        pytest.param(100, slice(0, 100, 100), True, id="step-past-end-one-element"),
        # A size-1 axis: any step past the end still selects the whole axis
        pytest.param(1, slice(0, 1, 5), False, id="size1-big-step"),
        # A size-0 axis is whole whatever the slice
        (0, slice(None), False),
        pytest.param(0, slice(0, 5), False, id="size0-stop-past"),
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

    keys_against_numpy = (
        pytest.param((slice(0, 0), slice(None)), id="empty-first"),
        pytest.param((slice(None), slice(0, 0)), id="empty-last"),
        pytest.param((slice(0, 0), slice(0, 0)), id="empty-both"),
        pytest.param((slice(3, 3), slice(None)), id="empty-at-end"),
        pytest.param((slice(1, 3), slice(None, None, 2)), id="partial"),
        pytest.param((slice(0, 10), slice(-4, None)), id="whole-other-spelling"),
        pytest.param((slice(None, None, -1), slice(None)), id="reversed"),
    )

    # Whatever path a key takes, the result must be what numpy gives for it.
    @pytest.mark.parametrize("key", keys_against_numpy)
    def test_getitem_matches_numpy(self, array, key):
        actual = array[BasicIndexer(key)]

        assert actual.shape == self.reference[key].shape
        np.testing.assert_array_equal(actual, self.reference[key])

    # The whole read still applies the key, so even a key wrongly taken as
    # whole gives the right data, only through a slower read (issue #62).
    @pytest.mark.parametrize("key", keys_against_numpy)
    def test_getitem_is_right_even_when_every_key_is_taken_as_whole(
        self, array, key, monkeypatch
    ):
        monkeypatch.setattr(
            "xarray_binfile.read.array._is_coord_sliced", lambda size, key: False
        )

        actual = array[BasicIndexer(key)]

        assert actual.shape == self.reference[key].shape
        np.testing.assert_array_equal(actual, self.reference[key])

    def test_getitem_outer_indexer_with_array(self, array):
        key = (np.array([0, 2]), slice(None))
        actual = array[OuterIndexer(key)]
        np.testing.assert_array_equal(actual, self.reference[[0, 2], :])

    def test_getitem_vectorized_indexer(self, array):
        key = (np.array([0, 1, 2]), np.array([0, 1, 2]))
        actual = array[VectorizedIndexer(key)]
        np.testing.assert_array_equal(actual, self.reference[[0, 1, 2], [0, 1, 2]])


EMPTY_FILE = pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="a partial key on a file with a zero-length axis goes through "
    "numpy.memmap, which cannot map an empty file",
)


@pytest.mark.parametrize(
    "key",
    [
        pytest.param((slice(0, 0), slice(None)), id="empty-axis"),
        pytest.param((slice(None), slice(0, 0)), marks=EMPTY_FILE, id="empty-other"),
        pytest.param((slice(None), slice(1, 3)), marks=EMPTY_FILE, id="partial"),
        pytest.param((slice(None), 2), marks=EMPTY_FILE, id="integer"),
    ],
)
def test_partial_read_of_a_file_with_a_zero_length_axis(tmp_path, key):
    # Zero elements, zero bytes: there is nothing to map.
    reference = np.zeros((0, 4))
    file_path = tmp_path / "test.bin"
    reference.tofile(file_path)
    array = BinaryEngineBackendArray(
        file_read_specs_getter_factory(coords={"x": [], "y": range(4)})(file_path)
    )

    actual = array[BasicIndexer(key)]

    assert actual.shape == reference[key].shape


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
                return array._wrap_numpy_fromfile(file)

        result = benchmark(helper)
        assert np.array_equal(result, write_array)

    # Worst case: the whole 76 MiB array through the memmap path. memray counts
    # the file mapping (file-backed, released after the read) plus the copy
    # handed to the caller, so about twice the array. The copy is what keeps
    # loaded data independent of the file (issue #61).
    @pytest.mark.limit_memory("160 MB")
    def test_read_array__numpy_memmap(self, file_path, array, write_array, benchmark):
        keys = tuple(slice(None) for _ in array.metadata.shape)

        def helper():
            with open(file_path, "rb") as file:
                return array._wrap_numpy_memmap(file, key=keys)

        result = benchmark(helper)
        assert np.array_equal(result, write_array)


class TestMemoryOrder:
    @pytest.fixture
    def fortran_file(self, tmp_path):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        file_path = tmp_path / "ux.bin"
        np.asfortranarray(values).ravel(order="F").tofile(file_path)
        return file_path, values

    def _specs(self, file_path, order):
        return ReadSpecs(
            filepath=file_path,
            dtype=np.float32,
            coords={"x": range(2), "y": range(3), "z": range(4)},
            name="ux",
            order=order,
        )

    def test_whole_read_honours_fortran_order(self, fortran_file):
        file_path, values = fortran_file
        dataset = BinaryEngineBackendArray(
            self._specs(file_path, "F")
        ).get_xarray_dataset()

        np.testing.assert_array_equal(dataset["ux"].values, values)

    def test_sliced_read_honours_fortran_order(self, fortran_file):
        file_path, values = fortran_file
        dataset = BinaryEngineBackendArray(
            self._specs(file_path, "F")
        ).get_xarray_dataset()

        np.testing.assert_array_equal(
            dataset["ux"].isel(x=1, z=slice(1, 3)).values, values[1, :, 1:3]
        )

    def test_c_order_reads_the_same_bytes_with_reversed_dims(self, fortran_file):
        file_path, values = fortran_file
        specs = ReadSpecs(
            filepath=file_path,
            dtype=np.float32,
            coords={"z": range(4), "y": range(3), "x": range(2)},
            name="ux",
        )
        dataset = BinaryEngineBackendArray(specs).get_xarray_dataset()

        np.testing.assert_array_equal(dataset["ux"].transpose("x", "y", "z"), values)


def test_coord_attrs_are_attached(tmp_path):
    file_path = tmp_path / "ux.bin"
    np.zeros((2, 3), dtype=np.float32).tofile(file_path)
    specs = ReadSpecs(
        filepath=file_path,
        dtype=np.float32,
        coords={"x": range(2), "y": range(3)},
        name="ux",
        coord_attrs={"x": {"units": "m", "long_name": "streamwise"}},
    )

    dataset = BinaryEngineBackendArray(specs).get_xarray_dataset()

    assert dataset["x"].attrs == {"units": "m", "long_name": "streamwise"}
    assert dataset["y"].attrs == {}


def test_coord_attrs_ignore_coordinates_the_file_does_not_have(tmp_path):
    file_path = tmp_path / "epsi.bin"
    np.zeros((2, 3), dtype=np.float32).tofile(file_path)
    specs = ReadSpecs(
        filepath=file_path,
        dtype=np.float32,
        coords={"x": range(2), "y": range(3)},
        name="epsi",
        coord_attrs={"x": {"units": "m"}, "time": {"units": "s"}},
    )

    dataset = BinaryEngineBackendArray(specs).get_xarray_dataset()

    assert dataset["x"].attrs == {"units": "m"}
    assert "time" not in dataset.coords
