"""
One ``to_file`` call must not write two arrays to the same file.

From the pyOpenSci review, https://github.com/fschuch/xarray-binfile/issues/67:
the duplicate check only looked inside one array's time split, so a stacked
``u`` and a separate ``ux`` in one Dataset both went to ``ux-0000.bin``, and
``U`` and ``u`` collided on a filesystem that ignores case. Every destination
of a call is now checked before the first file is written.
"""

import os
import unicodedata

import numpy as np
import pytest
import xarray as xr

from xarray_binfile.conventions import Layout, StepIndexedFiles, VariableStack
from xarray_binfile.write import WriteSpecs, accessor

LAYOUT = Layout({"x": [0, 1]})
# From CodeRabbit on https://github.com/fschuch/xarray-binfile/pull/86.
REVIEW_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="#86 review: links and Unicode forms are compared as written",
)
CAFE_NFC = unicodedata.normalize("NFC", "café.bin")
CAFE_NFD = unicodedata.normalize("NFD", "café.bin")


def _writing(*filenames):
    def writer(array: xr.DataArray):
        for value, filename in enumerate(filenames):
            yield WriteSpecs(filename=filename, sub_array=array + value)

    return writer


def _linked_folder(link, target):
    target.mkdir()
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:  # no cover - Windows without symlink rights
        pytest.skip("symbolic links are not available")


def _written(folder) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


def _snapshot(value: float) -> xr.DataArray:
    return xr.DataArray(
        np.full((2, 1), value), coords={"x": [0, 1], "time": [0]}, dims=("x", "time")
    )


def test_the_reviewers_stack_and_variable_are_refused(tmp_path):
    velocity = VariableStack("i", "{name}{i}", values=("x", "y", "z"))
    convention = StepIndexedFiles(LAYOUT, stacks=[velocity])
    dataset = xr.Dataset(
        {
            "u": (("i", "x", "time"), np.ones((2, 2, 1))),
            "ux": (("x", "time"), np.full((2, 1), 7.0)),
        },
        coords={"i": ["x", "y"], "x": [0, 1], "time": [0]},
    )

    with pytest.raises(ValueError, match="same file"):
        dataset.binary_engine.to_file(convention.writer, tmp_path)
    assert _written(tmp_path) == []


def test_a_relative_and_an_absolute_path_to_one_file_are_refused(tmp_path):
    def writer(array: xr.DataArray):
        yield WriteSpecs(filename="ux.bin", sub_array=array)
        yield WriteSpecs(filename=tmp_path / "ux.bin", sub_array=array * 2)

    ux = _snapshot(1.0)

    with pytest.raises(ValueError, match="same file"):
        ux.binary_engine.to_file(writer, tmp_path)
    assert _written(tmp_path) == []


def test_names_differing_only_in_case_are_refused_where_case_is_ignored(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(accessor, "_ignores_case", lambda folder: True)
    dataset = xr.Dataset({"U": _snapshot(1.0), "u": _snapshot(7.0)})
    writer = StepIndexedFiles(LAYOUT).writer

    with pytest.raises(ValueError, match="only in case"):
        dataset.binary_engine.to_file(writer, tmp_path)
    assert _written(tmp_path) == []


def test_names_differing_only_in_case_are_written_where_case_matters(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(accessor, "_ignores_case", lambda folder: False)
    dataset = xr.Dataset({"U": _snapshot(1.0), "u": _snapshot(7.0)})

    dataset.binary_engine.to_file(StepIndexedFiles(LAYOUT).writer, tmp_path)

    assert _written(tmp_path)


def test_the_case_probe_matches_the_filesystem_and_leaves_nothing(tmp_path):
    (tmp_path / "CaseCheck").touch()
    expected = (tmp_path / "casecheck").exists()
    (tmp_path / "CaseCheck").unlink()

    assert accessor._ignores_case(tmp_path / "not" / "yet") is expected
    assert _written(tmp_path) == []


def test_a_later_invalid_variable_stops_the_call_before_any_file_is_written(
    tmp_path,
):
    dataset = xr.Dataset(
        {
            "ux": _snapshot(1.0),
            # A dimension the layout does not know.
            "uy": _snapshot(2.0).expand_dims(y=3),
        }
    )
    writer = StepIndexedFiles(LAYOUT).writer

    with pytest.raises(ValueError, match="'y'"):
        dataset.binary_engine.to_file(writer, tmp_path)
    assert _written(tmp_path) == []


@REVIEW_XFAIL
def test_two_paths_through_a_linked_folder_are_refused(tmp_path):
    _linked_folder(tmp_path / "alias", tmp_path / "real")
    ux = _snapshot(1.0)

    with pytest.raises(ValueError, match="same file"):
        ux.binary_engine.to_file(_writing("alias/ux.bin", "real/ux.bin"), tmp_path)
    assert _written(tmp_path / "real") == []


def test_linked_folders_to_different_places_are_written(tmp_path):
    _linked_folder(tmp_path / "a", tmp_path / "first")
    _linked_folder(tmp_path / "b", tmp_path / "second")

    _snapshot(1.0).binary_engine.to_file(_writing("a/ux.bin", "b/ux.bin"), tmp_path)

    assert _written(tmp_path / "first") == ["ux.bin"]
    assert _written(tmp_path / "second") == ["ux.bin"]


@REVIEW_XFAIL
def test_unicode_forms_of_one_name_are_refused_where_the_form_is_ignored(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(accessor, "_ignores_case", lambda folder: False)
    monkeypatch.setattr(
        accessor, "_ignores_normalization", lambda folder: True, raising=False
    )
    ux = _snapshot(1.0)

    with pytest.raises(ValueError, match="Unicode form"):
        ux.binary_engine.to_file(_writing(CAFE_NFC, CAFE_NFD), tmp_path)
    assert _written(tmp_path) == []


def test_unicode_forms_of_one_name_are_written_where_the_form_matters(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(accessor, "_ignores_case", lambda folder: False)
    monkeypatch.setattr(
        accessor, "_ignores_normalization", lambda folder: False, raising=False
    )

    _snapshot(1.0).binary_engine.to_file(_writing(CAFE_NFC, CAFE_NFD), tmp_path)

    assert _written(tmp_path)


@REVIEW_XFAIL
def test_the_normalization_probe_matches_the_filesystem_and_leaves_nothing(
    tmp_path,
):
    (tmp_path / CAFE_NFC).touch()
    expected = (tmp_path / CAFE_NFD).exists()
    (tmp_path / CAFE_NFC).unlink()

    assert accessor._ignores_normalization(tmp_path / "not" / "yet") is expected
    assert _written(tmp_path) == []
