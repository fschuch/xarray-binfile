import pytest
from hypothesis import given
from hypothesis import strategies as st

from xarray_binfile.conventions import FilenamePattern

NAMES = st.from_regex(r"\A[A-Za-z_][A-Za-z0-9_]{0,7}\Z")
STEPS = st.integers(min_value=0, max_value=10**9)
TIMES = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)


@pytest.mark.parametrize(
    ("template", "filename", "expected"),
    [
        ("{name}-{step:04d}.bin", "ux-0007.bin", {"name": "ux", "step": 7}),
        ("{name}-{step:04d}.bin", "ux-12345.bin", {"name": "ux", "step": 12345}),
        ("{name}{step:03d}", "ux001", {"name": "ux", "step": 1}),
        ("{name}.{step:d}", "phi1.42", {"name": "phi1", "step": 42}),
        ("{name}-{time:.3f}.bin", "ux-0.250.bin", {"name": "ux", "time": 0.25}),
        ("{name}-{time:e}.bin", "ux-2.500000e-01.bin", {"name": "ux", "time": 0.25}),
        ("{name}-{time:g}.bin", "ux-1e+06.bin", {"name": "ux", "time": 1e6}),
        ("{name}.bin", "epsi.bin", {"name": "epsi"}),
    ],
)
def test_parse(template, filename, expected):
    assert FilenamePattern(template).parse(filename) == expected


@pytest.mark.parametrize(
    "filename",
    ["ux-0001.bin.bak", "ux-001.bin", "ux_0001.bin", "ux-0001.dat", "prefix/ux-0001.bin"],
)
def test_parse_rejects_non_matching(filename):
    pattern = FilenamePattern("{name}-{step:04d}.bin")

    assert not pattern.matches(filename)
    with pytest.raises(ValueError, match="does not match the pattern"):
        pattern.parse(filename)


@pytest.mark.parametrize(
    "template",
    ["no-fields.bin", "{}-{step:04d}.bin", "{name!r}.bin", "{name}-{name}.bin", "{step:x}"],
)
def test_invalid_templates_are_rejected(template):
    with pytest.raises(ValueError):
        FilenamePattern(template)


@given(name=NAMES, step=STEPS)
def test_step_roundtrip(name, step):
    pattern = FilenamePattern("{name}-{step:04d}.bin")

    assert pattern.parse(pattern.format(name=name, step=step)) == {
        "name": name,
        "step": step,
    }


@given(name=st.from_regex(r"\A[A-Za-z_][A-Za-z0-9_]{0,6}[A-Za-z_]\Z"), step=STEPS)
def test_step_roundtrip_without_separator(name, step):
    # Without a separator the name must not end with a digit, or the split
    # between name and step is ambiguous.
    pattern = FilenamePattern("{name}{step:03d}.bin")

    assert pattern.parse(pattern.format(name=name, step=step)) == {
        "name": name,
        "step": step,
    }


@given(name=NAMES, time=TIMES)
def test_time_roundtrip_within_precision(name, time):
    pattern = FilenamePattern("{name}-{time:.3f}.bin")

    parsed = pattern.parse(pattern.format(name=name, time=time))

    assert parsed["name"] == name
    assert parsed["time"] == pytest.approx(time, abs=0.5e-3)


@given(name=NAMES, time=TIMES)
def test_time_roundtrip_is_exact_with_repr_precision(name, time):
    pattern = FilenamePattern("{name}-{time:.17g}.bin")

    assert pattern.parse(pattern.format(name=name, time=time)) == {
        "name": name,
        "time": time,
    }
