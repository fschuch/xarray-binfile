import pytest
from hypothesis import given
from hypothesis import strategies as st

from xarray_binfile.conventions import FilenamePattern

NAMES = st.from_regex(r"\A[A-Za-z_][A-Za-z0-9_]{0,7}\Z")
STEPS = st.integers(min_value=-(10**9), max_value=10**9)
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
        ("{name}-{time:.0f}.bin", "ux-1.bin", {"name": "ux", "time": 1.0}),
        ("{name}-{step:04d}.bin", "ux--005.bin", {"name": "ux", "step": -5}),
        ("{name}-{step:04d}.bin", "ux--12345.bin", {"name": "ux", "step": -12345}),
        ("{name}.bin", "epsi.bin", {"name": "epsi"}),
    ],
)
def test_parse(template, filename, expected):
    assert FilenamePattern(template).parse(filename) == expected


@pytest.mark.parametrize(
    "filename",
    [
        "ux-0001.bin.bak",
        "ux-001.bin",
        "ux--01.bin",
        "ux_0001.bin",
        "ux-0001.dat",
        "prefix/ux-0001.bin",
    ],
)
def test_parse_rejects_non_matching(filename):
    pattern = FilenamePattern("{name}-{step:04d}.bin")

    assert not pattern.matches(filename)
    with pytest.raises(ValueError, match="does not match the pattern"):
        pattern.parse(filename)


@pytest.mark.parametrize(
    "template",
    [
        "no-fields.bin",
        "{}-{step:04d}.bin",
        "{name!r}.bin",
        "{name}-{name}.bin",
        "{step:x}",
    ],
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


@pytest.mark.parametrize(
    ("template", "fields"),
    [
        ("{name}-{step:04d}.bin", {"name": "u.x", "step": 1}),
        ("{name}-{step:04d}.bin", {"name": "u-x", "step": 1}),
        ("{name}-{step:04d}.bin", {"name": "", "step": 1}),
        ("{name}-{time:.3f}.bin", {"name": "ux", "time": float("nan")}),
    ],
)
def test_format_rejects_values_that_do_not_parse_back(template, fields):
    pattern = FilenamePattern(template)

    with pytest.raises(ValueError, match="cannot be parsed back"):
        pattern.format(**fields)


@pytest.mark.parametrize("precision", [0, 1, 3])
@given(name=NAMES, time=TIMES)
def test_fixed_precision_roundtrip(precision, name, time):
    pattern = FilenamePattern(f"{{name}}-{{time:.{precision}f}}.bin")

    parsed = pattern.parse(pattern.format(name=name, time=time))

    assert parsed["name"] == name
    assert parsed["time"] == pytest.approx(time, abs=0.5 * 10**-precision)


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


class TestExactWidth:
    def test_parse_keeps_digits_in_the_name(self):
        pattern = FilenamePattern("{name}{step:03d}", exact_width=True)

        assert pattern.parse("phi1000") == {"name": "phi1", "step": 0}
        assert pattern.parse("ux001") == {"name": "ux", "step": 1}

    def test_default_is_minimum_width(self):
        assert FilenamePattern("{name}{step:03d}").parse("phi1000") == {
            "name": "phi",
            "step": 1000,
        }

    def test_rejects_wider_steps_at_format_time(self):
        pattern = FilenamePattern("{name}-{step:03d}.bin", exact_width=True)

        with pytest.raises(ValueError, match="cannot be parsed back"):
            pattern.format(name="ux", step=1000)

    def test_signed_values_count_the_sign(self):
        pattern = FilenamePattern("{name}-{step:03d}.bin", exact_width=True)

        assert pattern.parse("ux--05.bin") == {"name": "ux", "step": -5}
        assert not pattern.matches("ux--005.bin")

    @given(name=NAMES, step=st.integers(min_value=0, max_value=999))
    def test_roundtrip(self, name, step):
        pattern = FilenamePattern("{name}{step:03d}", exact_width=True)

        assert pattern.parse(pattern.format(name=name, step=step)) == {
            "name": name,
            "step": step,
        }


class TestGlob:
    @pytest.mark.parametrize(
        ("template", "fixed", "expected"),
        [
            ("{name}-{step:04d}.bin", {}, "*-*.bin"),
            ("{name}-{step:04d}.bin", {"name": "ux"}, "ux-*.bin"),
            ("{name}-{step:04d}.bin", {"step": 7}, "*-0007.bin"),
            ("{name}{step:03d}", {}, "**"),
            ("{name}.bin", {"name": "epsi"}, "epsi.bin"),
        ],
    )
    def test_glob(self, template, fixed, expected):
        assert FilenamePattern(template).glob(**fixed) == expected

    def test_glob_rejects_unknown_field(self):
        with pytest.raises(ValueError, match="Unknown field"):
            FilenamePattern("{name}.bin").glob(step=1)

    def test_glob_matches_formatted_names(self, tmp_path):
        pattern = FilenamePattern("{name}-{step:04d}.bin")
        for name, step in (("ux", 1), ("uy", 2)):
            (tmp_path / pattern.format(name=name, step=step)).touch()
        (tmp_path / "epsi.bin").touch()

        assert sorted(p.name for p in tmp_path.glob(pattern.glob())) == [
            "ux-0001.bin",
            "uy-0002.bin",
        ]
        assert [p.name for p in tmp_path.glob(pattern.glob(name="ux"))] == [
            "ux-0001.bin"
        ]


class TestFolderPrefix:
    def test_format_keeps_folder_segments(self):
        pattern = FilenamePattern("{name}-{step:04d}.bin")

        assert pattern.format(name="geometry/epsi", step=0) == "geometry/epsi-0000.bin"
        assert pattern.format(name="a/b/c", step=0) == "a/b/c-0000.bin"

    @pytest.mark.parametrize("name", ["../epsi", "./epsi", "/epsi", "a//epsi"])
    def test_format_rejects_malformed_prefix(self, name):
        with pytest.raises(ValueError, match="Invalid folder prefix"):
            FilenamePattern("{name}.bin").format(name=name)

    def test_parse_still_works_on_bare_filenames_only(self):
        pattern = FilenamePattern("{name}.bin")

        assert not pattern.matches("geometry/epsi.bin")
        assert pattern.parse("epsi.bin") == {"name": "epsi"}
