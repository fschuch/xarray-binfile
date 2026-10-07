"""
Single-source filename conventions.

A :class:`FilenamePattern` is built from one ``str.format``-style template such
as ``"{name}-{step:04d}.bin"``. Both the formatter used when writing and the
regular expression used when reading are derived from that single template, so
the two can never drift apart (for example a zero-fill width that the regex
does not accept).
"""

import re
import string
from dataclasses import dataclass, field
from typing import Any

_STRING_FIELD = r"\w+?"
_INT_SPEC = re.compile(r"^(?P<fill>0?)(?P<width>\d*)d$")
_FLOAT_SPEC = re.compile(
    r"^(?P<fill>0?)(?P<width>\d*)(?:\.(?P<precision>\d+))?(?P<kind>[feEgG])$"
)
_SIGN = r"[-+]?"


def _field_regex(spec: str) -> tuple[str, type]:
    """
    Translate one format specification into a regex fragment and a Python type.

    Args:
        spec: The format specification found after ``:`` in the template.

    Returns:
        The regex fragment matching values written with ``spec`` and the type
        used to convert matched text back into a Python value.

    Raises:
        ValueError: If ``spec`` is not supported.
    """
    if spec in ("", "s"):
        return _STRING_FIELD, str

    if int_match := _INT_SPEC.match(spec):
        width = int(int_match.group("width") or 0)
        # Zero-fill is a *minimum* width for ``str.format``: wider numbers are
        # written in full, so the regex must accept them as well. The sign
        # counts towards the width, so signed values have one digit less.
        if width:
            return rf"(?:\d{{{width},}}|[-+]\d{{{max(width - 1, 1)},}})", int
        return rf"{_SIGN}\d+", int

    if float_match := _FLOAT_SPEC.match(spec):
        kind = float_match.group("kind").lower()
        precision = float_match.group("precision")
        if kind == "f":
            if precision == "0":
                return rf"{_SIGN}\d+", float  # ``.0f`` writes no decimal point
            decimals = rf"\d{{{precision}}}" if precision else r"\d{6}"
            return rf"{_SIGN}\d+\.{decimals}", float
        if kind == "e":
            return rf"{_SIGN}\d(?:\.\d+)?[eE][-+]\d+", float
        return rf"{_SIGN}(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", float

    error_message = (
        f"Unsupported format specification {spec!r}. Supported specifications are "
        "plain strings (``{name}``), integers with optional zero-fill "
        "(``{step:04d}``), and floats (``{time:.3f}``, ``{time:e}``, ``{time:g}``)."
    )
    raise ValueError(error_message)


@dataclass(frozen=True)
class FilenamePattern:
    """
    Format and parse filenames from a single ``str.format``-style template.

    The regular expression used by :meth:`parse` is derived from ``template``,
    so writing with :meth:`format` and reading with :meth:`parse` are always
    consistent. Matching is anchored to the whole filename, so a trailing
    ``.bak`` or a different suffix is rejected.

    Supported field specifications:

    - ``{name}`` or ``{name:s}``: a word made of letters, digits and underscores.
      It is matched lazily, so ``"{name}{step:03d}"`` splits ``ux001`` into
      ``ux`` and ``1``; without a separator, names ending in a digit are
      ambiguous and should be avoided.
    - ``{step:d}``, ``{step:04d}``: an integer. A zero-fill width is treated as
      a minimum width, matching ``str.format`` semantics, so step ``12345``
      written with ``04d`` can still be read back.
    - ``{time:.3f}``, ``{time:f}``, ``{time:e}``, ``{time:g}``: a float.

    Examples:
        >>> pattern = FilenamePattern("{name}-{step:04d}.bin")
        >>> pattern.format(name="ux", step=7)
        'ux-0007.bin'
        >>> pattern.parse("ux-0007.bin")
        {'name': 'ux', 'step': 7}
        >>> pattern.parse("ux-12345.bin")
        {'name': 'ux', 'step': 12345}
        >>> FilenamePattern("{name}{step:03d}").parse("ux001")
        {'name': 'ux', 'step': 1}

    Attributes:
        template: The ``str.format`` template.
        fields: Field names in the order they appear in ``template``.
        regex: Compiled regular expression equivalent to ``template``.
    """

    template: str
    fields: tuple[str, ...] = field(init=False, repr=False)
    regex: re.Pattern[str] = field(init=False, repr=False)
    _types: dict[str, type] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Derive ``fields``, ``regex`` and the field types from ``template``.

        Raises:
            ValueError: If the template has no fields, repeats a field, uses a
                conversion flag (``!r``), or uses an unsupported specification.
        """
        parts: list[str] = []
        fields: list[str] = []
        types: dict[str, type] = {}
        for literal, name, spec, conversion in string.Formatter().parse(self.template):
            parts.append(re.escape(literal))
            if name is None:
                continue
            if not name or conversion is not None:
                error_message = (
                    f"Invalid template {self.template!r}: every field must be "
                    "named and conversion flags such as '!r' are not supported."
                )
                raise ValueError(error_message)
            if name in types:
                error_message = (
                    f"Invalid template {self.template!r}: field {name!r} is repeated."
                )
                raise ValueError(error_message)
            fragment, type_ = _field_regex(spec or "")
            parts.append(f"(?P<{name}>{fragment})")
            fields.append(name)
            types[name] = type_
        if not fields:
            error_message = f"Template {self.template!r} does not declare any field."
            raise ValueError(error_message)
        object.__setattr__(self, "fields", tuple(fields))
        object.__setattr__(self, "regex", re.compile("".join(parts)))
        object.__setattr__(self, "_types", types)

    def format(self, **fields: Any) -> str:
        """
        Build a filename from field values.

        The result is checked against the derived regular expression, so a
        filename that :meth:`parse` could not read back (for example a name
        containing ``.`` or ``-``) is rejected here, at write time.

        Args:
            **fields: One value per field declared in the template.

        Returns:
            The formatted filename.

        Raises:
            ValueError: If the formatted filename does not match the pattern.
        """
        filename = self.template.format(**fields)
        if not self.matches(filename):
            error_message = (
                f"Formatted filename {filename!r} cannot be parsed back by the "
                f"pattern {self.template!r}; check the field values {fields!r}."
            )
            raise ValueError(error_message)
        return filename

    def matches(self, filename: str) -> bool:
        """
        Check whether ``filename`` follows the template.

        Args:
            filename: The file name (without directories) to test.

        Returns:
            True if the whole filename matches the template.
        """
        return self.regex.fullmatch(filename) is not None

    def parse(self, filename: str) -> dict[str, Any]:
        """
        Extract typed field values from ``filename``.

        Args:
            filename: The file name (without directories) to parse.

        Returns:
            A mapping from field name to its value, converted to ``str``,
            ``int`` or ``float`` according to the template specification.

        Raises:
            ValueError: If ``filename`` does not follow the template.
        """
        match = self.regex.fullmatch(filename)
        if match is None:
            error_message = (
                f"Filename {filename!r} does not match the pattern {self.template!r}."
            )
            raise ValueError(error_message)
        return {
            name: self._types[name](value) for name, value in match.groupdict().items()
        }
