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
# Folder segments accepted in front of a ``name`` value at format time
# (``geometry/epsilon``). A segment cannot start with ``.``, so ``..`` never
# sneaks into a filename; the write accessor rejects escaping paths anyway.
_FOLDER_PREFIX = re.compile(r"^(?:\w[\w.-]*/)+$")
_INT_SPEC = re.compile(r"^(?P<fill>0?)(?P<width>\d*)d$")
_FLOAT_SPEC = re.compile(
    r"^(?P<fill>0?)(?P<width>\d*)(?:\.(?P<precision>\d+))?(?P<kind>[feEgG])$"
)
_SIGN = r"[-+]?"


def _int_regex(width: int, *, exact_width: bool = False) -> str:
    """
    Regex fragment for an integer written with a zero-fill ``width``.

    Zero-fill is a *minimum* width for ``str.format``: wider numbers are
    written in full, so by default the fragment accepts them as well. With
    ``exact_width`` the fragment accepts exactly ``width`` characters, which
    removes the ambiguity of templates without a separator between a name
    ending in a digit and the number (``phi1`` + ``000``). The sign counts
    towards the width, so signed values carry one digit less.

    Args:
        width: Width declared in the format specification, 0 if none.
        exact_width: Match exactly ``width`` characters instead of at least.

    Returns:
        The regex fragment.
    """
    if not width:
        return rf"{_SIGN}\d+"
    bound = "" if exact_width else ","
    return rf"(?:\d{{{width}{bound}}}|[-+]\d{{{max(width - 1, 1)}{bound}}})"


def _float_regex(kind: str, precision: str | None) -> str:
    """
    Regex fragment for a float written with the ``f``, ``e`` or ``g`` kinds.

    Args:
        kind: Lower-case presentation type.
        precision: Digits after the decimal point, or ``None`` when omitted.

    Returns:
        The regex fragment.
    """
    if kind == "f":
        if precision == "0":
            return rf"{_SIGN}\d+"  # ``.0f`` writes no decimal point
        decimals = rf"\d{{{precision}}}" if precision else r"\d{6}"
        return rf"{_SIGN}\d+\.{decimals}"
    if kind == "e":
        return rf"{_SIGN}\d(?:\.\d+)?[eE][-+]\d+"
    return rf"{_SIGN}(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _field_regex(spec: str, *, exact_width: bool = False) -> tuple[str, type]:
    """
    Translate one format specification into a regex fragment and a Python type.

    Args:
        spec: The format specification found after ``:`` in the template.
        exact_width: Match zero-filled integers with exactly their width.

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
        return _int_regex(width, exact_width=exact_width), int
    if float_match := _FLOAT_SPEC.match(spec):
        kind = float_match.group("kind").lower()
        return _float_regex(kind, float_match.group("precision")), float
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
      ``ux`` and ``1``. When formatting, the value may carry folder segments
      (``geometry/epsilon``), which are kept in front of the filename and
      turned into sub-folders by the write accessor; parsing always works on
      the bare filename.
    - ``{step:d}``, ``{step:04d}``: an integer. By default a zero-fill width is
      treated as a minimum width, matching ``str.format`` semantics, so step
      ``12345`` written with ``04d`` can still be read back. Without a
      separator, names ending in a digit are then ambiguous (``phi1000`` reads
      as ``phi`` + ``1000``); set ``exact_width=True`` to match exactly the
      declared width instead, so ``phi1000`` reads as ``phi1`` + ``000`` and
      steps wider than the template are refused at write time.
    - ``{time:.3f}``, ``{time:f}``, ``{time:e}``, ``{time:g}``: a float.

    Examples:
        >>> pattern = FilenamePattern("{name}-{step:04d}.bin")
        >>> pattern.format(name="ux", step=7)
        'ux-0007.bin'
        >>> pattern.parse("ux-0007.bin")
        {'name': 'ux', 'step': 7}
        >>> pattern.parse("ux-12345.bin")
        {'name': 'ux', 'step': 12345}
        >>> pattern.format(name="geometry/epsi", step=0)
        'geometry/epsi-0000.bin'
        >>> pattern.glob()
        '*-*.bin'
        >>> pattern.glob(name="ux")
        'ux-*.bin'
        >>> FilenamePattern("{name}{step:03d}").parse("ux001")
        {'name': 'ux', 'step': 1}
        >>> FilenamePattern("{name}{step:03d}").parse("phi1000")
        {'name': 'phi', 'step': 1000}
        >>> FilenamePattern("{name}{step:03d}", exact_width=True).parse("phi1000")
        {'name': 'phi1', 'step': 0}

    Attributes:
        template: The ``str.format`` template.
        exact_width: Whether zero-filled integers match exactly their width.
        fields: Field names in the order they appear in ``template``.
        regex: Compiled regular expression equivalent to ``template``.
    """

    template: str
    exact_width: bool = False
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
            fragment, type_ = _field_regex(spec or "", exact_width=self.exact_width)
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
        containing ``.`` or ``-``) is rejected here, at write time. A ``name``
        value may start with folder segments (``"geometry/epsi"``); they are
        kept in front of the checked filename (``"geometry/epsi.bin"``).

        Args:
            **fields: One value per field declared in the template.

        Returns:
            The formatted filename.

        Raises:
            ValueError: If the formatted filename does not match the pattern,
                or if the folder prefix of ``name`` is malformed.
        """
        folder = ""
        if "name" in fields and "/" in str(fields["name"]):
            head, _, base = str(fields["name"]).rpartition("/")
            folder = f"{head}/"
            if not _FOLDER_PREFIX.match(folder):
                error_message = (
                    f"Invalid folder prefix {head!r} in name {fields['name']!r}: "
                    "segments must be non-empty and cannot start with '.'."
                )
                raise ValueError(error_message)
            fields = {**fields, "name": base}
        filename = self.template.format(**fields)
        if not self.matches(filename):
            error_message = (
                f"Formatted filename {filename!r} cannot be parsed back by the "
                f"pattern {self.template!r}; check the field values {fields!r}."
            )
            raise ValueError(error_message)
        return folder + filename

    def glob(self, **fixed: Any) -> str:
        """
        Build a glob pattern that matches filenames following the template.

        Every field is replaced by ``*`` unless a value for it is given, in
        which case the value is formatted with the field specification. The
        result is coarser than :meth:`matches` (``*`` also accepts text the
        regular expression would reject), so filter candidates with
        :meth:`matches` or :meth:`parse` after globbing.

        Args:
            **fixed: Values for the fields to pin, for example ``name="ux"``.

        Returns:
            A pattern for :func:`glob.glob` or :meth:`pathlib.Path.glob`.

        Raises:
            ValueError: If ``fixed`` names a field the template does not declare.
        """
        unknown = set(fixed) - set(self.fields)
        if unknown:
            error_message = f"Unknown field(s) {sorted(unknown)} for the template {self.template!r}."
            raise ValueError(error_message)
        parts: list[str] = []
        for literal, name, spec, _ in string.Formatter().parse(self.template):
            parts.append(literal)
            if name is None:
                continue
            if name in fixed:
                parts.append(format(fixed[name], spec or ""))
            else:
                parts.append("*")
        return "".join(parts)

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
