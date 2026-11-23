"""
Sphinx extension: show the ``pattern`` field with its constructor type.

The shipped conventions declare ``pattern`` as a descriptor-typed dataclass
field, so autodoc would render the private descriptor class in their
signatures. Readers care about what the constructor accepts, which is a
template string or a ``FilenamePattern``.
"""

from typing import Any

_DESCRIPTOR = "_PatternField"
_CONSTRUCTOR_TYPE = "FilenamePattern | str"


def _rewrite_signature(
    app: Any,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    signature: str | None,
    return_annotation: str | None,
) -> tuple[str | None, str | None]:
    if signature is not None:
        signature = signature.replace(_DESCRIPTOR, _CONSTRUCTOR_TYPE)
    return signature, return_annotation


def setup(app: Any) -> dict[str, bool]:
    app.connect("autodoc-process-signature", _rewrite_signature)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
