"""Parser for the standardized formula docstring contract (issue #248).

Every public function in :mod:`vaft.formula` documents itself in one layout:
a one-line summary, a *Definition* paragraph (math in ``$..$``), numpydoc
``Parameters`` / ``Returns`` items whose description paragraph closes with a
unit tag such as ``[Wb/rad]`` or ``[-]``, and any of a fixed set of
underlined sections -- ``Convention``, ``Physical interpretation``,
``Assumptions``, ``Validity``, ``Limitations``, ``Numerical notes``,
``References``.  The docstring is the single source of truth: the catalog
in :mod:`vaft.formula.catalog` and the generated reference pages are both
read out of it through this module.

Only the standard library is used here, and nothing in ``vaft.formula``
imports this module at package-import time.  ``parse_docstring`` never
raises: every contract violation is collected in ``ParsedDocstring.errors``
so the enforcement test can report all of them per function.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field

#: Section titles the contract allows, in the order they are rendered.
SECTION_VOCABULARY: tuple[str, ...] = (
    "Parameters",
    "Returns",
    "Yields",
    "Raises",
    "Convention",
    "Physical interpretation",
    "Assumptions",
    "Validity",
    "Limitations",
    "Numerical notes",
    "References",
    "Notes",
    "See Also",
    "Examples",
    "Warnings",
)

#: The provenance sections issue #248 adds on top of numpydoc.
CUSTOM_SECTIONS: tuple[str, ...] = (
    "Convention",
    "Physical interpretation",
    "Assumptions",
    "Validity",
    "Limitations",
    "Numerical notes",
)

#: Sections whose body is a list of ``name : type`` items.
_ITEM_SECTIONS = frozenset({"Parameters", "Returns", "Yields", "Raises"})

#: Sections whose items must carry a unit tag.
_UNIT_SECTIONS = frozenset({"Parameters", "Returns", "Yields"})

#: An empirical formula opens its ``Validity`` section with this sentence.
EMPIRICAL_MARKER = "Empirical fit."

#: Section titles a module docstring may use.
MODULE_SECTION_VOCABULARY: tuple[str, ...] = (
    "Notation",
    "Conventions",
    "Notes",
    "References",
    "Examples",
    "See Also",
)

_UNDERLINE = re.compile(r"^-{3,}\s*$")
_PARAM_HEAD = re.compile(
    r"^(?P<name>\*{0,2}[A-Za-z_]\w*(?:\s*,\s*\*{0,2}[A-Za-z_]\w*)*)"
    r"\s*(?::\s*(?P<type>.+?))?\s*$"
)
_UNIT_TAG = re.compile(r"\[(?P<unit>[^\[\]]+)\]\.?\s*$")
_REFERENCE = re.compile(r"^\.\.\s*\[(?P<label>[^\]]+)\]\s*(?P<text>.*)$")
_CITATION = re.compile(r"\[(\w+)\]_")
_ROLE = re.compile(r":(?:func|class|mod|attr|meth|data|obj|const):`~?([^`]+)`")
_NOTATION_ROW = re.compile(r"^(?P<symbol>\S[^:]*?)\s*:\s+(?P<rest>.+?)\s*$")


@dataclass(frozen=True)
class ParamDoc:
    """One ``Parameters`` item."""

    name: str
    type: str
    unit: str | None
    description: str


@dataclass(frozen=True)
class ReturnDoc:
    """One ``Returns`` / ``Yields`` item; ``name`` is ``None`` for a bare type."""

    name: str | None
    type: str
    unit: str | None
    description: str


@dataclass(frozen=True)
class RaiseDoc:
    """One ``Raises`` item."""

    type: str
    description: str


@dataclass(frozen=True)
class Reference:
    """One ``.. [label] text`` entry of a ``References`` section."""

    label: str
    text: str


@dataclass(frozen=True)
class ParsedDocstring:
    """A function docstring split into the contract's parts.

    ``sections`` holds every section present, as ``(title, text)`` in the
    order of :data:`SECTION_VOCABULARY`, including the item sections in their
    raw text form so callers can render everything uniformly.
    """

    summary: str
    description: str
    parameters: tuple[ParamDoc, ...]
    returns: tuple[ReturnDoc, ...]
    raises: tuple[RaiseDoc, ...]
    sections: tuple[tuple[str, str], ...]
    references: tuple[Reference, ...]
    deprecated: bool
    empirical: bool
    convention_sensitive: bool
    errors: tuple[str, ...]

    def section(self, title: str) -> str | None:
        """Text of the section called ``title``, or ``None``."""
        for name, text in self.sections:
            if name == title:
                return text
        return None


@dataclass(frozen=True)
class NotationRow:
    """One row of a module docstring's ``Notation`` table."""

    symbol: str
    description: str
    unit: str


@dataclass(frozen=True)
class ModuleDoc:
    """A submodule docstring: title, overview prose and notation table."""

    title: str
    overview: str
    notation: tuple[NotationRow, ...] = ()
    conventions: str = ""
    errors: tuple[str, ...] = field(default=())


def strip_roles(text: str) -> str:
    """Turn Sphinx markup into plain text for the site export.

    Roles (``:func:`x```) become literals (````x````) and citation references
    (``[1]_``) lose their trailing underscore.  The site renders kramdown, not
    reST, so the generated pages must not carry either; the source docstrings
    keep them for Sphinx-style tooling.
    """
    return _CITATION.sub(r"[\1]", _ROLE.sub(r"``\1``", text))


def _split_sections(
    lines: list[str], vocabulary: tuple[str, ...]
) -> tuple[list[str], list[tuple[str, list[str]]], list[str]]:
    """Split ``lines`` into preamble and ``(title, body_lines)`` sections.

    A header is an unindented line immediately followed by an unindented
    line of three or more dashes.  A header whose title is not in
    ``vocabulary`` is reported and its body is still consumed, so one typo
    cannot cascade into "missing section" errors downstream.
    """
    errors: list[str] = []
    headers: list[tuple[int, str]] = []
    for index in range(len(lines) - 1):
        line, underline = lines[index], lines[index + 1]
        if not line or line[0].isspace() or underline[:1].isspace():
            continue
        if _UNDERLINE.match(underline) and not _UNDERLINE.match(line):
            headers.append((index, line.strip()))

    preamble = lines[: headers[0][0]] if headers else list(lines)
    sections: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for position, (start, title) in enumerate(headers):
        end = headers[position + 1][0] if position + 1 < len(headers) else len(lines)
        body = _trim(lines[start + 2 : end])
        if title not in vocabulary:
            errors.append(f"unknown section header {title!r}")
            continue
        if title in seen:
            errors.append(f"duplicate section {title!r}")
            continue
        seen.add(title)
        sections.append((title, body))
    return preamble, sections, errors


def _trim(lines: list[str]) -> list[str]:
    """Drop leading and trailing blank lines."""
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def _items(body: list[str]) -> list[tuple[str, list[str]]]:
    """Group an item section into ``(head_line, body_lines)`` pairs."""
    groups: list[tuple[str, list[str]]] = []
    for line in body:
        if line and not line[0].isspace():
            groups.append((line.strip(), []))
        elif groups:
            groups[-1][1].append(line.strip())
        elif line.strip():
            groups.append(("", [line.strip()]))
    return groups


def _unit_and_text(body: list[str], where: str, errors: list[str]) -> tuple[str | None, str]:
    """Pull the unit tag off the first body line; join the rest as prose."""
    if not any(line for line in body):
        errors.append(f"{where}: missing description and unit tag")
        return None, ""
    # The tag closes the first paragraph: the end of its last line is the
    # canonical place, the end of its first line the other accepted one.
    paragraph = 0
    while paragraph < len(body) and body[paragraph]:
        paragraph += 1
    candidates = [paragraph - 1] if paragraph else []
    if paragraph > 1:
        candidates.append(0)
    for index in candidates:
        match = _UNIT_TAG.search(body[index])
        if match is None:
            continue
        unit = match.group("unit").strip()
        stripped = body[index][: match.start()].rstrip()
        if match.group(0).rstrip().endswith(".") and stripped and not stripped.endswith("."):
            stripped += "."
        lines = list(body)
        lines[index] = stripped
        return unit, " ".join(line for line in lines if line).strip()
    errors.append(f"{where}: the first description paragraph must end with a unit tag like [m]")
    return None, " ".join(line for line in body if line)


def _parse_parameters(body: list[str], errors: list[str]) -> tuple[ParamDoc, ...]:
    parsed: list[ParamDoc] = []
    for head, lines in _items(body):
        match = _PARAM_HEAD.match(head)
        if match is None:
            errors.append(f"Parameters: malformed item head {head!r}")
            continue
        name = match.group("name")
        unit, text = _unit_and_text(lines, f"Parameters: {name}", errors)
        parsed.append(ParamDoc(name, (match.group("type") or "").strip(), unit, text))
    return tuple(parsed)


def _parse_returns(body: list[str], title: str, errors: list[str]) -> tuple[ReturnDoc, ...]:
    parsed: list[ReturnDoc] = []
    for head, lines in _items(body):
        if not head:
            errors.append(f"{title}: item without a type line")
            continue
        name: str | None = None
        type_ = head
        if ":" in head:
            candidate, _, rest = head.partition(":")
            if _PARAM_HEAD.match(candidate.strip()) and "," not in candidate:
                name, type_ = candidate.strip(), rest.strip()
        unit, text = _unit_and_text(lines, f"{title}: {name or type_}", errors)
        parsed.append(ReturnDoc(name, type_, unit, text))
    return tuple(parsed)


def _parse_raises(body: list[str]) -> tuple[RaiseDoc, ...]:
    return tuple(
        RaiseDoc(head, " ".join(line for line in lines if line).strip())
        for head, lines in _items(body)
        if head
    )


def _parse_references(body: list[str], errors: list[str]) -> tuple[Reference, ...]:
    parsed: list[Reference] = []
    for line in body:
        if not line.strip():
            continue
        if line[0].isspace():
            if not parsed:
                errors.append(f"References: continuation line before any entry: {line.strip()!r}")
                continue
            last = parsed[-1]
            parsed[-1] = Reference(last.label, f"{last.text} {line.strip()}".strip())
            continue
        match = _REFERENCE.match(line)
        if match is None:
            errors.append(f"References: entries must look like '.. [1] text', got {line.strip()!r}")
            continue
        parsed.append(Reference(match.group("label").strip(), match.group("text").strip()))
    return tuple(parsed)


def parse_docstring(text: str | None) -> ParsedDocstring:
    """Parse a function docstring against the contract.

    Never raises.  A missing docstring yields an empty result whose
    ``errors`` say so; every other violation -- unknown section title,
    missing unit tag, malformed reference -- is appended to ``errors`` and
    parsing continues.
    """
    if not text or not text.strip():
        return ParsedDocstring(
            "", "", (), (), (), (), (), False, False, False, ("missing docstring",)
        )

    lines = inspect.cleandoc(text).splitlines()
    preamble, sections, errors = _split_sections(lines, SECTION_VOCABULARY)

    preamble = _trim(preamble)
    summary_lines: list[str] = []
    for line in preamble:
        if not line.strip():
            break
        summary_lines.append(line.strip())
    summary = " ".join(summary_lines)
    description = "\n".join(_trim(preamble[len(summary_lines) :]))
    if not summary:
        errors.append("missing summary line")

    parameters: tuple[ParamDoc, ...] = ()
    returns: tuple[ReturnDoc, ...] = ()
    raises: tuple[RaiseDoc, ...] = ()
    references: tuple[Reference, ...] = ()
    rendered: dict[str, str] = {}
    for title, body in sections:
        rendered[title] = "\n".join(body)
        if title == "Parameters":
            parameters = _parse_parameters(body, errors)
        elif title in ("Returns", "Yields"):
            returns = returns + _parse_returns(body, title, errors)
        elif title == "Raises":
            raises = _parse_raises(body)
        elif title == "References":
            references = _parse_references(body, errors)

    ordered = tuple((title, rendered[title]) for title in SECTION_VOCABULARY if title in rendered)
    validity = rendered.get("Validity", "").lstrip()
    return ParsedDocstring(
        summary=summary,
        description=description,
        parameters=parameters,
        returns=returns,
        raises=raises,
        sections=ordered,
        references=references,
        deprecated=summary.lower().startswith("deprecated"),
        empirical=validity.startswith(EMPIRICAL_MARKER),
        convention_sensitive="Convention" in rendered,
        errors=tuple(errors),
    )


def _parse_notation(body: list[str]) -> tuple[NotationRow, ...]:
    rows: list[NotationRow] = []
    for line in body:
        if not line.strip():
            continue
        match = _NOTATION_ROW.match(line)
        if match is None:
            continue
        rest = match.group("rest")
        parts = re.split(r"\s{2,}", rest, maxsplit=1)
        description = parts[0].strip()
        unit = parts[1].strip() if len(parts) > 1 else ""
        if unit.startswith("[") and unit.endswith("]") and unit.count("[") == 1:
            unit = unit[1:-1]
        rows.append(NotationRow(match.group("symbol").strip(), description, unit))
    return tuple(rows)


def parse_module_docstring(text: str | None) -> ModuleDoc:
    """Parse a submodule docstring into title, overview and notation table."""
    if not text or not text.strip():
        return ModuleDoc("", "", errors=("missing module docstring",))
    lines = inspect.cleandoc(text).splitlines()
    preamble, sections, errors = _split_sections(lines, MODULE_SECTION_VOCABULARY)
    preamble = _trim(preamble)
    title = preamble[0].strip() if preamble else ""
    overview = "\n".join(_trim(preamble[1:]))
    notation: tuple[NotationRow, ...] = ()
    conventions = ""
    for name, body in sections:
        if name == "Notation":
            notation = _parse_notation(body)
        elif name == "Conventions":
            conventions = "\n".join(body)
    if not title:
        errors.append("missing module title line")
    return ModuleDoc(title, overview, notation, conventions, tuple(errors))
