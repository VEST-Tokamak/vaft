"""GPEC's ASCII output: sectioned tables, transcribed rather than interpreted.

The ``gpec_singcoup_matrix_n*.out``, ``gpec_singcoup_svd_n*.out``,
``gpec_response_n*.out`` and ``gpec_response_fun_n*.out`` files all share one
layout, so one parser reads them::

    GPEC_SINGCOUP_MATRIX: Coupling matrices between resonant ...   <- kind
    v1.5.5-378-gf06e6ab                                            <- version

       jac_out = boozer     tmag_out = 1                           <- file scalars
         msing =   4      mpert = 129 ...

     The coupling matrix to effective resonant fields              <- section

      q = 2.000  psi =  5.93643838E-001                            <- block scalars

        m        real(C_f)        imag(C_f)                        <- columns
      -64  2.38809557E-005  4.32442929E-006                        <- rows

A file is :class:`GpecAsciiOutput`; a section is a titled group of blocks; a
block is a :class:`GpecAsciiTable` -- its own scalars, its column names, and
a 2-D float array in the file's own column order.  ``singcoup`` files put one
block per rational surface in each section; ``response`` files usually put one
block in each.

What this module deliberately does not do:

- **No reinterpretation.**  Columns keep their file names, including GPEC's
  ``real(X)``/``imag(X)`` pairs and its typos (one section really is titled
  "The coupling matrix to to penetrated resonant fields").  Pairing a
  ``real``/``imag`` column pair into a complex array is available on request
  (:meth:`GpecAsciiTable.complex_column`) because it composes stored columns
  and decides nothing; applying a helicity or phase convention is a separate
  question and is not answered here.
- **No dropping.**  Every line of the file is accounted for: as the kind
  line, the version, a note, a scalar, a section title, a legend line, a
  column header, or a data row.  Anything the parser cannot classify is kept
  verbatim in :attr:`GpecAsciiOutput.unparsed` with its line number, so a
  format change surfaces as data rather than as silence.  A numeric row whose
  width does not match the open block's columns goes there too, rather than
  being padded or truncated into a shape it does not have -- the trailing
  summary rows of ``gpec_recon_integration`` are the real example.
- **No section collapsing.**  Sections are an ordered tuple, not a mapping:
  ``gpec_response_n*.out`` has two different sections both titled
  "Eigenvectors", and a mapping would lose one.  The legacy reader
  (``gpec_analysis_parsers.parse_gpec_matrix``) collapsed all five singcoup
  coupling matrices into one flat list of surfaces, so a caller could not
  tell an effective-resonant-field coupling from an island-width one; that
  loss is what this container exists to prevent.

Fortran fixed-form comment lines (``c  ...``, which ``gpec_recon_*`` writes
between a column header and its rows) are kept verbatim as notes or as the
section's legend.

Scalars are read from ``key = value`` pairs when the whole line is pairs.  A
prose gloss such as ``rho = Reluctance (power norm)`` leaves text over and is
kept as a legend line instead of being mangled into a scalar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

__all__ = [
    "GpecAsciiOutput",
    "GpecAsciiSection",
    "GpecAsciiTable",
    "read_gpec_ascii",
    "read_gpec_response",
    "read_gpec_singcoup",
]

#: The first line names the file; GPEC is not consistent about case
#: (``GPEC_SINGCOUP_MATRIX``, but ``GPEC_dw``), so only line 1 is matched
#: against this and any colon-terminated first token counts.
_KIND = re.compile(r"^\s*(\S+):\s*(.*?)\s*$")
_VERSION = re.compile(r"^\s*(v\d[\w.\-+]*)\s*$")
#: ``key = value``.  Keys may carry the punctuation GPEC uses in names
#: (``sweet-spot``, ``K_x^L``); the value must be one token, so a prose
#: gloss leaves text over and is not mistaken for a scalar.
_PAIR = re.compile(r"([A-Za-z_][\w^.\-]*)\s*=\s*(\S+)")
_COMPLEX_COLUMN = re.compile(r"^(real|imag)\((?P<name>.+)\)$")
#: Fortran fixed-form comment: ``gpec_recon_*`` writes its notes this way, in
#: between the column header and the data it describes.
_COMMENT = re.compile(r"^\s*c\s")


def _as_scalar(text: str) -> Any:
    """``"1"`` -> 1, ``"5.9E-1"`` -> 0.59, anything else stays a string."""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _scalars(line: str) -> Optional[dict[str, Any]]:
    """``key = value`` pairs, but only when they account for the whole line."""
    pairs = list(_PAIR.finditer(line))
    if not pairs:
        return None
    leftover = _PAIR.sub("", line).strip()
    if leftover:
        return None
    return {match.group(1): _as_scalar(match.group(2)) for match in pairs}


def _is_data(line: str) -> bool:
    tokens = line.split()
    if not tokens:
        return False
    for token in tokens:
        try:
            float(token)
        except ValueError:
            return False
    return True


@dataclass
class GpecAsciiTable:
    """One block: its scalars, its column names, and its rows.

    ``data`` is ``(rows, len(columns))`` in the file's own column order.
    """

    columns: tuple[str, ...]
    data: np.ndarray
    header: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def column(self, name: str) -> np.ndarray:
        """One column by its file name."""
        try:
            index = self.columns.index(name)
        except ValueError:
            raise KeyError(f"no column {name!r}; this block has {list(self.columns)}") from None
        return self.data[:, index]

    def complex_column(self, name: str) -> np.ndarray:
        """``real(name)`` and ``imag(name)`` combined, without reinterpreting them.

        GPEC stores a complex quantity as two adjacent real columns; this
        composes them and nothing more.  Whether that number should be
        conjugated for a given helicity is a separate question this container
        does not answer.
        """
        return self.column(f"real({name})") + 1j * self.column(f"imag({name})")

    @property
    def complex_names(self) -> tuple[str, ...]:
        """Names that appear as a ``real``/``imag`` column pair."""
        seen: dict[str, set[str]] = {}
        for column in self.columns:
            match = _COMPLEX_COLUMN.match(column)
            if match:
                seen.setdefault(match.group("name"), set()).add(match.group(1))
        return tuple(name for name, parts in seen.items() if parts == {"real", "imag"})

    def to_dict(self) -> dict[str, np.ndarray]:
        """Columns as a name -> array mapping, in file order."""
        return {name: self.data[:, index] for index, name in enumerate(self.columns)}


@dataclass
class GpecAsciiSection:
    """A titled group of blocks, with the free text GPEC printed under the title."""

    title: str
    tables: tuple[GpecAsciiTable, ...] = ()
    legend: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.tables)

    @property
    def table(self) -> GpecAsciiTable:
        """The only block, for the sections that have exactly one."""
        if len(self.tables) != 1:
            raise ValueError(
                f"section {self.title!r} holds {len(self.tables)} blocks; use .tables"
            )
        return self.tables[0]


@dataclass
class GpecAsciiOutput:
    """One GPEC ASCII output file.

    ``kind`` is the leading token of the first line (``GPEC_SINGCOUP_MATRIX``
    and so on) and ``attrs`` the file-level scalars printed above the first
    section.  ``n_tor`` is *not* a field: these files do not record the
    toroidal mode number, so a caller that needs it takes it from the netCDF
    output or from the run directory rather than from the file name.
    """

    kind: str
    description: str = ""
    version: str = ""
    notes: tuple[str, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)
    sections: tuple[GpecAsciiSection, ...] = ()
    unparsed: tuple[tuple[int, str], ...] = ()
    path: Optional[str] = None

    @property
    def titles(self) -> tuple[str, ...]:
        return tuple(section.title for section in self.sections)

    def section(self, title: str) -> GpecAsciiSection:
        """The first section with this exact title."""
        for section in self.sections:
            if section.title == title:
                return section
        raise KeyError(f"no section {title!r}; this file has {list(self.titles)}")

    def sections_named(self, title: str) -> tuple[GpecAsciiSection, ...]:
        """Every section with this title -- ``gpec_response`` repeats "Eigenvectors"."""
        return tuple(section for section in self.sections if section.title == title)


def read_gpec_ascii(path: str | Path) -> GpecAsciiOutput:
    """Read any of GPEC's sectioned ASCII outputs."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    kind, description, version = "", "", ""
    notes: list[str] = []
    attrs: dict[str, Any] = {}
    sections: list[GpecAsciiSection] = []
    unparsed: list[tuple[int, str]] = []

    title: Optional[str] = None
    legend: list[str] = []
    tables: list[GpecAsciiTable] = []
    pending: dict[str, Any] = {}
    columns: Optional[tuple[str, ...]] = None
    rows: list[list[float]] = []

    def close_table() -> None:
        """Emit the open block, if any.  Scalars seen since the last block are
        that block's header, so they are cleared only when a block is emitted."""
        nonlocal columns, rows, pending
        if columns is None:
            return
        data = np.asarray(rows, dtype=float) if rows else np.empty((0, len(columns)))
        tables.append(GpecAsciiTable(columns=columns, data=data, header=dict(pending)))
        columns, rows, pending = None, [], {}

    def close_section() -> None:
        nonlocal title, legend, tables
        close_table()
        if title is not None or tables:
            sections.append(
                GpecAsciiSection(title=title or "", tables=tuple(tables), legend=tuple(legend))
            )
        title, legend, tables = None, [], []

    for number, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        if not line.strip():
            continue

        if _COMMENT.match(line):
            # Kept verbatim: a comment says nothing about structure, and it
            # can sit between a column header and its rows.
            (legend if (title is not None or tables or columns is not None) else notes).append(
                line.strip()
            )
            continue

        if number == 1:
            match = _KIND.match(line)
            if match:
                kind, description = match.group(1), match.group(2)
                continue

        match = _VERSION.match(line)
        if match and not version:
            version = match.group(1)
            continue

        if _is_data(line):
            if columns is None:
                # Data with no header above it: keep the line rather than lose it.
                unparsed.append((number, raw))
                continue
            values = [float(token) for token in line.split()]
            if len(values) != len(columns):
                unparsed.append((number, raw))
                continue
            rows.append(values)
            continue

        scalars = _scalars(line)
        if scalars is not None:
            if title is None and not tables and not sections:
                attrs.update(scalars)
            else:
                # A scalar line after data starts the next block of this section.
                close_table()
                pending.update(scalars)
            continue

        # Free text: a column header when the next line is data, otherwise a
        # section title (when we are between sections) or a legend line.
        # Index scan rather than a slice: these files run to millions of lines.
        following = None
        for index in range(number, len(lines)):
            candidate = lines[index]
            if candidate.strip() and not _COMMENT.match(candidate):
                following = candidate.rstrip()
                break
        # A column header is the line above data.  At end of file there is no
        # data to look at, and a run whose count is zero (``msing = 0`` in
        # ``gpec_vsingfld``) writes exactly that: a header and no rows.  Read
        # it as a header so the columns survive and the block is empty rather
        # than the line becoming a stray title.
        if following is not None and _is_data(following):
            close_table()
            columns = tuple(line.split())
            continue
        if following is None and columns is None and not tables:
            close_table()
            columns = tuple(line.split())
            continue

        if not version:
            notes.append(line.strip())
            continue

        if title is None and not tables:
            title = line.strip()
        elif columns is None and not rows and not tables:
            legend.append(line.strip())
        else:
            close_section()
            title = line.strip()

    close_section()
    return GpecAsciiOutput(
        kind=kind,
        description=description,
        version=version,
        notes=tuple(notes),
        attrs=attrs,
        sections=tuple(sections),
        unparsed=tuple(unparsed),
        path=str(path),
    )


def read_gpec_singcoup(path: str | Path) -> GpecAsciiOutput:
    """Read ``gpec_singcoup_matrix_n*.out`` or ``gpec_singcoup_svd_n*.out``."""
    return read_gpec_ascii(path)


def read_gpec_response(path: str | Path) -> GpecAsciiOutput:
    """Read ``gpec_response_n*.out`` or ``gpec_response_fun_n*.out``."""
    return read_gpec_ascii(path)
