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
section's legend, and a file with no version line at all (the same family)
reads the same way.

Three rules decide what a line means, and each is forced by the files:

- **A line carrying ``=`` is never a section title.**  GPEC writes titles as
  bare prose, and ``gpec_control``'s ``vacuum energy =  3.28697546E+000`` was
  being read as one.
- **Scalars** are either several ``key = value`` pairs accounting for a whole
  line (``jac_out = boozer     tmag_out = 1``, ``q = 2.000  psi = 5.93E-1``)
  or a single pair with a numeric value, whose key may contain spaces
  (``vacuum energy = ...``, ``sweet-spot = 0``).  A single pair with a
  non-numeric value is prose: that is the shape of GPEC's glossaries
  (``Lambda = Inductance``, ``P = Permeability``).  The cost is that a real
  ``jac_type = hamada`` reads as prose too -- it is kept verbatim in the
  section's legend, and no guess is made either way.
- **Scalars printed before the first block** are the file's, and when the very
  next line is that block's column header they are recorded as the block's as
  well: ``gpec_response_fun`` prints ``isol = 1  n = 1`` there, exactly where
  ``gpec_singfld`` prints ``msing = 4`` for the whole file.  Which GPEC means
  is not knowable from the text, so both readings are kept rather than one
  being guessed at.

Numbers are read as Fortran writes them, including ``1.0D+00`` and a denormal
that has lost its exponent marker (``5.84973725-321``, which occurs in a real
run).  Rejecting such a row would promote it to a column header and split its
section in two.
"""

from __future__ import annotations

import re
from array import array
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


#: Fortran writes exponents GPEC's own output sometimes cannot round-trip:
#: ``1.0D+00``, and denormals with the ``E`` dropped (``5.84973725-321``,
#: which really occurs in ``gpec_response``).  ``float`` rejects both.
_FORTRAN_EXPONENT = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+))([+-]\d+)$")


def _float(token: str) -> float:
    """``float`` for a Fortran-written number; raises ``ValueError`` like it."""
    try:
        return float(token)
    except ValueError:
        pass
    if "D" in token or "d" in token:
        return float(token.replace("D", "E").replace("d", "e"))
    match = _FORTRAN_EXPONENT.match(token)
    if match:
        return float(f"{match.group(1)}E{match.group(2)}")
    raise ValueError(f"could not read {token!r} as a number")


def _as_scalar(text: str) -> Any:
    """``"1"`` -> 1, ``"5.9E-1"`` -> 0.59, anything else stays a string."""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return _float(text)
    except ValueError:
        return text


def _scalars(line: str) -> Optional[dict[str, Any]]:
    """Scalars from a ``key = value`` line, or ``None`` when it is prose.

    Two shapes count, and the distinction is forced on us by the files:

    * **Several pairs on one line** that account for the whole line --
      ``jac_out = boozer     tmag_out = 1``, ``q = 2.000  psi = 5.93E-1``.
    * **One pair with a numeric value**, where the key may contain spaces:
      ``vacuum energy =  3.28697546E+000``, ``sweet-spot = 0``.

    A single pair with a non-numeric value is *not* a scalar, because that is
    exactly the shape of GPEC's glossaries (``Lambda = Inductance``,
    ``P = Permeability``, ``rho = Reluctance``).  Those stay prose.  The cost
    is that a genuine ``jac_type = hamada`` is also read as prose; it is kept
    verbatim in the section's legend, and no guess is made either way.
    """
    if line.count("=") == 1:
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and len(value.split()) == 1 and not isinstance(_as_scalar(value), str):
            return {key: _as_scalar(value)}
        return None
    pairs = list(_PAIR.finditer(line))
    if not pairs:
        return None
    if _PAIR.sub("", line).strip():
        return None
    return {match.group(1): _as_scalar(match.group(2)) for match in pairs}


def _is_data(line: str) -> bool:
    tokens = line.split()
    if not tokens:
        return False
    for token in tokens:
        try:
            _float(token)
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

    def column(self, name: str, occurrence: Optional[int] = None) -> np.ndarray:
        """One column by its file name.

        A header may repeat a name -- ``gpec_singfld``'s overlap block writes
        ``overlap(%)`` once per coupling matrix -- so an ambiguous lookup is
        refused rather than silently answered with the first one.  Pass
        ``occurrence`` to choose among them, in file order.
        """
        indices = [index for index, column in enumerate(self.columns) if column == name]
        if not indices:
            raise KeyError(f"no column {name!r}; this block has {list(self.columns)}")
        if len(indices) > 1 and occurrence is None:
            raise KeyError(
                f"column {name!r} appears {len(indices)} times in this block (at {indices}); "
                "pass occurrence= to choose one"
            )
        try:
            return self.data[:, indices[occurrence or 0]]
        except IndexError:
            raise KeyError(
                f"column {name!r} appears {len(indices)} times; no occurrence {occurrence}"
            ) from None

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
        """Columns as a name -> array mapping, in file order.

        Refuses a block whose header repeats a name: a mapping cannot hold
        both, and dropping one silently is how four of ``gpec_singfld``'s
        sixteen columns would disappear.  Use :attr:`columns` and
        :attr:`data` for those.
        """
        duplicated = sorted({name for name in self.columns if self.columns.count(name) > 1})
        if duplicated:
            raise ValueError(
                f"columns {duplicated} appear more than once in this block; "
                "read it through .columns and .data instead"
            )
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
    section.

    ``n_tor`` is deliberately *not* a field.  Some of these files do record
    the toroidal mode number and most do not: ``gpec_response_fun`` prints
    ``isol = 1  n = 1`` on every block and ``gpec_brzphi_n123`` writes
    ``n = 123`` at file level, while ``gpec_singcoup_matrix`` and
    ``gpec_response`` write none.  Rather than sometimes having it and
    sometimes inventing it from the file name, this container surfaces
    whatever the file said -- in ``attrs`` or in a block ``header`` -- and
    leaves the run's mode number to the netCDF output, which always carries
    it as a global attribute.
    """

    kind: str
    description: str = ""
    version: str = ""
    notes: tuple[str, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)
    sections: tuple[GpecAsciiSection, ...] = ()
    unparsed: tuple[tuple[int, str], ...] = ()
    path: str = ""

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
    """Read any of GPEC's sectioned ASCII outputs.

    Streams the file: these outputs reach hundreds of megabytes, and rows are
    accumulated in a typed buffer rather than as Python floats.
    """
    path = Path(path)
    if path.is_dir():
        raise IsADirectoryError(
            f"{path} is a directory; read_gpec_ascii takes one .out file "
            "(read_gpec_netcdf is the one that takes a run directory)"
        )

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
    values = array("d")
    row_count = 0

    def close_table() -> None:
        nonlocal columns, values, row_count, pending
        if columns is None:
            return
        data = np.frombuffer(values, dtype=float).reshape(row_count, len(columns)).copy()
        tables.append(GpecAsciiTable(columns=columns, data=data, header=dict(pending)))
        columns, values, row_count, pending = None, array("d"), 0, {}

    def close_section() -> None:
        nonlocal title, legend, tables, pending
        close_table()
        if title is not None or tables or legend:
            sections.append(
                GpecAsciiSection(title=title or "", tables=tuple(tables), legend=tuple(legend))
            )
        # Scalars that never reached a block belong to the section that
        # printed them, not to the next one.
        title, legend, tables, pending = None, [], [], {}

    def in_section() -> bool:
        return title is not None or bool(tables) or columns is not None or bool(legend)

    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        stream = enumerate(handle, start=1)
        buffer: list[tuple[int, str]] = []

        def peek_meaningful(ahead: int = 1) -> Optional[str]:
            """The n-th line that is neither blank nor a comment, unconsumed."""
            seen = 0
            for _, candidate in buffer:
                if candidate.strip() and not _COMMENT.match(candidate):
                    seen += 1
                    if seen == ahead:
                        return candidate.rstrip()
            for item in stream:
                buffer.append(item)
                if item[1].strip() and not _COMMENT.match(item[1]):
                    seen += 1
                    if seen == ahead:
                        return item[1].rstrip()
            return None

        def heads_a_block() -> bool:
            """Whether the next meaningful line is a column header with rows under it."""
            first, second = peek_meaningful(1), peek_meaningful(2)
            return (
                first is not None
                and not _is_data(first)
                and _scalars(first) is None
                and second is not None
                and _is_data(second)
            )

        while True:
            if buffer:
                number, raw = buffer.pop(0)
            else:
                try:
                    number, raw = next(stream)
                except StopIteration:
                    break
            line = raw.rstrip()
            if not line.strip():
                continue

            if number == 1:
                match = _KIND.match(line)
                if match:
                    kind, description = match.group(1), match.group(2)
                    continue

            if _COMMENT.match(line):
                # A comment says nothing about structure and can sit between a
                # column header and its rows; keep it verbatim.
                (legend if in_section() else notes).append(line.strip())
                continue

            match = _VERSION.match(line)
            if match and not version:
                version = match.group(1)
                continue

            if _is_data(line):
                if columns is None:
                    unparsed.append((number, raw.rstrip("\n")))
                    continue
                row = line.split()
                if len(row) != len(columns):
                    # Never padded or truncated into a shape it does not have.
                    unparsed.append((number, raw.rstrip("\n")))
                    continue
                values.extend(_float(token) for token in row)
                row_count += 1
                continue

            if "=" in line:
                # A line carrying '=' is a scalar or a glossary entry, never a
                # section title: GPEC writes titles as bare prose.
                scalars = _scalars(line)
                if scalars is None:
                    (legend if in_section() else notes).append(line.strip())
                elif not in_section() and not sections:
                    # Printed before any section began, so these are the file's
                    # scalars.  When the very next thing is a column header they
                    # are *also* that block's -- ``gpec_response_fun`` prints
                    # ``isol = 1  n = 1`` there, exactly where ``gpec_singfld``
                    # prints ``msing = 4`` for the whole file.  Which one GPEC
                    # means is not knowable from the text, so both readings are
                    # recorded rather than one being guessed at.
                    attrs.update(scalars)
                    if heads_a_block():
                        pending.update(scalars)
                else:
                    close_table()
                    pending.update(scalars)
                continue

            following = peek_meaningful()
            # A column header is the line above data.  At end of file there is
            # no data to look at, and a run whose count is zero writes exactly
            # that: a header and no rows.
            if (following is not None and _is_data(following)) or (
                following is None and columns is None and not tables
            ):
                close_table()
                columns = tuple(line.split())
                continue

            if not version:
                notes.append(line.strip())
                continue

            if columns is not None or tables:
                # A title arriving while a block is open ends the section that
                # block belongs to; otherwise an untitled leading block would
                # be filed under the next section's name.
                close_section()
            if title is None:
                title = line.strip()
            else:
                legend.append(line.strip())

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


def _read_kind(path: str | Path, prefix: str) -> GpecAsciiOutput:
    output = read_gpec_ascii(path)
    if not output.kind.startswith(prefix):
        raise ValueError(
            f"{Path(path).name} is a {output.kind or 'kindless'} file, not {prefix}*; "
            "use read_gpec_ascii to read any of them"
        )
    return output


def read_gpec_singcoup(path: str | Path) -> GpecAsciiOutput:
    """Read ``gpec_singcoup_matrix_n*.out`` or ``gpec_singcoup_svd_n*.out``.

    Refuses a file whose kind line says otherwise, so passing the wrong path
    fails here rather than later as a missing section title.
    """
    return _read_kind(path, "GPEC_SINGCOUP")


def read_gpec_response(path: str | Path) -> GpecAsciiOutput:
    """Read ``gpec_response_n*.out`` or ``gpec_response_fun_n*.out``.

    Refuses a file whose kind line says otherwise (see
    :func:`read_gpec_singcoup`).
    """
    return _read_kind(path, "GPEC_RESPONSE")
