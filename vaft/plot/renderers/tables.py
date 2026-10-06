"""Renderers of the non-graphical views: tables and text summaries (issue #1180).

A ``table`` or ``text`` view is a scientific presentation like any figure --
the adapter selects and reduces the data into a :class:`~vaft.plot.models.Table`
or :class:`~vaft.plot.models.TextSummary` -- but what presents it is text, not
Matplotlib.  :func:`render_table` and :func:`render_text_summary` return a
:class:`RenderedTable` / :class:`RenderedTextSummary`: it prints as fixed-width
text (``str()``), shows as an HTML table in a notebook (``_repr_html_``), and
exports deterministically with ``.text()``, ``.markdown()`` and ``.html()``
(``.save(path)`` picks one by extension).

The renderer formats and nothing else.  Numbers are written here, through the
display policy of :mod:`vaft.plot.display` -- a current stored in ampere is
shown in kA, a beta by its convention -- so the model keeps the stored value
and unit.  A status is printed as a word; no colour is drawn, so no colour is
chosen.  This module imports no Matplotlib.
"""

from __future__ import annotations

import html as _html
from pathlib import Path
from typing import Any

from ..models import Table, TableCell, TextItem, TextSummary
from ..registry import renderer

__all__ = [
    "RenderedTable",
    "RenderedTextSummary",
    "TextView",
    "equilibrium_table_fit_quality",
    "equilibrium_table_summary",
    "equilibrium_table_validation",
    "equilibrium_text_summary",
    "format_quantity",
    "render_table",
    "render_text_summary",
]

#: The default number format of a displayed value: four significant figures,
#: what the slice summary panel of ``equilibrium_overview`` writes too.
DEFAULT_FORMAT = ".4g"

#: How a status reads in the text forms.
STATUS_LABELS = {
    "": "", "pass": "PASS", "warn": "WARN", "fail": "FAIL", "info": "INFO",
    "indeterminate": "INDETERMINATE", "not_available": "NOT_AVAILABLE",
}


# ---------------------------------------------------------------------------
# One value
# ---------------------------------------------------------------------------


def _display(value: float, item: TableCell | TextItem) -> tuple[float, str]:
    """``value`` scaled into the display unit the policy picks, and that unit."""
    if not (item.unit or item.quantity or item.display_unit):
        return value, ""
    from ..display import resolve_display

    try:
        spec = resolve_display(
            item.unit,
            unit=item.display_unit,
            subject=item.subject or None,
            quantity=item.quantity or None,
        )
    except ValueError:
        # A unit the policy has no conversion for is shown as stored.
        return value, item.unit
    return value * spec.scale, spec.unit


def format_quantity(item: TableCell | TextItem) -> tuple[str | None, str]:
    """``(text, unit)`` of one structured value, or ``(None, "")`` when it is missing.

    A word is shown as written and a bool as ``yes``/``no``; a number is
    converted to its display unit and written with the item's ``format``
    (:data:`DEFAULT_FORMAT` when empty; an integer with no conversion keeps
    its digits).  Deterministic: the same item always reads the same.
    """
    if item.missing:
        return None, ""
    value = item.value
    if isinstance(value, bool):
        return ("yes" if value else "no"), ""
    if isinstance(value, str):
        return value, item.unit
    shown, unit = _display(value, item)
    if item.format:
        spec = item.format
    elif isinstance(shown, int):
        spec = "d"
    else:
        spec = DEFAULT_FORMAT
    try:
        text = format(shown, spec)
    except (TypeError, ValueError):
        # An integer format for a value the display scaled to a float.
        text = format(float(shown), DEFAULT_FORMAT)
    return text, unit


# ---------------------------------------------------------------------------
# The rendered objects
# ---------------------------------------------------------------------------


class TextView:
    """A rendered non-graphical view: text in a terminal, HTML in a notebook.

    ``model`` is the view model it presents.  ``text()``, ``markdown()`` and
    ``html()`` are deterministic; ``save(path)`` writes the one the extension
    names (``.txt``, ``.md``, ``.html``).
    """

    __slots__ = ("model",)

    def __init__(self, model: Any) -> None:
        self.model = model

    def text(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def markdown(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def html(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def __str__(self) -> str:
        return self.text()

    def __repr__(self) -> str:
        return self.text()

    def _repr_html_(self) -> str:
        return self.html()

    def _repr_markdown_(self) -> str:
        return self.markdown()

    def show(self) -> None:
        """Print the fixed-width text form."""
        print(self.text())

    def save(self, path: Any) -> str:
        """Write the form ``path``'s extension names; returns ``path`` as a string.

        ``.txt``/``.text`` writes :meth:`text`, ``.md``/``.markdown`` writes
        :meth:`markdown`, ``.html``/``.htm`` writes :meth:`html`; any other
        extension is refused -- a table view has no image to save.
        """
        target = Path(path)
        suffix = target.suffix.lower()
        if suffix in (".txt", ".text"):
            content = self.text()
        elif suffix in (".md", ".markdown"):
            content = self.markdown()
        elif suffix in (".html", ".htm"):
            content = self.html()
        else:
            raise ValueError(
                f"a {type(self.model).__name__} view is text; save it as .txt, .md or .html, "
                f"not {target.name!r}"
            )
        target.write_text(content + "\n", encoding="utf-8")
        return str(path)


# -- tables -----------------------------------------------------------------


def _cell_texts(model: Table) -> tuple[list[str], list[list[str]], list[str], list[str]]:
    """``(headers, rows, aligns, notes)`` with every cell written out.

    A ``value`` column with ``units="column"`` gains a ``Unit`` column beside
    it; ``units="header"`` moves a column's single display unit into its
    header (``Value [kA]``), falling back to ``inline`` when the cells do not
    agree.  Cell notes are marked in row-major order, each text once, by a
    symbol right after the cell's value (:func:`note_marker`) -- never in a
    unit column, where it would read as a unit.
    """
    note_numbers: dict[str, int] = {}
    formatted = [[format_quantity(cell) for cell in row] for row in model.rows]
    headers: list[str] = []
    aligns: list[str] = []
    plan: list[tuple[int, str]] = []  # (column index, "cell" | "unit")
    header_units: dict[int, str] = {}
    for index, column in enumerate(model.columns):
        if column.kind == "value" and column.units == "header":
            units = {formatted[r][index][1] for r in range(len(model.rows)) if formatted[r][index][0] is not None}
            if len(units) == 1:
                header_units[index] = units.pop()
        name = column.name
        if header_units.get(index):
            name = f"{name} [{header_units[index]}]"
        headers.append(name)
        default_align = "right" if column.kind == "value" else "left"
        aligns.append(column.align or default_align)
        plan.append((index, "cell"))
        if column.kind == "value" and column.units == "column":
            headers.append("Unit")
            aligns.append("left")
            plan.append((index, "unit"))
    rows: list[list[str]] = []
    for r, row in enumerate(model.rows):
        texts: list[str] = []
        for index, part in plan:
            cell = row[index]
            column = model.columns[index]
            text, unit = formatted[r][index]
            if part == "unit":
                texts.append(unit if text is not None else "")
                continue
            if column.kind == "status":
                shown = STATUS_LABELS[cell.status]
            elif text is None:
                shown = model.missing
            elif column.kind == "value" and column.units == "inline" or (
                column.kind == "value" and column.units == "header" and index not in header_units
            ):
                shown = f"{text} {unit}".rstrip()
            elif column.kind == "text":
                shown = f"{text} {unit}".rstrip() if unit else text
            else:
                shown = text
            if cell.note:
                shown = f"{shown}{note_marker(note_numbers, cell.note)}"
            texts.append(shown)
        rows.append(texts)
    notes = [f"{_marker(number)} {note}" for note, number in note_numbers.items()]
    return headers, rows, aligns, notes + list(model.notes)


def _pad(text: str, width: int, align: str) -> str:
    if align == "right":
        return text.rjust(width)
    if align == "center":
        return text.center(width)
    return text.ljust(width)


class RenderedTable(TextView):
    """A :class:`~vaft.plot.models.Table`, presented."""

    __slots__ = ()

    def text(self) -> str:
        """Fixed-width columns, a rule under the header, the caption and notes after."""
        headers, rows, aligns, notes = _cell_texts(self.model)
        headers = [_one_line(header) for header in headers]
        rows = [[_one_line(cell) for cell in row] for row in rows]
        widths = [max([len(header)] + [len(row[i]) for row in rows]) for i, header in enumerate(headers)]
        lines: list[str] = []
        if self.model.title:
            lines += [self.model.title, ""]
        lines.append("  ".join(_pad(h, w, a) for h, w, a in zip(headers, widths, aligns)).rstrip())
        lines.append("  ".join("-" * w for w in widths))
        for row in rows:
            lines.append("  ".join(_pad(t, w, a) for t, w, a in zip(row, widths, aligns)).rstrip())
        if self.model.caption:
            lines += ["", self.model.caption]
        if notes:
            lines += [""] + notes
        return "\n".join(lines)

    def markdown(self) -> str:
        """A GitHub-flavoured Markdown table, alignment in the rule row."""
        headers, rows, aligns, notes = _cell_texts(self.model)
        rule = {"left": "---", "right": "---:", "center": ":---:"}

        def line(cells: list[str]) -> str:
            return "| " + " | ".join(_markdown_escape(cell) for cell in cells) + " |"

        lines: list[str] = []
        if self.model.title:
            lines += [f"**{_markdown_escape(self.model.title)}**", ""]
        lines.append(line(headers))
        lines.append("|" + "|".join(rule[a] for a in aligns) + "|")
        lines += [line(row) for row in rows]
        if self.model.caption:
            lines += ["", _markdown_escape(self.model.caption)]
        if notes:
            lines += [""] + [f"{_markdown_escape(note)}  " for note in notes[:-1]] + [_markdown_escape(notes[-1])]
        return "\n".join(lines)

    def html(self) -> str:
        """An HTML ``<table>``; a status cell carries ``data-status``, never a colour."""
        headers, rows, aligns, notes = _cell_texts(self.model)
        model = self.model
        # The written-out position of each status column (a Unit column added
        # beside a value column shifts the ones after it).
        status_positions: dict[int, int] = {}
        position = 0
        for index, column in enumerate(model.columns):
            if column.kind == "status":
                status_positions[position] = index
            position += 2 if column.kind == "value" and column.units == "column" else 1
        esc = _html.escape
        parts = ['<div class="vaft-table-view">', '<table class="vaft-table">']
        if model.title:
            parts.append(f"<caption>{esc(model.title)}</caption>")
        parts.append("<thead><tr>" + "".join(
            f'<th style="text-align: {a}">{esc(h)}</th>' for h, a in zip(headers, aligns)
        ) + "</tr></thead>")
        parts.append("<tbody>")
        for r, row in enumerate(rows):
            cells = []
            for i, (text, align) in enumerate(zip(row, aligns)):
                if i in status_positions:
                    status = model.rows[r][status_positions[i]].status
                    cells.append(
                        f'<td class="vaft-status" data-status="{esc(status)}" '
                        f'style="text-align: {align}">{esc(text)}</td>'
                    )
                else:
                    cells.append(f'<td style="text-align: {align}">{esc(text)}</td>')
            parts.append("<tr>" + "".join(cells) + "</tr>")
        parts.append("</tbody>")
        parts.append("</table>")
        if model.caption:
            parts.append(f'<p class="vaft-table-caption">{esc(model.caption)}</p>')
        if notes:
            parts.append('<p class="vaft-table-notes">' + "<br>".join(esc(n) for n in notes) + "</p>")
        parts.append("</div>")
        return "\n".join(parts)


def _markdown_escape(text: str) -> str:
    """``text`` safe inside a Markdown table cell or line.

    Backslash and ``|`` are escaped, ``&``, ``<`` and ``>`` become entities
    (so no cell is read as HTML), and a line break becomes ``<br>``.
    """
    text = text.replace("\\", "\\\\").replace("|", "\\|")
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text.replace("\r\n", "\n").replace("\n", "<br>")


def _one_line(text: str) -> str:
    """``text`` on one line, for the fixed-width form whose columns must align."""
    return " ".join(text.replace("\r\n", "\n").split("\n"))


#: The note markers, in order; past them a note is marked ``[n]``.
_MARKERS = ("*", "†", "‡", "§", "¶")


def _marker(number: int) -> str:
    return _MARKERS[number - 1] if number <= len(_MARKERS) else f"[{number}]"


def note_marker(numbers: dict[str, int], note: str) -> str:
    """The marker of ``note``, numbering it on first sight."""
    return _marker(numbers.setdefault(note, len(numbers) + 1))


# -- text summaries -----------------------------------------------------------


def _item_text(item: TextItem, missing: str, notes: dict[str, int]) -> str:
    """One item's value as it reads after its label; its note becomes a numbered marker."""
    if item.is_statement:
        text = "" if item.missing else str(item.value)
    else:
        value, unit = format_quantity(item)
        text = missing if value is None else f"{value} {unit}".rstrip()
    if item.status:
        text = f"{text} [{STATUS_LABELS[item.status]}]"
    if item.note:
        text = f"{text}{note_marker(notes, item.note)}"
    return text


def _summary_texts(model: TextSummary) -> tuple[list[tuple[str, list[tuple[TextItem, str]]]], list[str]]:
    """Every section's items written out, and the numbered notes they refer to."""
    notes: dict[str, int] = {}
    sections = [
        (section.title, [(item, _item_text(item, model.missing, notes)) for item in section.items])
        for section in model.sections
    ]
    return sections, [f"{_marker(number)} {note}" for note, number in notes.items()]


class RenderedTextSummary(TextView):
    """A :class:`~vaft.plot.models.TextSummary`, presented."""

    __slots__ = ()

    def text(self) -> str:
        """The title, then each section as a heading over aligned ``label  value`` lines."""
        model = self.model
        sections, notes = _summary_texts(model)
        lines: list[str] = []
        if model.title:
            lines += [model.title, "=" * len(model.title)]
        for title, items in sections:
            if lines:
                lines.append("")
            lines += [title, "-" * len(title)]
            width = max((len(item.label) for item, _ in items if not item.is_statement), default=0)
            for item, value in items:
                value = _one_line(value)
                if item.is_statement:
                    lines.append(f"  - {value}")
                else:
                    lines.append(f"  {_one_line(item.label).ljust(width)}  {value}".rstrip())
        if model.caption:
            lines += ["", model.caption]
        if notes:
            lines += [""] + notes
        return "\n".join(lines)

    def markdown(self) -> str:
        """A heading per section and one list item per line."""
        model = self.model
        sections, notes = _summary_texts(model)
        lines: list[str] = []
        if model.title:
            lines += [f"### {_markdown_escape(model.title)}"]
        for title, items in sections:
            if lines:
                lines.append("")
            lines += [f"#### {_markdown_escape(title)}", ""]
            for item, value in items:
                if item.is_statement:
                    lines.append(f"- {_markdown_escape(value)}")
                else:
                    lines.append(f"- **{_markdown_escape(item.label)}:** {_markdown_escape(value)}")
        if model.caption:
            lines += ["", _markdown_escape(model.caption)]
        if notes:
            lines += [""] + [f"{_markdown_escape(note)}  " for note in notes[:-1]] + [_markdown_escape(notes[-1])]
        return "\n".join(lines)

    def html(self) -> str:
        """A heading per section over a two-column table of label and value."""
        model = self.model
        sections, notes = _summary_texts(model)
        esc = _html.escape
        parts = ['<div class="vaft-text-summary">']
        if model.title:
            parts.append(f"<h3>{esc(model.title)}</h3>")
        for title, items in sections:
            parts.append(f"<h4>{esc(title)}</h4>")
            parts.append('<table class="vaft-text-section"><tbody>')
            for item, value in items:
                status = f' data-status="{esc(item.status)}"' if item.status else ""
                if item.is_statement:
                    parts.append(f'<tr><td colspan="2"{status}>{esc(value)}</td></tr>')
                else:
                    parts.append(
                        f'<tr><th style="text-align: left">{esc(item.label)}</th>'
                        f'<td style="text-align: left"{status}>{esc(value)}</td></tr>'
                    )
            parts.append("</tbody></table>")
        if model.caption:
            parts.append(f"<p>{esc(model.caption)}</p>")
        if notes:
            parts.append('<p class="vaft-text-notes">' + "<br>".join(esc(n) for n in notes) + "</p>")
        parts.append("</div>")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Low-level renderers
# ---------------------------------------------------------------------------


def render_table(model: Table, *, show: bool = False) -> RenderedTable:
    """Present a :class:`~vaft.plot.models.Table`; ``show=True`` prints it.

    Returns a :class:`RenderedTable`: text in a terminal, an HTML table in a
    notebook, ``.text()``/``.markdown()``/``.html()`` for export.  It draws no
    figure and takes no Matplotlib keyword.
    """
    if not isinstance(model, Table):
        raise TypeError(
            f"expected a vaft.plot.models.Table; got {type(model).__name__}. "
            "Adapters such as vaft.omas.plot_* build the model from data objects."
        )
    rendered = RenderedTable(model)
    if show:
        rendered.show()
    return rendered


def render_text_summary(model: TextSummary, *, show: bool = False) -> RenderedTextSummary:
    """Present a :class:`~vaft.plot.models.TextSummary`; ``show=True`` prints it."""
    if not isinstance(model, TextSummary):
        raise TypeError(
            f"expected a vaft.plot.models.TextSummary; got {type(model).__name__}. "
            "Adapters such as vaft.omas.plot_* build the model from data objects."
        )
    rendered = RenderedTextSummary(model)
    if show:
        rendered.show()
    return rendered


# ---------------------------------------------------------------------------
# Canonical table and text views
# ---------------------------------------------------------------------------

_SLICE_REQUIRED = ("equilibrium.time_slice.{i}.profiles_2d.0.psi",)
_SLICE_OPTIONAL = (
    "equilibrium.time_slice.{i}.global_quantities.ip",
    "equilibrium.time_slice.{i}.global_quantities.beta_pol",
    "equilibrium.time_slice.{i}.global_quantities.q_95",
    "equilibrium.time_slice.{i}.boundary.minor_radius",
)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="table",
    quantity="summary",
    model=Table,
    description=(
        "One equilibrium slice's global quantities as a table -- the representative "
        "slice, or the stored slice time= snaps to -- in the display units of the "
        "slice overview."
    ),
    ids=("equilibrium", "wall"),
    required_paths=_SLICE_REQUIRED,
    optional_paths=_SLICE_OPTIONAL,
)
def equilibrium_table_summary(model: Table, *, show: bool = False) -> RenderedTable:
    """The selected equilibrium slice's global quantities, tabulated."""
    return render_table(model, show=show)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="table",
    quantity="fit_quality",
    model=Table,
    description=(
        "EFIT goodness of fit at one slice, per constraint family: fit role, channel "
        "states, share of the total chi-square and residuals normalized by the "
        "uncertainty EFIT was given; the reduced chi-square in the caption."
    ),
    ids=("equilibrium",),
    required_paths=("equilibrium.time_slice.{i}.constraints.bpol_probe.{j}.chi_squared",),
    optional_paths=(
        "equilibrium.time_slice.{i}.constraints.flux_loop.{j}.chi_squared",
        "equilibrium.time_slice.{i}.constraints.ip.chi_squared",
        "equilibrium.time_slice.{i}.constraints.diamagnetic_flux.chi_squared",
    ),
)
def equilibrium_table_fit_quality(model: Table, *, show: bool = False) -> RenderedTable:
    """EFIT goodness of fit at one slice, by constraint family."""
    return render_table(model, show=show)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="table",
    quantity="validation",
    model=Table,
    description=(
        "Every registered validation check's verdict at one equilibrium slice -- "
        "verification, diagnostic fit, physical validity, independent validation: "
        "the number each status was decided on, the registry criterion, the status "
        "(NOT_AVAILABLE and INDETERMINATE kept as such) and its reason; the "
        "aggregate status and the count per status in the caption."
    ),
    ids=("equilibrium", "magnetics", "core_profiles", "thomson_scattering"),
    required_paths=_SLICE_REQUIRED,
    optional_paths=(
        "equilibrium.time_slice.{i}.profiles_1d.q",
        "equilibrium.time_slice.{i}.profiles_1d.pressure",
        "equilibrium.time_slice.{i}.constraints.bpol_probe.{j}.chi_squared",
        "magnetics.diamagnetic_flux.{j}.data",
    ),
)
def equilibrium_table_validation(model: Table, *, show: bool = False) -> RenderedTable:
    """The validation verdicts of one equilibrium slice, one row per registered check."""
    return render_table(model, show=show)


@renderer(
    domain="equilibrium",
    subject="equilibrium",
    view="text",
    quantity="summary",
    model=TextSummary,
    description=(
        "One equilibrium slice as a text summary: which slice and why, its global "
        "quantities, and its boundary shape."
    ),
    ids=("equilibrium", "wall"),
    required_paths=_SLICE_REQUIRED,
    optional_paths=_SLICE_OPTIONAL,
)
def equilibrium_text_summary(model: TextSummary, *, show: bool = False) -> RenderedTextSummary:
    """The selected equilibrium slice, summarised in text."""
    return render_text_summary(model, show=show)

