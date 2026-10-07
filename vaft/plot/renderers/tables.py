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
    "magnetics_table_vacuum_benchmark",
    "magnetics_table_vacuum_benchmark_aggregate",
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
    """The selected equilibrium slice's global quantities, tabulated.

    Parameters
    ----------
    model : Table
        The table the adapter built.
    show : bool
        Print the table as well as returning it.

    Returns
    -------
    RenderedTable
        Text, Markdown and HTML renderings of the table.

    Interpretation
    --------------
    The global quantities of one reconstructed slice -- the slice
    :func:`equilibrium_overview` draws -- as a Quantity / Value / Unit table
    for reports and for comparing slices or shots number by number.  A quantity
    the slice does not store but that can be derived from it is shown with a
    note saying so; one that is neither is shown as not stored.

    Options
    -------
    ``time=`` and ``time_slice=`` choose the slice exactly as for
    :func:`equilibrium_overview`; ``units=`` sets the flux display unit.

    Limitations
    -----------
    The numbers are reconstruction outputs without their uncertainties, and a
    derived value is computed from the stored slice, not refitted.

    See Also
    --------
    equilibrium_table_fit_quality : how well the same slice fits its constraints.
    """
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
    """EFIT goodness of fit at one slice, by constraint family.

    Parameters
    ----------
    model : Table
        The table the adapter built.
    show : bool
        Print the table as well as returning it.

    Returns
    -------
    RenderedTable
        Text, Markdown and HTML renderings of the table.

    Interpretation
    --------------
    For one EFIT slice and each constraint family -- the magnetic sensor arrays
    and the scalar plasma-current and diamagnetic-flux constraints -- the table
    states whether it was fitted, how many channels were enabled, disabled or
    missing, its chi-square and share of the total, and the normalized
    residuals z = (measured - reconstructed) * weight / k: their RMS, mean bias
    and largest magnitude with the channel that has it.  The caption gives the
    reduced chi-square over EFIT's own count of degrees of freedom.  It answers
    which family dominates the fit, whether a family is systematically offset
    (flagged when the bias exceeds two standard errors), and which channel to
    inspect.

    Options
    -------
    ``time_slice=`` selects the stored slice.

    Limitations
    -----------
    Chi-square and z are measured in units of the uncertainties EFIT was given,
    so they judge the fit against those uncertainties, not against the truth: a
    small chi-square can mean generous uncertainties, and a family given small
    uncertainties dominates the total whatever its information content.  A good
    fit to external magnetics does not validate the internal profiles, which
    those constraints determine only weakly.

    See Also
    --------
    equilibrium_overview_fit_quality : the same metrics across slices, plotted.
    equilibrium_table_summary : the global quantities of the slice.
    """
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
    """The validation verdicts of one equilibrium slice, one row per registered check.

    Parameters
    ----------
    model : Table
        The table the adapter built.
    show : bool
        Print the text form.

    Returns
    -------
    RenderedTable
        Text, Markdown and HTML renderings of the table.

    Interpretation
    --------------
    For one equilibrium slice, every check of the validation registry in its
    registry order, one row each: the category (verification, diagnostic fit,
    physical validity, independent validation), the check, the measure and
    the number the status was decided on, the registry's criterion (a
    tolerance pair, or "rule" with the method as a note), the status, and the
    reason the check itself gave.  The status is printed as a word -- pass,
    warn, fail, indeterminate or not_available -- and carried as a
    ``data-status`` attribute in the HTML form; no colour is drawn.
    ``not_available`` means the evidence was never produced (the input lacks
    what the check reads) and ``indeterminate`` that it was produced but did
    not decide; neither is a pass, and both keep their reason in the note
    rather than being dropped.  The caption aggregates the rows the way
    :func:`vaft.validation.equilibrium.aggregate_status` does -- the worst of
    fail, warn and indeterminate wins, and pass needs every row to pass, so a
    mix of pass and not_available reads indeterminate -- and counts each
    status.  The table is read to find which check holds a slice back and
    why, and which checks the input could not even be put to.  It flags and
    never edits: the numbers are the stored equilibrium's and the statuses
    are the registered checks' own, with their tolerances; the view adds no
    physics and no threshold.

    Options
    -------
    ``time=`` and ``time_slice=`` select the slice the checks run on, by
    nearest stored time or by index; without either the representative slice
    of the overview is used, and the title says which and why.  ``title=``
    replaces the title.  The set of checks is the registry's and is not
    selectable here.

    Limitations
    -----------
    A row judges the slice against the check's tolerance in the units the
    check uses, so an all-pass table states that the registered checks found
    nothing, not that the reconstruction is right: the internal profiles are
    weakly constrained by external magnetics whatever the fit quality says,
    and a check that is not_available contributes no evidence.  Continuity
    is the one row decided over the whole IDS rather than the slice.  A value
    shown in the Value column was graded; a measure the verdict was not
    decided on is moved to the note so it does not read as a grade.  The
    verdicts come from the validation functions as they are: a tolerance
    that is a round figure rather than a qualified threshold is marked so in
    the registry, not here.

    See Also
    --------
    equilibrium_table_fit_quality : the goodness of fit per constraint family, in numbers.
    equilibrium_overview_verification : the verification checks drawn across slices.
    """
    return render_table(model, show=show)


#: What the vacuum benchmark needs: the PF currents on their time base (the
#: interval and the wall solve) and the passive loops' resistances (the wall
#: model it solves); the magnetics it scores vary by shot and are optional.
_BENCHMARK_IDS = ("pf_active", "pf_passive", "magnetics", "em_coupling", "wall")
_BENCHMARK_REQUIRED = (
    "pf_active.time",
    "pf_active.coil.{i}.current.data",
    "pf_passive.loop.{i}.resistance",
)
_BENCHMARK_OPTIONAL = (
    "magnetics.b_field_pol_probe.{i}.field.data",
    "magnetics.flux_loop.{i}.flux.data",
    "magnetics.ip.{i}.data",
    "em_coupling.mutual_passive_active",
)


@renderer(
    domain="magnetics",
    subject="magnetics",
    view="table",
    quantity="vacuum_benchmark",
    model=Table,
    description=(
        "The plasma-free vacuum benchmark of one shot (issue #190), one row per channel: "
        "family, kind, status (evaluated, flagged, excluded), eddy improvement, normalized "
        "residual, correlation, wall authority and the reason a channel is flagged or "
        "excluded; windows, solver history, coil drive and scored medians in the caption."
    ),
    ids=_BENCHMARK_IDS,
    required_paths=_BENCHMARK_REQUIRED,
    optional_paths=_BENCHMARK_OPTIONAL,
)
def magnetics_table_vacuum_benchmark(model: Table, *, show: bool = False) -> RenderedTable:
    """The vacuum benchmark's per-channel scores of one shot, one row per channel.

    Parameters
    ----------
    model : Table
        The table the adapter built.
    show : bool
        Print the text form.

    Returns
    -------
    RenderedTable
        Text, Markdown and HTML renderings of the table.

    Interpretation
    --------------
    The shot is run through :func:`vaft.validation.vacuum_benchmark.run_benchmark_case`:
    the passive wall is re-solved from the measured PF currents alone over the
    plasma-free stretch, and every usable B probe and flux loop is compared
    with the coil-only and coil+eddy forward response inside the validation
    window.  One row per scored channel, in the benchmark's order (grouped by
    family): the eddy improvement 1 - RMS(coil+eddy)/RMS(coil) -- what the wall
    model is worth on that channel -- the residual RMS as a fraction of the
    channel's swing, the correlation of measured with coil+eddy (near 1 with a
    large residual points at a gain, not the wall), and the wall authority, the
    eddy term's share of the reading in whose light a small or negative
    improvement is read.  Status is a fact, not a grade: ``flagged`` is a probe
    that contradicts its own array, scored but kept out of the medians as a
    sensor finding; ``excluded`` had too few usable samples.  The caption
    states the case type, the validation window inside the solver-input
    window, the solver history against the slowest wall time constant, the
    coil drive inside the window and the scored medians, so a row can be
    traced to the conditions it was measured under.  The table answers which
    channels and families the wall model reproduces and which it does not.

    Options
    -------
    ``per_family=`` limits the case to that many channels per family (the
    default scores every usable channel, which qualifying a machine model
    needs).  ``resistance_scale=`` multiplies every passive-loop resistance by
    one global factor -- the benchmark's resistance study, never a per-loop
    fit.  ``n_tau=`` sets how many slowest wall time constants of solver
    history must elapse before the validation window opens.  ``title=``
    replaces the title.

    Limitations
    -----------
    The benchmark states no acceptance threshold and no verdict, and neither
    does the table: #190 defers acceptance bounds until the VEST benchmark
    distribution has been inspected.  A score is measured against one shot's
    plasma-free stretch, so it says nothing about the model with plasma, and a
    low improvement where the wall authority is small is a rounding of the
    model's error rather than a finding about the wall.  A flagged probe is a
    sensor finding from its array neighbours, not a proof that the probe is
    faulty.  A case the coils did not drive inside its window is reported (the
    caption says so) and its improvements are ratios of noise.

    See Also
    --------
    magnetics_overview_vacuum_benchmark : the same scores drawn per channel.
    magnetics_table_vacuum_benchmark_aggregate : the benchmark across shots.
    magnetics_overview_vacuum : one shot's measured, coil and coil+eddy waveforms.
    """
    return render_table(model, show=show)


@renderer(
    domain="magnetics",
    subject="magnetics",
    view="table",
    quantity="vacuum_benchmark_aggregate",
    model=Table,
    description=(
        "The plasma-free vacuum benchmark across shots (issue #190): median eddy "
        "improvement, normalized residual and correlation and the worst channel by case, "
        "family, PF excitation and machine era; undriven cases, flagged channels and the "
        "summary in the caption."
    ),
    ids=_BENCHMARK_IDS,
    required_paths=_BENCHMARK_REQUIRED,
    optional_paths=_BENCHMARK_OPTIONAL,
)
def magnetics_table_vacuum_benchmark_aggregate(model: Table, *, show: bool = False) -> RenderedTable:
    """The vacuum benchmark across shots, by case, family, excitation and machine era.

    Parameters
    ----------
    model : Table
        The table the adapter built.
    show : bool
        Print the text form.

    Returns
    -------
    RenderedTable
        Text, Markdown and HTML renderings of the table.

    Interpretation
    --------------
    Every entry is run through :func:`vaft.validation.vacuum_benchmark.run_benchmark_case`
    and the cases are cross-tabulated by
    :func:`vaft.validation.vacuum_benchmark.aggregate_benchmark`.  One row per
    group of each axis -- case (the entry's label), magnetic family, PF
    excitation (the coils that carried current) and machine era (the VEST era
    the pulse falls in, ``unknown`` without a pulse) -- with the number of cases
    and channels in it, the median eddy improvement, normalized residual and
    correlation, and the channel with the lowest improvement.  Which axis a poor
    median concentrates on is the diagnosis the cross-tabs exist for: one
    channel across excitations points at probe calibration or geometry, many
    channels whenever one PF coil dominates at that coil, a change across eras
    at the static model's provenance, one shot among consistent neighbours at
    that shot's acquisition.  An entry that cannot supply a plasma-free case
    -- it lacks the PF or passive-loop circuits the case is solved from, or
    the benchmark refuses it -- is a case row with the reason, never dropped.  The caption lists
    the cases the coils did not drive, the flagged probes per case and the
    summary medians over driven, unflagged channel rows.

    Options
    -------
    ``per_family=``, ``resistance_scale=`` and ``n_tau=`` are passed to every
    case as in :func:`magnetics_table_vacuum_benchmark`; ``label=`` names the
    entries, and so the case rows.  ``title=`` replaces the title.

    Limitations
    -----------
    No threshold and no verdict, as in the benchmark itself.  The cross-tab
    medians count every evaluated channel, the flagged and undriven ones too
    (the notes say so); only the caption's summary leaves them out.  A median
    over a group of one case is that case, not a distribution, and the
    packaged samples are few: a cross-tab over them shows the structure of the
    analysis, not the VEST benchmark distribution.

    See Also
    --------
    magnetics_table_vacuum_benchmark : one case, channel by channel.
    magnetics_overview_vacuum_benchmark : one case's scores drawn per channel.
    """
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

