"""Several canonical plots arranged into one figure (issue #1467).

A *composition* says which canonical plots a figure holds and where: a grid
of ``rows x cols`` cells, each :class:`FigureCell` naming one plot, its
options and the region it covers (``rowspan``/``colspan``), plus the axes that
move together (:class:`AxisLink`, or ``share_x``/``share_y`` for the common
cases).  It holds no data and no drawing objects, so the same composition is
drawn from an ODS, a native IMAS entry or several shots, by Matplotlib or by
Plotly, and round-trips through :meth:`FigureComposition.to_dict`.

This is a different question from a plot's own ``layout=`` (#260): that
spreads the series of *one* plot over axes; a composition places *several*
independent plots.  A cell therefore holds one panel -- a plot that draws a
grid of its own (an overview, ``layout="subplots"``) is refused by name.

Drawing goes through the existing pieces: every cell is built by the plot's
own recipe (:func:`vaft.plot.backend.recipes.build_model`), the cells become
one :class:`~vaft.plot.models.Panels` with spans and axis links, and the
panels renderer of the chosen backend draws it -- a Matplotlib ``Figure``
over a ``GridSpec``, or one Plotly figure from ``make_subplots``.  Figure
presentation (``format=``, ``theme=``, ``figsize=``) is the Matplotlib
renderer's, as for any plot.

The namespace entry points take the data: :func:`vaft.omas.compose` and
:func:`vaft.imas.compose`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

__all__ = [
    "AxisLink",
    "FigureCell",
    "FigureComposition",
    "as_composition",
    "build_composition",
    "render_composition",
]

AXES = ("x", "y")


def _plain(value: Any) -> Any:
    """``value`` as JSON data: tuples become lists, mappings dicts, numpy values Python ones."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    tolist = getattr(value, "tolist", None)  # numpy scalars and arrays
    if callable(tolist) and type(value).__module__ == "numpy":
        return _plain(tolist())
    return value


#: Keywords that shape the whole figure, set on ``compose(...)``, never on a cell.
FIGURE_LEVEL_OPTIONS = frozenset({
    "format", "theme", "figsize", "figure_options", "save_path", "row_heights", "ax", "show", "backend",
    "interactive", "animation", "fps", "duration", "interval_ms", "controls", "interaction_backend",
})


@dataclass(frozen=True)
class FigureCell:
    """One canonical plot in one region of a composed figure.

    ``row``/``col`` are the top-left grid cell (from 0); ``rowspan``/
    ``colspan`` how many grid cells it covers.  ``options`` are the plot's own
    keyword arguments, exactly as ``plot_<name>(..., **options)`` takes them.
    ``name`` identifies the cell for :class:`AxisLink`; it defaults to the
    plot name, so a plot used twice needs explicit names.
    """

    plot: str
    row: int = 0
    col: int = 0
    rowspan: int = 1
    colspan: int = 1
    options: Mapping[str, Any] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.plot, str) or not self.plot:
            raise ValueError(f"FigureCell.plot must name a canonical plot; got {self.plot!r}")
        for attribute in ("row", "col", "rowspan", "colspan"):
            object.__setattr__(self, attribute, int(getattr(self, attribute)))
        if self.row < 0 or self.col < 0:
            raise ValueError(f"FigureCell {self.plot!r}: row and col start at 0; got ({self.row}, {self.col})")
        if self.rowspan < 1 or self.colspan < 1:
            raise ValueError(f"FigureCell {self.plot!r}: rowspan and colspan are at least 1")
        figure_level = sorted(FIGURE_LEVEL_OPTIONS & set(self.options))
        if figure_level:
            raise ValueError(
                f"FigureCell {self.plot!r}: {', '.join(figure_level)} shape the whole figure; "
                "set them on compose(...), not on a cell"
            )
        # Stored as the JSON form to_dict() writes, so a composition rebuilt
        # from its dict compares equal and builds the same figure.
        object.__setattr__(self, "options", MappingProxyType(_plain(dict(self.options))))
        object.__setattr__(self, "name", str(self.name or self.plot))

    #: Equal by value, not hashable: the options are a mapping.
    __hash__ = None  # type: ignore[assignment]

    @property
    def region(self) -> tuple[int, int, int, int]:
        """``(row, col, rowspan, colspan)``."""
        return (self.row, self.col, self.rowspan, self.colspan)

    def covers(self) -> set[tuple[int, int]]:
        """Every grid cell this region occupies."""
        return {
            (row, col)
            for row in range(self.row, self.row + self.rowspan)
            for col in range(self.col, self.col + self.colspan)
        }

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"plot": self.plot, "row": self.row, "col": self.col}
        if self.rowspan != 1:
            data["rowspan"] = self.rowspan
        if self.colspan != 1:
            data["colspan"] = self.colspan
        if self.options:
            data["options"] = _plain(self.options)
        if self.name != self.plot:
            data["name"] = self.name
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FigureCell":
        known = {"plot", "row", "col", "rowspan", "colspan", "options", "name"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"FigureCell takes no {', '.join(unknown)}; known: {', '.join(sorted(known))}")
        return cls(**dict(data))


@dataclass(frozen=True)
class AxisLink:
    """Cells whose ``axis`` (``"x"`` or ``"y"``) zooms and pans together.

    ``cells`` are :attr:`FigureCell.name` values.  An ``x`` link also keeps
    the tick labels on the lowest linked cell of each column only.
    """

    axis: str
    cells: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.axis not in AXES:
            raise ValueError(f"AxisLink.axis must be one of {', '.join(AXES)}; got {self.axis!r}")
        cells = tuple(str(cell) for cell in self.cells)
        if len(set(cells)) < 2:
            raise ValueError(f"AxisLink {self.axis!r} must link at least two cells; got {cells!r}")
        object.__setattr__(self, "cells", cells)

    def to_dict(self) -> dict[str, Any]:
        return {"axis": self.axis, "cells": list(self.cells)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AxisLink":
        return cls(axis=data["axis"], cells=tuple(data["cells"]))


@dataclass(frozen=True)
class FigureComposition:
    """A ``rows x cols`` grid of canonical plots, drawn as one figure.

    Cells may span several grid cells and must not overlap; grid cells no
    cell covers stay empty.  ``share_x`` links the x axes of the cells in each
    column (stacked time traces), ``share_y`` the y axes of the cells in each
    row; :class:`AxisLink` entries name any other group.  ``title`` is the
    figure title -- ``None`` puts the shot number there, ``""`` nothing.
    ``panel_labels`` marks the cells ``(a)``, ``(b)``, ... in cell order.
    """

    shape: tuple[int, int]
    cells: tuple[FigureCell, ...]
    share_x: bool = False
    share_y: bool = False
    links: tuple[AxisLink, ...] = ()
    title: str | None = None
    panel_labels: bool = False

    def __post_init__(self) -> None:
        try:
            rows, cols = (int(value) for value in self.shape)
        except (TypeError, ValueError):
            raise ValueError(f"FigureComposition.shape is (rows, cols); got {self.shape!r}") from None
        if rows < 1 or cols < 1:
            raise ValueError(f"FigureComposition.shape needs at least one row and column; got {(rows, cols)}")
        object.__setattr__(self, "shape", (rows, cols))
        cells = tuple(_as_cell(cell) for cell in self.cells)
        if not cells:
            raise ValueError("FigureComposition.cells must hold at least one plot")
        names: dict[str, FigureCell] = {}
        occupied: dict[tuple[int, int], str] = {}
        for cell in cells:
            if cell.row + cell.rowspan > rows or cell.col + cell.colspan > cols:
                raise ValueError(f"cell {cell.name!r} at {cell.region} does not fit the {rows}x{cols} grid")
            if cell.name in names:
                raise ValueError(f"two cells are named {cell.name!r}; give a plot used twice distinct names")
            names[cell.name] = cell
            for spot in sorted(cell.covers()):
                if spot in occupied:
                    raise ValueError(f"cells {occupied[spot]!r} and {cell.name!r} overlap at grid cell {spot}")
                occupied[spot] = cell.name
        object.__setattr__(self, "cells", cells)
        links = tuple(link if isinstance(link, AxisLink) else AxisLink(**link) for link in self.links)
        for link in links:
            unknown = [name for name in link.cells if name not in names]
            if unknown:
                raise ValueError(
                    f"AxisLink {link.axis!r} names no cell {', '.join(map(repr, unknown))}; "
                    f"cells: {', '.join(names)}"
                )
        object.__setattr__(self, "links", links)

    #: Equal by value, not hashable: the cells' options are mappings.
    __hash__ = None  # type: ignore[assignment]

    # -- convenient shapes ---------------------------------------------------
    @classmethod
    def stack(cls, plots: Sequence[str | FigureCell], *, share_x: bool = True, **kwargs: Any) -> "FigureComposition":
        """``m x 1``: the plots one above the other, sharing the x axis by default."""
        cells = [_placed(plot, row=index, col=0) for index, plot in enumerate(plots)]
        return cls(shape=(len(cells), 1), cells=tuple(cells), share_x=share_x, **kwargs)

    @classmethod
    def side_by_side(cls, plots: Sequence[str | FigureCell], **kwargs: Any) -> "FigureComposition":
        """``1 x n``: the plots next to each other."""
        cells = [_placed(plot, row=0, col=index) for index, plot in enumerate(plots)]
        return cls(shape=(1, len(cells)), cells=tuple(cells), **kwargs)

    @classmethod
    def grid(
        cls, plots: Sequence[str | FigureCell | None], shape: tuple[int, int], **kwargs: Any
    ) -> "FigureComposition":
        """``m x n`` filled row by row; ``None`` leaves a cell empty."""
        rows, cols = (int(value) for value in shape)
        if len(plots) > rows * cols:
            raise ValueError(f"{len(plots)} plots do not fit a {rows}x{cols} grid")
        cells = [
            _placed(plot, row=index // cols, col=index % cols)
            for index, plot in enumerate(plots) if plot is not None
        ]
        return cls(shape=(rows, cols), cells=tuple(cells), **kwargs)

    # -- what the renderers need ----------------------------------------------
    def resolved_links(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """Every axis link as ``(axis, cell indices)``, groups that touch merged.

        ``share_x`` contributes one group per column (every cell covering
        it, so a full-width trace joins the columns beneath it), ``share_y``
        one per row; a group that shares a cell with another group on the
        same axis is joined to it, since an axis follows one anchor only.
        """
        index = {cell.name: position for position, cell in enumerate(self.cells)}
        groups: list[tuple[str, set[int]]] = []
        if self.share_x:
            for col in range(self.shape[1]):
                groups.append(("x", {i for i, cell in enumerate(self.cells) if cell.col <= col < cell.col + cell.colspan}))
        if self.share_y:
            for row in range(self.shape[0]):
                groups.append(("y", {i for i, cell in enumerate(self.cells) if cell.row <= row < cell.row + cell.rowspan}))
        groups += [(link.axis, {index[name] for name in link.cells}) for link in self.links]
        merged: list[tuple[str, set[int]]] = []
        for axis, members in groups:
            if len(members) < 2:
                continue
            joined = set(members)
            keep = []
            for other_axis, other in merged:
                if other_axis == axis and other & joined:
                    joined |= other
                else:
                    keep.append((other_axis, other))
            merged = keep + [(axis, joined)]
        return tuple((axis, tuple(sorted(members))) for axis, members in merged)

    # -- reproducible form -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable form; :meth:`from_dict` rebuilds an equal composition."""
        data: dict[str, Any] = {"shape": list(self.shape), "cells": [cell.to_dict() for cell in self.cells]}
        if self.share_x:
            data["share_x"] = True
        if self.share_y:
            data["share_y"] = True
        if self.links:
            data["links"] = [link.to_dict() for link in self.links]
        if self.title is not None:
            data["title"] = self.title
        if self.panel_labels:
            data["panel_labels"] = True
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FigureComposition":
        known = {"shape", "cells", "share_x", "share_y", "links", "title", "panel_labels"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"FigureComposition takes no {', '.join(unknown)}; known: {', '.join(sorted(known))}")
        return cls(
            shape=tuple(data["shape"]),
            cells=tuple(FigureCell.from_dict(cell) for cell in data["cells"]),
            share_x=bool(data.get("share_x", False)),
            share_y=bool(data.get("share_y", False)),
            links=tuple(AxisLink.from_dict(link) for link in data.get("links", ())),
            title=data.get("title"),
            panel_labels=bool(data.get("panel_labels", False)),
        )


def _as_cell(cell: Any) -> FigureCell:
    if isinstance(cell, FigureCell):
        return cell
    if isinstance(cell, Mapping):
        return FigureCell.from_dict(cell)
    return FigureCell(cell)


def _placed(plot: str | FigureCell, *, row: int, col: int) -> FigureCell:
    if isinstance(plot, FigureCell):
        return FigureCell(
            plot.plot, row=row, col=col, rowspan=plot.rowspan, colspan=plot.colspan,
            options=plot.options, name=plot.name,
        )
    return FigureCell(str(plot), row=row, col=col)


def as_composition(value: Any) -> FigureComposition:
    """``value`` as a :class:`FigureComposition`: one already, or its dict form."""
    if isinstance(value, FigureComposition):
        return value
    if isinstance(value, Mapping):
        return FigureComposition.from_dict(value)
    raise TypeError(
        "a composition is a vaft.plot.FigureComposition or its to_dict() form; "
        f"got {type(value).__name__}"
    )


def build_composition(
    composition: FigureComposition | Mapping[str, Any],
    entries: Sequence[tuple[str, Any]],
    *,
    namespace: str = "vaft.omas",
    subject: str = "ods",
) -> Any:
    """The :class:`~vaft.plot.models.Panels` model of ``composition`` over ``entries``.

    Each cell is built by its plot's own recipe, as a member of a composite
    (short titles, no ``layout=`` of its own), with its options checked the
    way ``plot_<name>`` checks them.  A plot that draws several panels by
    itself is refused: a cell holds one.
    """
    from vaft.plot.backend.options import split_options, validate_options
    from vaft.plot.backend.recipes import _entry_shot, build_model
    from vaft.plot.backend.render import refuse_when_unsupported
    from vaft.plot.models import Panels

    composition = as_composition(composition)
    models, styles = [], []
    for cell in composition.cells:
        options = dict(cell.options)
        if "layout" in options:
            raise ValueError(
                f"cell {cell.name!r}: layout= spreads one plot over several axes; "
                "a composed cell holds one panel -- add the plots as cells instead"
            )
        validate_options(cell.plot, options)
        refuse_when_unsupported(cell.plot, entries, namespace=namespace, subject=subject)
        extraction, style = split_options(options)
        model = build_model(cell.plot, entries, _panel_member=True, **extraction)
        if isinstance(model, Panels):
            raise ValueError(
                f"cell {cell.name!r}: {cell.plot} draws {len(model.models)} panels of its own here; "
                "a composed cell holds one panel -- draw it as its own figure, or compose its "
                "member plots as cells"
            )
        models.append(model)
        styles.append(style)
    title = composition.title
    if title is None:
        shot = _entry_shot(entries)
        title = f"#{shot}" if shot else ""
    rows, cols = composition.shape
    return Panels(
        models=tuple(models),
        nrows=rows,
        ncols=cols,
        share_x=False,
        share_y=False,
        suptitle=title,
        member_styles=tuple(styles),
        spans=tuple(cell.region for cell in composition.cells),
        links=composition.resolved_links(),
        panel_labels=composition.panel_labels,
    )


#: Presentation keywords the Matplotlib panels renderer takes for the figure.
FIGURE_OPTIONS = ("format", "theme", "figsize", "figure_options")


def render_composition(
    composition: FigureComposition | Mapping[str, Any],
    entries: Sequence[tuple[str, Any]],
    *,
    backend: str | None = None,
    show: bool = False,
    namespace: str = "vaft.omas",
    subject: str = "ods",
    **presentation: Any,
) -> Any:
    """Draw ``composition`` over ``entries`` as one figure.

    Matplotlib (the default) returns ``(Figure, ndarray[Axes])`` -- one axes
    per cell, in cell order -- and takes the figure's ``format=``,
    ``theme=`` and ``figsize=``.  ``backend="plotly"`` returns one
    :class:`plotly.graph_objects.Figure`; it applies no Matplotlib
    presentation and refuses it, as single plots do.
    """
    from vaft.plot.backends import resolve_render_backend

    unknown = sorted(set(presentation) - set(FIGURE_OPTIONS))
    if unknown:
        raise TypeError(
            f"a composed figure takes {', '.join(FIGURE_OPTIONS)}; plot options belong "
            f"to its cells (got {', '.join(unknown)})"
        )
    from vaft.plot.figure_options import as_figure_options, figure_options_scope

    backend = resolve_render_backend(backend)
    model = build_composition(composition, entries, namespace=namespace, subject=subject)
    figure_options = as_figure_options(presentation.pop("figure_options", None))
    given = {key: value for key, value in presentation.items() if value is not None}
    if backend == "plotly":
        if given:
            raise TypeError(f"{', '.join(given)} apply to Matplotlib; backend='plotly' does not apply them")
        from vaft.plot.plotly import PLOTLY_MODELS, require_plotly

        require_plotly()
        missing = sorted({type(member).__name__ for member in model.models if type(member) not in PLOTLY_MODELS})
        if missing:
            raise NotImplementedError(
                f"backend='plotly' cannot draw this composition (no Plotly rendering for {', '.join(missing)})"
            )
        figure = PLOTLY_MODELS[type(model)].render(model, show=False)
        if figure_options:
            figure_options.apply_plotly(figure)
        if show:
            figure.show()
        return figure
    from vaft.plot.renderers.panels import render_panels

    if figure_options.panel_labels is not None:
        # The composition's own marks, at render time: one style, never doubled.
        import dataclasses

        model = dataclasses.replace(model, panel_labels=figure_options.panel_labels)
        figure_options = dataclasses.replace(figure_options, panel_labels=None)
    with figure_options_scope(figure_options):
        result = render_panels(model, show=False, **given)
    if figure_options:
        figure_options.apply(result[0])
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return result
