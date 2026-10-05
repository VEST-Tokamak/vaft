"""The composition editor: several plots arranged into one figure (#1467, phase 5).

The editor holds no figure of its own: it edits a
:class:`vaft.plot.FigureComposition` -- a preset or custom grid, one plot per
cell from the loaded sources' catalog, row and column spans, shared axes,
panel labels and a title -- which the app draws with ``compose`` and the
reproduction writes as code.  Validation is the composition's: an
overlapping span or a cell outside the grid is reported, not drawn.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from ._require import require_panel

EMPTY = "(empty)"

#: The common grids (rows, columns) a figure starts from.
PRESETS: dict[str, tuple[int, int]] = {
    "1 × 1": (1, 1), "2 × 1": (2, 1), "3 × 1": (3, 1), "4 × 1": (4, 1),
    "1 × 2": (1, 2), "1 × 3": (1, 3), "2 × 2": (2, 2), "2 × 3": (2, 3),
}


class CompositionEditor:
    """Widgets that build a :class:`~vaft.plot.FigureComposition`."""

    def __init__(self, on_draw: Callable[[Any], None]) -> None:
        pn = require_panel()

        self._on_draw = on_draw
        self._plots: list[str] = []
        self.preset = pn.widgets.Select(label="Layout", options=[*PRESETS, "custom"], value="2 × 1")
        self.rows = pn.widgets.IntInput(label="Rows", value=2, start=1, end=6)
        self.cols = pn.widgets.IntInput(label="Columns", value=1, start=1, end=4)
        self.share_x = pn.widgets.Checkbox(label="share x (per column)", value=True)
        self.share_y = pn.widgets.Checkbox(label="share y (per row)", value=False)
        self.panel_labels = pn.widgets.Checkbox(label="panel labels (a), (b), ...", value=False)
        self.title = pn.widgets.TextInput(label="Title", value="", placeholder="the shot number")
        self.draw = pn.widgets.Button(label="Draw composition", color="primary")
        self.cells_box = pn.Column(sizing_mode="stretch_width")
        #: (row, col) -> (plot select, rowspan, colspan)
        self.cells: dict[tuple[int, int], tuple[Any, Any, Any]] = {}
        self.preset.param.watch(self._on_preset, "value")
        self.rows.param.watch(lambda _event: self._rebuild(), "value")
        self.cols.param.watch(lambda _event: self._rebuild(), "value")
        self.draw.on_click(lambda _event: self._draw())
        self._rebuild()

    # -- catalog -----------------------------------------------------------------------
    def set_plots(self, names: Sequence[str]) -> None:
        """The plots a cell may hold: the catalog of the loaded sources."""
        self._plots = list(names)
        for select, _, _ in self.cells.values():
            current = select.value
            select.options = [EMPTY, *self._plots]
            select.value = current if current in self._plots else EMPTY

    # -- grid --------------------------------------------------------------------------
    def _on_preset(self, event: Any) -> None:
        if event.new in PRESETS:
            rows, cols = PRESETS[event.new]
            self.share_x.value = cols == 1 or rows > 1
            self.rows.value, self.cols.value = rows, cols

    def _rebuild(self) -> None:
        pn = require_panel()
        rows, cols = int(self.rows.value or 1), int(self.cols.value or 1)
        kept = {spot: widgets for spot, widgets in self.cells.items() if spot[0] < rows and spot[1] < cols}
        cells: dict[tuple[int, int], tuple[Any, Any, Any]] = {}
        layout = []
        for row in range(rows):
            for col in range(cols):
                if (row, col) in kept:
                    cells[(row, col)] = kept[(row, col)]
                else:
                    cells[(row, col)] = (
                        pn.widgets.Select(label=f"Cell ({row + 1}, {col + 1})", options=[EMPTY, *self._plots], value=EMPTY),
                        pn.widgets.IntInput(label="rows spanned", value=1, start=1, end=rows),
                        pn.widgets.IntInput(label="columns spanned", value=1, start=1, end=cols),
                    )
                select, rowspan, colspan = cells[(row, col)]
                rowspan.end, colspan.end = rows, cols
                layout.append(pn.Column(select, rowspan, colspan, sizing_mode="stretch_width"))
        self.cells = cells
        self.cells_box.objects = layout

    def assign(self, plots: Sequence[str | None]) -> None:
        """Fill the cells row by row; ``None`` leaves one empty."""
        for spot, plot in zip(sorted(self.cells), plots):
            self.cells[spot][0].value = plot if plot in self._plots else EMPTY

    # -- the composition ------------------------------------------------------------------
    def composition(self) -> Any:
        """The :class:`FigureComposition` the widgets say; ``ValueError`` when invalid."""
        from vaft.plot import FigureCell, FigureComposition

        cells = []
        used: dict[str, int] = {}
        for (row, col), (select, rowspan, colspan) in sorted(self.cells.items()):
            if select.value == EMPTY:
                continue
            used[select.value] = used.get(select.value, 0) + 1
            name = select.value if used[select.value] == 1 else f"{select.value} ({used[select.value]})"
            cells.append(FigureCell(
                select.value, row=row, col=col, rowspan=int(rowspan.value or 1), colspan=int(colspan.value or 1),
                name=name,
            ))
        if not cells:
            raise ValueError("choose a plot for at least one cell")
        return FigureComposition(
            shape=(int(self.rows.value), int(self.cols.value)), cells=tuple(cells),
            share_x=bool(self.share_x.value), share_y=bool(self.share_y.value),
            title=self.title.value or None, panel_labels=bool(self.panel_labels.value),
        )

    def _draw(self) -> None:
        self._on_draw(self)

    def widgets(self) -> list[Any]:
        return [
            self.preset, self.rows, self.cols, self.cells_box,
            self.share_x, self.share_y, self.panel_labels, self.title, self.draw,
        ]


__all__ = ["CompositionEditor", "EMPTY", "PRESETS"]
