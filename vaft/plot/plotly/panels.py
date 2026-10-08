"""Plotly drawing of :class:`~vaft.plot.models.Panels` with ``make_subplots``.

The grid, the spans, the shared time base and the bottom-row-only time label
follow the model exactly as the Matplotlib panels renderer does; each member
is drawn into its cell by the ``add`` function of its kind.
"""

from __future__ import annotations

from typing import Any

from ..models import Geometry3DLayers, Panels
from . import PLOTLY_MODELS, require_plotly

__all__ = ["add_panels", "render_panels"]


def _cells(model: Panels) -> list[tuple[int, int, int, int]]:
    if model.spans is not None:
        return [tuple(span) for span in model.spans]
    return [(slot // model.ncols, slot % model.ncols, 1, 1) for slot in range(len(model.models))]


def _specs(model: Panels, cells: list[tuple[int, int, int, int]]) -> list[list[Any]]:
    specs: list[list[Any]] = [[None] * model.ncols for _ in range(model.nrows)]
    for (row, col, rowspan, colspan), member in zip(cells, model.models):
        specs[row][col] = {"rowspan": rowspan, "colspan": colspan}
        if isinstance(member, Geometry3DLayers):
            # A 3-D member needs a scene cell, not x/y axes.
            specs[row][col]["type"] = "scene"
    return specs


def render_panels(model: Panels, *, show: bool = False, **style: Any) -> Any:
    go = require_plotly()
    from plotly.subplots import make_subplots

    if not isinstance(model, Panels):
        raise TypeError(f"expected a vaft.plot.models.Panels; got {type(model).__name__}.")
    for member in model.models:
        if type(member) not in PLOTLY_MODELS:
            raise NotImplementedError(f"no Plotly rendering for {type(member).__name__}")
    cells = _cells(model)
    # make_subplots hands titles out in row-major cell order, whatever order
    # the models are in (a spans overview lists them column by column).
    from ._style import plain_text

    titles = {cell[:2]: plain_text(getattr(member, "title", "")) for cell, member in zip(cells, model.models)}
    ordered = [titles[key] for key in sorted(titles)]
    figure = make_subplots(
        rows=model.nrows, cols=model.ncols, specs=_specs(model, cells),
        shared_xaxes=model.share_x, shared_yaxes=model.share_y,
        subplot_titles=ordered,
        vertical_spacing=min(0.12, 0.6 / max(model.nrows, 1)),
        horizontal_spacing=0.1 if model.ncols > 1 else 0.02,
    )
    add_panels(figure, model, **style)
    _link_axes(figure, model, cells)
    if model.panel_labels:
        _label_panels(figure, model, cells)
    figure.update_layout(title={"text": plain_text(model.suptitle)} if model.suptitle else None, template="plotly_white",
                         height=max(320, 260 * model.nrows), width=max(480, 420 * model.ncols))
    if show:
        figure.show()
    return figure


def _link_axes(figure: Any, model: Panels, cells: list[tuple[int, int, int, int]]) -> None:
    """Tie each link's panels together with ``matches`` (issue #1467).

    The panel that set its own range anchors the link, whatever its place in
    it, so the range it asked for is the range all of them show.
    """
    from .._panel_grid import linked_above

    for axis_name, members in model.links:
        def axis_of(slot: int) -> Any:
            row, col, _, _ = cells[slot]
            return getattr(figure.get_subplot(row + 1, col + 1), f"{axis_name}axis")

        anchor_slot = next((slot for slot in members if axis_of(slot).range is not None), members[0])
        anchor = axis_of(anchor_slot).plotly_name.replace("axis", "")
        update = figure.update_xaxes if axis_name == "x" else figure.update_yaxes
        for slot in members:
            if slot != anchor_slot:
                row, col, _, _ = cells[slot]
                update(matches=anchor, row=row + 1, col=col + 1)
    for slot in linked_above(model):
        row, col, _, _ = cells[slot]
        figure.update_xaxes(showticklabels=False, row=row + 1, col=col + 1)


def _label_panels(figure: Any, model: Panels, cells: list[tuple[int, int, int, int]]) -> None:
    from .._panel_grid import panel_label

    for index, ((row, col, _, _), member) in enumerate(zip(cells, model.models)):
        subplot = figure.get_subplot(row + 1, col + 1)
        if isinstance(member, Geometry3DLayers):
            x0, y1 = subplot.domain.x[0], subplot.domain.y[1]
        else:
            x0, y1 = subplot.xaxis.domain[0], subplot.yaxis.domain[1]
        figure.add_annotation(
            text=f"<b>{panel_label(index)}</b>", xref="paper", yref="paper", x=x0, y=y1,
            xanchor="right", yanchor="bottom", showarrow=False,
        )


def add_panels(figure: Any, model: Panels, **style: Any) -> None:
    cells = _cells(model)
    # With a shared time base the time label sits on the lowest panel of each
    # column only, as in the Matplotlib renderer; an x link does the same for
    # the panels it ties.
    lowest = {}
    for (row, col, rowspan, _), member in zip(cells, model.models):
        if row + rowspan > lowest.get(col, (-1, None))[0]:
            lowest[col] = (row + rowspan, member)
    from .._panel_grid import linked_above as linked

    linked_above = linked(model)
    for index, ((row, col, _, _), member) in enumerate(zip(cells, model.models)):
        member_style = dict(model.member_styles[index]) if model.member_styles else {}
        x_title = (not model.share_x or lowest[col][1] is member) and index not in linked_above
        PLOTLY_MODELS[type(member)].add(figure, member, row=row + 1, col=col + 1, x_title=x_title,
                                        **{**member_style, **style})
