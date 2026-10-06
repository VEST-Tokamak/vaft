"""Edge-q proxies against time, for shots with or without an equilibrium (#1583).

Each renderer draws a :class:`~vaft.plot.models.LineSeries` built by
:func:`vaft.omas.edge_q.edge_q_estimate`: the estimated q95 (labelled with its
scaling, the equilibrium q95 overlaid when the ODS has one), Menard's
cylindrical q*, Freidberg's kink q* and the normalised current I_N. Each is
named by its own quantity; none is q_a.
"""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..models import LineSeries
from ..registry import renderer
from .lines import render_line_series

__all__ = [
    "summary_time_estimated_q95",
    "summary_time_normalized_current",
    "summary_time_q_star_cylindrical",
    "summary_time_q_star_kink",
]

#: Either an equilibrium (shape, Ip, vacuum field) or magnetics Ip with the TF
#: field; ``available`` on the recipe states the either-or.
_IDS = ("equilibrium", "magnetics", "tf")
_OPTIONAL = (
    "equilibrium.time",
    "equilibrium.time_slice.{i}.boundary.outline.r",
    "equilibrium.time_slice.{i}.global_quantities.ip",
    "equilibrium.time_slice.{i}.global_quantities.q_95",
    "equilibrium.vacuum_toroidal_field.b0",
    "magnetics.ip.0.data",
    "tf.b_field_tor_vacuum_r.data",
)


def _edge_q_renderer(quantity: str, description: str):
    return renderer(
        domain="summary",
        subject="summary",
        view="time",
        quantity=quantity,
        model=LineSeries,
        description=description,
        ids=_IDS,
        required_paths=(),
        optional_paths=_OPTIONAL,
    )


@_edge_q_renderer(
    "estimated_q95",
    "q95 estimated from global shape, I_p and B_T (START scaling by default, from vest.yaml), "
    "the equilibrium q95 overlaid; works without an equilibrium (issue #1583).",
)
def summary_time_estimated_q95(
    model: LineSeries, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Estimated q95 against time, labelled with its scaling, equilibrium q95 overlaid."""
    return render_line_series(model, ax=ax, show=show, **style)


@_edge_q_renderer(
    "q_star_cylindrical",
    "Menard's cylindrical safety factor q* = pi a^2 B_T (1 + kappa^2)/(mu0 R I_p) against time; "
    "a shape-weighted proxy, not q95 or q_a (issue #1583).",
)
def summary_time_q_star_cylindrical(
    model: LineSeries, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Menard's cylindrical q* against time."""
    return render_line_series(model, ax=ax, show=show, **style)


@_edge_q_renderer(
    "q_star_kink",
    "Freidberg's kink safety factor q* = 2 pi a^2 kappa B_T/(mu0 R I_p) against time; "
    "not q95 or q_a (issue #1583).",
)
def summary_time_q_star_kink(
    model: LineSeries, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Freidberg's kink q* against time."""
    return render_line_series(model, ax=ax, show=show, **style)


@_edge_q_renderer(
    "normalized_current",
    "Normalised current I_N = I_p[MA]/(a B_T) against time, from the same shape as the "
    "edge-q estimates (issue #1583).",
)
def summary_time_normalized_current(
    model: LineSeries, *, ax: Axes | None = None, show: bool = False, **style: Any
) -> tuple[Figure, Axes]:
    """Normalised current I_N against time."""
    return render_line_series(model, ax=ax, show=show, **style)
