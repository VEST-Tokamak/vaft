"""Internal inductance against edge q: the empirical and the theoretical l_i-q diagrams (#1422).

Two literature pictures share the look of this plane and nothing else:

* ``reference="wesson_1989"``: the JET *empirical* operating space of
  Wesson et al., Nucl. Fusion 29 (1989) 641, Fig. 6, on $(q_\\psi,\\ l_i)$
  with $l_i = 2\\int B_\\theta^2\\,d\\tau/(\\mu_0^2 R I^2)$. Below the lower,
  saw-tooth boundary rotating kink and double-tearing modes grow during the
  current rise; above the upper one density-limit disruptions occur. The
  $q_\\psi = 2$ edge is the registered ``low_q``, which is on this same axis.
* ``reference="cheng_1987"``: the *theoretical* domain of MHD-stable current
  profiles of Cheng, Furth and Boozer, Plasma Phys. Control. Fusion 29 (1987)
  351, Fig. 4: a pressureless straight cylinder without a wall, $q(0) = 1.01$,
  on the cylinder's $(q(a),\\ l_i)$. The jig-saw lower bound is the ideal
  external kink, the upper bound low-order resistive kinks. The paper plots
  $l_i/2$; this diagram plots $l_i$. Its $q(a) = 2$ edge is ``cheng_1987_qa_min``.

Both sets of lines are the registered boundaries of
:mod:`vaft.formula.boundaries`, taken through the ``li_qa_wesson`` and
``li_qa_cheng`` projections; nothing is digitized here.
"""

from __future__ import annotations

import numpy as np

from vaft.formula import boundaries as _b

from ._chart import Chart, nice_ticks as _nice_ticks, render_chart as _render_chart
from ._op_space import get_projection
from ._render import Diagram

_REFERENCES = {
    "wesson_1989": dict(
        projection="li_qa_wesson",
        x_range=(0.0, 15.0), y_range=(0.0, 2.0),
        x_label="$q_\\psi$", y_label="$l_i$",
        labels={"operating": (6.8, 1.12), "upper": (3.2, 1.75), "lower": (6.5, 0.1)},
        text={"operating": "Operating space",
              "upper": "\\begin{tabular}{c}Density-limit\\\\disruptions\\end{tabular}",
              "lower": "Kink and double tearing"},
        note="Wesson et al., Nucl. Fusion 29 (1989) 641, Fig. 6: JET empirical boundaries",
    ),
    "cheng_1987": dict(
        projection="li_qa_cheng",
        x_range=(1.0, 8.0), y_range=(0.2, 2.6),
        x_label="$q(a)$ (cylinder)", y_label="$l_i$ (cylinder)",
        labels={"operating": (4.9, 1.3), "upper": (3.1, 2.05), "lower": (5.5, 0.55)},
        text={"operating": "MHD stable",
              "upper": "\\begin{tabular}{c}Unstable\\\\(resistive kink)\\end{tabular}",
              "lower": "Unstable (ideal kink)"},
        note="Cheng, Furth and Boozer, PPCF 29 (1987) 351, Fig. 4: theory, $q(0) = 1.01$, plotted as $l_i$",
    ),
}


def _samples(lo: float, hi: float) -> np.ndarray:
    """A grid that also resolves the vertical edges of a saw-tooth boundary at integer q."""
    integers = np.arange(np.ceil(lo), np.floor(hi) + 1)
    return np.unique(np.concatenate([np.linspace(lo, hi, 401), integers, integers - 1e-6]))


def li_qa(*, reference: str = "wesson_1989", labels: bool = True) -> Diagram:
    r"""The $l_i$-$q$ diagram of one source: Wesson 1989 (JET, empirical) or Cheng 1987 (theory).

    Parameters
    ----------
    reference : {"wesson_1989", "cheng_1987"}
        Which published diagram. The two use different quantities: Wesson's
        $q_\psi$ and $l_i(3)$, Cheng's straight-cylinder $q(a)$ and $l_i$.
    labels : bool
        Draw region labels and the source note.

    Returns
    -------
    Diagram
        ``Diagram.model`` is a :class:`Chart` with the ``lower`` and ``upper``
        boundary curves in data coordinates and the projection key and
        boundary keys in ``parameters``.
    """
    try:
        spec = _REFERENCES[reference]
    except KeyError:
        raise ValueError(f"reference must be one of {sorted(_REFERENCES)}, not {reference!r}") from None
    projection = get_projection(spec["projection"])
    chart = Chart(x_range=spec["x_range"], y_range=spec["y_range"])
    keys = []
    edges = []
    for key in projection.default_boundaries:
        boundary = _b.get_boundary(key)
        keys.append(key)
        if boundary.form == "threshold":  # a vertical q edge, drawn between the two branches below
            edges.append(boundary)
            continue
        lo, hi = boundary.applicability.ranges[boundary.input_names[0]]
        curve = _b.boundary_curve(boundary, boundary.input_names[0], _samples(lo, hi))
        xy = curve.xy[np.isfinite(curve.xy).all(axis=1)]
        # 9 decimals: the TikZ source (and the committed SVG's hash) must not depend on the platform's
        # last-bit floating-point differences, which flipped one printed coordinate (5.029 vs 5.0291)
        chart.curves[boundary.branch] = np.round(xy, 9)
    for boundary in edges:
        q_edge = float(_b.boundary_value(boundary))
        span = sorted(float(np.interp(q_edge, *chart.curves[b].T)) for b in ("lower", "upper"))
        chart.curves["edge"] = _b.threshold_curve(boundary, projection.y, span, target_axis="x").xy
    chart.labels.update(spec["labels"])
    chart.parameters.update({"reference": reference, "projection": projection.key, "boundaries": tuple(keys)})
    x_hi, y_hi = chart.x_range[1], chart.y_range[1]
    scene = _render_chart(
        chart,
        x_label=spec["x_label"],
        y_label=spec["y_label"],
        curve_styles={name: "boundary" for name in ("lower", "upper", "edge") if name in chart.curves},
        region_text=spec["text"] if labels else {},
        x_ticks=[t for t in _nice_ticks(x_hi) if t >= chart.x_range[0]],
        y_ticks=[t for t in _nice_ticks(y_hi) if t >= chart.y_range[0]],
        note=spec["note"] if labels else "",
    )
    return Diagram("li_qa", scene, model=chart)
