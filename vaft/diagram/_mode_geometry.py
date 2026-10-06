"""How MHD mode families map across slab, cylindrical and toroidal geometry (#1574).

``mhd_mode_geometry_map``
    one map of the pressure-driven, current-driven, resonant and axisymmetric
    branches in each geometry, with four kinds of arrow kept apart: an exact
    relabelling of coordinates and mode numbers, a limit or coordinate
    continuation, a physical analogue, and a branch that adds physics or
    localisation. It is a map of relationships, not a claim that connected
    modes are the same eigenmode.

The spectral relations in the header are :mod:`vaft.formula.geometry` and
``vaft.formula.stability.helical_phase``; the classification text is
annotation. Mode morphology, layers and eigenfunctions stay in the
specialised diagrams (``cylindrical_mode_morphology``, ``slab_parity``,
``ballooning_eigenfunction``, ``kink_mode``, ...).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from vaft.formula.geometry import cylindrical_parallel_wavenumber, slab_parallel_wavenumber
from vaft.formula.stability import helical_phase

from ._concept import band, box, connector
from ._equations import formula_equation
from ._geometry import _check_labels
from ._render import Diagram
from ._scene import Arrow, Label, Scene

#: arrow kind -> (TikZ style, legend text)
RELATIONS: Dict[str, Tuple[str, str]] = {
    "exact": ("map exact", "exact relabelling of mode labels"),
    "limit": ("map limit", "limit or continuation, head at the more general model"),
    "analogue": ("map analogue", "physical analogue, not the same eigenproblem"),
    "branch": ("map branch", "branch that adds physics or localisation"),
}

#: family band -> (y top, y bottom, label)
FAMILIES: Dict[str, Tuple[float, float, str]] = {
    "pressure": (-2.85, -7.6, "pressure / curvature driven"),
    "current": (-8.0, -14.3, "current driven"),
    "resonant": (-14.7, -17.4, "resonant and reconnecting"),
    "axisymmetric": (-17.8, -20.1, "axisymmetric, free boundary ($n = 0$)"),
}

_SLAB, _CYL, _T1, _T2, _T3 = 0.0, 6.6, 13.0, 18.2, 23.4
GEOMETRY_COLUMNS = {"slab": _SLAB, "cylinder": _CYL, "torus": (_T1, _T3)}
_W, _H = 4.2, 1.25
#: slab and cylinder boxes have more room than the three torus sub-columns
_W_LOCAL = 4.8
#: legend text: wrapped at word boundaries, never hyphenated
_LEGEND_STYLE = "small label,text width=5.2cm,align=left,execute at begin node={\\hyphenpenalty=10000}"

#: node -> (x, y, family, text); a family of None is the header row
_NODES: Dict[str, Tuple[float, float, object, str]] = {
    # header: the coordinates and spectral labels of each geometry
    "slab": (_SLAB, 0.0, None, "$(x, y, z)$\\\\ $k_y$, $k_\\parallel$"),
    "cylinder": (_CYL, 0.0, None, "$(r, \\theta, z)$\\\\ $m$, $k_z$"),
    "torus": (_T2, 0.0, None, "$(\\psi, \\theta, \\phi)$\\\\ $(m, n)$"),
    # pressure / curvature
    "rayleigh_taylor": (_SLAB, -4.4, "pressure", "curved slab:\\\\ Rayleigh--Taylor analogue"),
    "interchange": (_CYL, -4.4, "pressure", "flute / interchange,\\\\ $k_\\parallel \\simeq 0$ (Suydam)"),
    "mercier": (_T1, -4.4, "pressure", "interchange / Mercier,\\\\ localised"),
    "infernal": (_T1, -6.6, "pressure", "infernal: low $n$,\\\\ weak shear"),
    "ballooning": (_T2, -4.4, "pressure", "ballooning: high $n$,\\\\ bad curvature, shear"),
    "peeling_ballooning": (_T3, -6.6, "pressure", "peeling--ballooning:\\\\ coupled edge branch"),
    # current
    "sausage": (_CYL, -6.6, "pressure", "$m = 0$ sausage: $p'$ against\\\\ $B_\\theta$ curvature; $B_z$ stabilises"),
    "kink": (_CYL, -10.3, "current", "internal / external kink:\\\\ $m = 1$, $m \\ge 2$ helical"),
    "internal_kink": (_T1, -9.3, "current", "toroidal internal kink:\\\\ Bussac $\\delta W$; resistive"),
    "external_kink": (_T1, -11.3, "current", "toroidal external kink:\\\\ boundary displacement"),
    "rwm": (_T1, -13.3, "current", "RWM: external kink\\\\ + resistive wall"),
    "peeling": (_T2, -11.3, "current", "peeling: edge current,\\\\ external-kink-like"),
    # resonant
    "slab_layer": (_SLAB, -16.05, "resonant", "sheared-slab layer, $k_\\parallel = 0$:\\\\ tearing / twisting parity"),
    "cylindrical_tearing": (_CYL, -16.05, "resonant", "cylindrical tearing:\\\\ outer region, $\\Delta'$"),
    "toroidal_tearing": (_T1, -16.05, "resonant", "toroidal tearing:\\\\ $m$ coupled at fixed $n$"),
    "ntm": (_T2, -16.05, "resonant", "NTM: nonlinear island,\\\\ bootstrap-current hole"),
    # axisymmetric
    "rigid_shift": (_CYL, -18.95, "axisymmetric", "rigid transverse shift:\\\\ intuition only"),
    "vde": (_T1, -18.95, "axisymmetric", "vertical instability\\\\ (VDE), free boundary"),
}

#: (start, end, relation, label or "")
_EDGES: Tuple[Tuple[str, str, str, str], ...] = (
    ("cylinder", "slab", "exact", "at $r_0$:\\\\ $m \\mapsto k_y$"),
    ("torus", "cylinder", "exact", "$z = R_0\\phi$: $n \\mapsto k_z$"),
    ("rayleigh_taylor", "interchange", "analogue", ""),
    ("interchange", "mercier", "limit", ""),
    ("mercier", "ballooning", "branch", ""),
    ("interchange", "infernal", "branch", ""),
    ("ballooning", "peeling_ballooning", "branch", ""),
    ("peeling", "peeling_ballooning", "branch", ""),
    ("kink", "internal_kink", "limit", ""),
    ("kink", "external_kink", "limit", ""),
    ("external_kink", "rwm", "branch", ""),
    ("external_kink", "peeling", "branch", ""),
    ("slab_layer", "cylindrical_tearing", "limit", ""),
    ("cylindrical_tearing", "toroidal_tearing", "limit", ""),
    ("toroidal_tearing", "ntm", "branch", ""),
    ("rigid_shift", "vde", "analogue", ""),
)


def mhd_mode_geometry_map(*, labels: bool = True) -> Diagram:
    r"""MHD mode families across slab, cylindrical and toroidal geometry, with typed relations.

    Columns are geometries, each headed by its coordinates, spectral labels
    and spectral relation: ``slab_parallel_wavenumber``,
    ``cylindrical_parallel_wavenumber`` ($k_\parallel = 0 \Leftrightarrow
    q = m/n$) and ``helical_phase``. The header arrows relabel mode labels
    exactly; the geometric reductions behind them are limits
    (``geometry_ordering_map``). Bands are physical families: pressure /
    curvature driven (Rayleigh--Taylor analogue, interchange, the $m = 0$
    sausage, Mercier,
    ballooning, infernal, peeling--ballooning), current driven (internal and
    external kink in the cylinder and the torus, peeling, RWM), resonant
    (layer parity, outer tearing, toroidal tearing, NTM) and the $n = 0$
    vertical instability. Arrows are typed: an exact relabelling, a limit or
    coordinate continuation, a physical analogue, or a branch that adds
    physics or localisation. A connection is a relationship, not identity.
    """
    labels = _check_labels(labels)
    left, right = -2.6, 25.9
    items: List = []
    for name, (top, bottom, text) in FAMILIES.items():
        items += band(left, right, bottom, top, text, role=f"family:{name}")
    titles = {"slab": (_SLAB, "slab"), "cylinder": (_CYL, "cylinder / screw pinch"),
              "torus": (_T2, "torus (axisymmetric tokamak)")}
    for name, (x, text) in titles.items():
        items.append(Label((x, 1.05), text, "subtitle", anchor="south", role=f"geometry:{name}"))
    nodes = {}
    for name, (x, y, family, text) in _NODES.items():
        width = 4.4 if family is None else (_W if x > _CYL else _W_LOCAL)
        style = "concept strong" if family is None else "concept box"
        nodes[name] = box(x, y, width, 1.35 if family is None else _H, text, style=style,
                          role=f"node:{name}", latex=True)
        items += list(nodes[name].items)
    # each geometry's resonance, from the catalog
    for name, (x, function) in {"slab": (_SLAB, slab_parallel_wavenumber),
                                "cylinder": (_CYL, cylindrical_parallel_wavenumber),
                                "torus": (_T2, helical_phase)}.items():
        items.append(Label((x, -1.05), f"$\\displaystyle {formula_equation(function)}$", "formula box",
                           anchor="north", role=f"equation:{name}"))
    for start, end, relation, text in _EDGES:
        arrow = connector(nodes[start], nodes[end], style=RELATIONS[relation][0],
                          role=f"edge:{relation}:{start}->{end}")
        items.append(arrow)
        if text:
            mid = 0.5 * (np.array(arrow.start) + np.array(arrow.end))
            items.append(Label((float(mid[0]), float(mid[1]) + 0.15), text, "small label,align=center", anchor="south",
                               role=f"edge:{relation}:{start}->{end}"))
    # global current-driven modes have no local-slab form
    items.append(Label((_SLAB, -10.3), "global: no local-slab\\\\ counterpart", "concept annotation",
                       role="note:no_slab_kink"))
    if labels:
        y = -20.9
        x0 = left + 0.3
        for i, (relation, (style, text)) in enumerate(RELATIONS.items()):
            x = x0 + 7.0 * i
            items.append(Arrow((x, y), (x + 1.2, y), style, role=f"legend:{relation}"))
            items.append(Label((x + 1.4, y), text, _LEGEND_STYLE, anchor="west", role=f"legend:{relation}"))
        items.append(Label((0.5 * (left + right), -21.75),
                           "A map of relationships, not of identity: header arrows relabel mode labels exactly, "
                           "while the geometric reductions themselves are limits (geometry\\_ordering\\_map).\\\\ "
                           "A mode is a "
                           "spectral label + drive + localisation + resonance + parity + physical model",
                           "note", anchor="north", role="note"))
    model = {
        "columns": dict(GEOMETRY_COLUMNS),
        "nodes": {k: (b.x, b.y) for k, b in nodes.items()},
        "families": {k: v[2] for k, v in _NODES.items() if v[2] is not None},
        "edges": tuple((a, b, rel) for a, b, rel, _ in _EDGES),
        "relations": {k: v[1] for k, v in RELATIONS.items()},
    }
    return Diagram("mhd_mode_geometry_map", Scene(tuple(items)), model=model)
