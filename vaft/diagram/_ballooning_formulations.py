"""Reduced and full-geometry high-$n$ ballooning formulations side by side (#1637).

``ballooning_formulation_hierarchy``
    the general local ballooning equation; the chain of approximations that
    reduces it to the Connor-Hastie-Taylor $s$-$\\alpha$ equation VAFT solves;
    the two full-geometry implementations (Fortran DCON's $C_A$, GPEC.jl's
    BALOO-style $\\Delta'$) that keep the geometry; and the common
    normalisation they must share before their marginal boundaries can be
    compared.

The equations are :mod:`vaft.formula` entries; implementation facts are from
the DCON (``dcon/bal.f``) and GPEC.jl (``src/LocalStability/Ballooning.jl``)
sources, documented on the Ballooning formulations reference page.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from vaft.formula.equilibrium import ballooning_alpha_from_volume, shear_from_volume
from vaft.formula.stability import s_alpha_ballooning_stable

from ._concept import box, connector
from ._equations import formula_equation
from ._geometry import _check_labels
from ._render import Diagram
from ._scene import Label, Scene

#: the approximations taking the general equation to the s-alpha one, in order
REDUCTION_STEPS: Tuple[Tuple[str, str], ...] = (
    ("large_aspect_ratio", "large aspect ratio, $\\epsilon \\ll 1$"),
    ("shifted_circles", "shifted circular surfaces"),
    ("metric", "metric: $|\\nabla\\beta|^2/B^2 \\propto 1 + \\Lambda^2$"),
    ("curvature", "curvature: $\\kappa_w \\propto \\cos\\theta + \\Lambda\\sin\\theta$"),
)

#: implementation -> (index, stable sign, boundary treatment)
IMPLEMENTATIONS: Dict[str, Tuple[str, str, str]] = {
    "vaft": ("Newcomb test; $\\alpha_1$, $\\alpha_2$", "no zero crossing", "even start, finite $\\theta$ range"),
    "dcon": ("$C_A$", "$C_A > 0$", "asymptotic small solution"),
    "gpec_jl": ("ballooning $\\Delta'$", "$\\Delta' < 0$", "Dirichlet at $\\pm\\theta_{max}$"),
}

_X_LEFT, _X_DCON, _X_JL = -4.6, 4.2, 10.4


def ballooning_formulation_hierarchy(*, labels: bool = True) -> Diagram:
    r"""How the reduced $s$-$\alpha$ model and the full-geometry ballooning codes relate.

    The general local high-$n$ ideal ballooning equation (field-line bending
    against the pressure-curvature drive) either loses its geometry step by
    step -- large aspect ratio, shifted circles, the $1 + \Lambda^2$ metric,
    the $\cos\theta + \Lambda\sin\theta$ curvature -- to become the
    Connor-Hastie-Taylor equation of ``s_alpha_ballooning_stable``, or keeps
    it, as Fortran DCON (asymptotic matching, $C_A$, stable when positive)
    and GPEC.jl (Dirichlet matching, ballooning $\Delta'$, stable when
    negative, with poles) do. Their marginal boundaries are comparable only
    after a common normalisation: ``shear_from_volume`` and
    ``ballooning_alpha_from_volume``, which reduce exactly to $\hat s$ and the
    CHT $\alpha$ for circles.
    """
    labels = _check_labels(labels)
    items: List = []
    nodes = {
        "general": box(2.9, 0.0, 13.5, 1.3, "general local high-$n$ ideal ballooning equation:\\\\ field-line bending "
                       "$|\\nabla\\beta|^2/B^2$ against the drive $P'\\kappa_w$ along an extended field line",
                       style="concept strong", role="node:general", latex=True),
    }
    y = -2.2
    for key, text in REDUCTION_STEPS:
        nodes[key] = box(_X_LEFT, y, 6.2, 0.9, text, role=f"node:{key}", latex=True)
        y -= 1.3
    nodes["vaft"] = box(_X_LEFT, y - 0.35, 6.2, 1.6, "VAFT reduced $s$--$\\alpha$ (CHT):\\\\ Newcomb test, "
                        "$\\alpha_1$ and $\\alpha_2$;\\\\ any $\\hat s$, $\\alpha$; no geometry", style="concept leaf",
                        role="node:vaft", latex=True)
    nodes["dcon"] = box(_X_DCON, -3.6, 5.6, 2.6, "Fortran DCON\\\\ full geometry, $\\theta_0 = 0$\\\\ asymptotic "
                        "small solution\\\\ index $C_A$, stable when $> 0$;\\\\ only where its own $D_I < 0$",
                        style="concept leaf", role="node:dcon", latex=True)
    nodes["gpec_jl"] = box(_X_JL, -3.6, 5.6, 2.6, "GPEC.jl local ballooning\\\\ full geometry, any $\\theta_k$\\\\ "
                           "Dirichlet at $\\pm\\theta_{max}$\\\\ $\\Delta'$, stable when $< 0$;\\\\ poles; "
                           "$\\alpha_{crit,1}$, $\\alpha_{crit,2}$ scans", style="concept leaf",
                           role="node:gpec_jl", latex=True)
    compare_y = nodes["vaft"].y - 2.0
    nodes["compare"] = box(2.9, compare_y, 12.0, 1.3, "compare marginal boundaries on one normalisation, not Boolean labels:\\\\ "
                           "$C_A$ and $\\Delta'$ zeros agree only as $\\theta_{max} \\to \\infty$ with $D_I < 0$", style="concept strong", role="node:compare",
                           latex=True)
    for b in nodes.values():
        items += list(b.items)
    edges = [("general", "large_aspect_ratio", "reduce geometry"), ("large_aspect_ratio", "shifted_circles", ""),
             ("shifted_circles", "metric", ""), ("metric", "curvature", ""), ("curvature", "vaft", ""),
             ("general", "dcon", ""), ("general", "gpec_jl", ""),
             ("vaft", "compare", ""), ("dcon", "compare", ""), ("gpec_jl", "compare", "")]
    for a, b, text in edges:
        arrow = connector(nodes[a], nodes[b], role=f"edge:{a}->{b}")
        items.append(arrow)
        if text and labels:
            mx, my = 0.5 * (arrow.start[0] + arrow.end[0]), 0.5 * (arrow.start[1] + arrow.end[1])
            side = "east" if mx < nodes["general"].x - 2.0 else "west"
            items.append(Label((mx + (-0.15 if side == "east" else 0.15), my), text, "small label", anchor=side,
                               role=f"edge:{a}->{b}"))
    if labels:
        # one label for both full-geometry arrows, right of the GPEC.jl one
        items.append(Label((8.5, -1.35), "both retain the geometry", "small label", anchor="west",
                           role="edge:general->full"))
    # the equations: what VAFT solves, and the normalisation all three must share
    eq_y = compare_y - 1.6
    items += [
        Label((-3.6, eq_y), f"$\\displaystyle {formula_equation(s_alpha_ballooning_stable)}$",
              "formula box", anchor="north", role="equation:cht"),
        Label((5.3, eq_y), f"$\\displaystyle {formula_equation(shear_from_volume)}$", "formula box", anchor="north",
              role="equation:shear"),
        Label((10.9, eq_y), f"$\\displaystyle {formula_equation(ballooning_alpha_from_volume)}$", "formula box",
              anchor="north", role="equation:alpha"),
    ]
    if labels:
        items += [Label((-3.6, eq_y + 0.1), "what VAFT solves", "small label", anchor="south", role="equation:cht"),
                  Label((8.1, eq_y + 0.1), "the normalisation all three must share", "small label", anchor="south",
                        role="equation:shear")]
    if labels:
        items.append(Label((2.9, eq_y - 1.9), "Common normalisation: both reduce exactly to $\\hat s$ and the CHT "
                           "$\\alpha$ for circular large-aspect-ratio surfaces ($\\psi$ per radian)", "note",
                           anchor="north", role="note"))
    model = {"steps": tuple(k for k, _ in REDUCTION_STEPS), "implementations": dict(IMPLEMENTATIONS),
             "nodes": {k: (b.x, b.y) for k, b in nodes.items()}, "edges": tuple((a, b) for a, b, _ in edges)}
    return Diagram("ballooning_formulation_hierarchy", Scene(tuple(items)), model=model)
