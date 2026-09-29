"""MARFE: a localized radiation condensation on an equilibrium, and the condition that makes it (#1209).

Left, a prescribed MARFE on the equilibrium's edge: a band just inside the
last closed surface, about 30 degrees wide poloidally and a tenth of the minor
radius deep, on the high-field side or next to the X-point -- the location,
size and state Lipschultz (1987) reports. Right, Drake's (1987) constant-
pressure condensation condition in dimensionless form, computed from
``vaft.formula.sol.radiative_condensation_growth_rate``; the constant-density
flute limit from ``radiative_thermal_instability_growth_rate``.

The MARFE is prescribed, not solved: no reaction-diffusion problem, no
cooling curve, no density-limit formula (those belong to #1068). The
diagram explains why a cool, dense, strongly radiating spot forms and where,
and what stabilises it.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from vaft.formula.sol import radiative_condensation_growth_rate, radiative_thermal_instability_growth_rate

from ._chart import Chart, render_chart
from ._render import Diagram
from ._scene import Label, Marker, Polyline, Scene

#: Lipschultz (1987) pp. 15-17: poloidal width ~30 degrees, radial extent ~10 % of the minor radius
POLOIDAL_WIDTH_DEG = 30.0
RADIAL_FRACTION = 0.1
#: drawing scale of the cross-section [cm per m]
_SCALE = 14.0


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _psi_n(eq):
    psi = np.asarray(eq.psi, float)
    return (psi - float(eq.psi_axis)) / (float(eq.psi_boundary) - float(eq.psi_axis))


def _surface(eq, level):
    """The closed flux surface psi_N = level around the magnetic axis (the longest contour through it)."""
    from contourpy import LineType, contour_generator
    from matplotlib.path import Path as _Path

    gen = contour_generator(x=eq.r, y=eq.z, z=_psi_n(eq).T, line_type=LineType.Separate)
    axis = tuple(float(v) for v in eq.magnetic_axis)
    loops = [np.asarray(l) for l in gen.lines(level) if len(l) > 10]
    around = [l for l in loops if _Path(l).contains_point(axis)]
    if not around:
        raise ValueError(f"no closed flux surface psi_N = {level:g} around the axis")
    return max(around, key=len)


def _arc(loop, axis, centre, half):
    """The part of a closed loop within +-half of poloidal angle centre (radians, from the outboard midplane)."""
    theta = np.arctan2(loop[:, 1] - axis[1], loop[:, 0] - axis[0])
    rel = (theta - centre + math.pi) % (2.0 * math.pi) - math.pi
    keep = np.abs(rel) <= half
    pts, rel = loop[keep], rel[keep]
    order = np.argsort(rel)
    return pts[order]


def marfe_region(equilibrium=None, *, localization: str = "hfs", poloidal_width_deg: float = POLOIDAL_WIDTH_DEG,
                 radial_fraction: float = RADIAL_FRACTION) -> dict:
    """The prescribed MARFE band: equilibrium, its LCFS, and the band's outline and centre."""
    from vaft.process.equilibrium import solovev_example

    if localization not in ("hfs", "xpoint"):
        raise ValueError(f"localization must be 'hfs' or 'xpoint', not {localization!r}")
    if not 0.0 < poloidal_width_deg < 180.0:
        raise ValueError("poloidal_width_deg must lie in (0, 180)")
    if not 0.0 < radial_fraction < 0.5:
        raise ValueError("radial_fraction must lie in (0, 0.5)")
    if equilibrium is None:
        equilibrium = solovev_example("limited" if localization == "hfs" else "single_null", a_parameter=0.0)
    eq = equilibrium
    axis = tuple(float(v) for v in eq.magnetic_axis)
    lcfs = _surface(eq, 0.999)
    a_minor = 0.5 * float(np.ptp(lcfs[:, 0]))
    low = lcfs[int(np.argmin(lcfs[:, 1]))]
    if localization == "hfs":
        centre = math.pi
    else:
        centre = math.atan2(low[1] - axis[1], low[0] - axis[0])
    # the inner edge: the surface a radial_fraction of the minor radius inside the LCFS along the band's centre
    ray = np.array([math.cos(centre), math.sin(centre)])
    from scipy.interpolate import RectBivariateSpline
    from scipy.optimize import brentq

    sp = RectBivariateSpline(eq.r, eq.z, _psi_n(eq))
    if localization == "hfs":
        edge = brentq(lambda t: float(sp.ev(axis[0] + t * ray[0], axis[1] + t * ray[1])) - 0.999, 1e-3,
                      2.0 * a_minor)
    else:
        # toward the X-point psi_N peaks at the saddle (the private region lies on the core's side of it), so the
        # edge is the boundary point the ray was drawn through
        edge = float(np.hypot(*(low - np.asarray(axis))))
    t_in = edge - radial_fraction * a_minor
    level_in = float(sp.ev(axis[0] + t_in * ray[0], axis[1] + t_in * ray[1]))
    inner = _surface(eq, level_in)
    half = math.radians(0.5 * poloidal_width_deg)
    outer_arc, inner_arc = _arc(lcfs, axis, centre, half), _arc(inner, axis, centre, half)
    outline = np.concatenate([outer_arc, inner_arc[::-1]])
    return {"equilibrium": eq, "lcfs": lcfs, "axis": axis, "outline": outline, "centre_angle": centre,
            "inner_level": level_in, "minor_radius": a_minor, "localization": localization,
            "poloidal_width_deg": float(poloidal_width_deg), "radial_fraction": float(radial_fraction),
            "centre_point": tuple(np.asarray(axis) + (edge - 0.5 * radial_fraction * a_minor) * ray)}


def condensation_boundary(slopes: np.ndarray) -> np.ndarray:
    """The marginal conduction ratio $k_\\parallel^2\\kappa_\\parallel T/L$ at each radiation slope
    $\\partial\\ln L/\\partial\\ln T$, where ``radiative_condensation_growth_rate`` vanishes."""
    n, T, L = 1e19, 5.0, 1e5  # any positive state: the boundary is dimensionless
    k = 1.0
    out = []
    for x in np.asarray(slopes, float):
        g = lambda y: radiative_condensation_growth_rate(n, T, L, x * L / T, k, y * L / T)
        # gamma is linear in the conduction ratio: two evaluations find its zero
        g0, g1 = g(0.0), g(1.0)
        out.append(g0 / (g0 - g1))
    return np.asarray(out)


def marfe(equilibrium=None, *, localization: str = "hfs", poloidal_width_deg: float = POLOIDAL_WIDTH_DEG,
          radial_fraction: float = RADIAL_FRACTION, labels: bool = True) -> Diagram:
    r"""A MARFE on the edge of an equilibrium, and Drake's condensation condition.

    Left: a prescribed radiating band inside the last closed surface, on the
    high-field side (``localization="hfs"``) or next to the X-point
    (``"xpoint"``); width and depth default to Lipschultz's ~30 degrees and
    ~10 % of $a$. Inside, $T_e$ falls to a few eV and the density rises at
    constant pressure; the band is toroidally symmetric, a ring. Right: the
    plane of radiation slope $\partial\ln L/\partial\ln T$ and conduction
    ratio $k_\parallel^2\kappa_\parallel T/L$, with the condensation boundary
    where ``radiative_condensation_growth_rate`` vanishes and the flute
    boundary where ``radiative_thermal_instability_growth_rate`` does.
    """
    labels = _check_labels(labels)
    m = marfe_region(equilibrium, localization=localization, poloidal_width_deg=poloidal_width_deg,
                     radial_fraction=radial_fraction)
    lcfs, axis = m["lcfs"], np.asarray(m["axis"])
    ox, oy = -_SCALE * float(np.min(lcfs[:, 0])) + 0.6, -_SCALE * float(np.min(lcfs[:, 1])) + 0.4

    def cm(pts):
        pts = np.asarray(pts, dtype=float)
        return np.stack([_SCALE * pts[..., 0] + ox, _SCALE * pts[..., 1] + oy], -1)

    items: List = [Polyline.of(cm(lcfs), "lcfs", role="lcfs", closed=True),
                   Polyline.of(cm(m["outline"]), "layer", role="marfe", closed=True),
                   Marker(tuple(cm(axis)), "o", "opoint", role="magnetic_axis")]
    left_width = float(cm([np.max(lcfs[:, 0]), 0.0])[0]) + 0.6
    if labels:
        c = cm(m["centre_point"])
        side = -1.0 if math.cos(m["centre_angle"]) < 0 else 1.0
        # outside the plasma, on the MARFE's own side, so the label never covers the boundary
        tip = (float(c[0]) + side * 0.9, float(c[1]) - 1.6)
        items += [Polyline.of([tuple(c), tip], "leader line", role="marfe"),
                  Label(tip, "\\begin{tabular}{c}MARFE: $T_e < 10$ eV,\\\\ $n_e$ up at constant $p$,\\\\"
                        " strongly radiating,\\\\ a toroidal ring\\end{tabular}", "small label",
                        anchor="north east" if side < 0 else "north west", role="marfe")]
    # right: the condensation plane
    x = np.linspace(-3.0, 3.0, 241)
    y_marg = condensation_boundary(x)
    chart = Chart(x_range=(-3.0, 3.0), y_range=(0.0, 5.0))
    chart.curves["condensation"] = np.stack([x, y_marg], -1)
    # the flute limit: unstable where dL/dT < 0, whatever the conduction (it has no k_parallel)
    flute_zero = -radiative_thermal_instability_growth_rate(1e19, 0.0)  # = 0: the boundary is dL/dT = 0
    chart.curves["flute"] = np.array([[flute_zero, 0.0], [flute_zero, 5.0]])
    chart.labels.update({"stable": (1.6, 3.6), "unstable": (-1.6, 1.4), "flute": (-1.0, 4.6)})
    chart.parameters.update({"poloidal_width_deg": m["poloidal_width_deg"], "radial_fraction": m["radial_fraction"]})
    scene = render_chart(
        chart, x_label="$\\partial\\ln L/\\partial\\ln T$", y_label="$k_\\parallel^2\\kappa_\\parallel T/L$",
        curve_styles={"flute": "approx", "condensation": "boundary"},
        region_text={"stable": "stable", "unstable": "\\begin{tabular}{c}condensation\\\\(MARFE)\\end{tabular}",
                     "flute": "\\small flute unstable ($k_\\parallel = 0$)"}
        if labels else {},
        x_ticks=[-2.0, 0.0, 2.0], y_ticks=[0.0, 2.0, 4.0],
        note=("Drake: unstable below $2 - \\partial\\ln L/\\partial\\ln T$; dashed: flute limit, unstable left of it"
              if labels else ""),
    )
    items += list(scene.transformed(offset=(left_width + 2.0, 0.4)).items)
    model = {k: v for k, v in m.items() if k != "equilibrium"}
    model["chart"] = chart
    return Diagram("marfe", Scene(tuple(items)), model=model)
