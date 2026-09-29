"""MARFE: a localized radiation condensation on an equilibrium, and the condition that makes it (#1209).

Left, a prescribed MARFE on the equilibrium's edge. On the high-field side
(the usual location, Lipschultz 1987) the band straddles the last closed
surface, about 30 degrees wide poloidally and a tenth of the minor radius
deep, as Lipschultz reports; next to the X-point (Greenwald 2002, p. R35) it
is drawn on the closed side above the X-point with the same, borrowed,
sizes. Right, Drake's (1987) constant-pressure condensation condition in
dimensionless form, computed from
``vaft.formula.sol.radiative_condensation_growth_rate``, with the
constant-density flute limit of ``radiative_thermal_instability_growth_rate``
on its $k_\\parallel = 0$ axis.

The MARFE is prescribed, not solved: no reaction-diffusion problem, no
cooling curve, no density-limit formula (those belong to #1068).
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


def _arc(eq, level, axis, centre, half, near):
    """The piece of the psi_N = level contour within +-half of poloidal angle centre, closest to ``near``.

    Works on closed surfaces and on open ones outside the separatrix alike."""
    from contourpy import LineType, contour_generator

    gen = contour_generator(x=eq.r, y=eq.z, z=_psi_n(eq).T, line_type=LineType.Separate)
    best, best_d = None, np.inf
    for line in gen.lines(level):
        line = np.asarray(line)
        theta = np.arctan2(line[:, 1] - axis[1], line[:, 0] - axis[0])
        rel = (theta - centre + math.pi) % (2.0 * math.pi) - math.pi
        keep = np.abs(rel) <= half
        if keep.sum() < 3:
            continue
        pts, rel = line[keep], rel[keep]
        d = float(np.min(np.hypot(pts[:, 0] - near[0], pts[:, 1] - near[1])))
        if d < best_d:
            best, best_d = pts[np.argsort(rel)], d
    if best is None:
        raise ValueError(f"no flux surface psi_N = {level:.4f} in the band's angular window")
    return best


def marfe_region(equilibrium=None, *, localization: str = "hfs", poloidal_width_deg: float = POLOIDAL_WIDTH_DEG,
                 radial_fraction: float = RADIAL_FRACTION) -> dict:
    """The prescribed MARFE band: equilibrium, its LCFS, and the band's outline and centre."""
    from scipy.interpolate import RectBivariateSpline
    from scipy.optimize import brentq

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
    sp = RectBivariateSpline(eq.r, eq.z, _psi_n(eq))
    if localization == "hfs":
        centre = math.pi
        ray = np.array([math.cos(centre), math.sin(centre)])
        edge = brentq(lambda t: float(sp.ev(axis[0] + t * ray[0], axis[1] + t * ray[1])) - 1.0, 1e-3,
                      2.0 * a_minor)
        # straddles r = a (Lipschultz p. 16): half the depth inside, half outside
        t_in, t_out = edge - 0.5 * radial_fraction * a_minor, edge + 0.5 * radial_fraction * a_minor
    else:
        centre = math.atan2(low[1] - axis[1], low[0] - axis[0])
        ray = np.array([math.cos(centre), math.sin(centre)])
        # toward the X-point psi_N peaks at the saddle (the private region lies on the core's side of it), so the
        # band is drawn on the closed side, ending at the boundary point the ray was drawn through
        edge = float(np.hypot(*(low - np.asarray(axis))))
        t_in, t_out = edge - radial_fraction * a_minor, None
    point = lambda t: (axis[0] + t * ray[0], axis[1] + t * ray[1])
    half = math.radians(0.5 * poloidal_width_deg)
    level_in = float(sp.ev(*point(t_in)))
    inner_arc = _arc(eq, level_in, axis, centre, half, point(t_in))
    if t_out is None:
        level_out, outer_arc = 0.999, _arc(eq, 0.999, axis, centre, half, point(edge))
    else:
        level_out = float(sp.ev(*point(t_out)))
        outer_arc = _arc(eq, level_out, axis, centre, half, point(t_out))
    outline = np.concatenate([outer_arc, inner_arc[::-1]])
    t_mid = 0.5 * (t_in + (t_out if t_out is not None else edge))
    return {"equilibrium": eq, "lcfs": lcfs, "axis": axis, "outline": outline, "centre_angle": centre,
            "inner_level": level_in, "outer_level": level_out, "minor_radius": a_minor,
            "localization": localization, "poloidal_width_deg": float(poloidal_width_deg),
            "radial_fraction": float(radial_fraction), "centre_point": point(t_mid), "x_point_side": tuple(low)}


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

    Left: a prescribed radiating band. On the high-field side
    (``localization="hfs"``) it straddles the last closed surface, ~30
    degrees wide and ~10 % of $a$ deep (Lipschultz 1987); next to the X-point
    (``"xpoint"``, Greenwald 2002 p. R35) it sits on the closed side with the
    same borrowed sizes. Its $T_e$ is below ~10 eV (Greenwald p. R34); the
    density rise at constant pressure is Drake's model (his Eq. 17); it is
    toroidally symmetric, a ring. Right: the plane of radiation slope
    $\partial\ln L/\partial\ln T$ and conduction ratio
    $k_\parallel^2\kappa_\parallel T/L$, with the condensation boundary where
    ``radiative_condensation_growth_rate`` vanishes; on the $k_\parallel = 0$
    axis, where the flute limit applies, the segment on which
    ``radiative_thermal_instability_growth_rate`` is positive.
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
                   Polyline.of(cm(m["outline"]), "inner solution", role="marfe_edge", closed=True),
                   Marker(tuple(cm(axis)), "o", "opoint", role="magnetic_axis")]
    left_width = float(cm([np.max(lcfs[:, 0]), 0.0])[0]) + 0.6
    if labels:
        c = cm(m["centre_point"])
        side = -1.0 if math.cos(m["centre_angle"]) < 0 else 1.0
        # outside the plasma, on the MARFE's own side, so the label never covers the boundary
        tip = (float(c[0]) + side * 0.9, float(c[1]) - 1.6)
        name = "MARFE" if localization == "hfs" else "X-point MARFE"
        items += [Polyline.of([tuple(c), tip], "leader line", role="marfe"),
                  Label(tip, f"\\begin{{tabular}}{{c}}{name}: $T_e < 10$ eV,\\\\ strongly radiating,\\\\"
                        " a toroidal ring\\end{tabular}", "small label",
                        anchor="north east" if side < 0 else "north west", role="marfe")]
    # right: the condensation plane (k_parallel > 0) and the flute limit on its k_parallel = 0 axis
    x = np.linspace(-3.0, 3.0, 241)
    y_marg = condensation_boundary(x)
    chart = Chart(x_range=(-3.0, 3.0), y_range=(0.0, 5.0))
    chart.curves["condensation"] = np.stack([x, y_marg], -1)
    # the flute growth rate at slope x (any L/T > 0): unstable where it is positive
    flute_growth = np.array([radiative_thermal_instability_growth_rate(1e19, xi * 1e5 / 5.0) for xi in x])
    # drawn just above the k_parallel = 0 axis so the axis line does not hide it
    chart.curves["flute"] = np.stack([x[flute_growth > 0], np.full(int(np.sum(flute_growth > 0)), 0.1)], -1)
    chart.labels.update({"stable": (1.6, 3.6), "unstable": (-1.6, 1.6), "flute": (-1.5, 0.42)})
    chart.parameters.update({"poloidal_width_deg": m["poloidal_width_deg"], "radial_fraction": m["radial_fraction"]})
    scene = render_chart(
        chart, x_label="$\\partial\\ln L/\\partial\\ln T$", y_label="$k_\\parallel^2\\kappa_\\parallel T/L$",
        curve_styles={"condensation": "boundary", "flute": "inner solution"},
        region_text={"stable": "stable", "unstable": "\\begin{tabular}{c}condensation\\\\(MARFE)\\end{tabular}",
                     "flute": "\\small flute, $k_\\parallel = 0$"} if labels else {},
        x_ticks=[-2.0, 0.0, 2.0], y_ticks=[0.0, 2.0, 4.0],
        note=("Drake: for $k_\\parallel > 0$ unstable below $2 - \\partial\\ln L/\\partial\\ln T$; red on the axis: "
              "the flute ($k_\\parallel = 0$) limit, unstable where $L$ falls with $T$" if labels else ""),
    )
    items += list(scene.transformed(offset=(left_width + 2.0, 0.4)).items)
    model = {k: v for k, v in m.items() if k != "equilibrium"}
    model["chart"] = chart
    return Diagram("marfe", Scene(tuple(items)), model=model)
