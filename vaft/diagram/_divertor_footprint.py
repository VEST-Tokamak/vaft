"""Divertor heat-flux footprint: an Eich target profile mapped onto a diverted equilibrium (#1209).

The geometry comes from the equilibrium: the X-point, the separatrix legs,
the strike point on a horizontal target, and the total flux expansion from
the outer midplane to the target, $f_x = (\\partial\\psi/\\partial R)_\\mathrm{OMP}
/ (\\partial\\psi/\\partial s)_\\mathrm{target}$, evaluated on the flux itself.
The heat-flux profile is ``vaft.formula.sol.eich_target_heat_flux_profile``
with that $f_x$; nothing of the Eich form is restated here. $\\lambda_q$ and
$S$ are inputs -- a schematic, not an IR measurement (that is ``vaft.plot``).
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from vaft.formula.sol import eich_integral_width, eich_target_heat_flux_profile

from ._chart import Chart, render_chart
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: drawing scale of the divertor region [cm per m]
_SCALE = 50.0


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def footprint_model(equilibrium=None, *, lambda_q: float = 0.004, spreading: float = 0.0015,
                    target: str = "outer", target_z: Optional[float] = None) -> dict:
    """Separatrix geometry, strike point, flux expansion and the Eich profile on a horizontal target.

    ``equilibrium`` is an ``EquilibriumData`` with a lower X-point; ``None`` is
    the single-null Solov'ev of ``solovev_example``. ``target_z`` defaults to
    the lowest point of the limiter outline.
    """
    from scipy.interpolate import RectBivariateSpline
    from scipy.optimize import brentq, minimize

    if target not in ("outer", "inner"):
        raise ValueError(f"target must be 'outer' or 'inner', not {target!r}")
    for name, value in (("lambda_q", lambda_q), ("spreading", spreading)):
        if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be a positive length in metres, not {value!r}")
    if equilibrium is None:
        from vaft.process.equilibrium import solovev_example

        equilibrium = solovev_example("single_null", a_parameter=0.0)
    eq = equilibrium
    sp = RectBivariateSpline(eq.r, eq.z, np.asarray(eq.psi, float) / (2.0 * math.pi))
    R_ax, Z_ax = (float(v) for v in eq.magnetic_axis)
    # X-point: the zero of grad psi next to the lowest point of the boundary
    low = int(np.argmin(np.asarray(eq.lcfs.z)))
    guess = [float(eq.lcfs.r[low]), float(eq.lcfs.z[low])]
    xp = minimize(lambda p: sp.ev(p[0], p[1], dx=1) ** 2 + sp.ev(p[0], p[1], dy=1) ** 2, guess,
                  method="Nelder-Mead", options={"xatol": 1e-11, "fatol": 1e-30}).x
    psi_x = float(sp.ev(*xp))
    psi_ax = float(sp.ev(R_ax, Z_ax))
    span = abs(psi_ax - psi_x)
    a_minor = 0.5 * float(np.ptp(np.asarray(eq.lcfs.r)))
    grad = math.hypot(float(sp.ev(*xp, dx=1)), float(sp.ev(*xp, dy=1)))
    hessian = float(sp.ev(*xp, dx=2)) * float(sp.ev(*xp, dy=2)) - float(sp.ev(*xp, dx=1, dy=1)) ** 2
    psi_b = float(eq.psi_boundary) / (2.0 * math.pi)
    # a saddle (negative Hessian determinant), with a vanishing gradient on the machine's length scale, on the
    # boundary flux, below the axis
    if (xp[1] > Z_ax or grad * a_minor > 1e-4 * span or hessian >= 0.0
            or abs(psi_x - psi_b) > 1e-2 * abs(psi_ax - psi_b)):
        raise ValueError("the equilibrium has no lower X-point on its boundary: a lower-single-null equilibrium "
                         "is needed")
    z_t = float(np.min(eq.limiter.z)) if target_z is None else float(target_z)
    if not z_t < xp[1]:
        raise ValueError("the target must lie below the X-point")
    r_min, r_max = float(np.min(eq.r)), float(np.max(eq.r))
    f_t = lambda R: float(sp.ev(R, z_t)) - psi_x
    # strike points: the separatrix legs cross the target on either side of the X-point
    grid = np.linspace(r_min, r_max, 2001)
    vals = np.array([f_t(R) for R in grid])
    crossings = [brentq(f_t, grid[i], grid[i + 1], xtol=1e-14) for i in np.where(np.diff(np.sign(vals)) != 0)[0]]
    outer = [c for c in crossings if c > xp[0]]
    inner = [c for c in crossings if c < xp[0]]
    if not outer or not inner:
        raise ValueError("the separatrix legs do not both reach the target")
    from matplotlib.path import Path as _Path

    wall = _Path(np.stack([np.asarray(eq.limiter.r), np.asarray(eq.limiter.z)], -1))
    for R in (min(outer), max(inner)):
        # just above the target: the strike point must be on the plate, inside the vessel
        if not wall.contains_point((R, z_t + 1e-4)):
            raise ValueError(f"a separatrix leg reaches z = {z_t:g} m at R = {R:.4f} m, outside the limiter: "
                             "the legs meet a wall before this target")
    R_t = min(outer) if target == "outer" else max(inner)
    # outer midplane separatrix
    r_lcfs = float(np.max(np.asarray(eq.lcfs.r)))
    R_u = brentq(lambda R: float(sp.ev(R, Z_ax)) - psi_x, R_ax + 1e-6, min(r_lcfs + 0.1 * a_minor, r_max - 1e-6),
                 xtol=1e-14)
    dpsi_u = abs(float(sp.ev(R_u, Z_ax, dx=1)))
    dpsi_t = abs(float(sp.ev(R_t, z_t, dx=1)))
    f_x = dpsi_u / dpsi_t
    # target coordinate s into the SOL: outward on the outer target, inward on the inner one
    direction = 1.0 if target == "outer" else -1.0
    s = np.linspace(-4.0 * spreading, 5.0 * lambda_q * f_x + 3.0 * spreading, 801)
    q = np.asarray(eich_target_heat_flux_profile(s, 1.0, lambda_q, spreading, flux_expansion=f_x))
    q_unspread = np.where(s >= 0.0, np.exp(-s / (lambda_q * f_x)), 0.0)
    # SOL flux surfaces one, two and three lambda_q outside the separatrix at the outer midplane
    sol_levels = [float(sp.ev(R_u + k * lambda_q, Z_ax)) for k in (1, 2, 3)]
    return {"equilibrium": eq, "spline": sp, "x_point": (float(xp[0]), float(xp[1])), "psi_x": psi_x,
            "target_z": z_t, "strike_points": {"outer": min(outer), "inner": max(inner)}, "target": target,
            "R_strike": R_t, "R_omp": R_u, "Z_axis": Z_ax, "psi_axis": psi_ax, "flux_expansion": f_x, "direction": direction,
            "lambda_q": float(lambda_q), "spreading": float(spreading), "s": s, "q": q, "q_unspread": q_unspread,
            "lambda_int": float(eich_integral_width(lambda_q, spreading, flux_expansion=f_x)),
            "sol_levels": sol_levels}


def _contour(model, level, window):
    from contourpy import LineType, contour_generator

    eq = model["equilibrium"]
    gen = contour_generator(x=eq.r, y=eq.z, z=(np.asarray(eq.psi, float) / (2.0 * math.pi)).T,
                            line_type=LineType.Separate)
    r0, r1, z0, z1 = window
    runs = []
    for line in gen.lines(level):
        line = np.asarray(line)
        ok = (line[:, 0] >= r0) & (line[:, 0] <= r1) & (line[:, 1] >= z0) & (line[:, 1] <= z1)
        start = None
        for i, good in enumerate(ok):
            if good and start is None:
                start = i
            if (not good or i == len(ok) - 1) and start is not None:
                end = i + 1 if good else i
                if end - start >= 3:
                    runs.append(line[start:end])
                start = None
    return runs


def divertor_heat_footprint(equilibrium=None, *, lambda_q: float = 0.004, spreading: float = 0.0015,
                            target: str = "outer", target_z: Optional[float] = None, labels: bool = True) -> Diagram:
    r"""An Eich heat-flux profile on the divertor target of a diverted equilibrium.

    Left, the divertor region of the equilibrium: separatrix, X-point, a
    horizontal target, the strike point, and the SOL flux surfaces one, two
    and three $\lambda_q$ outside the separatrix at the outer midplane -- fanned
    out by the flux expansion before they reach the target. Right, the
    target profile ``eich_target_heat_flux_profile`` along $s$ with the
    equilibrium's $f_x$: the unspread exponential of width $\lambda_q f_x$
    and its Gaussian spreading $S$ drawn apart. ``lambda_q`` (outer
    midplane) and ``spreading`` (at the target) are illustrative inputs [m].
    """
    labels = _check_labels(labels)
    m = footprint_model(equilibrium, lambda_q=lambda_q, spreading=spreading, target=target, target_z=target_z)
    xp, z_t, R_t, f_x = m["x_point"], m["target_z"], m["R_strike"], m["flux_expansion"]
    inner, outer = m["strike_points"]["inner"], m["strike_points"]["outer"]
    d, lam_fx = m["direction"], m["lambda_q"] * f_x
    pad = 0.06
    # widened on the SOL side of the chosen target, so the footprint and three lambda_q f_x fit
    r0 = min(inner - pad, R_t - 3.6 * lam_fx) if d < 0 else inner - pad
    r1 = max(outer + pad, R_t + 3.6 * lam_fx) if d > 0 else outer + pad
    window = (r0, r1, z_t, xp[1] + 0.12)
    below = 1.6  # cm of drawing under the plate: the footprint and the s axis
    ox, oy = -_SCALE * window[0], -_SCALE * window[2] + below

    def cm(pts):
        pts = np.asarray(pts, dtype=float)
        return np.stack([_SCALE * pts[..., 0] + ox, _SCALE * pts[..., 1] + oy], -1)

    items: List = []
    # the separatrix a hair on the core side of the saddle, relative to the flux range: the core loop and the two
    # legs, without a contour through the X-point cell
    sep_level = m["psi_x"] + 1e-4 * (m["psi_axis"] - m["psi_x"])
    for run in _contour(m, sep_level, window):
        items.append(Polyline.of(cm(run), "separatrix", role="separatrix"))
    for k, level in enumerate(m["sol_levels"], start=1):
        for run in _contour(m, level, window):
            items.append(Polyline.of(cm(run), "surface", role=f"sol_surface_{k}"))
    items.append(Polyline.of(cm([[window[0], z_t], [window[1], z_t]]), "machine", role="target"))
    items.append(Marker(tuple(cm(xp)), "x", "xpoint", role="x_point"))
    items.append(Marker(tuple(cm([R_t, z_t])), "o", "opoint", role="strike_point"))
    # where the SOL surfaces meet the target: ticks at ~ k lambda_q f_x
    from scipy.optimize import brentq

    sp = m["spline"]
    crossings = []
    for k, level in enumerate(m["sol_levels"], start=1):
        f = lambda x, level=level: float(sp.ev(R_t + d * x, z_t)) - level
        crossings.append(brentq(f, 1e-9, 20.0 * k * lam_fx))
        tick = cm([R_t + d * crossings[-1], z_t])
        items.append(Polyline.of([tuple(tick), (float(tick[0]), float(tick[1]) + 0.2)], "tick", role="sol_crossing"))
        if labels and k != 2:  # 2 lambda_q sits between the two labelled ticks; its label would collide
            items.append(Label((float(tick[0]), float(tick[1]) + 0.22), f"${k if k > 1 else ''}\\lambda_q$",
                               "small label", anchor="south", role="sol_crossing"))
    m["sol_crossings"] = crossings
    # the footprint on the target itself: q(s) drawn below the plate, in the tile, 1 cm for the peak
    depth = 1.0 / _SCALE
    prof = np.stack([R_t + d * m["s"], z_t - depth * m["q"] / m["q"].max()], -1)
    shown = (prof[:, 0] >= window[0]) & (prof[:, 0] <= window[1])
    items.append(Polyline.of(cm(prof[shown]), "inner solution", role="footprint"))
    # the target coordinate: s from the strike point into the SOL
    s_start = cm([R_t, z_t])
    s_end = cm([R_t + d * 3.0 * lam_fx, z_t])
    y_axis = float(s_start[1]) - 1.35
    items.append(Arrow((float(s_start[0]), y_axis), (float(s_end[0]), y_axis), "connector", role="target_coordinate"))
    if labels:
        peak = cm(prof[int(np.argmax(m["q"]))])
        items += [Label(tuple(cm(xp) + [-0.15, 0.1]), "X-point", "small label", anchor="south east", role="x_point"),
                  Label(tuple(cm([R_t, z_t]) + [-d * 0.1, 0.12]), "strike point", "small label",
                        anchor="south east" if d > 0 else "south west", role="strike_point"),
                  Label(tuple(cm([R_t - d * 0.03, z_t]) + [0.0, -0.1]), "target", "small label",
                        anchor="north east" if d > 0 else "north west", role="target"),
                  Label((float(peak[0]) + d * 0.2, float(peak[1])), "$q_t(s)$", "small label",
                        anchor="west" if d > 0 else "east", role="footprint"),
                  Label((float(s_end[0]) + d * 0.1, y_axis), "$s$", "small label",
                        anchor="west" if d > 0 else "east", role="target_coordinate")]
    left_width = float(cm([window[1], 0.0])[0])
    # right: the profile along s, from the formula
    s_mm, q = 1e3 * m["s"], m["q"]
    lam_t = 1e3 * m["lambda_q"] * f_x
    s_max = float(np.ceil(4.0 * lam_t / 10.0) * 10.0)
    keep = s_mm <= s_max
    chart = Chart(x_range=(float(s_mm[0]), s_max), y_range=(0.0, 1.15))
    chart.curves.update({"eich": np.stack([s_mm[keep], q[keep]], -1),
                         "unspread": np.stack([s_mm[keep], m["q_unspread"][keep]], -1)})
    chart.parameters.update({"lambda_q": m["lambda_q"], "spreading": m["spreading"], "flux_expansion": f_x,
                             "lambda_int": m["lambda_int"], "R_strike": R_t, "R_omp": m["R_omp"]})
    ticks = [float(v) for v in np.arange(0.0, s_max + 1e-9, 20.0 if s_max > 60 else 10.0)]
    scene = render_chart(
        chart, x_label="$s$ along the target [mm]", y_label="$q_t/q_0$",
        curve_styles={"unspread": "approx", "eich": "inner solution"},
        region_text={}, x_ticks=ticks, x_tick_text=[f"${t:.0f}$" for t in ticks], y_ticks=[0.0, 0.5, 1.0],
        note=(f"$\\lambda_q = {1e3 * lambda_q:g}$ mm at the outer midplane, $f_x = {f_x:.1f}$ from the equilibrium; "
              "left: surfaces $\\lambda_q, 2\\lambda_q, 3\\lambda_q$ outside it there; dashed: no spreading"
              if labels else ""),
    )
    # lambda_q f_x as a dimension arrow at q = 1/e of the unspread profile; S by a leader to the rising edge it sets
    e_level = math.exp(-1.0)
    s_cm = [tuple(chart.to_cm(np.array([x, y]))) for x, y in ((0.0, e_level), (lam_t, e_level))]
    extra: List = [Arrow(s_cm[0], s_cm[1], "connector", role="lambda_q_f_x", both=True)]
    rise = int(np.argmax(q[: int(np.argmax(q)) + 1] >= 0.5))
    edge = tuple(chart.to_cm(np.array([s_mm[rise], q[rise]])))
    note_at = tuple(chart.to_cm(np.array([0.45 * lam_t, 1.05])))
    extra.append(Polyline.of([note_at, edge], "leader line", role="spreading"))
    if labels:
        extra += [Label((s_cm[1][0] + 0.1, s_cm[1][1]), f"$\\lambda_q f_x = {lam_t:.1f}$ mm", "small label",
                        anchor="west", role="lambda_q_f_x"),
                  Label((note_at[0] + 0.05, note_at[1]), f"rise set by $S = {1e3 * spreading:g}$ mm", "small label",
                        anchor="south west", role="spreading")]
    scene = Scene(tuple(scene.items) + tuple(extra))
    items += list(scene.transformed(offset=(left_width + 2.2, 0.3)).items)
    return Diagram("divertor_heat_footprint", Scene(tuple(items)),
                   model={k: v for k, v in m.items() if k not in ("equilibrium", "spline")})


__all__ = ["divertor_heat_footprint", "footprint_model"]
