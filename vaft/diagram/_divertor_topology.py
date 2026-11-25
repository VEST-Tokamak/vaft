"""Divertor topology and upper/lower separatrix balance.

Five diagrams for the signed outboard-midplane separatrix separation
$dR_\\mathrm{sep} = R_\\mathrm{out}(\\psi_{X,\\mathrm{lower}}) - R_\\mathrm{out}(\\psi_{X,\\mathrm{upper}})$
at the magnetic-axis height (negative for a lower single null, zero for a
balanced double null, positive for an upper single null):

``divertor_topology_classification``
    LIMITED, LSN, balanced DN, USN and near-DN side by side.
``secondary_separatrix_balance``
    primary and secondary separatrices in the R-Z plane, in flux and along
    the outboard midplane, for both signs and for DN.
``sol_magnetic_connectivity``
    which targets each open field line reaches, traced against a wall.
``separatrix_balance_transport_scale``
    $dR_\\mathrm{sep}$ against $\\lambda_q$; magnetic balance is not power balance.
``separatrix_balance_calculation``
    the computation as a procedure, with the values it gives on one case.

Every equilibrium is the analytic family
$\\psi = (R - 1)^2 + Z^2 - Z^4/a^2 + s Z^3$ ($a = 0.65$) of
``test/test_separatrix_balance.py``: an O-point at $(1, 0)$ and two saddles
on $R = 1$ that the skew $s$ biases ($s > 0$ lowers the lower saddle's flux,
a lower single null). It is an illustrative flux map, not a VEST
reconstruction. The topology, the X-points with their ``role`` and ``branch``,
and $dR_\\mathrm{sep}$ with its quality all come from
``vaft.process.equilibrium.derive_boundary_representation``; the diagram
code searches the flux for no saddle. (The family's closed-form saddle flux
only sets the boundary flux handed to the process, as an equilibrium solver
would.) The diagrams draw the process output and contour the flux. Open
field lines are followed along their flux contour (the poloidal projection of
an axisymmetric field line) until they meet the wall.

This is axisymmetric balance. The non-axisymmetric separatrix splitting of
``separatrix_lobes`` (``_magnetic_topology``) is a different phenomenon.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vaft.formula.equilibrium import separatrix_radial_separation
from vaft.formula.sol import secondary_separatrix_exponential_factor, separatrix_imbalance_to_heat_flux_width

from ._chart import Chart, render_chart
from ._concept import box, connector
from ._equations import formula_equation
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: half-height parameter a of the analytic family [m]
FAMILY_A = 0.65
#: the vessel: an octagon; its edges at |Z| >= 0.45 m (floor, roof and their chamfers) are the divertor targets [m]
WALL: Tuple[Tuple[float, float], ...] = (
    (0.60, -0.45), (0.75, -0.60), (1.40, -0.60), (1.45, -0.50),
    (1.45, 0.50), (1.40, 0.60), (0.75, 0.60), (0.60, 0.45),
)
_DIVERTOR_Z = 0.45
#: the largest skew the family keeps both saddles inside the vessel for
_MAX_SKEW = 0.2
#: the diverted LCFS contour sits this fraction of the boundary flux inside the saddle flux (a tie guard)
_LCFS_INSET = 1e-6

#: line styles of the connectivity classes (inline TikZ options, no template change)
_CLASS_STYLE = {
    "primary_lower": "driftred,line width=1.1pt",
    "primary_upper": "absequil,line width=1.1pt",
    "common_outboard": "islandblue,line width=1.1pt",
    "common_inboard": "abskinetic,line width=1.1pt",
    "private_lower": "black!55,densely dashed,line width=0.9pt",
    "private_upper": "black!55,densely dashed,line width=0.9pt",
    "other": "black,dotted,line width=0.9pt",
}
_CLASS_TEXT = {
    "primary_lower": "primary SOL: both ends on the lower divertor",
    "primary_upper": "primary SOL: both ends on the upper divertor",
    "common_outboard": "common SOL, outboard: lower outer to upper outer",
    "common_inboard": "common SOL, inboard: lower inner to upper inner",
    "private_lower": "private flux: inner to outer target, beyond an X-point",
    "other": "any other connection (main-chamber wall, or both ends on one target)",
}
_TOPOLOGY_TEXT = {
    "limited": "LIMITED", "lower_single_null": "LSN", "upper_single_null": "USN",
    "double_null": "DN", "diverted": "diverted", "ambiguous": "ambiguous",
}
#: a contour run within this distance of an X-point belongs to its separatrix [m]
_NEAR_X = 0.03
_BLACK_PRIMARY = "black,line width=1.3pt"
_BLACK_SECONDARY = "black,densely dashed,line width=1.0pt"
_INACTIVE_X = "draw=black!45,line width=0.8pt"
_SECONDARY = "separatrix,densely dashed"
_EXTERNAL = "black!45,densely dashed,line width=0.7pt"


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _check_skew(skew, name: str = "skew") -> float:
    if isinstance(skew, bool) or not (isinstance(skew, (int, float)) and math.isfinite(skew)
                                      and abs(skew) <= _MAX_SKEW):
        raise ValueError(f"{name} must be a finite number with |{name}| <= {_MAX_SKEW}, not {skew!r}")
    return float(skew)


# --------------------------------------------------------------------------
# the analytic family and its process output


def family_flux(R, Z, skew: float):
    """$\\psi = (R - 1)^2 + Z^2 - Z^4/a^2 + s Z^3$, the flux of the analytic family [arbitrary]."""
    R, Z = np.asarray(R, dtype=float), np.asarray(Z, dtype=float)
    return (R - 1.0) ** 2 + Z**2 - Z**4 / FAMILY_A**2 + skew * Z**3


def _family_saddles(skew: float) -> List[Tuple[float, float]]:
    """The family's saddles as (Z, psi) on R = 1, from its definition: the roots of
    $\\partial_Z\\psi/Z = 2 - 4Z^2/a^2 + 3sZ = 0$."""
    roots = sorted(np.roots([4.0 / FAMILY_A**2, -3.0 * skew, -2.0]).real)
    return [(float(z), float(family_flux(1.0, z, skew))) for z in roots]


def _family_boundary_flux(skew: float) -> float:
    """The flux of the lower-flux saddle, where the family's plasma ends when nothing limits it.

    The family is defined with its saddles on $R = 1$ at the roots of
    $\\partial_Z\\psi/Z = 2 - 4Z^2/a^2 + 3sZ = 0$; this only sets the
    boundary flux an equilibrium solver would hand over. The X-points the
    diagrams draw are the ones ``derive_boundary_representation`` finds.
    """
    return min(psi for _, psi in _family_saddles(skew))


def analytic_balance_equilibrium(skew: float = 0.0, *, limiter_tip: Optional[float] = None, n: int = 201):
    """An ``EquilibriumData`` of the analytic family inside :data:`WALL`.

    ``skew`` biases the saddles: positive is lower single null, negative upper
    single null, zero a balanced double null. ``limiter_tip`` [m] adds an
    inboard limiter tip at the midplane whose flux surface becomes the LCFS
    (a limited plasma with two external nulls). ``n`` is the radial grid size.
    """
    from vaft.data.equilibrium import Contour, EquilibriumConvention, EquilibriumData
    from vaft.process.equilibrium import extract_flux_surface_contours

    skew = _check_skew(skew)
    if not (isinstance(n, int) and n >= 33):
        raise ValueError(f"n must be an integer of at least 33, not {n!r}")
    r = np.linspace(0.5, 1.5, n)
    z = np.linspace(-0.8, 0.8, n + 40)
    psi = family_flux(*np.meshgrid(r, z, indexing="ij"), skew)
    wall = list(WALL)
    boundary = _family_boundary_flux(skew)
    if limiter_tip is not None:
        tip = float(limiter_tip)
        if not (math.isfinite(tip) and WALL[0][0] < tip < 1.0 and family_flux(tip, 0.0, skew) < boundary):
            raise ValueError(f"limiter_tip must lie between the inboard wall (R = {WALL[0][0]} m) and the "
                             f"separatrix, inboard of the magnetic axis (R = 1 m), not {limiter_tip!r}")
        wall += [(WALL[0][0], 0.08), (tip, 0.0), (WALL[0][0], -0.08)]
        boundary = float(family_flux(tip, 0.0, skew))
    # the LCFS handed over is the closed surface. Limited: the limiter's own flux surface. Diverted: the
    # separatrix contour (a hair inside the saddle flux, so platform last bits cannot decide it) comes back
    # leg-core-leg through the saddle cell; its part between the bounding saddles' heights is the LCFS
    level = boundary if limiter_tip is not None else boundary * (1.0 - _LCFS_INSET)
    lines = extract_flux_surface_contours(psi, r, z, 0.0, level, [1.0])[1.0]
    if limiter_tip is None:
        rb, zb = _diverted_lcfs(lines, skew, boundary)
    else:
        closed = [(rb, zb) for rb, zb in lines if math.hypot(rb[0] - rb[-1], zb[0] - zb[-1]) < 1e-9]
        if not closed:
            raise ValueError("the limiter's flux surface does not close on this grid")
        rb, zb = max(closed, key=lambda pair: pair[0].size)
    w = np.asarray(wall, dtype=float)
    return EquilibriumData(
        r=r, z=z, psi=psi, psi_axis=0.0, psi_boundary=boundary, magnetic_axis=(1.0, 0.0), lcfs=Contour(rb, zb),
        limiter=Contour(w[:, 0], w[:, 1]),
        convention=EquilibriumConvention(1, (1,), True, False, 1, 1, 1, "analytic separatrix-balance family"))


def _diverted_lcfs(lines, skew: float, boundary: float) -> Tuple[np.ndarray, np.ndarray]:
    """The closed part of the separatrix contour ``lines``: between the heights of the saddles on the boundary flux.

    Only pieces that pass a bounding saddle count -- the same flux level also
    lies in the other divertor's private flux. A double null gives two halves,
    joined into one loop through both saddles.
    """
    bounding = [(1.0, z_s) for z_s, psi_s in _family_saddles(skew) if psi_s <= boundary * (1.0 + 1e-12)]
    z_low = max([z for _, z in bounding if z < 0.0], default=-np.inf)
    z_high = min([z for _, z in bounding if z > 0.0], default=np.inf)
    pieces = []
    for rb, zb in lines:
        for piece, core in _split_at_x_points(np.stack([rb, zb], -1), (z_low, z_high)):
            near = min(float(np.min(np.hypot(piece[:, 0] - r_s, piece[:, 1] - z_s))) for r_s, z_s in bounding)
            if core and near < _NEAR_X:
                pieces.append(piece)
    if not pieces:
        raise ValueError("the separatrix contour has no closed part on this grid")
    loop = pieces[0]
    for piece in sorted(pieces[1:], key=lambda p: float(np.mean(p[:, 0]))):
        flip = np.hypot(*(piece[-1] - loop[-1])) < np.hypot(*(piece[0] - loop[-1]))
        loop = np.concatenate([loop, piece[::-1] if flip else piece])
    loop = np.concatenate([loop, loop[:1]])
    return loop[:, 0], loop[:, 1]


def balance_state(skew: float = 0.0, *, limiter_tip: Optional[float] = None, n: int = 201) -> dict:
    """The equilibrium and its canonical boundary representation, with what the diagrams read off it."""
    from scipy.interpolate import RectBivariateSpline

    from vaft.process.equilibrium import derive_boundary_representation

    eq = analytic_balance_equilibrium(skew, limiter_tip=limiter_tip, n=n)
    rep = derive_boundary_representation(eq)
    return {"skew": float(skew), "equilibrium": eq, "representation": rep, "topology": rep.topology.value,
            "spline": RectBivariateSpline(eq.r, eq.z, np.asarray(eq.psi, dtype=float)),
            "wall": np.asarray(np.stack([eq.limiter.r, eq.limiter.z], -1), dtype=float),
            "d_r_sep": rep.d_r_sep.value, "available": bool(rep.d_r_sep.available),
            "quality": dict(rep.d_r_sep.quality or {}), "reason": rep.d_r_sep.reason}


# --------------------------------------------------------------------------
# geometry: contours clipped at the wall, and field lines followed to it


def _wall_path(wall):
    from matplotlib.path import Path as _Path

    return _Path(np.asarray(wall, dtype=float))


def _segment_hit(p, q, wall) -> Tuple[np.ndarray, int]:
    """Where the segment ``p`` (inside) to ``q`` (outside) crosses the wall, and the wall edge it crosses."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    best = (2.0, None, -1)
    n = len(wall)
    for k in range(n):
        a, b = np.asarray(wall[k], float), np.asarray(wall[(k + 1) % n], float)
        d, e = q - p, b - a
        den = d[0] * e[1] - d[1] * e[0]
        if den == 0.0:
            continue
        w = a - p
        t = (w[0] * e[1] - w[1] * e[0]) / den
        u = (w[0] * d[1] - w[1] * d[0]) / den
        if -1e-12 <= t <= 1.0 + 1e-12 and -1e-12 <= u <= 1.0 + 1e-12 and t < best[0]:
            best = (t, p + t * d, k)
    if best[1] is None:
        raise ValueError(f"the segment {tuple(p)} -> {tuple(q)} does not cross the wall")
    return best[1], best[2]


def _contour_lines(eq, level: float) -> List[np.ndarray]:
    from contourpy import LineType, contour_generator

    gen = contour_generator(x=eq.r, y=eq.z, z=np.asarray(eq.psi, dtype=float).T, line_type=LineType.Separate)
    return [np.asarray(line, dtype=float) for line in gen.lines(level)]


def _inside_runs(eq, level: float, wall) -> List[np.ndarray]:
    """The parts of the flux contour ``level`` inside the wall, each ended exactly on the wall."""
    path = _wall_path(wall)
    runs = []
    for line in _contour_lines(eq, level):
        inside = path.contains_points(line)
        start = None
        for i in range(len(line) + 1):
            ok = i < len(line) and inside[i]
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                run = list(line[start:i])
                if start > 0:
                    run.insert(0, _segment_hit(line[start], line[start - 1], wall)[0])
                if i < len(line):
                    run.append(_segment_hit(line[i - 1], line[i], wall)[0])
                if len(run) >= 3:
                    runs.append(np.asarray(run))
                start = None
    return runs


def _target_name(point, edge: int, state) -> str:
    """``"lower outer target"`` and the like; a hit on the main-chamber wall (|Z| < 0.45 m) is ``"lower wall"``."""
    z_axis = float(state["equilibrium"].magnetic_axis[1]) if state["equilibrium"].magnetic_axis is not None else 0.0
    side = "lower" if point[1] < z_axis else "upper"
    xp = state["representation"].separatrix_x_points.get(side)
    r_split = xp.r if xp is not None else float(state["equilibrium"].magnetic_axis[0])
    wall = state["wall"]
    a, b = wall[edge], wall[(edge + 1) % len(wall)]
    if min(abs(a[1]), abs(b[1])) < _DIVERTOR_Z - 1e-12:
        return f"{side} wall"
    return f"{side} {'inner' if point[0] < r_split else 'outer'} target"


def trace_field_line(state: dict, seed) -> dict:
    """Follow the axisymmetric field line through ``seed`` [m] in both directions to the wall.

    In axisymmetry a field line stays on its flux surface, so its poloidal
    projection is the flux contour through ``seed``. Returns the path between
    its two wall hits (``closed`` when the contour never leaves the vessel),
    the targets it reaches, and whether it crosses the magnetic-axis height.
    """
    eq, wall = state["equilibrium"], state["wall"]
    seed = np.asarray(seed, dtype=float)
    path = _wall_path(wall)
    if not path.contains_point(seed):
        raise ValueError(f"the seed {tuple(seed)} lies outside the wall")
    level = float(state["spline"].ev(*seed))
    lines = _contour_lines(eq, level)
    distance = [float(np.min(np.hypot(*(line - seed).T))) for line in lines]
    line = lines[int(np.argmin(distance))]
    i0 = int(np.argmin(np.hypot(*(line - seed).T)))
    closed = bool(np.allclose(line[0], line[-1]))
    if closed:
        ring = line[:-1]
        line = np.concatenate([ring[i0:], ring[:i0 + 1]])
        i0 = 0

    def walk(points):
        out = [points[0]]
        for k in range(1, len(points)):
            if not path.contains_point(points[k]):
                hit, edge = _segment_hit(points[k - 1], points[k], wall)
                out.append(hit)
                return np.asarray(out), edge
            out.append(points[k])
        return np.asarray(out), None

    if closed:
        forward, f_edge = walk(line)
        if f_edge is None:
            return {"seed": tuple(seed), "level": level, "closed": True, "path": forward, "ends": None,
                    "targets": None, "crosses_axis_height": True}
        backward, b_edge = walk(line[::-1])
    else:
        forward, f_edge = walk(line[i0:])
        backward, b_edge = walk(line[: i0 + 1][::-1])
    if f_edge is None or b_edge is None:
        raise ValueError(f"the field line through {tuple(seed)} leaves the flux grid before it reaches the wall")
    full = np.concatenate([backward[::-1], forward[1:]])
    ends = (tuple(full[0]), tuple(full[-1]))
    targets = (_target_name(full[0], b_edge, state), _target_name(full[-1], f_edge, state))
    z_axis = float(state["equilibrium"].magnetic_axis[1])
    return {"seed": tuple(seed), "level": level, "closed": False, "path": full, "ends": ends,
            "targets": targets,
            "crosses_axis_height": bool(np.any(full[:, 1] > z_axis) and np.any(full[:, 1] < z_axis))}


def connection_class(trace: dict) -> str:
    """The connectivity class of a traced field line, from the targets it reaches."""
    if trace["closed"]:
        return "closed"
    a, b = sorted(trace["targets"])
    if a.endswith("target") and b.endswith("target"):
        sa, sb = a.split()[0], b.split()[0]
        if sa == sb and {a.split()[1], b.split()[1]} == {"inner", "outer"}:
            return f"{'primary' if trace['crosses_axis_height'] else 'private'}_{sa}"
        if sa != sb and a.split()[1] == b.split()[1] == "outer":
            return "common_outboard"
        if sa != sb and a.split()[1] == b.split()[1] == "inner":
            return "common_inboard"
    return "other"


def _midplane_crossing(state, level: float, side: str) -> float:
    """The radius where the flux ``level`` crosses the magnetic-axis height, inboard or outboard of the axis."""
    from scipy.optimize import brentq

    sp = state["spline"]
    r_ax, z_ax = (float(v) for v in state["equilibrium"].magnetic_axis)
    wall_r = state["wall"][:, 0]
    edge = float(np.min(wall_r)) if side == "inboard" else float(np.max(wall_r))

    def f(R):
        return float(sp.ev(R, z_ax)) - level

    return float(brentq(f, r_ax, edge, xtol=1e-12) if side == "outboard" else brentq(f, edge, r_ax, xtol=1e-12))


# --------------------------------------------------------------------------
# drawing an equilibrium panel


def _x_points(state) -> Dict[str, object]:
    """The separatrix X-points the process selected, by branch."""
    return dict(state["representation"].separatrix_x_points)


def _separatrix_level(point, state) -> float:
    """Just on the core side of the saddle flux: a separatrix contoured exactly at a saddle value is decided by
    last-bit differences between platforms."""
    psi_axis = float(state["equilibrium"].psi_axis)
    return float(point.psi) + 1e-6 * (psi_axis - float(point.psi))


def _separatrix_runs(state, point) -> List[np.ndarray]:
    """The contour of ``point``'s flux inside the wall, only the runs through the X-point itself.

    The same flux level also appears away from the separatrix (inside the other
    divertor's private flux, say); those pieces are not this X-point's
    separatrix and are left out.
    """
    runs = _inside_runs(state["equilibrium"], _separatrix_level(point, state), state["wall"])
    x = np.array([point.r, point.z])
    return [run for run in runs if float(np.min(np.hypot(*(run - x).T))) < _NEAR_X]


def _core_bounds(state) -> Tuple[float, float]:
    """The heights of the active X-points below and above the axis: the confined region lies between."""
    z_axis = float(state["equilibrium"].magnetic_axis[1])
    low, high = -np.inf, np.inf
    for point in state["representation"].x_points:
        if point.active and point.z < z_axis:
            low = max(low, point.z)
        elif point.active:
            high = min(high, point.z)
    return low, high


def _split_at_x_points(run, bounds) -> List[Tuple[np.ndarray, bool]]:
    """``run`` cut where it crosses the active X-point heights: (piece, inside the confined height range).

    The separatrix contour can come back from the grid as one line through
    the saddle cell -- leg, closed part, leg -- or as separate lines; cutting
    by height gives the same pieces either way.
    """
    inside = (run[:, 1] >= bounds[0]) & (run[:, 1] <= bounds[1])
    pieces, start = [], 0
    for i in range(1, len(run) + 1):
        if i == len(run) or inside[i] != inside[start]:
            piece = run[start:min(i + 1, len(run))]
            if len(piece) >= 2:
                pieces.append((piece, bool(inside[start])))
            start = i
    return pieces


def _panel_items(state, cm, *, lcfs: bool = True, secondary: bool = True, primary_style: str = "separatrix",
                 secondary_style: str = _SECONDARY) -> List:
    """Wall, LCFS, the separatrix of each selected X-point, every saddle and the axis.

    With ``lcfs`` the closed surface is the equilibrium's LCFS contour (on the
    limiter, or a hair inside the primary separatrix). Without it, the closed
    part of the primary separatrix is drawn in the primary style instead: the
    part between the active X-point heights of the X-point whose flux bounds
    the plasma -- in a double null the grid cannot resolve, the other active
    X-point contributes its legs only.
    """
    eq, rep = state["equilibrium"], state["representation"]
    items: List = [Polyline.of(cm(state["wall"]), "machine", role="wall", closed=True)]
    selected = _x_points(state)
    bounds = _core_bounds(state)
    psi_axis = float(eq.psi_axis)
    active = sorted((p for p in selected.values() if p.active), key=lambda p: (abs(p.psi - psi_axis), p.z))
    bounding = active[0] if active else None
    core_pieces: Dict[Tuple, np.ndarray] = {}
    for branch in ("lower", "upper"):
        point = selected.get(branch)
        if point is None or (not point.active and not secondary):
            continue
        if point.active:
            style, role = primary_style, "primary_separatrix"
        elif rep.topology.is_diverted:
            style, role = secondary_style, "secondary_separatrix"
        else:
            style, role = _EXTERNAL, "external_separatrix"
        x = np.array([point.r, point.z])
        for run in _separatrix_runs(state, point):
            if not point.active:
                items.append(Polyline.of(cm(run), style, role=role))
                continue
            for piece, core in _split_at_x_points(run, bounds):
                if core:
                    if point is bounding:
                        core_pieces[tuple(np.round(np.concatenate([piece[0], piece[-1]]), 4))] = piece
                elif float(np.min(np.hypot(*(piece - x).T))) < _NEAR_X:
                    # this X-point's own legs; another active X-point draws its own
                    items.append(Polyline.of(cm(piece), style, role=role))
    if lcfs:
        items.append(Polyline.of(cm(np.stack([eq.lcfs.r, eq.lcfs.z], -1)[:-1]), "lcfs", role="lcfs", closed=True))
    else:
        # one piece that closes through its X-point (single null), or halves that meet at both (double null)
        pieces = [core_pieces[k] for k in sorted(core_pieces)]
        for piece in pieces:
            closes = len(pieces) == 1 and float(np.hypot(*(piece[0] - piece[-1]))) < 2 * _NEAR_X
            items.append(Polyline.of(cm(piece), primary_style, role="primary_separatrix", closed=closes))
    # in drawing order by position: the process orders saddles by flux, which ties in a balanced double null
    for point in sorted(rep.x_points, key=lambda p: (round(p.z, 6), round(p.r, 6))):
        style, role = ("xpoint", "active_x_point") if point.active else (_INACTIVE_X, "inactive_x_point")
        items.append(Marker(tuple(cm([point.r, point.z])), "x", style, role=role))
    items.append(Marker(tuple(cm(state["equilibrium"].magnetic_axis)), "o", "fill=black", role="magnetic_axis"))
    return items


#: region fills, light enough to carry the field lines drawn over them (inline TikZ options)
REGION_FILL = {
    "common_sol": "fill=islandblue!9,draw=none",
    "limiter_sol": "fill=islandblue!9,draw=none",
    "primary_sol_lower": "fill=driftred!16,draw=none",
    "primary_sol_upper": "fill=absequil!20,draw=none",
    "core": "fill=qblue!35,draw=none",
    "private": "fill=black!11,draw=none",
}


def _filled_below(state, level: float, keep) -> List[np.ndarray]:
    """Outer boundaries of the regions where psi < ``level`` and ``keep`` (a node mask) holds."""
    from contourpy import FillType, contour_generator

    eq = state["equilibrium"]
    psi = np.asarray(eq.psi, dtype=float)
    z = np.ma.masked_array(psi.T, mask=~keep.T)
    gen = contour_generator(x=eq.r, y=eq.z, z=z, fill_type=FillType.OuterCode)
    polygons, codes = gen.filled(float(np.min(psi)) - 1.0, level)
    outers = []
    for points, code in zip(polygons, codes):
        starts = np.flatnonzero(code == 1)
        end = starts[1] if len(starts) > 1 else len(points)
        if end >= 4:
            outers.append(np.asarray(points[:end]))
    return outers


def region_fills(state, cm) -> List:
    """Filled regions in drawing order, each later one painted over the earlier.

    Diverted: the vessel as common SOL; inside the secondary separatrix, on the
    primary X-point's side of the secondary one, the primary SOL; the core
    (the LCFS); and beyond each separatrix X-point, below its own flux, the
    private flux. Limited: the vessel as limiter SOL and the core. Regions
    come from the flux and the process's X-points, not from hand-drawn shapes.
    """
    eq, rep = state["equilibrium"], state["representation"]
    rr, zz = np.meshgrid(eq.r, eq.z, indexing="ij")
    wall_path = _wall_path(state["wall"])
    inside = wall_path.contains_points(np.stack([rr.ravel(), zz.ravel()], -1)).reshape(rr.shape)
    diverted = rep.topology.is_diverted
    items: List = [Polyline.of(cm(state["wall"]), REGION_FILL["common_sol" if diverted else "limiter_sol"],
                               role="region:common_sol" if diverted else "region:limiter_sol", closed=True)]
    selected = _x_points(state)
    if diverted and len(selected) == 2:
        secondary = [p for p in selected.values() if not p.active]
        if secondary:
            s = secondary[0]
            primary_side = zz <= s.z if s.branch == "upper" else zz >= s.z
            primary_branch = "lower" if s.branch == "upper" else "upper"
            for poly in _filled_below(state, _separatrix_level(s, state), inside & primary_side):
                items.append(Polyline.of(cm(poly), REGION_FILL[f"primary_sol_{primary_branch}"],
                                         role=f"region:primary_sol_{primary_branch}", closed=True))
    items.append(Polyline.of(cm(np.stack([eq.lcfs.r, eq.lcfs.z], -1)[:-1]), REGION_FILL["core"], role="region:core",
                             closed=True))
    if diverted:
        for branch, point in sorted(selected.items()):
            beyond = zz < point.z if branch == "lower" else zz > point.z
            for poly in _filled_below(state, _separatrix_level(point, state), inside & beyond):
                items.append(Polyline.of(cm(poly), REGION_FILL["private"], role=f"region:private_{branch}",
                                         closed=True))
    return items


def _pointer(text: str, at_cm, label_cm, *, anchor: str, role: str) -> List:
    """A label and an arrow from it to the thing it names."""
    return [Arrow(tuple(label_cm), tuple(at_cm), "leader,black!80", role=f"pointer:{role}"),
            Label(tuple(label_cm), text, "small label,align=center", anchor=anchor, role=f"pointer:{role}")]


def _topology_text(state) -> str:
    """The process's topology in short; a double null the grid cannot resolve says so."""
    text = _TOPOLOGY_TEXT.get(state["topology"], state["topology"])
    if state["representation"].topology_evidence.get("double_null_resolution_limited"):
        text += " within resolution"
    return text


def _mm(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    text = f"{1e3 * value:+.1f}".replace("+0.0", "0").replace("-0.0", "0")
    return text.replace("-", "$-$").replace("+", "$+$") + " mm"


def _scaler(scale: float, origin: Tuple[float, float], r0: float = 0.55, z0: float = -0.6):
    def cm(points):
        p = np.asarray(points, dtype=float)
        return np.stack([scale * (p[..., 0] - r0) + origin[0], scale * (p[..., 1] - z0) + origin[1]], -1)
    return cm


# --------------------------------------------------------------------------
# 1. divertor topology classification

#: (key, title, skew, limiter tip) of the five classification panels
CLASSIFICATION_CASES: Tuple[Tuple[str, str, float, Optional[float]], ...] = (
    ("limited", "LIMITED", 0.08, 0.72),
    ("lower_single_null", "LSN", 0.08, None),
    ("near_double_null", "near-DN", 0.01, None),
    ("double_null", "balanced DN", 0.0, None),
    ("upper_single_null", "USN", -0.08, None),
)


def divertor_topology_classification(*, labels: bool = True) -> Diagram:
    r"""LIMITED, LSN, near-DN, balanced DN and USN on one analytic flux family.

    Five equilibria of the family (``analytic_balance_equilibrium``) in one
    vessel, each classified by ``derive_boundary_representation``: the
    caption under a panel is the process's topology and $dR_\mathrm{sep}$,
    not an assertion of the drawing. Solid: the primary separatrix through an
    active X-point (black cross). Dashed: the secondary separatrix of the
    inactive X-point (grey cross) on the other side. The LIMITED panel has
    an inboard limiter tip: its LCFS rests on the tip, both saddles are
    external nulls, and its $dR_\mathrm{sep}$ is geometric only -- not a
    diverted magnetic-balance sample. Near-DN is a single null whose
    secondary separatrix lies a few millimetres outside the primary one.
    """
    labels = _check_labels(labels)
    scale, pitch = 4.0, 4.6
    items: List = []
    model = {}
    for i, (key, title, skew, tip) in enumerate(CLASSIFICATION_CASES):
        state = balance_state(skew, limiter_tip=tip)
        cm = _scaler(scale, (i * pitch, 0.0))
        items += [_tag(item, key) for item in _panel_items(state, cm)]
        model[key] = {"topology": state["topology"], "d_r_sep": state["d_r_sep"], "available": state["available"],
                      "applicability": state["quality"].get("applicability"), "skew": skew,
                      "x_points": [(p.r, p.z, p.active, p.role, p.branch) for p in state["representation"].x_points]}
        if labels:
            top = float(cm([1.0, 0.6])[1]) + 0.2
            bottom = float(cm([1.0, -0.6])[1]) - 0.15
            centre = float(cm([1.0, 0.0])[0])
            items.append(Label((centre, top), title, "label", anchor="south", role=f"title:{key}"))
            process = _topology_text(state)
            lines = [f"process: {process}", f"$dR_\\mathrm{{sep}}$ = {_mm(state['d_r_sep'])}"]
            applicability = state["quality"].get("applicability")
            if applicability:
                lines.append(applicability.replace("_", " "))
            else:
                lines.append("unavailable")
            items.append(Label((centre, bottom), "\\\\ ".join(lines), "small label,align=center", anchor="north",
                               role=f"caption:{key}"))
    if labels:
        items += _classification_legend(5 * pitch - 0.6)
    return Diagram("divertor_topology_classification", Scene(tuple(items)), model=model)


def _tag(item, key: str):
    """``item`` with the panel key appended to its role, so a test can ask for one panel."""
    from dataclasses import replace

    return replace(item, role=f"{item.role}:{key}")


def _legend(entries, x0: float, y0: float, column: float, columns: int = 2) -> List:
    """Legend rows, black text: each entry is (kind, style, text) with kind "line", "x" or "dot"."""
    items: List = []
    for k, (kind, style, text) in enumerate(entries):
        cx = x0 + (k % columns) * column
        cy = y0 - 0.55 * (k // columns)
        if kind == "line":
            items.append(Polyline.of([(cx, cy), (cx + 0.8, cy)], style, role="legend"))
        else:
            items.append(Marker((cx + 0.4, cy), "x" if kind == "x" else "o", style, role="legend"))
        items.append(Label((cx + 0.95, cy), text, "small label", anchor="west", role="legend"))
    return items


def _classification_legend(width: float) -> List:
    return _legend((
        ("line", "separatrix", "primary separatrix legs (active X-point)"),
        ("line", _SECONDARY, "secondary separatrix (inactive X-point)"),
        ("line", _EXTERNAL, "external null's separatrix (limited)"),
        ("line", "lcfs", "LCFS: closed part of the primary separatrix, or on the limiter"),
        ("x", "xpoint", "active X-point"),
        ("x", _INACTIVE_X, "inactive X-point"),
        ("dot", "fill=black", "magnetic axis"),
    ), 0.2, -2.6, 0.5 * width)


# --------------------------------------------------------------------------
# 2. secondary separatrix balance


def _balance_cases(bias: float):
    bias = _check_skew(bias, "bias")
    if not bias > 0.0:
        raise ValueError(f"bias must be positive (it is the skew of the lower-biased case), not {bias!r}")
    return (("lower_biased", "lower-biased", bias), ("balanced", "balanced DN", 0.0),
            ("upper_biased", "upper-biased", -bias))


def secondary_separatrix_balance(bias: float = 0.08, *, labels: bool = True) -> Diagram:
    r"""Primary and secondary separatrices, their flux difference and the signed $dR_\mathrm{sep}$.

    Columns: lower-biased (skew ``bias``), balanced DN, upper-biased
    (``-bias``). Rows: (a) the R-Z plane with both separatrices and the
    reference line $Z = Z_\mathrm{axis}$; (b) flux space, the normalized
    fluxes $\psi_N$ of the two X-points and their difference
    $\Delta\psi_X = \psi_L - \psi_U$; (c) the outboard midplane, $\psi_N(R)$
    with the intersections $R_L$ and $R_U$ and $dR_\mathrm{sep} = R_L - R_U$
    (``separatrix_radial_separation``), negative when the lower X-point is
    primary. Every value is read from ``derive_boundary_representation``.
    """
    labels = _check_labels(labels)
    cases = _balance_cases(bias)
    pitch = 6.2
    items: List = []
    model = {}
    for i, (key, title, skew) in enumerate(cases):
        state = balance_state(skew)
        q = state["quality"]
        x0 = i * pitch
        cm = _scaler(4.6, (x0 + 0.5, 0.0))
        items += [_tag(item, key) for item in _panel_items(state, cm, lcfs=False)]
        z_ax = float(state["equilibrium"].magnetic_axis[1])
        r_l, r_u = q["r_lower_m"], q["r_upper_m"]
        d = float(separatrix_radial_separation(r_l, r_u))
        items.append(_tag(Polyline.of(cm([[1.0, z_ax], [1.45, z_ax]]), "cut", role="reference_line"), key))
        items += [_tag(Marker(tuple(cm([r_l, z_ax])), "o", "fill=driftred", role="r_lower"), key),
                  _tag(Marker(tuple(cm([r_u, z_ax])), "o", "fill=islandblue", role="r_upper"), key)]
        model[key] = {"topology": state["topology"], "d_r_sep": state["d_r_sep"], "r_lower": r_l, "r_upper": r_u,
                      "psi_n_lower": q["psi_n_lower"], "psi_n_upper": q["psi_n_upper"],
                      "flux_imbalance": q["flux_imbalance"], "linearized": q.get("linearized_m"),
                      "resolution": q.get("resolution_m"), "d_r_sep_formula": d}
        if labels:
            centre = float(cm([1.0, 0.0])[0])
            items.append(Label((centre, float(cm([1.0, 0.6])[1]) + 0.2), title, "label", anchor="south",
                               role=f"title:{key}"))
            items.append(Label((centre, float(cm([1.0, -0.6])[1]) - 0.1),
                               f"process: {_topology_text(state)}",
                               "small label", anchor="north", role=f"caption:{key}"))
        items += [_tag(item, key) for item in _flux_space(state, x0, -1.6, labels)]
        items += [_tag(item, key) for item in _midplane_profile(state, x0, -8.2, labels)]
    if labels:
        items += [Label((-0.3, float(_scaler(4.6, (0.5, 0.0))([0.55, 0.0])[1])), "(a) R-Z", "label",
                        anchor="east", role="row"),
                  Label((-0.3, -2.4), "(b) flux", "label", anchor="east", role="row"),
                  Label((-0.3, -5.4), "(c) midplane", "label", anchor="east", role="row")]
        items += _legend((
            ("line", "separatrix", "primary separatrix (active X-point)"),
            ("line", _SECONDARY, "secondary separatrix (inactive X-point)"),
            ("dot", "fill=driftred", "$R_L$: where $\\psi = \\psi_L$ (lower X-point) meets $Z_\\mathrm{axis}$"),
            ("dot", "fill=islandblue", "$R_U$: where $\\psi = \\psi_U$ (upper X-point) meets $Z_\\mathrm{axis}$"),
            ("line", "cut", "reference line $Z = Z_\\mathrm{axis}$, outboard"),
            ("line", "connector", "$dR_\\mathrm{sep} = R_L - R_U$: LSN $<0$, USN $>0$; (c) centred on $(R_L+R_U)/2$"),
        ), 0.0, -10.2, 9.3)
    return Diagram("secondary_separatrix_balance", Scene(tuple(items)), model=model)


def _flux_space(state, x0: float, y0: float, labels: bool) -> List:
    """A psi_N axis with the two X-point fluxes and the flux difference between them."""
    q = state["quality"]
    top = max(q["psi_n_lower"], q["psi_n_upper"])
    step = 0.1 if top < 1.25 else 0.2
    lo, hi = 0.94, max(1.22, top + 0.06)
    width = 5.0

    def x(v):
        return x0 + 0.5 + (v - lo) / (hi - lo) * width

    y = y0 - 0.9
    items: List = [Arrow((x(lo), y), (x(hi) + 0.2, y), "frame axis arrow", role="flux_axis")]
    for v in np.round(np.arange(1.0, hi, step), 6):
        items += [Polyline.of([(x(v), y), (x(v), y - 0.12)], "tick", role="ticks"),
                  Label((x(v), y - 0.18), f"${v:g}$", "small label", anchor="north", role="ticks")]
    points = {"lower": (q["psi_n_lower"], "fill=driftred"), "upper": (q["psi_n_upper"], "fill=islandblue")}
    for side, (value, style) in points.items():
        items.append(Marker((x(value), y), "o", style, role=f"psi_{side}"))
    low, high = sorted((q["psi_n_lower"], q["psi_n_upper"]))
    if high - low > 1e-6:
        items.append(Arrow((x(low), y + 0.45), (x(high), y + 0.45), "connector", role="flux_imbalance", both=True))
    if labels:
        items.append(Label((x(hi) + 0.25, y), "$\\psi_N$", "small label", anchor="west", role="flux_axis"))
        balanced = high - low <= 1e-6
        sign = "=" if balanced else ("<" if q["psi_n_lower"] < q["psi_n_upper"] else ">")
        text = f"$\\psi_L {sign} \\psi_U$"
        if not balanced:
            text = f"$\\Delta\\psi_X = \\psi_L - \\psi_U$, {text}"
        items.append(Label((x0 + 0.5 + 0.5 * width, y + 0.6), text, "small label", anchor="south",
                           role="flux_imbalance"))
        items += [Label((x(q["psi_n_lower"]), y - 0.6), "$\\psi_L$", "small label,text=driftred", anchor="north",
                        role="psi_lower"),
                  Label((x(q["psi_n_upper"]), y - 0.6 - (0.45 if high - low < 0.03 else 0.0)), "$\\psi_U$",
                        "small label,text=islandblue", anchor="north", role="psi_upper")]
    return items


def _midplane_profile(state, x0: float, y0: float, labels: bool) -> List:
    """psi_N(R) on the outboard midplane around the separatrices, with R_L, R_U and dR_sep."""
    q, sp = state["quality"], state["spline"]
    eq = state["equilibrium"]
    z_ax = float(state["equilibrium"].magnetic_axis[1])
    r_l, r_u = q["r_lower_m"], q["r_upper_m"]
    centre = 0.5 * (r_l + r_u)
    half = 0.05 if abs(r_l - r_u) < 0.06 else math.ceil(1e2 * (0.8 * abs(r_l - r_u) + 0.01)) / 1e2
    r0, r1 = centre - half, centre + half
    span = float(eq.psi_boundary - eq.psi_axis)
    R = np.linspace(r0, r1, 301)
    psi_n = (sp.ev(R, np.full_like(R, z_ax)) - float(eq.psi_axis)) / span
    # the window's own psi_N range, so both X-point levels and the curve stay on the axes
    lo = min(0.75, float(np.min(psi_n)))
    hi = max(1.3, float(np.max(psi_n)), q["psi_n_lower"] + 0.05, q["psi_n_upper"] + 0.05)
    width, height = 5.0, 2.6

    def xy(r, v):
        return (x0 + 0.5 + (r - r0) / (r1 - r0) * width, y0 + (v - lo) / (hi - lo) * height)

    curve = np.array([xy(r, v) for r, v in zip(R, psi_n) if lo <= v <= hi])
    items: List = [Arrow(xy(r0, lo), (xy(r1, lo)[0] + 0.2, xy(r1, lo)[1]), "frame axis arrow", role="axes"),
                   Arrow(xy(r0, lo), (xy(r0, lo)[0], xy(r0, hi)[1] + 0.2), "frame axis arrow", role="axes"),
                   Polyline.of(curve, "boundary,line width=1.2pt", role="psi_profile")]
    for side, r, value, colour in (("lower", r_l, q["psi_n_lower"], "driftred"),
                                   ("upper", r_u, q["psi_n_upper"], "islandblue")):
        items += [Polyline.of([xy(r0, value), xy(r, value), xy(r, lo)], f"{colour},densely dashed,line width=0.8pt",
                              role=f"psi_{side}"),
                  Marker(xy(r, lo), "o", f"fill={colour}", role=f"r_{side}")]
    d = float(separatrix_radial_separation(r_l, r_u))
    y_arrow = xy(r0, lo)[1] - 0.25
    if abs(d) > 1e-6:
        items.append(Arrow((xy(r_u, lo)[0], y_arrow), (xy(r_l, lo)[0], y_arrow), "connector", role="d_r_sep"))
    if labels:
        resolution = q.get("resolution_m")
        text = f"$dR_\\mathrm{{sep}}$ = {_mm(state['d_r_sep'])}"
        if resolution is not None and abs(d) <= resolution:
            text += f"\\\\ within the {1e3 * resolution:.2f} mm resolution"
        items += [Label((x0 + 0.5 + 0.5 * width, y_arrow - 0.15), text, "small label,align=center", anchor="north",
                        role="d_r_sep"),
                  Label((xy(r1, lo)[0] + 0.25, xy(r1, lo)[1]), "$R$", "small label", anchor="west", role="axes"),
                  Label((xy(r0, hi)[0] + 0.1, xy(r0, hi)[1] + 0.2), "$\\psi_N$ at $Z_\\mathrm{axis}$", "small label",
                        anchor="west", role="axes"),
                  Label((xy(r0, lo)[0], xy(r0, lo)[1] - 0.05), f"{1e3 * (r0 - centre):+.0f}".replace("-", "$-$"),
                        "small label", anchor="north", role="ticks"),
                  Label((xy(r1, lo)[0], xy(r1, lo)[1] - 0.05), f"{1e3 * (r1 - centre):+.0f} mm".replace("+", "$+$"),
                        "small label", anchor="north", role="ticks")]
    return items


# --------------------------------------------------------------------------
# 3. SOL magnetic connectivity


def connectivity_traces(state: dict) -> List[dict]:
    """Field lines traced from seeds in every region the separatrices bound.

    Outboard-midplane seeds between the two separatrices (when they are
    resolved apart) and outside the secondary one; an inboard-midplane seed
    outside both; one seed beyond each X-point in its private flux; two
    closed surfaces in the core.
    """
    eq = state["equilibrium"]
    q = state["quality"]
    z_ax = float(eq.magnetic_axis[1])
    psi_ax = float(eq.psi_axis)
    xps = _x_points(state)
    psi_p, psi_s = sorted((xps["lower"].psi, xps["upper"].psi), key=lambda v: abs(v - psi_ax))
    seeds: List[Tuple[str, Tuple[float, float]]] = []
    for frac in (0.4, 0.8):
        r = _midplane_crossing(state, psi_ax + frac * (psi_p - psi_ax), "outboard")
        seeds.append(("core", (r, z_ax)))
    if abs(q["r_lower_m"] - q["r_upper_m"]) > max(q.get("resolution_m", 0.0), 1e-6):
        seeds.append(("between", (0.5 * (q["r_lower_m"] + q["r_upper_m"]), z_ax)))
    r_s = max(q["r_lower_m"], q["r_upper_m"])
    r_wall = float(np.max(state["wall"][:, 0]))
    for frac in (0.25, 0.6):
        seeds.append(("outboard", (r_s + frac * (r_wall - r_s), z_ax)))
    r_in = _midplane_crossing(state, psi_s, "inboard")
    r_wall_in = float(np.min(state["wall"][:, 0]))
    seeds.append(("inboard", (r_in - 0.5 * (r_in - r_wall_in), z_ax)))
    for side, sign in (("lower", -1.0), ("upper", 1.0)):
        point = xps[side]
        seeds.append((f"private_{side}", (point.r, point.z + sign * 0.06)))
    traces = []
    for region, seed in seeds:
        trace = trace_field_line(state, seed)
        trace["region"] = region
        trace["class"] = connection_class(trace)
        traces.append(trace)
    return traces


def _connectivity_items(state, cm, key: str, traces) -> Tuple[List, List[str]]:
    """Region fills, traced field lines with their wall hits, separatrices and X-points of one panel."""
    items: List = [_tag(item, key) for item in region_fills(state, cm)]
    seen: List[str] = []
    for trace in traces:
        cls = trace["class"]
        if cls == "closed":
            items.append(_tag(Polyline.of(cm(trace["path"]), "surface", role="closed_surface"), key))
            continue
        items.append(_tag(Polyline.of(cm(trace["path"]), _CLASS_STYLE.get(cls, _CLASS_STYLE["other"]),
                                      role=f"field_line:{cls}"), key))
        for end in trace["ends"]:
            items.append(_tag(Marker(tuple(cm(end)), "o", "fill=black", role="target_hit"), key))
        if cls not in seen:
            seen.append(cls)
    items += [_tag(item, key) for item in _panel_items(state, cm, lcfs=False, primary_style=_BLACK_PRIMARY,
                                                       secondary_style=_BLACK_SECONDARY)]
    return items, seen


def _region_pointers(state, cm, key: str, traces) -> List:
    """Arrows from region names, outside the vessel, into each region the panel has."""
    q, eq = state["quality"], state["equilibrium"]
    z_ax = float(eq.magnetic_axis[1])
    xps = _x_points(state)
    r_s = max(q["r_lower_m"], q["r_upper_m"])
    r_wall = float(np.max(state["wall"][:, 0]))
    inboard = next(t["seed"] for t in traces if t["region"] == "inboard")
    items: List = [Label(tuple(cm([float(eq.magnetic_axis[0]), z_ax + 0.12])), "core", "small label",
                         anchor="south", role=f"pointer:core:{key}")]
    items += _pointer("common SOL", cm([0.5 * (r_s + r_wall), z_ax + 0.3]), cm([1.62, z_ax + 0.36]),
                      anchor="west", role=f"common_sol_outboard:{key}")
    items += _pointer("common SOL", cm(inboard), cm([0.36, z_ax + 0.2]), anchor="east",
                      role=f"common_sol_inboard:{key}")
    between = [t for t in traces if t["region"] == "between"]
    if between:
        side = "lower" if between[0]["class"] == "primary_lower" else "upper"
        items += _pointer(f"primary SOL\\\\({side} divertor only)", cm(between[0]["seed"]), cm([1.62, z_ax - 0.1]),
                          anchor="west", role=f"primary_sol:{key}")
    for branch, sign in (("lower", -1.0), ("upper", 1.0)):
        point = xps[branch]
        items += _pointer("private flux", cm([point.r - 0.03, point.z + sign * 0.06]),
                          cm([0.36, point.z + sign * 0.04]), anchor="east", role=f"private_{branch}:{key}")
    return items


def sol_magnetic_connectivity(bias: float = 0.15, *, labels: bool = True) -> Diagram:
    r"""Which divertor targets the open field lines reach, on each side of the secondary separatrix.

    Lower-biased (skew ``bias``), balanced DN and upper-biased equilibria of
    the analytic family. Shading marks the regions (``region_fills``): core,
    primary SOL, common SOL and private flux. Field lines are traced from
    seeds in each region along their flux contour until they meet the wall
    (``trace_field_line``), and coloured by the targets they reach
    (``connection_class``); arrows name the regions.

    Between the primary and the secondary separatrix lies the primary SOL:
    both ends on the primary divertor, inner and outer target. Outside the
    secondary separatrix is the common SOL: an outboard line runs from the
    lower outer to the upper outer target, an inboard one from lower inner to
    upper inner. Beyond each X-point, below its own flux, lies a private flux
    region whose lines join that divertor's inner and outer targets without
    passing the main chamber. In a balanced DN the primary SOL has zero width,
    so the inboard and outboard SOL are disconnected. ``bias`` is large so the
    primary SOL is visible; the topology does not depend on its width.
    """
    labels = _check_labels(labels)
    cases = _balance_cases(bias)
    scale, pitch = 4.8, 9.2
    items: List = []
    model = {}
    seen: List[str] = []
    for i, (key, title, skew) in enumerate(cases):
        state = balance_state(skew)
        cm = _scaler(scale, (i * pitch + 2.3, 0.0))
        traces = connectivity_traces(state)
        panel, classes = _connectivity_items(state, cm, key, traces)
        items += panel
        seen += [c for c in classes if c not in seen]
        model[key] = {"topology": state["topology"], "d_r_sep": state["d_r_sep"],
                      "connections": [(t["region"], t["class"], t["targets"]) for t in traces]}
        if labels:
            centre = float(cm([1.0, 0.0])[0])
            top = float(cm([1.0, 0.6])[1])
            items += [Label((centre, top + 0.75), title, "label", anchor="south", role=f"title:{key}"),
                      Label((centre, float(cm([1.0, -0.6])[1]) - 0.15),
                            f"{_topology_text(state)}, "
                            f"$dR_\\mathrm{{sep}}$ = {_mm(state['d_r_sep'])}", "small label", anchor="north",
                            role=f"caption:{key}")]
            for name, r in (("IU", 0.80), ("OU", 1.33)):
                items.append(_tag(Label(tuple(cm([r, 0.6])), name, "small label", anchor="south",
                                        role="target_label"), key))
            items += _region_pointers(state, cm, key, traces)
    if labels:
        order = [c for c in _CLASS_TEXT if c in seen or (c == "private_lower" and "private_upper" in seen)]
        entries = [("line", _CLASS_STYLE[c], _CLASS_TEXT[c]) for c in order]
        entries += [("line", _BLACK_PRIMARY, "primary separatrix; dashed: secondary"),
                    ("dot", "fill=black", "where a field line meets the wall; IU/OU: upper targets")]
        items += _legend(entries, 1.0, -1.6, 1.5 * pitch)
    return Diagram("sol_magnetic_connectivity", Scene(tuple(items)), model=model)


# --------------------------------------------------------------------------
# single-topology anatomy: LIMITED, LSN, DN


def _run_point(runs, z: float, outboard: bool) -> np.ndarray:
    """The point of ``runs`` on the outboard (or inboard) side whose height is closest to ``z``."""
    points = np.concatenate(runs)
    side = points[:, 0] > 1.0 if outboard else points[:, 0] < 1.0
    candidates = points[side]
    return candidates[int(np.argmin(np.abs(candidates[:, 1] - z)))]


def _strike_points(state) -> Dict[str, np.ndarray]:
    """Where the primary separatrix legs meet the divertor targets, named like ``"lower inner"``."""
    bounds = _core_bounds(state)
    out: Dict[str, np.ndarray] = {}
    selected = _x_points(state)
    for point in selected.values():
        if not point.active:
            continue
        for run in _separatrix_runs(state, point):
            for piece, core in _split_at_x_points(run, bounds):
                if core:
                    continue
                for end in (piece[0], piece[-1]):
                    if _on_wall(end, state["wall"]) and abs(end[1]) >= _DIVERTOR_Z - 1e-9:
                        # a double null's separatrix runs through both X-points: name the end by its own side
                        side = "lower" if end[1] < 0.0 else "upper"
                        split = selected[side].r if side in selected else point.r
                        out[f"{side} {'inner' if end[0] < split else 'outer'}"] = np.asarray(end)
    return out


def _on_wall(point, wall, tolerance: float = 1e-6) -> bool:
    """Whether ``point`` lies on an edge of the wall polygon."""
    p = np.asarray(point, dtype=float)
    for k in range(len(wall)):
        a, b = np.asarray(wall[k], float), np.asarray(wall[(k + 1) % len(wall)], float)
        t = float(np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0))
        if np.hypot(*(p - (a + t * (b - a)))) < tolerance:
            return True
    return False


def _outboard_level_point(state, level: float, z: float) -> np.ndarray:
    """The outboard point at height ``z`` where psi equals ``level``."""
    from scipy.optimize import brentq

    sp = state["spline"]
    r_wall = float(np.max(state["wall"][:, 0]))
    return np.array([brentq(lambda R: float(sp.ev(R, z)) - level, 1.0 + 1e-6, r_wall, xtol=1e-12), z])


_ANATOMY = {
    "limited": ("LIMITED", 0.08, 0.72),
    "lower_single_null": ("Lower single null (LSN)", 0.12, None),
    "double_null": ("Balanced double null (DN)", 0.0, None),
}


def _anatomy(kind: str, labels: bool) -> Diagram:
    labels = _check_labels(labels)
    title, skew, tip = _ANATOMY[kind]
    state = balance_state(skew, limiter_tip=tip)
    eq, q = state["equilibrium"], state["quality"]
    scale = 7.0
    cm = _scaler(scale, (4.2, 0.0))
    z_ax = float(eq.magnetic_axis[1])
    diverted = state["representation"].topology.is_diverted
    items: List = []
    traces: List[dict] = []
    seen: List[str] = []
    if diverted:
        traces = connectivity_traces(state)
        items, seen = _connectivity_items(state, cm, kind, traces)
    else:
        items = [_tag(i, kind) for i in region_fills(state, cm) + _panel_items(state, cm)]
    # the outboard reference line and the two separatrix crossings
    r_l, r_u = q["r_lower_m"], q["r_upper_m"]
    items += [Polyline.of(cm([[float(eq.magnetic_axis[0]), z_ax], [float(np.max(state["wall"][:, 0])), z_ax]]),
                          "cut", role="reference_line"),
              Marker(tuple(cm([r_l, z_ax])), "o", "fill=driftred", role="r_lower"),
              Marker(tuple(cm([r_u, z_ax])), "o", "fill=islandblue", role="r_upper")]
    strikes = _strike_points(state) if diverted else {}
    for name, at in sorted(strikes.items()):
        items.append(Marker(tuple(cm(at)), "o", "fill=driftred", role=f"strike_point:{name}"))
    model = {"topology": state["topology"], "d_r_sep": state["d_r_sep"],
             "applicability": q.get("applicability"), "strike_points": {k: (float(v[0]), float(v[1])) for k, v in strikes.items()},
             "connections": [(t["region"], t["class"], t["targets"]) for t in traces]}
    if labels:
        items += _anatomy_labels(kind, state, cm, traces, strikes)
        centre = float(cm([1.0, 0.0])[0])
        items += [Label((centre, float(cm([1.0, 0.6])[1]) + 0.75), title, "label", anchor="south", role="title"),
                  Label((centre, float(cm([1.0, -0.6])[1]) - 0.75),
                        f"process: {_topology_text(state)}, $dR_\\mathrm{{sep}} = R_L - R_U$ = {_mm(state['d_r_sep'])}"
                        f" ({(q.get('applicability') or 'unavailable').replace('_', ' ')})", "small label",
                        anchor="north", role="caption")]
        entries = [("line", _CLASS_STYLE[c], _CLASS_TEXT[c]) for c in _CLASS_TEXT
                   if c in seen or (c == "private_lower" and "private_upper" in seen)]
        if entries:
            items += _legend(entries, 0.0, float(cm([1.0, -0.6])[1]) - 1.9, 10.2)
    return Diagram(f"{kind}_topology" if kind != "limited" else "limited_plasma_topology", Scene(tuple(items)),
                   model=model)


def _anatomy_labels(kind, state, cm, traces, strikes) -> List:
    """Named arrows for every element of a single-topology panel; left column R = 0.3 m, right R = 1.7 m."""
    eq, q = state["equilibrium"], state["quality"]
    z_ax = float(eq.magnetic_axis[1])
    xps = _x_points(state)
    lcfs = np.stack([eq.lcfs.r, eq.lcfs.z], -1)
    seed = {t["region"]: t["seed"] for t in traces}
    left, right = 0.3, 1.7
    items: List = [Label(tuple(cm([float(eq.magnetic_axis[0]), z_ax + 0.14])), "core", "small label", anchor="south",
                         role="pointer:core"),
                   Label(tuple(cm([float(eq.magnetic_axis[0]), z_ax - 0.03])), "magnetic axis", "font=\\scriptsize",
                         anchor="north", role="pointer:axis")]
    r_mid = 0.5 * (q["r_lower_m"] + q["r_upper_m"])

    def point(text, target, y, side):
        return _pointer(text, cm(target), cm([right if side == "right" else left, y]),
                        anchor="west" if side == "right" else "east", role=text.split("\\\\")[0])

    if kind == "limited":
        tip = np.array([0.72, 0.0])
        ext = [run for p in xps.values() for run in _separatrix_runs(state, p)]
        items += point("limiter tip:\\\\the LCFS touches it", tip, 0.12, "left")
        items += point("upper external null\\\\(inactive, not a boundary)", [xps["upper"].r, xps["upper"].z],
                       xps["upper"].z + 0.03, "left")
        items += point("lower external null\\\\(inactive)", [xps["lower"].r, xps["lower"].z], xps["lower"].z - 0.03,
                       "left")
        items += point("LCFS: the surface\\\\through the limiter tip", _run_point([lcfs], 0.25, True), 0.42, "right")
        items += point("separatrices of the\\\\external nulls", _run_point(ext, -0.3, True), -0.32, "right")
        items += point("limiter SOL: open lines\\\\end on limiter and wall", [1.4, 0.3], 0.22, "right")
        items += point(f"$R_L$, $R_U$: {_mm(state['d_r_sep'])}\\\\geometric only, no balance", [r_mid, z_ax],
                       -0.05, "right")
        return items
    upper, lower = xps["upper"], xps["lower"]
    if kind == "lower_single_null":
        items += point("lower X-point\\\\(active: primary)", [lower.r, lower.z], lower.z - 0.06, "right")
        items += point("upper X-point\\\\(inactive: secondary)", [upper.r, upper.z], 0.64, "left")
        sec = _separatrix_runs(state, upper)
        items += point("secondary separatrix\\\\$\\psi = \\psi_U$", _run_point(sec, 0.3, True), 0.5, "right")
        items += point("LCFS = primary\\\\separatrix, $\\psi = \\psi_L$", _run_point([lcfs], 0.15, True), 0.3, "right")
        band = _outboard_level_point(state, 0.5 * (lower.psi + upper.psi), -0.18)
        items += point("primary SOL:\\\\lower divertor only", band, -0.2, "right")
        items += point("common SOL", [0.5 * (max(q["r_lower_m"], q["r_upper_m"]) + 1.45), 0.12], 0.1, "right")
        items += point(f"$dR_\\mathrm{{sep}} = R_L - R_U$\\\\= {_mm(state['d_r_sep'])} $<0$", [r_mid, z_ax],
                       -0.02, "right")
        items += point("private flux\\\\(upper divertor)", [upper.r - 0.03, upper.z + 0.06], 0.4, "left")
        items += point("common SOL\\\\(inboard)", seed["inboard"], 0.12, "left")
        items += point("private flux\\\\(lower divertor)", [lower.r - 0.03, lower.z - 0.06], -0.5, "left")
        if "lower inner" in strikes:
            items += point("inner strike point", strikes["lower inner"], -0.66, "left")
        if "lower outer" in strikes:
            items += point("outer strike point", strikes["lower outer"], -0.66, "right")
        return items
    # balanced double null
    items += point("upper X-point (active)", [upper.r, upper.z], upper.z + 0.06, "right")
    items += point("lower X-point (active)", [lower.r, lower.z], lower.z - 0.06, "right")
    items += point("LCFS through\\\\both X-points", _run_point([lcfs], 0.2, True), 0.3, "right")
    items += point("common SOL (outboard):\\\\lower outer $\\leftrightarrow$ upper outer",
                   [0.5 * (q["r_upper_m"] + 1.45), -0.15], -0.15, "right")
    items += point(f"$R_L = R_U$: $dR_\\mathrm{{sep}}$ = {_mm(state['d_r_sep'])}\\\\no primary SOL", [r_mid, z_ax],
                   0.05, "right")
    items += point("common SOL (inboard):\\\\lower inner $\\leftrightarrow$ upper inner", seed["inboard"], 0.05,
                   "left")
    items += point("private flux (upper)", [upper.r - 0.03, upper.z + 0.06], 0.45, "left")
    items += point("private flux (lower)", [lower.r - 0.03, lower.z - 0.06], -0.45, "left")
    for name, y, side in (("upper inner", 0.66, "left"), ("lower inner", -0.66, "left"),
                          ("upper outer", 0.66, "right"), ("lower outer", -0.66, "right")):
        if name in strikes:
            items += point(f"{name} strike point", strikes[name], y, side)
    return items


def limited_plasma_topology(*, labels: bool = True) -> Diagram:
    r"""A limited plasma with two external nulls, one element at a time.

    The LCFS is the flux surface through the inboard limiter tip: the first
    closed surface that touches a material object from the magnetic axis
    out. Both saddles of the flux lie outside it; they are external nulls
    (``XPoint.active`` false), and their separatrices do not bound the
    plasma. Outside the LCFS the field lines are cut by the limiter and the
    wall: a limiter scrape-off layer with no divertor and no private flux.

    $R_L$ and $R_U$ -- where the two null fluxes cross the outboard midplane
    -- still exist, so the process keeps $dR_\mathrm{sep}$ with
    ``quality["applicability"] == "geometric_only"``: a geometric number, not
    a magnetic-balance sample. Analytic family, skew 0.08, tip at R = 0.72 m.
    """
    return _anatomy("limited", labels)


def lower_single_null_topology(*, labels: bool = True) -> Diagram:
    r"""A lower single null, one element at a time.

    The lower X-point is active (primary): its separatrix is the boundary,
    closed above it as the LCFS and continuing as two legs to the lower
    inner and outer strike points. The upper X-point is inactive
    (secondary): its separatrix lies outside, crossing the outboard midplane
    at $R_U > R_L$, so $dR_\mathrm{sep} = R_L - R_U < 0$.

    Between the two separatrices is the primary SOL: its field lines pass
    over the top of the plasma and end on both lower targets only. Outside
    the secondary separatrix is the common SOL, connecting lower and upper
    targets (outboard to outboard, inboard to inboard). Beyond each X-point,
    below its own flux, a private flux region joins that divertor's inner
    and outer targets without entering the main chamber. Analytic family,
    skew 0.12; every line drawn is traced along its flux contour.
    """
    return _anatomy("lower_single_null", labels)


def double_null_topology(*, labels: bool = True) -> Diagram:
    r"""A balanced double null, one element at a time.

    Both X-points are active and lie on the same flux: one separatrix
    closes through both as the LCFS, and the two crossings coincide on the
    outboard midplane, $R_L = R_U$, $dR_\mathrm{sep} = 0$ within the flux
    resolution. There is no primary SOL: the outboard SOL connects the lower
    outer to the upper outer target, the inboard SOL the lower inner to the
    upper inner, and the two do not meet. Each divertor has its private flux
    and two strike points, four in all. A balanced magnetic configuration
    does not imply balanced power: see ``separatrix_balance_transport_scale``.
    Analytic family, skew 0.
    """
    return _anatomy("double_null", labels)


# --------------------------------------------------------------------------
# 4. dR_sep against the heat-flux width


def separatrix_balance_transport_scale(lambda_q: float = 0.003, d_r_sep: Sequence[float] = (-0.001, -0.003, -0.009),
                                       *, labels: bool = True) -> Diagram:
    r"""$dR_\mathrm{sep}$ against the SOL heat-flux width: an illustrative scale, not a power split.

    Left, the model upstream profile $q_\parallel(x)/q_0 = \exp(-x/\lambda_q)$
    outside the primary separatrix on the outboard midplane, with the
    secondary separatrix at $x = |dR_\mathrm{sep}|$ for each value of
    ``d_r_sep`` and the factor ``secondary_separatrix_exponential_factor``
    there. Right, the equations (from the formulas' catalog entries) and
    $\chi = dR_\mathrm{sep}/\lambda_q$
    (``separatrix_imbalance_to_heat_flux_width``) on a signed axis:
    $|\chi| \lesssim 1$ is the transport-relevant near-double-null band,
    $|\chi| \gg 1$ behaves as a single null although both X-points exist.

    Reading it
    ----------
    In this model $e^{-|\chi|} = \int_{|dR_\mathrm{sep}|}^\infty q\,dx /
    \int_0^\infty q\,dx$ is also the share of the upstream heat flux that lies
    in the common SOL. The rest, $1 - e^{-|\chi|}$, flows in the primary SOL
    and reaches only the primary divertor.

    Magnetic balance is geometry: $dR_\mathrm{sep} = 0 \Leftrightarrow
    \psi_L = \psi_U$, and the two separatrices coincide; $|\chi|$ says how far
    apart they sit in units of the heat channel. Power balance,
    $P_\mathrm{lower} = P_\mathrm{upper}$, is transport and depends on more
    than geometry: $E\times B$ and $\nabla B$ drifts (and so the field
    direction), the inner/outer asymmetry, recycling and neutrals,
    detachment, and $\lambda_q$ itself. The common-SOL share divides between
    both divertors, so $e^{-|\chi|}$ is not the fraction of power that reaches
    the secondary divertor, and $\chi = 0$ is not thermal or power balance.

    Limitations
    -----------
    An illustrative model, not a transport prediction: the single
    exponential is an assumption, and nothing here predicts how the power
    divides (Brunner et al., Nucl. Fusion 58 (2018) 076010).
    """
    labels = _check_labels(labels)
    if isinstance(lambda_q, bool) or not (isinstance(lambda_q, (int, float)) and math.isfinite(lambda_q)
                                          and lambda_q > 0.0):
        raise ValueError(f"lambda_q must be a positive length in metres, not {lambda_q!r}")
    if isinstance(d_r_sep, (str, bytes)) or not isinstance(d_r_sep, Sequence) or not 1 <= len(d_r_sep) <= 4:
        raise ValueError(f"d_r_sep must be a sequence of one to four separations in metres, not {d_r_sep!r}")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v != 0.0
               and abs(v) <= 20.0 * lambda_q for v in d_r_sep):
        raise ValueError("d_r_sep must hold finite, non-zero separations in metres, at most 20 lambda_q")
    values = [float(v) for v in d_r_sep]
    chi = [float(separatrix_imbalance_to_heat_flux_width(v, lambda_q)) for v in values]
    factors = [float(secondary_separatrix_exponential_factor(v, lambda_q)) for v in values]
    lam_mm = 1e3 * lambda_q
    x_max = max(4.0 * lam_mm, 1.15 * 1e3 * max(abs(v) for v in values))
    x = np.linspace(0.0, x_max, 401)
    profile = np.asarray(secondary_separatrix_exponential_factor(1e-3 * x, lambda_q), dtype=float)
    chart = Chart(x_range=(0.0, x_max), y_range=(0.0, 1.15))
    chart.curves["profile"] = np.stack([x, profile], -1)
    step = 2.0 if x_max <= 12 else (5.0 if x_max <= 30 else 10.0)
    ticks = [float(t) for t in np.arange(0.0, x_max + 1e-9, step)]
    scene = render_chart(chart, x_label="$x$ outside the primary separatrix [mm]", y_label="$q/q_0$",
                         curve_styles={"profile": "boundary"}, region_text={}, x_ticks=ticks,
                         x_tick_text=[f"${t:g}$" for t in ticks], y_ticks=[0.0, 0.5, 1.0])
    items: List = list(scene.items)
    colours = ("driftred", "absequil", "islandblue", "abskinetic")
    for k, (v, c, f) in enumerate(zip(values, chi, factors)):
        xs = 1e3 * abs(v)
        top = tuple(chart.to_cm(np.array([xs, f])))
        base = tuple(chart.to_cm(np.array([xs, 0.0])))
        items += [Polyline.of([base, (top[0], top[1] + 0.25)], f"{colours[k]},densely dashed,line width=1.0pt",
                              role="secondary_separatrix"),
                  Marker(top, "o", f"fill={colours[k]}", role="factor")]
        if labels:
            items.append(Label((top[0] + 0.12, top[1] + 0.15),
                               f"$|\\chi| = {abs(c):.2g}$: $e^{{-|\\chi|}} = {f:.2f}$",
                               f"small label,text={colours[k]}", anchor="south west", role="factor"))
    lam_cm = [tuple(chart.to_cm(np.array([0.0, math.exp(-1.0)]))), tuple(chart.to_cm(np.array([lam_mm, math.exp(-1.0)])))]
    items.append(Arrow(lam_cm[0], lam_cm[1], "connector", role="lambda_q", both=True))
    if labels:
        items += [Label((0.5 * (lam_cm[0][0] + lam_cm[1][0]), lam_cm[0][1] - 0.12), f"$\\lambda_q = {lam_mm:g}$ mm",
                        "small label,fill=white,inner sep=1pt", anchor="north", role="lambda_q"),
                  Label(tuple(chart.to_cm(np.array([0.28 * x_max, 1.1]))),
                        "$x < |dR_\\mathrm{sep}|$: primary SOL, primary divertor only\\\\ beyond: common SOL, "
                        "both divertors", "small label,align=left", anchor="north west", role="regions")]
    items += [i for i in _chi_line(chi, values, colours, 11.5, labels)]
    if labels:
        items += _transport_explanation()
    return Diagram("separatrix_balance_transport_scale", Scene(tuple(items)),
                   model={"lambda_q": float(lambda_q), "d_r_sep": values, "chi": chi, "factor": factors})


def _transport_explanation() -> List:
    """The model's equations, from the formulas' catalog entries."""
    items: List = [Label((16.1, 7.6), "\\textbf{Model} (outer midplane, $x$ from the primary separatrix)\\\\[4pt]"
                         "\\mbox{$\\displaystyle \\frac{q_\\parallel(x)}{q_0} = e^{-x/\\lambda_q}$}\\\\[4pt]"
                         f"\\mbox{{$\\displaystyle {formula_equation(separatrix_imbalance_to_heat_flux_width)}$}}\\qquad"
                         f"\\mbox{{$\\displaystyle {formula_equation(secondary_separatrix_exponential_factor)}$}}"
                         "\\\\[4pt]"
                         "\\mbox{$\\displaystyle \\frac{\\int_{|dR_\\mathrm{sep}|}^{\\infty} q\\,dx}"
                         "{\\int_0^{\\infty} q\\,dx} = e^{-|\\chi|}$}: the share of the upstream heat flux "
                         "in the common SOL",
                         "formula box,text width=9.2cm", anchor="north", role="equations")]
    return items


def _chi_line(chi, values, colours, x0: float, labels: bool, y: float = 1.0) -> List:
    """chi = dR_sep / lambda_q on a signed axis, with the |chi| <~ 1 band shaded."""
    lim = max(4.0, math.ceil(max(abs(c) for c in chi) + 0.5))
    width = 8.0

    def x(c):
        return x0 + (c + lim) / (2 * lim) * width

    items: List = [Polyline.of([(x(-1.0), y - 0.5), (x(1.0), y - 0.5), (x(1.0), y + 0.5), (x(-1.0), y + 0.5)],
                               "fill=qblue!25,draw=none", role="near_dn_band", closed=True),
                   Arrow((x(-lim) - 0.2, y), (x(lim) + 0.2, y), "frame axis arrow", role="chi_axis", both=True)]
    step = 1 if lim <= 6 else int(math.ceil(lim / 5.0))
    for c in range(-int(lim) // step * step, int(lim) + 1, step):
        items += [Polyline.of([(x(c), y), (x(c), y - 0.12)], "tick", role="ticks"),
                  Label((x(c), y - 0.6), f"${c}$", "small label", anchor="north", role="ticks")]
    items.append(Polyline.of([(x(0.0), y - 0.5), (x(0.0), y + 0.5)], "black,line width=1.4pt", role="balance"))
    for k, c in enumerate(chi):
        items.append(Marker((x(c), y), "o", f"fill={colours[k]}", role="chi"))
    if labels:
        items += [Label((x(lim) + 0.3, y), "$\\chi = dR_\\mathrm{sep}/\\lambda_q$", "small label", anchor="west",
                        role="chi_axis"),
                  Label((x(0.0), y + 0.6), "$|\\chi| \\leq O(1)$: near-DN, both divertors in the near-SOL "
                        "heat channel", "small label", anchor="south", role="near_dn_band"),
                  Label((x(0.0), y + 1.2), "$\\chi = 0$: magnetic balance", "small label", anchor="south",
                        role="balance"),
                  Label((x(-lim), y - 1.2), "$\\leftarrow$ LSN side", "small label",
                        anchor="north west", role="sign"),
                  Label((x(lim), y - 1.2), "USN side $\\rightarrow$", "small label",
                        anchor="north east", role="sign"),
                  Label((x(0.0), y - 1.9), "$|\\chi| \\gg 1$: behaves as a single null, though both X-points exist",
                        "small label", anchor="north", role="single_null_limit")]
    return items


# --------------------------------------------------------------------------
# 5. the calculation as a procedure


def _glyph_scaler(gx: float, gy: float, height: float):
    """Map the vessel (R 0.6-1.45 m, Z +-0.6 m) into a glyph ``height`` cm tall, centred at (gx, gy)."""
    z_top = max(z for _, z in WALL)
    scale = height / (2.0 * z_top)
    centre_r = 0.5 * (min(r for r, _ in WALL) + max(r for r, _ in WALL))
    return _scaler(scale, (gx - scale * (centre_r - 0.55), gy - scale * z_top))


def _glyph_flux_map(state, gx, gy, height, *, saddles: bool) -> List:
    """Wall, a few flux surfaces and the magnetic axis; with ``saddles``, every saddle the process found."""
    cm = _glyph_scaler(gx, gy, height)
    eq = state["equilibrium"]
    psi_b = float(eq.psi_boundary)
    items: List = [Polyline.of(cm(state["wall"]), "machine,line width=0.7pt", role="glyph", closed=True)]
    for frac in (0.25, 0.6, 1.25, 1.6):
        for run in _inside_runs(eq, float(eq.psi_axis) + frac * (psi_b - float(eq.psi_axis)), state["wall"]):
            items.append(Polyline.of(cm(run), "surface", role="glyph"))
    items.append(Marker(tuple(cm(eq.magnetic_axis)), "o", "fill=black", role="glyph"))
    if saddles:
        for point in sorted(state["representation"].x_points, key=lambda p: (round(p.z, 6), round(p.r, 6))):
            items.append(Marker(tuple(cm([point.r, point.z])), "x", "draw=driftred,line width=1.0pt", role="glyph"))
    return items


def _glyph_roles(state, gx, gy, height) -> List:
    """The panel of ``divertor_topology_classification`` in miniature: roles drawn as the large figures draw them."""
    return [_retag(item, "glyph") for item in _panel_items(state, _glyph_scaler(gx, gy, height))]


def _retag(item, role: str):
    from dataclasses import replace

    return replace(item, role=role)


def _glyph_midplane(state, gx, gy, width, height) -> List:
    """psi_N(R) on the outboard midplane, zoomed so that R_L and R_U are apart."""
    q, sp, eq = state["quality"], state["spline"], state["equilibrium"]
    z_ax = float(eq.magnetic_axis[1])
    r_l, r_u = q["r_lower_m"], q["r_upper_m"]
    centre = 0.5 * (r_l + r_u)
    half = max(4.0 * abs(r_l - r_u), 0.004)
    R = np.linspace(centre - half, centre + half, 121)
    psi_n = (sp.ev(R, np.full_like(R, z_ax)) - float(eq.psi_axis)) / float(eq.psi_boundary - eq.psi_axis)
    lo, hi = float(psi_n.min()), float(psi_n.max())
    x0, y0 = gx - 0.5 * width, gy - 0.5 * height

    def xy(r, v):
        return (x0 + (r - R[0]) / (R[-1] - R[0]) * width, y0 + (v - lo) / (hi - lo) * height)

    items: List = [Arrow(xy(R[0], lo), (xy(R[-1], lo)[0] + 0.15, y0), "frame axis arrow", role="glyph"),
                   Arrow(xy(R[0], lo), (x0, xy(R[0], hi)[1] + 0.15), "frame axis arrow", role="glyph"),
                   Polyline.of([xy(r, v) for r, v in zip(R, psi_n)], "boundary,line width=1.0pt", role="glyph"),
                   Label((xy(R[-1], lo)[0] + 0.15, y0), "$R$", "font=\\scriptsize", anchor="west", role="glyph"),
                   Label((x0 + 0.08, xy(R[0], hi)[1] + 0.15), "$\\psi_N$", "font=\\scriptsize", anchor="west",
                         role="glyph")]
    for r, value, colour in ((r_l, q["psi_n_lower"], "driftred"), (r_u, q["psi_n_upper"], "islandblue")):
        items += [Polyline.of([xy(R[0], value), xy(r, value), xy(r, lo)], f"{colour},densely dashed,line width=0.7pt",
                              role="glyph"),
                  Marker(xy(r, lo), "o", f"fill={colour}", role="glyph")]
    return items


def _glyph_signed_line(gx, gy, width, value: Optional[float], resolution: Optional[float], *,
                       span: float) -> List:
    """A signed dR_sep axis of half-width ``span`` [m]: the resolution band, zero, and the value (if any)."""
    def x(v):
        return gx + v / span * 0.5 * width

    items: List = []
    if resolution:
        r = min(resolution, span)
        items.append(Polyline.of([(x(-r), gy - 0.25), (x(r), gy - 0.25), (x(r), gy + 0.25), (x(-r), gy + 0.25)],
                                 "fill=qblue!35,draw=none", role="glyph", closed=True))
    items += [Arrow((x(-span) - 0.1, gy), (x(span) + 0.1, gy), "frame axis arrow", role="glyph", both=True),
              Polyline.of([(x(0.0), gy - 0.3), (x(0.0), gy + 0.3)], "black,line width=1.1pt", role="glyph"),
              Label((x(-span), gy - 0.2), "LSN", "font=\\scriptsize", anchor="north west", role="glyph"),
              Label((x(span), gy - 0.2), "USN", "font=\\scriptsize", anchor="north east", role="glyph")]
    if value is None:
        items += [Polyline.of([(x(0.0) - 0.18, gy + 0.42), (x(0.0) + 0.18, gy + 0.78)], "driftred,line width=1.2pt",
                              role="glyph"),
                  Polyline.of([(x(0.0) - 0.18, gy + 0.78), (x(0.0) + 0.18, gy + 0.42)], "driftred,line width=1.2pt",
                              role="glyph"),
                  Label((x(0.0) + 0.3, gy + 0.6), "not 0", "font=\\scriptsize,text=driftred", anchor="west",
                        role="glyph")]
    else:
        items += [Arrow((x(0.0), gy + 0.5), (x(value), gy + 0.5), "connector", role="glyph"),
                  Marker((x(value), gy), "o", "fill=driftred", role="glyph")]
    return items


def separatrix_balance_calculation(bias: float = 0.01, *, labels: bool = True) -> Diagram:
    r"""How $dR_\mathrm{sep}$ is computed, with the values one near-double-null case gives at each step.

    Flux map, critical points (O- and X-points with the Hessian test
    $\det H_\psi < 0$), topology and the primary/secondary selection of
    ``XPoint.role`` and ``XPoint.branch``, the outboard intersections of
    both X-point fluxes at $Z = Z_\mathrm{axis}$, and the signed result with
    its applicability and resolution. Each box carries a small picture of its
    step, computed from ``derive_boundary_representation`` on the analytic
    family with skew ``bias``, and that case's values. Below the chain: the
    four outcomes, none of which reports a missing value as zero.

    Outcomes
    --------
    Diverted balance
        The topology is diverted (LSN, USN, DN) and both X-point fluxes
        cross the outboard midplane: ``d_r_sep.available`` with
        ``quality["applicability"] == "diverted_balance"`` and the flux
        resolution mapped to metres in ``quality["resolution_m"]``.
    DN within resolution
        The two X-point fluxes are closer than the grid can resolve: the flux
        at a saddle is only known to within a window of a few grid cells, and
        that window, carried to the midplane, is about
        $\delta_R$ = ``quality["resolution_m"]``. So when
        $|dR_\mathrm{sep}| \lesssim \delta_R$ the grid cannot tell an exact
        double null from a slightly biased one. The process calls it DN, sets
        ``topology_evidence["double_null_resolution_limited"]``, and keeps the
        small signed value with its $\pm\delta_R$ -- it is not rounded to
        zero. (The exact test compares the X-points' $\psi_N$ difference with
        the saddle flux window.) A finer grid may resolve it into LSN or USN.
    Geometric only
        LIMITED or AMBIGUOUS, while both crossings still exist: the value is
        kept with ``applicability == "geometric_only"`` and a reason. A
        limited plasma with external nulls is not a diverted balance sample.
    Unavailable
        No magnetic axis, no upper or no lower X-point, or an X-point flux
        that does not reach the outboard midplane: ``value is None`` with a
        reason. Missing is never zero.
    """
    labels = _check_labels(labels)
    state = balance_state(_check_skew(bias, "bias"))
    rep, q = state["representation"], state["quality"]
    xps = _x_points(state)
    width, height, gap = 4.2, 5.5, 0.8
    glyph_y = 0.75
    d, resolution = state["d_r_sep"], q.get("resolution_m")
    span = max(2.0 * abs(d or 0.0), 3.0 * (resolution or 0.0), 1e-3)
    steps = (
        ("flux_map", "1. Flux map", _glyph_flux_map(state, 0.0, glyph_y, 2.2, saddles=False),
         f"$\\psi(R,Z)$ on the grid with the magnetic axis and the wall\\\\[2pt]"
         f"here: {state['equilibrium'].r.size} $\\times$ {state['equilibrium'].z.size} grid"),
        ("critical_points", "2. Critical points", _glyph_flux_map(state, 0.0, glyph_y, 2.2, saddles=True),
         "$\\nabla\\psi = 0$; a saddle (X-point) where $\\det H_\\psi < 0$\\\\[2pt]"
         f"here: {len(rep.x_points)} saddles"),
        ("selection", "3. Topology and roles", _glyph_roles(state, 0.0, glyph_y, 2.2),
         "the active X-point is primary, the other side's secondary (XPoint.role)\\\\[2pt]"
         f"here: {_topology_text(state)}, lower {xps['lower'].role}"),
        ("intersections", "4. Midplane crossings", _glyph_midplane(state, 0.0, glyph_y - 0.1, 3.0, 1.7),
         "$\\psi(R_L, Z_\\mathrm{axis}) = \\psi_L$, $\\psi(R_U, Z_\\mathrm{axis}) = \\psi_U$\\\\[2pt]"
         f"here: $R_L$ = {q['r_lower_m']:.4f} m, $R_U$ = {q['r_upper_m']:.4f} m"),
        ("result", "5. Signed result", _glyph_signed_line(0.0, glyph_y, 3.2, d, resolution, span=span),
         "$dR_\\mathrm{sep} = R_L - R_U$, its applicability and flux resolution (shaded)\\\\[2pt]"
         f"here: {_mm(d)} $\\pm$ {1e3 * (resolution or float('nan')):.2f} mm"),
    )
    items: List = []
    boxes = []
    for k, (role, title, glyph, body) in enumerate(steps):
        cx = k * (width + gap)
        boxes.append(_glyph_box(items, cx, 0.0, width, height, title, glyph, body, glyph_y, role, labels,
                                style="concept box"))
    for a, b in zip(boxes, boxes[1:]):
        items.append(connector(a, b, role="flow"))
    resolved = balance_state(0.001)
    limited = balance_state(0.08, limiter_tip=0.72)
    lim_q = limited["quality"]
    res_q = resolved["quality"]
    outcomes = (
        ("available", "Diverted balance",
         _glyph_signed_line(0.0, 0.0, 3.2, d, resolution, span=span),
         "\\textit{when} the topology is diverted (LSN, USN, DN) and both X-point fluxes cross the midplane"
         "\\\\[2pt]\\textit{gives} the signed value in metres, $\\pm\\delta_R$ (resolution\\_m)"),
        ("resolution_limited", "DN within resolution",
         _glyph_signed_line(0.0, 0.0, 3.2, resolved["d_r_sep"], res_q.get("resolution_m"),
                            span=3.0 * res_q["resolution_m"]),
         "\\textit{when} the X-point fluxes are too close for the grid: $|dR_\\mathrm{sep}|$ within "
         "$\\pm\\delta_R$ (shaded)"
         f"\\\\[2pt]\\textit{{gives}} DN, flagged; the small value is kept, not set to 0 "
         f"(here {_mm(resolved['d_r_sep'])})"),
        ("geometric_only", "Geometric only", _glyph_roles(limited, 0.0, 0.0, 2.0),
         "\\textit{when} LIMITED (here on an inboard tip) or AMBIGUOUS, yet both crossings exist"
         f"\\\\[2pt]\\textit{{gives}} the value ({_mm(limited['d_r_sep'])}) as geometric\\_only, with a reason"),
        ("unavailable", "Unavailable: None",
         _glyph_signed_line(0.0, 0.0, 3.2, None, None, span=span),
         "\\textit{when} no magnetic axis, no upper or lower X-point, or its flux misses the midplane"
         "\\\\[2pt]\\textit{gives} None and a reason, never 0"),
    )
    out_height = 6.2
    y = -0.5 * height - 1.6 - 0.5 * out_height
    bus = -0.5 * height - 0.8
    # every outcome leaves step 5: one bus from its bottom edge, an arrow down to each outcome
    middle = 0.5 * (boxes[0].x + boxes[-1].x)
    centres = [middle + (k - 0.5 * (len(outcomes) - 1)) * (width + gap) for k in range(len(outcomes))]
    last = boxes[-1]
    items.append(Polyline.of([(last.x, -0.5 * height - 0.08), (last.x, bus), (centres[0], bus)], "connector line",
                             role="outcome"))
    for cx, (role, title, glyph, body) in zip(centres, outcomes):
        glyph_at = y + 0.5 * out_height - 1.75
        shifted = [_shift(item, cx, glyph_at) for item in glyph]
        _glyph_box(items, cx, y, width, out_height, title, shifted, body, glyph_at + 0.2, role, labels,
                   style="concept leaf",
                   offset=False)
        items.append(Arrow((cx, bus), (cx, y + 0.5 * out_height + 0.08), "connector", role="outcome"))
    return Diagram("separatrix_balance_calculation", Scene(tuple(items)),
                   model={"topology": state["topology"], "d_r_sep": d, "quality": q,
                          "roles": {k: v.role for k, v in xps.items()}, "linearized": q.get("linearized_m"),
                          "outcome_examples": {"resolution_limited": (resolved["topology"], resolved["d_r_sep"]),
                                               "geometric_only": (limited["topology"], lim_q.get("applicability"))}})


def _shift(item, dx: float, dy: float):
    return Scene((item,)).transformed(offset=(dx, dy)).items[0]


def _glyph_box(items: List, cx: float, cy: float, width: float, height: float, title: str, glyph: List, body: str,
               glyph_y: float, role: str, labels: bool, *, style: str, offset: bool = True):
    """A box with its title on top, ``glyph`` (drawn around x = 0, or already placed) and the text below it."""
    b = box(cx, cy, width, height, "", role=role, style=style)
    items.append(b.items[0])
    items += [_retag(_shift(item, cx, 0.0) if offset else item, f"glyph:{role}") for item in glyph]
    if labels:
        top = cy + 0.5 * height
        items += [Label((cx, top - 0.15), f"\\textbf{{{title}}}", "concept text,font=\\small", anchor="north",
                        role=role),
                  Label((cx, glyph_y - 1.4), body, f"concept text,font=\\small,text width={width - 0.3:.2f}cm",
                        anchor="north", role=role)]
    return b


__all__ = [
    "CLASSIFICATION_CASES",
    "FAMILY_A",
    "WALL",
    "analytic_balance_equilibrium",
    "balance_state",
    "connection_class",
    "connectivity_traces",
    "divertor_topology_classification",
    "family_flux",
    "secondary_separatrix_balance",
    "separatrix_balance_calculation",
    "separatrix_balance_transport_scale",
    "sol_magnetic_connectivity",
    "trace_field_line",
    "limited_plasma_topology",
    "lower_single_null_topology",
    "double_null_topology",
    "region_fills",
]
