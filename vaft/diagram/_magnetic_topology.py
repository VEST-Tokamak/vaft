"""Reduced Hamiltonian field-line topology on an arbitrary equilibrium (#1209).

``stochastic_layer``
    two resonant harmonics on an equilibrium's own $q$ profile: isolated
    islands, touching islands, then a stochastic layer, as the overlap
    parameter grows; a Poincare section of a field-line map in
    $(\\theta^*, \\psi_N)$, with where it sits in the equilibrium inset;
``separatrix_lobes``
    the stable and unstable manifolds of a perturbed X-point, traced on a
    single-null Solov'ev equilibrium: lobes, and a split strike point.

Model class: reduced Hamiltonian model. Field lines follow
$H(x, \\theta^*, \\phi) = \\int\\iota\\,dx - \\sum_k \\epsilon_k\\cos(m_k\\theta^* - n_k\\phi)$
with $x = \\psi_N$ and $\\iota = 1/q$ from the equilibrium:
$d\\theta^*/d\\phi = \\iota(x)$, $dx/d\\phi = -\\sum_k m_k\\epsilon_k\\sin(m_k\\theta^* - n_k\\phi)$.
Near each resonance this is the pendulum of ``island_pendulum_hamiltonian``,
full width $w_k = 4\\sqrt{\\epsilon_k/|\\iota'|}$ in $\\psi_N$. $x = \\psi_N$ stands in for
the toroidal-flux action (the canonical pair is $(\\theta^*, \\psi_\\mathrm{tor})$), so
$\\epsilon$ is a model amplitude, not a physical $\\delta B$, and area in the
section is not flux; the dynamics is area-preserving in $(\\theta^*, x)$. It explains the
topology prescribed harmonics would produce; it is not a GPEC, MARS or
vacuum-field trace, which belong to result plotting.
"""

from __future__ import annotations

import copy
import math
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np

from vaft.formula.stability import island_pendulum_hamiltonian

from ._equations import formula_equation
from ._chart import CHART_HEIGHT, CHART_WIDTH, Chart, render_chart
from ._equilibrium_geometry import _cocos, equilibrium_geometry
from ._mhd_mode import _S
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: default resonances (m, n) and the drawn overlap levels
RESONANCES = ((3, 2), (2, 1))
OVERLAPS = {"isolated": 0.5, "touching": 1.0, "overlapping": 1.6}
#: field-line map: symplectic steps per toroidal turn, turns per line, seed lines
_STEPS, _TURNS, _SEEDS = 96, 70, 22


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _note(text: str, x: float, y: float) -> Label:
    return Label((x, y), text, "note", anchor="north", role="note")


def _iota_table(geom):
    x = np.linspace(0.0, 0.96, 1201)
    return x, 1.0 / geom.q(np.sqrt(x))


def resonance_data(geom, resonances: Sequence[Tuple[int, int]]) -> dict:
    """$x_k = \\psi_N$ at $q = m_k/n_k$ and $|\\iota'(x_k)|$ there."""
    xg, iota = _iota_table(geom)
    diota = np.gradient(iota, xg)
    xs, slopes = [], []
    for m, n in resonances:
        rho = geom.rho_at_q(m / n)
        if rho is None or rho**2 > 0.93:
            raise ValueError(f"no q = {m}/{n} surface inside psi_N < 0.93 in this equilibrium")
        xs.append(rho**2)
        slopes.append(abs(float(np.interp(rho**2, xg, diota))))
    return {"x": np.array(xs), "iota_prime": np.array(slopes)}


def amplitudes_for_overlap(data: dict, overlap: float) -> np.ndarray:
    """Equal $\\epsilon$ for both harmonics giving the pair overlap $(w_1 + w_2)/(2|x_2 - x_1|)$ = ``overlap``."""
    a = 1.0 / np.sqrt(data["iota_prime"])
    spacing = abs(data["x"][1] - data["x"][0])
    root = overlap * spacing / (2.0 * a.sum())
    return np.full(2, root**2)


def field_line_map(geom, resonances, eps, x0, theta0, *, turns: int = _TURNS, steps: int = _STEPS) -> np.ndarray:
    """Poincare punctures at $\\phi = 2\\pi k$, shape (turns, lines, 2) as $(\\theta^*, x)$.

    Drift-kick-drift steps in $\\phi$ (a Strang composition of exact flows, so symplectic). The state is rounded to
    1e-9 after each turn: the map is chaotic where the islands overlap, and
    without the rounding a last-bit difference between platforms would grow
    into a different picture.
    """
    xg, iota_g = _iota_table(geom)
    x = np.asarray(x0, dtype=float).copy()
    t = np.asarray(theta0, dtype=float).copy()
    dphi = 2.0 * math.pi / steps
    out = np.empty((turns, x.size, 2))
    for k in range(turns):
        for s in range(steps):
            phi = (s + 0.5) * dphi
            t = t + 0.5 * dphi * np.interp(x, xg, iota_g)
            x = x - dphi * sum(m * e * np.sin(m * t - n * phi) for (m, n), e in zip(resonances, eps))
            x = np.clip(x, 0.0, 0.96)
            t = t + 0.5 * dphi * np.interp(x, xg, iota_g)
        x = np.round(x, 9)
        t = np.round(np.mod(t, 2.0 * math.pi), 9)
        out[k, :, 0], out[k, :, 1] = t, x
    return out


def stochastic_layer(equilibrium=None, regime: str = "touching", *, resonances=RESONANCES, overlap=None,
                     perturbations=None, labels: bool = True) -> Diagram:
    r"""Island overlap on an equilibrium: a Poincare section of two resonant harmonics.

    Field lines of the reduced Hamiltonian of this module, for the
    ``resonances`` (default $3/2$ and $2/1$) at $x_k = \psi_N(q = m_k/n_k)$ of
    ``equilibrium`` (default the Solov'ev of ``_equilibrium_geometry``).
    ``perturbations`` gives the $\epsilon_k$ directly; otherwise both are
    equal and set so that the pair overlap parameter
    $\sigma = (w_1 + w_2)/(2|x_2 - x_1|)$ is ``overlap`` -- by ``regime``,
    0.5 (isolated), 1 (touching) or 1.6 (overlapping); $\sigma$ itself is
    ``vaft.process.perturbation.chirikov`` of the widths. Punctures at
    $\phi = 0$ are drawn at $(\rho = \sqrt{x}, \theta^*)$ on the real surfaces.
    """
    if regime not in OVERLAPS:
        raise ValueError(f"regime must be one of {tuple(OVERLAPS)}, not {regime!r}")
    labels = _check_labels(labels)
    resonances = tuple((int(m), int(n)) for m, n in resonances)
    if len(resonances) != 2:
        raise ValueError("stochastic_layer takes exactly two resonances (m, n)")
    from vaft.process.perturbation import chirikov

    geom = equilibrium_geometry(equilibrium)
    data = resonance_data(geom, resonances)
    if perturbations is not None:
        eps = np.asarray(perturbations, dtype=float)
        if eps.shape != (2,) or np.any(eps <= 0.0):
            raise ValueError("perturbations must be two positive amplitudes")
    else:
        target = OVERLAPS[regime] if overlap is None else float(overlap)
        if not 0.0 < target <= 3.0:
            raise ValueError(f"overlap must lie in (0, 3], not {target!r}")
        eps = amplitudes_for_overlap(data, target)
    widths = 4.0 * np.sqrt(eps / data["iota_prime"])
    sigma = float(chirikov(data["x"], widths, definition="pair")[0])
    lo = max(0.05, data["x"].min() - 0.18)
    hi = min(0.93, data["x"].max() + 0.12)
    # uniform seeds, plus lines inside each island chain (from its O-point outward), so the islands show
    island_x, island_t = [], []
    for (m_k, _n), x_k, w_k in zip(resonances, data["x"], widths):
        for f in (0.0, 0.12, 0.25, 0.38):
            island_x.append(x_k + f * w_k)
            island_t.append(math.pi / m_k)
    x0 = np.concatenate([np.linspace(lo, hi, _SEEDS), island_x])
    t0 = np.concatenate([np.zeros(_SEEDS), island_t])
    punctures = field_line_map(geom, resonances, eps, x0, t0)
    items: List = []
    # the section itself, unwrapped: theta* across, psi_N up
    chart = Chart(x_range=(0.0, 2.0 * math.pi), y_range=(lo - 0.02, hi + 0.02))
    for k, x in enumerate(data["x"]):
        chart.curves[f"resonance_{k}"] = np.array([[0.0, x], [2.0 * math.pi, x]])
    scene = render_chart(chart, x_label="$\\theta^*$", y_label="$\\psi_N$",
                         curve_styles={f"resonance_{k}": "rational" for k in range(2)}, region_text={},
                         x_ticks=(0.0, math.pi, 2.0 * math.pi), x_tick_text=("$0$", "$\\pi$", "$2\\pi$"),
                         y_ticks=tuple(float(v) for v in np.round(data["x"], 2)),
                         y_tick_text=tuple(f"${v:.2f}$" for v in data["x"]))
    items += list(scene.items)
    flat = punctures.reshape(-1, 2)
    flat = flat[(flat[:, 1] > lo - 0.02) & (flat[:, 1] < hi + 0.02)]  # the section's window only
    cm = chart.to_cm(flat)
    for u, v in cm:
        items.append(Marker((float(u), float(v)), ".", "orbit electron", role="puncture"))
    # island widths of the pendulum, as brackets at the right edge
    for k, (x, w) in enumerate(zip(data["x"], widths)):
        a, b = chart.to_cm(np.array([[2.0 * math.pi, x - w / 2], [2.0 * math.pi, x + w / 2]]))
        items.append(Arrow((float(a[0]) + 0.3, float(a[1])), (float(b[0]) + 0.3, float(b[1])), "width arrow",
                           role=f"width_{k}", both=True))
    # where the layer sits in the equilibrium: a small (R, Z) inset
    inset_scale, inset_origin = 0.45, (-5.2, 0.4)
    ax_R, ax_Z = geom.axis

    def to_inset(R, Z):
        return np.stack([inset_origin[0] + inset_scale * _S * (np.asarray(R) - ax_R) + 2.0,
                         inset_origin[1] + inset_scale * _S * (np.asarray(Z) - ax_Z) + 3.0], -1)

    band_out, band_in = geom.surface(float(np.sqrt(hi))), geom.surface(float(np.sqrt(lo)))
    items.append(Polyline.of(np.concatenate([to_inset(band_out.R, band_out.Z), to_inset(band_in.R, band_in.Z)[::-1]]),
                             "layer", role="inset:section_range", closed=True))
    lcfs = geom.surface(0.95)
    items.append(Polyline.of(to_inset(lcfs.R, lcfs.Z), "boundary", role="inset:boundary", closed=True))
    for x in data["x"]:
        s_ = geom.surface(float(np.sqrt(x)))
        items.append(Polyline.of(to_inset(s_.R, s_.Z), "rational", role="inset:resonant_surface", closed=True))
    if labels:
        (m1, n1), (m2, n2) = resonances
        items += [
            Label((CHART_WIDTH / 2, CHART_HEIGHT + 0.5), f"{regime}: $\\sigma = {sigma:.2f}$, resonances "
                  f"${m1}/{n1}$ and ${m2}/{n2}$", "label", anchor="south", role="title"),
            Label((CHART_WIDTH + 0.8, CHART_HEIGHT), "dots: field-line punctures at $\\phi = 0$\\\\ dashed: "
                  "$q = m/n$ (unperturbed)\\\\ brackets: pendulum widths $w_k$\\\\ "
                  f"$w = {widths[0]:.2f}, {widths[1]:.2f}$ in $\\psi_N$", "small label,align=left",
                  anchor="north west", role="legend"),
            Label((CHART_WIDTH + 0.8, CHART_HEIGHT - 2.4), "$\\sigma = (w_1 + w_2)/(2|x_2 - x_1|)$, "
                  "$x = \\psi_N$\\\\ $w_k = 4\\sqrt{\\epsilon_k/|\\iota'|}$, "
                  f"$\\epsilon = {eps[0]:.1e}, {eps[1]:.1e}$", "small label,align=left", anchor="north west",
                  role="model"),
            Label((inset_origin[0] + 2.0, inset_origin[1] + 3.0 - 0.45 * _S * 0.45), "shaded: the range at "
                  "right", "small label", anchor="north", role="inset"),
            Label((CHART_WIDTH / 2, -1.45), f"$\\displaystyle {formula_equation(island_pendulum_hamiltonian)}$",
                  "formula box", anchor="north", role="equations"),
            _note("Reduced Hamiltonian model: prescribed harmonics on the equilibrium's $q$; not a GPEC, MARS or "
                  "vacuum-field trace. Each resonance alone is this pendulum", CHART_WIDTH / 2 - 1.5, -2.6),
        ]
    model = {"classification": "reduced_hamiltonian_model", "regime": regime, "resonances": resonances,
             "x_resonance": data["x"], "iota_prime": data["iota_prime"], "eps": eps, "widths": widths,
             "sigma": sigma, "punctures": punctures, "geometry": geom}
    return Diagram(f"stochastic_layer_{regime}", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# separatrix lobes: stable and unstable manifolds of a perturbed X-point
# ---------------------------------------------------------------------------

#: drawing scale of the X-point window [cm per m]
_S_ZOOM = 30.0


def _per_radian_scale(eq) -> float:
    """What ``eq.psi`` is divided by to give the per-radian flux the field-line equation needs.

    The unit of ``psi`` is a property of ``eq.convention`` (``EquilibriumData``): full weber for
    COCOS 11-18, already per radian for COCOS 1-8 -- which is what every g-file loaded through
    ``read_geqdsk().to_equilibrium()`` carries. ``_cocos`` raises when the record declares neither.
    """
    return 2.0 * math.pi if _cocos(eq) >= 11 else 1.0


class _FieldLineMap:
    """Field lines of $\\psi_0 + \\delta\\psi$ over one period $2\\pi/n$ of the perturbation.

    $dR/d\\phi = -(R/F)\\,\\partial_Z\\Psi$, $dZ/d\\phi = (R/F)\\,\\partial_R\\Psi$ with
    $\\Psi$ the per-radian flux and $F$ the boundary value of $RB_\\phi$;
    $\\delta\\psi = \\epsilon\\,\\Delta\\psi\\,(r/r_X)^m\\cos(m\\vartheta - n\\phi)$ about the
    magnetic axis, normalised at the X-point radius $r_X$. RK4 in $\\phi$.
    """

    def __init__(self, eq, m: int, n: int, eps: float, phase: float, steps: int = 32):
        from scipy.interpolate import RectBivariateSpline

        scale = _per_radian_scale(eq)
        self.sp = RectBivariateSpline(eq.r, eq.z, np.asarray(eq.psi, float) / scale)
        self.F = float(np.asarray(eq.f, float)[-1])
        self.axis = tuple(float(v) for v in eq.magnetic_axis)
        self.dpsi = float(eq.psi_boundary - eq.psi_axis) / scale
        self.m, self.n, self.eps, self.phase, self.steps = int(m), int(n), float(eps), float(phase), int(steps)
        self.r_x = 1.0

    def rhs(self, R, Z, phi):
        dR = self.sp.ev(R, Z, dx=1)
        dZ = self.sp.ev(R, Z, dy=1)
        if self.eps:
            x, z = R - self.axis[0], Z - self.axis[1]
            r = np.clip(np.hypot(x, z), 1e-9, 3.0 * self.r_x)  # escaping lines leave the grid; keep them finite
            th = np.arctan2(z, x)
            arg = self.m * th - self.n * phi + self.phase
            amp = self.eps * self.dpsi
            d_r = amp * self.m * (r / self.r_x) ** (self.m - 1) / self.r_x * np.cos(arg)
            d_t = -amp * (r / self.r_x) ** self.m * self.m * np.sin(arg) / r
            dR = dR + d_r * np.cos(th) - d_t * np.sin(th)
            dZ = dZ + d_r * np.sin(th) + d_t * np.cos(th)
        return -R / self.F * dZ, R / self.F * dR

    def __call__(self, R, Z, backward: bool = False):
        sign = -1.0 if backward else 1.0
        h = sign * 2.0 * math.pi / self.n / self.steps
        phi = 0.0
        for _ in range(self.steps):
            k1 = self.rhs(R, Z, phi)
            k2 = self.rhs(R + h / 2 * k1[0], Z + h / 2 * k1[1], phi + h / 2)
            k3 = self.rhs(R + h / 2 * k2[0], Z + h / 2 * k2[1], phi + h / 2)
            k4 = self.rhs(R + h * k3[0], Z + h * k3[1], phi + h)
            R = R + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            Z = Z + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            phi += h
        return R, Z


def _x_point_of(fmap, guess):
    from scipy.optimize import minimize

    sp = fmap.sp
    return minimize(lambda p: sp.ev(p[0], p[1], dx=1) ** 2 + sp.ev(p[0], p[1], dy=1) ** 2, guess,
                    method="Nelder-Mead", options={"xatol": 1e-11, "fatol": 1e-30}).x


def _fixed_point(fmap, x0):
    """Newton on $M(\\mathbf x) = \\mathbf x$; returns the point and the Jacobian's eigenpairs."""
    x = np.array(x0, dtype=float)

    def M(p):
        R, Z = fmap(np.array([p[0]]), np.array([p[1]]))
        return np.array([R[0], Z[0]])

    for _ in range(12):
        J = np.empty((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = 1e-7
            J[:, j] = (M(x + e) - M(x - e)) / 2e-7
        step = np.linalg.solve(J - np.eye(2), M(x) - x)
        x = x - step
        if np.linalg.norm(step) < 1e-13:
            break
    w, V = np.linalg.eig(J)
    return x, np.real(w), np.real(V)


def _thin(line: np.ndarray, min_step: float) -> np.ndarray:
    keep = [0]
    for i in range(1, len(line)):
        if np.hypot(*(line[i] - line[keep[-1]])) >= min_step:
            keep.append(i)
    return line[keep]


@lru_cache(maxsize=8)
def lobe_model(eps: float = 0.02, m: int = 8, n: int = 4, phase: float = 0.0, iterations: int = 26,
               points: int = 1000) -> dict:
    """Perturbed X-point, its multipliers and its stable/unstable manifolds, for the single-null Solov'ev."""
    from vaft.process.equilibrium import solovev_example

    eq = solovev_example("single_null", a_parameter=0.0)
    kappa_a = 1.1 * 1.7 * 0.5 * float(eq.lcfs.r.max() - eq.lcfs.r.min())
    return _lobe_model(eq, [float(eq.magnetic_axis[0]) - 0.1, -kappa_a], eps, m, n, phase, iterations, points)


def lobe_model_for(equilibrium, eps: float = 0.02, m: int = 8, n: int = 4, phase: float = 0.0, iterations: int = 26,
                   points: int = 1000) -> dict:
    """``lobe_model`` for any lower-single-null ``EquilibriumData``: the X-point is sought next to the lowest
    point of its boundary, and must be a saddle of the flux on the boundary value."""
    eq = equilibrium
    low = int(np.argmin(np.asarray(eq.lcfs.z)))
    guess = [float(eq.lcfs.r[low]), float(eq.lcfs.z[low])]
    fmap = _FieldLineMap(eq, m, n, 0.0, phase)
    x0 = _x_point_of(fmap, guess)
    sp = fmap.sp
    psi_x, psi_ax = float(sp.ev(*x0)), float(sp.ev(*eq.magnetic_axis))
    psi_b = float(eq.psi_boundary) / _per_radian_scale(eq)
    a_minor = 0.5 * float(np.ptp(np.asarray(eq.lcfs.r)))
    grad = math.hypot(float(sp.ev(*x0, dx=1)), float(sp.ev(*x0, dy=1)))
    hessian = float(sp.ev(*x0, dx=2)) * float(sp.ev(*x0, dy=2)) - float(sp.ev(*x0, dx=1, dy=1)) ** 2
    if (x0[1] > float(eq.magnetic_axis[1]) or grad * a_minor > 1e-4 * abs(psi_ax - psi_x) or hessian >= 0.0
            or abs(psi_x - psi_b) > 1e-2 * abs(psi_ax - psi_b)):
        raise ValueError("separatrix_lobes needs a lower-single-null equilibrium: no saddle of the flux was found "
                         "on the boundary below the axis")
    return _lobe_model(eq, x0, eps, m, n, phase, iterations, points)


def _lobe_model(eq, guess, eps, m, n, phase, iterations, points) -> dict:
    fmap = _FieldLineMap(eq, m, n, eps, phase)
    x0 = _x_point_of(fmap, guess)
    fmap.r_x = float(np.hypot(x0[0] - fmap.axis[0], x0[1] - fmap.axis[1]))
    xp, w, V = _fixed_point(fmap, x0)
    iu = int(np.argmax(np.abs(w)))
    lam = float(abs(w[iu]))
    wall = eq.limiter
    box = (float(np.min(wall.r)), float(np.max(wall.r)), float(np.min(wall.z)), float(np.max(wall.z)))
    manifolds = {}
    for name, idx, backward in (("unstable", iu, False), ("stable", 1 - iu, True)):
        v = V[:, idx] / np.linalg.norm(V[:, idx])
        branches = []
        for side in (1.0, -1.0):
            s = np.linspace(0.0, 1.0, points, endpoint=False)
            R = xp[0] + side * 2e-5 * v[0] * lam ** s
            Z = xp[1] + side * 2e-5 * v[1] * lam ** s
            seg = [np.stack([R, Z], -1)]
            for _ in range(iterations):
                R, Z = fmap(R, Z, backward=backward)
                # rounded every period to 10 nm: the map is chaotic near the X-point (lambda ~ 8), and a
                # last-bit difference between platforms must die in the rounding, not grow into the drawing
                R, Z = np.round(R, 8), np.round(Z, 8)
                # lines are kept a little past the target so its crossing can be found, then dropped
                inside = (R > box[0]) & (R < box[1]) & (Z > box[2] - 0.03) & (Z < box[3])
                R, Z = np.where(inside, R, np.nan), np.where(inside, Z, np.nan)
                seg.append(np.stack([R, Z], -1))
            branches.append(np.concatenate(seg))
        manifolds[name] = branches
    return {"equilibrium": eq, "x_point": tuple(xp), "x_point_unperturbed": tuple(x0), "multipliers": tuple(w),
            "lambda": lam, "manifolds": manifolds, "target_z": box[2], "box": box, "eps": eps, "m": m, "n": n,
            "psi_x": float(fmap.sp.ev(x0[0], x0[1]))}


def _runs(branch: np.ndarray, window) -> List[np.ndarray]:
    """Continuous pieces of a manifold branch inside the drawing window (NaN or a jump breaks a piece)."""
    r0, r1, z0, z1 = window
    ok = np.isfinite(branch).all(1) & (branch[:, 0] > r0) & (branch[:, 0] < r1) & (branch[:, 1] > z0) & (
        branch[:, 1] < z1)
    jumps = np.r_[False, np.hypot(*np.diff(branch, axis=0).T) > 0.01]
    runs, cur = [], []
    for p, good, jump in zip(branch, ok, jumps):
        if not good or jump:
            if len(cur) > 2:
                runs.append(np.array(cur))
            cur = []
        if good:
            cur.append(p)
    if len(cur) > 2:
        runs.append(np.array(cur))
    return runs


def crossing_segments(branches: Sequence[np.ndarray], target_z: float, max_gap: float = 0.01) -> List[np.ndarray]:
    """The (2, 2) segments, between neighbouring points of one piece, that cross the target plane."""
    out = []
    for b in branches:
        a, c = b[:-1], b[1:]
        ok = np.isfinite(a).all(1) & np.isfinite(c).all(1) & (np.hypot(*(c - a).T) < max_gap)
        cross = ok & ((a[:, 1] - target_z) * (c[:, 1] - target_z) < 0.0)
        out += [np.stack([p, q]) for p, q in zip(a[cross], c[cross])]
    return out


def strike_points(branches: Sequence[np.ndarray], target_z: float, max_gap: float = 0.01) -> np.ndarray:
    """Major radii where a manifold crosses the target plane, between neighbouring points of one piece."""
    hits = []
    for p, q in crossing_segments(branches, target_z, max_gap):
        t = (target_z - p[1]) / (q[1] - p[1])
        hits.append(p[0] + t * (q[0] - p[0]))
    return np.round(np.array(sorted(hits)), 6)


def separatrix_lobes(equilibrium=None, *, perturbation: float = 0.02, m: int = 8, n: int = 4, phase: float = 0.0,
                     labels: bool = True) -> Diagram:
    r"""Stable and unstable manifolds of a perturbed X-point, and the lobes between them.

    ``equilibrium`` is any lower-single-null ``EquilibriumData`` (``None``: the
    single-null Solov'ev equilibrium of ``solovev_example``) plus
    $\delta\psi = \epsilon\,\Delta\psi\,(r/r_X)^m\cos(m\vartheta - n\phi + \varphi_0)$
    ($\epsilon$ = ``perturbation``). The field-line map over one period
    $2\pi/n$ has a hyperbolic fixed point near the unperturbed X-point
    (Newton); its unstable manifold (grown forward in $\phi$) and stable
    manifold (backward) no longer coincide with the unperturbed separatrix
    (dashed) but cross it and each other, enclosing lobes; where the
    unstable manifold reaches the divertor target the strike point is split.
    Reduced Hamiltonian model: it does not reproduce a specific RMP
    discharge, whose traces belong to result plotting.
    """
    labels = _check_labels(labels)
    if isinstance(perturbation, bool) or not (isinstance(perturbation, (int, float)) and 0.0 <= perturbation <= 0.03):
        raise ValueError(f"perturbation must lie in [0, 0.03], not {perturbation!r}")
    for name, v, top in (("m", m, 16), ("n", n, 8)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 1 <= v <= top:
            raise ValueError(f"{name} must be an integer from 1 to {top}, not {v!r}")
    if equilibrium is None:
        model = lobe_model(float(perturbation), int(m), int(n), float(phase))
        size, zoom = 1.0, _S_ZOOM
    else:
        model = lobe_model_for(equilibrium, float(perturbation), int(m), int(n), float(phase))
        # the window and zoom scale with the machine: sized like the default at its minor radius
        default_eq = lobe_model(0.0)["equilibrium"]  # cached: the equilibrium the window was laid out for
        size = float(np.ptp(np.asarray(equilibrium.lcfs.r))) / float(np.ptp(np.asarray(default_eq.lcfs.r)))
        zoom = _S_ZOOM / size
    eq = model["equilibrium"]
    xp = np.array(model["x_point"])
    window = (xp[0] - 0.16 * size, xp[0] + 0.19 * size, model["target_z"] - 0.005 * size, xp[1] + 0.24 * size)
    ox, oy = -zoom * window[0], -zoom * window[2]

    def cm(pts):
        pts = np.asarray(pts, dtype=float)
        return np.stack([zoom * pts[..., 0] + ox, zoom * pts[..., 1] + oy], -1)

    items: List = []
    # the unperturbed separatrix, from the equilibrium flux
    from contourpy import LineType, contour_generator

    gen = contour_generator(x=eq.r, y=eq.z, z=(np.asarray(eq.psi, float) / _per_radian_scale(eq)).T,
                            line_type=LineType.Separate)
    for line in gen.lines(model["psi_x"] * (1.0 - 1e-6)):
        for run in _runs(np.asarray(line), window):
            items.append(Polyline.of(cm(_thin(run, 0.003)), "approx", role="unperturbed_separatrix"))
    for name, style in (("unstable", "trough"), ("stable", "crest")):
        for branch in model["manifolds"][name]:
            for run in _runs(branch, window):
                items.append(Polyline.of(cm(_thin(run, 0.0015)), style, role=f"{name}_manifold"))
    target = np.array([[window[0], model["target_z"]], [window[1], model["target_z"]]])
    items.append(Polyline.of(cm(target), "machine", role="target"))
    hits = {name: strike_points(model["manifolds"][name], model["target_z"]) for name in ("unstable", "stable")}
    # the split is sub-millimetre: a magnified inset of the target strip around the unstable manifold's hits
    inset = None
    if hits["unstable"].size:
        c = float(np.mean(hits["unstable"]))
        span = max(0.0025 * size, 0.7 * float(np.ptp(hits["unstable"])))
        inset = (c - span, c + span, model["target_z"], model["target_z"] + 1.2 * span)
        scale = 3.2 / (2 * span)  # the inset is 3.2 cm wide
        right_edge = float(cm([window[1], 0.0])[0])
        io = (right_edge + 0.8, 0.6)

        def icm(pts):
            pts = np.asarray(pts, dtype=float)
            return np.stack([io[0] + scale * (pts[..., 0] - inset[0]), io[1] + scale * (pts[..., 1] - inset[2])], -1)

        frame = np.array([[inset[0], inset[2]], [inset[1], inset[2]], [inset[1], inset[3]], [inset[0], inset[3]]])
        items.append(Polyline.of(icm(frame), "inset frame", role="inset", closed=True))
        for name, style in (("unstable", "trough"), ("stable", "crest")):
            # the manifold's last piece before the target, linear between its two neighbouring points, clipped
            for seg in crossing_segments(model["manifolds"][name], model["target_z"]):
                p, q = seg if seg[0, 1] > seg[1, 1] else seg[::-1]
                t_hit = (model["target_z"] - p[1]) / (q[1] - p[1])
                top_z = inset[3]
                t_top = max(0.0, (top_z - p[1]) / (q[1] - p[1])) if q[1] != p[1] else 0.0
                a_pt, b_pt = p + t_top * (q - p), p + t_hit * (q - p)
                if inset[0] <= b_pt[0] <= inset[1]:
                    a_pt[0] = min(max(a_pt[0], inset[0]), inset[1])
                    items.append(Polyline.of(icm(np.stack([a_pt, b_pt])), style, role=f"inset:{name}_manifold"))
        items.append(Polyline.of(icm(np.array([[inset[0], inset[2]], [inset[1], inset[2]]])), "machine",
                                 role="inset:target"))
        for r in hits["unstable"]:
            items.append(Marker(tuple(icm([r, model["target_z"]])), "o", "opoint", role="strike_point"))
        # where the inset is, on the main view
        items.append(Polyline.of(cm(frame), "inset frame", role="inset:locator", closed=True))
    items.append(Marker(tuple(cm(xp)), "x", "xpoint", role="x_point"))
    if labels:
        top = float(cm([0.0, window[3]])[1])
        right = float(cm([window[1], 0.0])[0])
        items += [
            Label((0.5 * right, top + 0.4), f"perturbed X-point: manifolds and lobes, $\\epsilon = {perturbation:g}$, "
                  f"$m/n = {m}/{n}$", "label", anchor="south", role="title"),
            Label((right + 0.4, top), "red: unstable manifold (forward in $\\phi$)\\\\ blue: stable manifold "
                  "(backward)\\\\ dashed: unperturbed separatrix\\\\ "
                  f"multipliers ${model['multipliers'][0]:.2f}$, ${model['multipliers'][1]:.3f}$\\\\ "
                  f"target hits: unstable {hits['unstable'].size}, stable {hits['stable'].size}",
                  "small label,align=left", anchor="north west", role="legend"),
            Label(tuple(cm([xp[0] + 0.012, xp[1]]) + [0.2, 0.0]), "X", "small label", anchor="west",
                  role="x_point"),
            Label((0.5 * right, -0.3), "divertor target", "small label", anchor="north", role="target"),
            *([Label((right + 0.8 + 1.6, 0.6 + 1.2 * 3.2 / 2 + 0.1), "inset: the target strip, $\\times"
                     f"{(3.2 / (inset[1] - inset[0])) / zoom:.0f}$\\\\ dots: the split strike points\\\\ "
                     f"({hits['unstable'].size} within {1e3 * float(np.ptp(hits['unstable'])):.1f} mm)",
                     "small label,align=center", anchor="south", role="inset")] if inset else []),
            _note("Reduced Hamiltonian model: a prescribed $\\delta\\psi$ on "
                  + ("an exact Solov'ev single null" if equilibrium is None else "the supplied equilibrium")
                  + ", field lines traced over one period $2\\pi/n$; not a specific RMP discharge",
                  0.5 * right + 2.0, -1.0),
        ]
    out = {"classification": "reduced_hamiltonian_model", **{k: model[k] for k in (
        "x_point", "x_point_unperturbed", "multipliers", "lambda", "target_z", "psi_x", "eps")},
        "strike_points": hits["unstable"], "strike_points_stable": hits["stable"],
        # a copy: the model is lru_cached, and a caller's edit must not reach the next build
        "manifolds": copy.deepcopy(model["manifolds"])}
    return Diagram("separatrix_lobes", Scene(tuple(items)), model=out)
