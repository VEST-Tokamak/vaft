"""The Grad--Shafranov problem: regions, boundaries, topology and problem classes (#1052).

``grad_shafranov_domain_decomposition``
    one flux function, three sources: plasma (force balance), vacuum
    (homogeneous) and coil (prescribed) regions, with their equations;
``fixed_vs_free_boundary_equilibrium``
    the boundary as an input (fixed) against the boundary as part of the
    solution (free);
``limiter_and_diverted_topologies``
    magnetic axis, closed surfaces and LCFS touching a limiter, against an
    X-point, separatrix, scrape-off layer and private-flux region;
``equilibrium_problem_taxonomy``
    forward/inverse and fixed/free as two separate axes;
``poloidal_flux_source_decomposition``
    $\\psi_\\mathrm{plasma} + \\psi_\\mathrm{coil} = \\psi_\\mathrm{total}$, and
    the LCFS and X-point only the total has.

The flux maps are a toy, not an equilibrium: a prescribed ring-current
distribution for the plasma and three coils, superposed through
``vaft.formula.green.green_psi_exact`` (full weber, $+2\\pi RA_\\phi$: the
COCOS-13 sign, a maximum on the axis for positive current). The topology
-- axis, X-point, limiter contact, LCFS -- is then found from the total flux
the way a free-boundary code finds it; a production solver would also make the
plasma current consistent with $p'(\\psi)$, $FF'(\\psi)$.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import List, Tuple

import numpy as np

from vaft.formula.equilibrium import grad_shafranov_source, toroidal_current_density_from_p_prime_ff_prime
from vaft.formula.green import green_psi_exact

from ._concept import box, connector
from ._equations import formula_equation
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

CONFIGURATIONS = ("limited", "diverted")
#: the toy machine [m]: plasma centre, minor radius, elongation, current; vessel; inboard limiter tip
_R0, _A, _KAPPA, _IP = 0.66, 0.18, 1.3, 1.0e5
_VESSEL = (0.25, 1.05, -0.8, 0.8)
_LIMITER = (0.42, 0.0)
#: coils (R, Z, current): an outboard vertical-field pair and, for the diverted case, a divertor coil
_VF_COILS = ((1.18, 0.6, -3.5e4), (1.18, -0.6, -3.5e4))
_DIVERTOR_COIL = (0.6, -0.92, 1.2e5)
_GRID_R = np.linspace(0.1, 1.3, 121)
_GRID_Z = np.linspace(-1.0, 1.0, 201)
#: drawing scale [cm per m]
_S = 5.0


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _note(text: str, x: float, y: float) -> Label:
    return Label((x, y), text, "note", anchor="north", role="note")


def plasma_rings() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The prescribed plasma current: rings on an elliptic cross-section, $j \\propto 1 - \\rho^2$."""
    rho = (np.arange(12) + 0.5) / 12 * _A
    theta = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
    rr, tt = np.meshgrid(rho, theta)
    weight = (1.0 - (rr / _A) ** 2) * rr  # current density times the ring's area element
    return (_R0 + rr * np.cos(tt)).ravel(), (_KAPPA * rr * np.sin(tt)).ravel(), (_IP * weight / weight.sum()).ravel()


def _coils(configuration: str):
    return _VF_COILS + ((_DIVERTOR_COIL,) if configuration == "diverted" else ())


def _ring_flux(r_src, z_src, currents) -> np.ndarray:
    RR, ZZ = np.meshgrid(_GRID_R, _GRID_Z, indexing="ij")
    out = np.zeros_like(RR)
    for r0, z0, current in zip(np.ravel(r_src), np.ravel(z_src), np.ravel(currents)):
        out += current * green_psi_exact(RR, ZZ, r0, z0)
    return out


@lru_cache(maxsize=None)
def flux_model(configuration: str = "diverted") -> dict:
    """$\\psi_\\mathrm{plasma}$, $\\psi_\\mathrm{coil}$, their sum on the grid, and its topology.

    The axis is the flux maximum near the plasma centre; the X-point, if the
    divertor coil is on, the null of $\\nabla\\psi$ below it; the boundary flux
    the larger of $\\psi_X$ and $\\psi$ at the limiter tip, since $\\psi$ falls
    from the axis outward and the first of the two reached bounds the plasma.
    """
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"configuration must be one of {CONFIGURATIONS}, not {configuration!r}")
    from scipy.interpolate import RectBivariateSpline
    from scipy.optimize import minimize

    psi_plasma = _ring_flux(*plasma_rings())
    coils = _coils(configuration)
    psi_coil = _ring_flux([c[0] for c in coils], [c[1] for c in coils], [c[2] for c in coils])
    total = psi_plasma + psi_coil
    spline = RectBivariateSpline(_GRID_R, _GRID_Z, total)

    def psi(p):
        return float(spline(p[0], p[1])[0, 0])

    def grad2(p):
        return float(spline(p[0], p[1], dx=1)[0, 0] ** 2 + spline(p[0], p[1], dy=1)[0, 0] ** 2)

    axis = minimize(lambda p: -psi(p), [_R0, 0.0], method="Nelder-Mead",
                    options={"xatol": 1e-9, "fatol": 1e-14}).x
    x_point = None
    if configuration == "diverted":
        found = minimize(grad2, [_R0, -0.45], method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-20}).x
        hessian = (spline(*found, dx=2)[0, 0] * spline(*found, dy=2)[0, 0]
                   - spline(*found, dx=1, dy=1)[0, 0] ** 2)
        r0, r1, z0, z1 = _VESSEL
        # a null, a saddle, inside the vessel -- a minimum of |grad psi|^2 elsewhere is not an X-point
        if not (grad2(found) < 1e-10 and hessian < 0.0 and r0 < found[0] < r1 and z0 < found[1] < z1):
            raise RuntimeError(f"no X-point found: |grad psi|^2 = {grad2(found):.2e}, det H = {hessian:.2e} "
                               f"at {tuple(found)}")
        x_point = (round(float(found[0]), 6), round(float(found[1]), 6))
    psi_limiter = psi(_LIMITER)
    psi_x = psi(x_point) if x_point else -math.inf
    limited_by = "x_point" if psi_x > psi_limiter else "limiter"
    psi_boundary = max(psi_x, psi_limiter)
    return {"configuration": configuration, "psi_plasma": psi_plasma, "psi_coil": psi_coil, "psi": total,
            "axis": (round(float(axis[0]), 6), round(float(axis[1]), 6)), "psi_axis": psi(axis),
            "x_point": x_point, "psi_x": psi_x if x_point else None, "psi_limiter": psi_limiter,
            "psi_boundary": psi_boundary, "limited_by": limited_by, "coils": coils}


# ---------------------------------------------------------------------------
# contour helpers
# ---------------------------------------------------------------------------


def _lines(field: np.ndarray, level: float) -> List[np.ndarray]:
    """Contour lines of ``field`` (indexed (R, Z)) at ``level``, as (R, Z) points [m]."""
    from contourpy import LineType, contour_generator

    gen = contour_generator(x=_GRID_R, y=_GRID_Z, z=field.T, line_type=LineType.Separate)
    return [np.round(np.asarray(line), 9) for line in gen.lines(level) if len(line) > 2]


def _encloses(line: np.ndarray, point) -> bool:
    from matplotlib.path import Path

    if not np.allclose(line[0], line[-1]):  # an open contour cannot enclose anything
        return False
    return bool(Path(line).contains_point(point))


def lcfs(model: dict) -> np.ndarray:
    """The last closed flux surface: the contour just inside $\\psi_b$ that encloses the axis."""
    level = model["psi_boundary"] + 1e-4 * (model["psi_axis"] - model["psi_boundary"])
    for line in _lines(model["psi"], level):
        if _encloses(line, model["axis"]):
            return line
    raise RuntimeError("no closed surface around the axis at the boundary flux")


def _clip(line: np.ndarray, box_: Tuple[float, float, float, float]) -> List[np.ndarray]:
    r0, r1, z0, z1 = box_
    inside = (line[:, 0] >= r0) & (line[:, 0] <= r1) & (line[:, 1] >= z0) & (line[:, 1] <= z1)
    runs, current = [], []
    for p, ok in zip(line, inside):
        if ok:
            current.append(p)
        elif current:
            runs.append(np.array(current))
            current = []
    if current:
        runs.append(np.array(current))
    return [r for r in runs if len(r) > 2]


def _cm(pts, offset=(0.0, 0.0)) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    return np.stack([offset[0] + _S * pts[..., 0], offset[1] + _S * pts[..., 1]], -1)


def _vessel_items(offset, limiter: bool = True) -> List:
    r0, r1, z0, z1 = _VESSEL
    items: List = [Polyline.of(_cm([(r0, z0), (r1, z0), (r1, z1), (r0, z1)], offset), "machine", role="vessel",
                               closed=True)]
    if limiter:
        tip = _LIMITER
        items.append(Polyline.of(_cm([(r0, tip[1] - 0.05), (tip[0], tip[1]), (r0, tip[1] + 0.05)], offset),
                                 "section fill", role="limiter", closed=True))
    return items


def _coil_items(coils, offset) -> List:
    h = 0.04
    return [Polyline.of(_cm([(r - h, z - h), (r + h, z - h), (r + h, z + h), (r - h, z + h)], offset),
                        "section fill", role="coil", closed=True) for r, z, _ in coils]


def _surfaces(model: dict, offset, *, inside_only: bool = False, n: int = 7) -> List:
    """Closed surfaces inside the LCFS (dark) and, unless ``inside_only``, open flux outside it (grey)."""
    items: List = []
    psi_ax, psi_b = model["psi_axis"], model["psi_boundary"]
    for f in np.linspace(0.15, 0.85, n):
        for line in _lines(model["psi"], psi_b + f * (psi_ax - psi_b)):
            if _encloses(line, model["axis"]):
                items.append(Polyline.of(_cm(line, offset), "orbit electron", role="closed_surface"))
    if not inside_only:
        for f in np.linspace(0.12, 0.9, 6):
            for line in _lines(model["psi"], psi_b - f * (psi_b - model["psi_limiter"] + 0.35 * (psi_ax - psi_b))):
                for run in _clip(line, _VESSEL):
                    items.append(Polyline.of(_cm(run, offset), "surface", role="open_flux"))
    return items


# ---------------------------------------------------------------------------
# diagrams
# ---------------------------------------------------------------------------


def grad_shafranov_domain_decomposition(*, labels: bool = True) -> Diagram:
    r"""Same flux function, different sources: plasma, vacuum and coil regions of one elliptic problem.

    The diverted toy equilibrium of ``flux_model``: inside the LCFS (shaded)
    the plasma source of ``toroidal_current_density_from_p_prime_ff_prime``;
    between plasma, coils and the outer boundary the homogeneous equation
    $\Delta^*\psi = 0$; in each coil the prescribed $-\mu_0RJ_{\phi,\mathrm{ext}}$
    -- all ``grad_shafranov_source`` of the region's $J_\phi$. The dashed frame
    is the computational boundary, where a free-boundary code imposes the
    Green's-function flux of all the currents.
    """
    labels = _check_labels(labels)
    model = flux_model("diverted")
    o = (0.0, 0.0)
    boundary = lcfs(model)
    items: List = [Polyline.of(_cm(boundary, o), "concept band", role="region:plasma", closed=True)]
    items += _vessel_items(o, limiter=False)
    items += _surfaces(model, o)
    items.append(Polyline.of(_cm(boundary, o), "boundary", role="lcfs", closed=True))
    items += _coil_items(model["coils"], o)
    r0, r1, z0, z1 = float(_GRID_R[0]), float(_GRID_R[-1]), float(_GRID_Z[0]), float(_GRID_Z[-1])
    items.append(Polyline.of(_cm([(r0, z0), (r1, z0), (r1, z1), (r0, z1)], o), "approx", role="computational_boundary",
                             closed=True))
    items.append(Marker(tuple(_cm(model["axis"], o)), "o", "opoint", role="axis"))
    if labels:
        xr = _S * r1 + 0.8
        rows = [
            ((_R0, 0.1), 3.0, "plasma: force balance fixes the source",
             f"$\\displaystyle {formula_equation(toroidal_current_density_from_p_prime_ff_prime)}$", "plasma"),
            ((0.97, -0.05), 0.9, "vacuum: homogeneous, Laplace-type (not $\\nabla^2$)",
             "$\\displaystyle \\Delta^*\\psi = 0$", "vacuum"),
            (model["coils"][1][:2], -1.2, "coil: prescribed external current",
             "$\\displaystyle \\Delta^*\\psi = -\\mu_0 R J_{\\phi,\\mathrm{ext}}$", "coil"),
        ]
        for point, y, head, eq, role in rows:
            p = _cm(point, o)
            items += [Polyline.of([tuple(p), (xr - 0.15, y)], "leader line", role=f"leader:{role}"),
                      Label((xr, y + 0.1), head, "small label", anchor="south west", role=f"region:{role}"),
                      Label((xr, y), eq, "formula box", anchor="north west", role="equations")]
        items += [
            Label((xr, 5.0), f"all regions: $\\displaystyle {formula_equation(grad_shafranov_source)}$",
                  "small label", anchor="west", role="equations"),
            Label(tuple(_cm((r1, z0), o) + [0.0, -0.15]), "computational boundary", "small label",
                  anchor="north east", role="computational_boundary"),
            _note("Same $\\psi$, different $J_\\phi$. Toy flux: prescribed ring currents and three coils "
                  "(\\texttt{green\\_psi\\_exact}), not a solved equilibrium", 0.5 * (xr + 4.0), _S * z0 - 0.8),
        ]
    return Diagram("grad_shafranov_domain_decomposition", Scene(tuple(items)),
                   model={k: model[k] for k in ("axis", "x_point", "psi_boundary", "limited_by")})


def fixed_vs_free_boundary_equilibrium(*, labels: bool = True) -> Diagram:
    r"""The boundary as an input against the boundary as part of the solution.

    Left, fixed boundary: the LCFS is given (thick) with $\psi = \psi_b$ on
    it, and only the plasma inside is solved -- nothing outside exists for the
    problem. Right, free boundary: coil currents and the plasma source give
    $\psi(R, Z)$ everywhere, and the LCFS -- here through an X-point -- is
    read off the solution's topology. Same toy flux as ``flux_model``.
    """
    labels = _check_labels(labels)
    model = flux_model("diverted")
    boundary = lcfs(model)
    left, right = (0.0, 0.0), (9.0, 0.0)
    items: List = []
    # fixed: the given boundary and the inside only
    items += _surfaces(model, left, inside_only=True)
    items.append(Polyline.of(_cm(boundary, left), "boundary", role="given_boundary", closed=True))
    items.append(Marker(tuple(_cm(model["axis"], left)), "o", "opoint", role="axis"))
    # free: everything, the boundary found
    items += _vessel_items(right, limiter=False)
    items += _surfaces(model, right)
    items += _coil_items(model["coils"], right)
    items.append(Polyline.of(_cm(boundary, right), "separatrix", role="found_boundary", closed=True))
    items.append(Marker(tuple(_cm(model["x_point"], right)), "x", "xpoint", role="x_point"))
    items.append(Marker(tuple(_cm(model["axis"], right)), "o", "opoint", role="axis"))
    if labels:
        top = _S * 0.8 + 1.0
        items.append(Label((_S * _R0, -_S * 0.6), "in practice a $\\psi_N \\approx 0.99$ surface, not the "
                           "separatrix", "small label", anchor="north", role="note:practice"))
        fixed_in = box(_S * _R0, top + 1.2, 5.6, 1.1, "given: LCFS shape, $\\psi_b$, $p'(\\psi)$, $FF'(\\psi)$",
                       role="fixed:input", latex=True)
        free_in = box(right[0] + _S * _R0, top + 1.2, 6.2, 1.1,
                      "given: coil currents, $p'(\\psi)$, $FF'(\\psi)$, $I_p$", role="free:input", latex=True)
        free_out = box(right[0] + _S * _R0, -_S * 0.92 - 1.5, 6.2, 1.1,
                       "found: $\\psi(R, Z)$ everywhere, then axis, X-point, LCFS", role="free:output", latex=True)
        items += list(fixed_in.items) + list(free_in.items) + list(free_out.items)
        items += [Arrow((_S * _R0, top + 0.6), (_S * _R0, top - 0.3), "connector", role="flow"),
                  Arrow((right[0] + _S * _R0, top + 0.6), (right[0] + _S * _R0, top - 0.3), "connector", role="flow"),
                  Arrow((right[0] + _S * 0.85, -_S * 0.8 - 0.1), (right[0] + _S * 0.85, -_S * 0.92 - 0.9),
                        "connector", role="flow"),
                  Label((_S * _R0, top + 2.1), "fixed boundary: boundary is an input", "label", anchor="south",
                        role="title"),
                  Label((right[0] + _S * _R0, top + 2.1), "free boundary: boundary is part of the solution", "label",
                        anchor="south", role="title"),
                  Label(tuple(_cm((_R0 + 0.2, 0.22), left)), "$\\psi = \\psi_b$ imposed", "small label",
                        anchor="west", role="given_boundary"),
                  Label((_S * _R0, -_S * 0.45), "solve inside only", "small label", anchor="north", role="note"),
                  _note("Same toy flux both sides; a fixed-boundary code (CHEASE) never sees the coils, "
                        "a free-boundary one (EFIT, TokaMaker) must", 0.5 * (right[0] + _S * 1.3), -_S * 0.92 - 2.4)]
    return Diagram("fixed_vs_free_boundary_equilibrium", Scene(tuple(items)),
                   model={"boundary": boundary, "x_point": model["x_point"], "axis": model["axis"]})


def limiter_and_diverted_topologies(*, labels: bool = True) -> Diagram:
    r"""Where the plasma ends: limiter contact against an X-point separatrix.

    Two toy equilibria of ``flux_model``, identical but for a divertor coil.
    Limited: the LCFS is the flux surface through the limiter tip, the first
    closed surface to touch a material object. Diverted: the divertor coil
    makes a null of $\nabla\psi$ (X-point) whose flux $\psi_X$ is reached
    before the limiter's, so the separatrix $\psi = \psi_X$ bounds the plasma;
    outside it the scrape-off layer carries open flux to the wall, and below
    the X-point lies the private-flux region.
    """
    labels = _check_labels(labels)
    items: List = []
    info = {}
    for i, configuration in enumerate(CONFIGURATIONS):
        o = (i * 9.0, 0.0)
        model = flux_model(configuration)
        boundary = lcfs(model)
        items += _vessel_items(o)
        items += _surfaces(model, o)
        items.append(Polyline.of(_cm(boundary, o), "boundary", role=f"lcfs:{configuration}", closed=True))
        items.append(Marker(tuple(_cm(model["axis"], o)), "o", "opoint", role="axis"))
        items += _coil_items(model["coils"], o)
        if configuration == "diverted":
            # a hair outside the saddle value: exactly at it the contour joins at the X-point are decided
            # by last-bit differences, which differ between platforms
            level = model["psi_x"] - 1e-6 * (model["psi_axis"] - model["psi_x"])
            for line in _lines(model["psi"], level):
                for run in _clip(line, _VESSEL):
                    items.append(Polyline.of(_cm(run, o), "separatrix", role="separatrix"))
            items.append(Marker(tuple(_cm(model["x_point"], o)), "x", "xpoint", role="x_point"))
        info[configuration] = {"axis": model["axis"], "x_point": model["x_point"], "limited_by": model["limited_by"],
                               "psi_boundary": model["psi_boundary"], "psi_limiter": model["psi_limiter"],
                               "lcfs": boundary}
        if labels:
            title = {"limited": "limited: LCFS on the limiter",
                     "diverted": "diverted: separatrix through an X-point"}[configuration]
            items.append(Label((o[0] + _S * 0.65, _S * 0.8 + 0.3), title, "label", anchor="south", role="title"))
            items += [Label(tuple(_cm((_VESSEL[0], 0.1), o) + [-0.1, 0.0]), "limiter", "small label", anchor="east",
                            role="limiter")]
            if configuration == "limited":
                items.append(Label(tuple(_cm((_R0, _KAPPA * _A + 0.12), o)), "LCFS", "small label", anchor="south",
                                   role="lcfs"))
            else:
                xp = _cm(model["x_point"], o)
                items += [Label(tuple(xp + [0.25, 0.0]), "X-point", "small label", anchor="west", role="x_point"),
                          Label(tuple(xp + [0.0, -0.95]), "private flux", "small label", anchor="north",
                                role="private_flux"),
                          Polyline.of(_cm([(0.97, -0.25), (1.1, -0.25)], o), "leader line", role="sol"),
                          Label(tuple(_cm((1.1, -0.25), o)), "SOL", "small label", anchor="west", role="sol"),
                          Polyline.of(_cm([(_R0 + 0.1, _KAPPA * _A + 0.2), (_R0 + 0.12, _KAPPA * _A + 0.04)], o),
                                      "leader line", role="separatrix"),
                          Label(tuple(_cm((_R0 + 0.1, _KAPPA * _A + 0.2), o)), "separatrix = LCFS", "small label",
                                anchor="south", role="separatrix")]
    if labels:
        items.append(_note("Dot: magnetic axis. Closed surfaces dark, open flux grey; the boundary is the larger of "
                           "$\\psi$ at the limiter tip and $\\psi_X$ -- the first reached from the axis", 8.0,
                           -_S * 0.92 - 0.5))
    return Diagram("limiter_and_diverted_topologies", Scene(tuple(items)), model=info)


def equilibrium_problem_taxonomy(*, labels: bool = True) -> Diagram:
    r"""Forward/inverse and fixed/free boundary are two axes, not one.

    Rows: forward (profiles and currents given, $\psi$ computed) and inverse
    (measurements given, profiles fitted). Columns: fixed boundary (LCFS
    given) and free boundary (coils given, LCFS found). EFIT is the inverse,
    free-boundary cell; a free-boundary problem need not be inverse
    (TokaMaker's forward solve, as VAFT uses it -- it also has a
    reconstruction mode) and an inverse one need not be free-boundary.
    """
    labels = _check_labels(labels)
    items: List = []
    cells = {
        ("forward", "fixed"): "prescribed-boundary equilibrium\\\\ (CHEASE)",
        ("forward", "free"): "coil-driven predictive equilibrium\\\\ (TokaMaker forward solve)",
        ("inverse", "fixed"): "boundary-constrained\\\\ reconstruction (given LCFS)",
        ("inverse", "free"): "magnetic / kinetic reconstruction\\\\ (EFIT)",
    }
    x_of = {"fixed": 3.2, "free": 9.2}
    y_of = {"forward": 0.0, "inverse": -2.2}
    for (row, col), text in cells.items():
        b = box(x_of[col], y_of[row], 5.6, 1.6, text, role=f"cell:{row}:{col}", latex=True)
        items += list(b.items)
    heads = [box(x_of["fixed"], 1.9, 5.6, 0.9, "fixed boundary: LCFS given", role="axis:fixed", latex=True,
                 style="concept leaf"),
             box(x_of["free"], 1.9, 5.6, 0.9, "free boundary: coils given, LCFS found", role="axis:free", latex=True,
                 style="concept leaf"),
             box(-1.9, y_of["forward"], 3.2, 1.6, "forward\\\\ sources $\\to\\psi$", role="axis:forward", latex=True,
                 style="concept leaf"),
             box(-1.9, y_of["inverse"], 3.2, 1.6, "inverse\\\\ data $\\to$ sources", role="axis:inverse", latex=True,
                 style="concept leaf")]
    for h in heads:
        items += list(h.items)
    if labels:
        items.append(_note("Free boundary and inverse are not synonyms: a code sits in a cell per use (TokaMaker "
                           "also reconstructs). DCON/GPEC take either kind as input", 3.6, y_of["inverse"] - 1.2))
    return Diagram("equilibrium_problem_taxonomy", Scene(tuple(items)), model={"cells": cells})


def poloidal_flux_source_decomposition(*, labels: bool = True) -> Diagram:
    r"""$\psi_\mathrm{plasma} + \psi_\mathrm{coil} = \psi_\mathrm{total}$: the topology belongs to the sum.

    The plasma ring currents and the three coils of the diverted
    ``flux_model``, each through ``green_psi_exact``, and their sum. Neither
    part alone has the X-point or the LCFS; both appear only in the total.
    Passive-structure (eddy) currents would add a further term,
    $\psi_\mathrm{passive}$, not drawn.
    """
    labels = _check_labels(labels)
    model = flux_model("diverted")
    items: List = []
    fields = (("psi_plasma", "plasma"), ("psi_coil", "coil"), ("psi", "total"))
    r0, r1, z0, z1 = _VESSEL
    for i, (key, name) in enumerate(fields):
        o = (i * 6.6, 0.0)
        field = model[key]
        inside = field[np.ix_((_GRID_R >= r0) & (_GRID_R <= r1), (_GRID_Z >= z0) & (_GRID_Z <= z1))]
        lo, hi = np.percentile(inside, [3.0, 97.0])
        for level in np.linspace(lo, hi, 14):
            for line in _lines(field, level):
                for run in _clip(line, _VESSEL):
                    items.append(Polyline.of(_cm(run, o), "surface", role=f"contour:{name}"))
        items.append(Polyline.of(_cm([(r0, z0), (r1, z0), (r1, z1), (r0, z1)], o), "machine", role="vessel",
                                 closed=True))
        if name != "coil":
            items.append(Polyline.of(_cm(np.stack([_R0 + _A * np.cos(np.linspace(0, 2 * math.pi, 61)),
                                                   _KAPPA * _A * np.sin(np.linspace(0, 2 * math.pi, 61))], -1), o),
                                     "approx", role="plasma_current", closed=True))
        if name != "plasma":
            items += _coil_items(model["coils"], o)
        if name == "total":
            items.append(Polyline.of(_cm(lcfs(model), o), "separatrix", role="lcfs", closed=True))
            items.append(Marker(tuple(_cm(model["x_point"], o)), "x", "xpoint", role="x_point"))
            items.append(Marker(tuple(_cm(model["axis"], o)), "o", "opoint", role="axis"))
        if labels:
            text = {"plasma": "$\\psi_\\mathrm{plasma}$", "coil": "$\\psi_\\mathrm{coil}$",
                    "total": "$\\psi_\\mathrm{total}$: LCFS, X-point"}[name]
            items.append(Label((o[0] + _S * 0.65, _S * z1 + 0.3), text, "label", anchor="south", role="title"))
        if i:
            items.append(Label((o[0] + _S * r0 - 0.45, 0.0), "$+$" if i == 1 else "$=$", "legend symbol",
                               role="operator"))
    if labels:
        items.append(_note("Each part through $\\psi = \\sum_k I_k\\,G(R, Z; R_k, Z_k)$ (\\texttt{green\\_psi\\_exact}); "
                           "dashed: the plasma current's support; contour levels per panel. $\\psi_\\mathrm{passive}$ "
                           "(eddy currents) not drawn. Only the sum has the X-point and LCFS", 6.6 + _S * 0.65, -_S * 0.92 - 0.5))
    return Diagram("poloidal_flux_source_decomposition", Scene(tuple(items)),
                   model={"axis": model["axis"], "x_point": model["x_point"]})
