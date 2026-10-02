"""Coordinates, geometry, meshes, mappings and topology: the spatial vocabulary (#1101).

Concept-oriented, not code-oriented: a structured grid or a logical mapping is
drawn as the numerical idea several equilibrium, transport and MHD codes share.

``tokamak_top_view``
    the torus seen from above: machine centre, $R$, $\\phi$ and its sense;
``cocos_orientation``
    what a COCOS index fixes -- the senses of $\\phi$ and $\\theta$, $I_p$ and
    $B_\\phi$ in or out of the page, where $\\psi$ increases and its unit --
    one panel per index, from :func:`vaft.data.cocos.cocos_spec`;
``machine_and_equilibrium_geometry``
    what the machine fixes (wall, limiter, coils, passive structure) against
    what the equilibrium decides (axis, surfaces, separatrix, X-point);
``structured_rz_grid``
    a rectangular $(R, Z)$ computational domain with the plasma boundary
    cutting through its cells;
``geometry_to_mesh``
    geometry, then regions, then an unstructured mesh whose resolution
    follows the region;
``logical_to_physical_mapping``
    a regular logical grid $(\\xi, \\eta)$ mapped onto flux-aligned
    curvilinear coordinates in $(R, Z)$;
``physical_to_flux_mapping``
    measurements at $(R, Z)$ relabelled by $\\psi_N$: the in- and outboard
    halves of a chord become one profile.

The machine and the flux are the toy free-boundary model of
:func:`vaft.diagram._gs_equilibrium.flux_model`, shared with the
Grad--Shafranov diagrams so a wall or an X-point looks the same everywhere.
:data:`SPATIAL_FAMILIES` files every builder of this vocabulary, here and in
earlier modules, under coordinate / geometry / mesh / mapping / topology.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

from vaft.data.cocos import cocos_spec
from vaft.formula.equilibrium import miller_surface

from ._chart import Chart, render_chart
from ._concept import box, connector
from ._gs_equilibrium import _GRID_R, _GRID_Z, _LIMITER, _VESSEL, _check_labels, _cm, _coil_items, _encloses
from ._gs_equilibrium import _lines, _note, _surfaces, flux_model, lcfs
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: The spatial vocabulary by family (#1101): each entry is a ``vaft.diagram`` builder.
SPATIAL_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "coordinate": ("tokamak_torus", "tokamak_top_view", "cocos_orientation", "flux_coordinates",
                   "poloidal_angle_comparison", "coordinates_vs_cocos"),
    "geometry": ("machine_and_equilibrium_geometry", "limiter_and_diverted_topologies", "shaping_family"),
    "mesh": ("structured_rz_grid", "geometry_to_mesh", "logical_to_physical_mapping", "sfl_coordinate_grids"),
    "mapping": ("physical_to_flux_mapping", "geometry_to_mesh", "logical_to_physical_mapping"),
    "topology": ("flux_surfaces", "x_point", "magnetic_island", "separatrix_lobes", "stochastic_layer"),
}


def _index(value, name: str) -> int:
    """``value`` as a plain int; bool is refused although it is an int subclass."""
    import operator

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not {value!r}")
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer, not {value!r}") from None


def _circle(cx: float, cy: float, r: float, n: int = 120, t0: float = 0.0, t1: float = 2.0 * math.pi) -> np.ndarray:
    t = np.linspace(t0, t1, n)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], -1)


def _out_of_page(symbol_at, *, out: bool, text: str, role: str, side: str = "east") -> List:
    """$\\odot$ (towards the reader) or $\\otimes$ (away), with its quantity."""
    glyph = "$\\odot$" if out else "$\\otimes$"
    return [Label(symbol_at, glyph, "legend symbol", role=role),
            Label((symbol_at[0] + (0.35 if side == "east" else -0.35), symbol_at[1]), text, "small label",
                  anchor="west" if side == "east" else "east", role=role)]


# ---------------------------------------------------------------------------
# coordinate
# ---------------------------------------------------------------------------


def _phi_counterclockwise_from_above(spec) -> bool:
    """$(R, \\phi, Z)$ right-handed ($\\sigma_{R\\phi Z} = +1$) puts $\\phi$ counter-clockwise seen from above."""
    return spec.sigma_rpz == 1


def tokamak_top_view(*, cocos: int = 11, n_coils: int = 12, labels: bool = True) -> Diagram:
    r"""The torus seen from above: machine centre, major radius $R$ and toroidal angle $\phi$.

    The plasma is the annulus $R_0 - a < R < R_0 + a$, the vessel the two
    circles around it and the ``n_coils`` toroidal-field coils cross it
    radially. $\phi$ is measured from the reference ray $\phi = 0$; its sense
    comes from ``cocos`` ($\sigma_{R\phi Z} = +1$: counter-clockwise seen from
    above, with $Z$ towards the reader). Representative, not any machine.
    """
    labels = _check_labels(labels)
    if not (isinstance(n_coils, int) and 4 <= n_coils <= 24):
        raise ValueError(f"n_coils must be an integer from 4 to 24, not {n_coils!r}")
    cocos = _index(cocos, "cocos")
    spec = cocos_spec(cocos)
    ccw = _phi_counterclockwise_from_above(spec)
    R0, a, r_in, r_out = 3.0, 1.0, 1.6, 4.5
    items: List = [Polyline.of(_circle(0, 0, R0 + a), "region plasma", role="plasma", closed=True),
                   Polyline.of(_circle(0, 0, R0 - a), "hole", role="plasma", closed=True)]
    for r in (r_in, r_out):
        items.append(Polyline.of(_circle(0, 0, r), "machine", role="vessel", closed=True))
    items.append(Polyline.of(_circle(0, 0, R0), "approx", role="magnetic_axis_circle", closed=True))
    w = 0.22
    for k in range(n_coils):
        t = 2.0 * math.pi * (k + 0.5) / n_coils
        c, s = math.cos(t), math.sin(t)
        corners = [(r * c - h * s, r * s + h * c) for r, h in ((r_in - 0.4, -w), (r_out + 0.4, -w),
                                                              (r_out + 0.4, w), (r_in - 0.4, w))]
        items.append(Polyline.of(corners, "machine", role="tf_coil", closed=True))
    phi_point = math.radians(50.0) * (1 if ccw else -1)
    items += [Polyline.of([(0, 0), (r_out + 1.0, 0)], "frame axis", role="phi_reference"),
              Arrow((0, 0), (R0 * math.cos(phi_point), R0 * math.sin(phi_point)), "vector", role="major_radius"),
              Polyline.of(_circle(0, 0, 2.1, 40, 0.0, phi_point), "angle arc", role="toroidal_angle"),
              Marker((R0 * math.cos(phi_point), R0 * math.sin(phi_point)), "o", "diagnostic point", role="point"),
              Polyline.of([(-0.15, -0.15), (0.15, 0.15)], "frame axis", role="centre"),
              Polyline.of([(-0.15, 0.15), (0.15, -0.15)], "frame axis", role="centre")]
    if labels:
        mid = 0.5 * phi_point
        items += [Label((2.45 * math.cos(mid), 2.45 * math.sin(mid)), "$\\phi$", "label", role="toroidal_angle"),
                  Label((0.6 * R0 * math.cos(phi_point) - 0.35 * math.sin(phi_point),
                         0.6 * R0 * math.sin(phi_point) + 0.35 * math.cos(phi_point)), "$R$", "label",
                        role="major_radius"),
                  Label((r_out + 1.05, 0), "$\\phi = 0$", "small label", anchor="west", role="phi_reference"),
                  Polyline.of([(0, 0), (-r_out - 0.6, -r_out + 0.4)], "leader line", role="leader:centre"),
                  Label((-r_out - 0.65, -r_out + 0.4), "machine centre, $Z$ out of page", "small label",
                        anchor="east", role="centre"),
                  Polyline.of([(-(R0 + 0.5) * 0.7071, (R0 + 0.5) * 0.7071), (-r_out - 0.6, r_out - 0.4)],
                              "leader line", role="leader:plasma"),
                  Label((-r_out - 0.65, r_out - 0.4), "plasma", "small label", anchor="east", role="plasma"),
                  Label((0, -r_out - 0.75), f"{n_coils} toroidal-field coils (representative)", "small label",
                        anchor="north", role="tf_coil"),
                  _note(f"Seen from above; $\\phi$ {'counter-clockwise' if ccw else 'clockwise'} "
                        f"($\\sigma_{{R\\phi Z}} = {spec.sigma_rpz:+d}$, COCOS {cocos}). "
                        "Dashed: the magnetic axis $R = R_0$", 0.0, -r_out - 1.5)]
    return Diagram("tokamak_top_view", Scene(tuple(items)),
                   model={"cocos": cocos, "phi_counterclockwise_from_above": ccw, "n_coils": n_coils,
                          "phi_point": phi_point})


def cocos_orientation_signs(cocos: int, *, sigma_ip: int = 1, sigma_b0: int = 1) -> dict:
    """The orientation a COCOS index fixes, as drawn by :func:`cocos_orientation`.

    With $R$ to the right and $Z$ up: $\\phi$ points out of the page when
    $(R, Z, \\phi)$ is right-handed ($\\sigma_{R\\phi Z} = -1$); $\\theta$ runs
    counter-clockwise when $\\sigma_{R\\phi Z}\\,\\sigma_{\\rho\\theta\\phi} = -1$
    (Sauter and Medvedev 2013, Table I). ``dpsi`` is the sign of
    $\\mathrm d\\psi/\\mathrm d\\rho$ and ``q`` of the safety factor, from
    :meth:`vaft.data.cocos.CocosSpec.expected_sign`.
    """
    if isinstance(sigma_ip, bool) or isinstance(sigma_b0, bool) or sigma_ip not in (1, -1) or sigma_b0 not in (1, -1):
        raise ValueError(f"sigma_ip and sigma_b0 must be +1 or -1, not {sigma_ip!r}, {sigma_b0!r}")
    cocos = _index(cocos, "cocos")
    spec = cocos_spec(cocos)
    return {
        "cocos": cocos,
        "phi_out_of_page": spec.sigma_rpz == -1,
        "phi_counterclockwise_from_above": spec.sigma_rpz == 1,
        "theta_counterclockwise": spec.sigma_rpz * spec.sigma_rhotp == -1,
        "ip_out_of_page": (spec.sigma_rpz == -1) == (sigma_ip == 1),
        "b0_out_of_page": (spec.sigma_rpz == -1) == (sigma_b0 == 1),
        "dpsi": spec.expected_sign("dpsi", sigma_ip=sigma_ip, sigma_b0=sigma_b0),
        "q": spec.expected_sign("q", sigma_ip=sigma_ip, sigma_b0=sigma_b0),
        "psi_per_radian": spec.exp_bp == 0,
        "sigma_bp": spec.sigma_bp, "sigma_rpz": spec.sigma_rpz, "sigma_rhotp": spec.sigma_rhotp,
        "sigma_ip": sigma_ip, "sigma_b0": sigma_b0,
    }


#: one COCOS panel [cm]: plasma centre, minor radius, horizontal and vertical pitch of the panel grid
_COCOS_CENTRE, _COCOS_R, _COCOS_DX, _COCOS_DY = (2.9, 0.0), 1.45, 7.6, 7.2
_COCOS_PER_ROW = 4


def _cocos_panel(s: dict, origin, labels: bool) -> List:
    """One generic poloidal cross-section drawing the orientation in ``s`` (:func:`cocos_orientation_signs`)."""
    ox, oy = origin
    cx, cy = ox + _COCOS_CENTRE[0], oy + _COCOS_CENTRE[1]
    r = _COCOS_R
    items: List = [Arrow((ox, oy - 2.1), (ox + 5.4, oy - 2.1), "frame axis arrow", role="R_axis"),
                   Arrow((ox + 0.2, oy - 2.3), (ox + 0.2, oy + 2.1), "frame axis arrow", role="Z_axis"),
                   Polyline.of(_circle(cx, cy, r), "lcfs", role="plasma_boundary", closed=True)]
    for f in (0.35, 0.68):
        items.append(Polyline.of(_circle(cx, cy, f * r), "surface", role="flux_surface", closed=True))
    sign = 1 if s["theta_counterclockwise"] else -1
    items.append(Polyline.of(_circle(cx, cy, 0.84 * r, 40, 0.0, sign * math.radians(80)), "angle arc",
                             role="poloidal_angle"))
    # the arrow points where psi increases: outward when sigma_Ip sigma_Bp > 0, inward otherwise
    t = -sign * math.radians(40)
    inner = (cx + 0.5 * r * math.cos(t), cy + 0.5 * r * math.sin(t))
    outer = (cx + 1.45 * r * math.cos(t), cy + 1.45 * r * math.sin(t))
    outward = s["dpsi"] > 0
    items.append(Arrow(inner if outward else outer, outer if outward else inner, "vector", role="psi_gradient"))
    # the magnetic axis carries the current marker: blue, the axis colour everywhere else
    items.append(Label((cx, cy), "$\\odot$" if s["ip_out_of_page"] else "$\\otimes$", "axis current", role="axis"))
    if labels:
        unit = "\\mathrm{Wb/rad}" if s["psi_per_radian"] else "\\mathrm{Wb}"
        items += [Label((ox + 5.45, oy - 2.1), "$R$", "label", anchor="west", role="R_axis"),
                  Label((ox + 0.2, oy + 2.15), "$Z$", "label", anchor="south", role="Z_axis"),
                  Label((cx + 0.6 * r * math.cos(sign * 0.65), cy + 0.6 * r * math.sin(sign * 0.65)),
                        "$\\theta$", "label", role="poloidal_angle"),
                  Label((cx - 0.3, cy - 0.12), "$I_p$", "small label", anchor="north east", role="ip"),
                  Label((outer[0] + 0.08, outer[1]), f"$\\psi\\ [{unit}]$", "small label",
                        anchor="south west" if t > 0 else "north west", role="psi_gradient")]
        items += _out_of_page((ox + 0.75, oy + 1.55), out=s["b0_out_of_page"], text="$B_\\phi$", role="b0")
        items += _out_of_page((ox + 0.75, oy - 1.55), out=s["phi_out_of_page"], text="$\\phi$", role="phi")
        items += [Label((ox + 2.7, oy + 3.3), f"COCOS = {s['cocos']}", "label", anchor="south", role="title"),
                  Label((ox + 2.7, oy + 2.75), f"$(\\sigma_{{B_p}}, \\sigma_{{R\\phi Z}}, \\sigma_{{\\rho\\theta\\phi}}, "
                        f"\\psi) = ({s['sigma_bp']:+d}, {s['sigma_rpz']:+d}, {s['sigma_rhotp']:+d}, {unit})$",
                        "subtitle", anchor="south", role="subtitle")]
    return items


def cocos_orientation(cocos=11, *, sigma_ip: int = 1, sigma_b0: int = 1, labels: bool = True) -> Diagram:
    r"""What a COCOS index fixes, on a generic poloidal cross-section: one panel per index.

    ``cocos`` is one index or a sequence of them, drawn as panels in rows of
    four at one scale. In each panel $R$ points right and $Z$ up;
    $\odot$/$\otimes$ say whether $\phi$, the toroidal field and the plasma
    current (the blue symbol on the magnetic axis) point out of or into the
    page; the arc is
    the sense of $\theta$; the arrow points where $\psi$ increases and carries
    its unit (Wb for full flux, $e_{B_p} = 1$; Wb/rad per radian). Title and
    subtitle give the index and
    $(\sigma_{B_p}, \sigma_{R\phi Z}, \sigma_{\rho\theta\phi}, \psi)$.
    ``sigma_ip`` and ``sigma_b0`` are the signs along $\phi$
    (:func:`cocos_orientation_signs`; Sauter and Medvedev,
    Comput. Phys. Commun. 184 (2013) 293). ``Diagram.model["panels"]`` holds
    the signs of every panel.
    """
    labels = _check_labels(labels)
    if isinstance(cocos, (str, bytes)):
        raise ValueError(f"cocos must be an index or a non-empty sequence of indices, not {cocos!r}")
    try:
        indices = (_index(cocos, "cocos"),)
    except ValueError:
        if isinstance(cocos, bool):
            raise
        try:
            indices = tuple(_index(i, "cocos") for i in cocos)
        except TypeError:
            raise ValueError(f"cocos must be an index or a non-empty sequence of indices, not {cocos!r}") from None
    if not indices:
        raise ValueError(f"cocos must be an index or a non-empty sequence of indices, not {cocos!r}")
    signs = tuple(cocos_orientation_signs(i, sigma_ip=sigma_ip, sigma_b0=sigma_b0) for i in indices)
    items: List = []
    for k, s in enumerate(signs):
        row, col = divmod(k, _COCOS_PER_ROW)
        items += _cocos_panel(s, (col * _COCOS_DX, -row * _COCOS_DY), labels)
    return Diagram("cocos_orientation", Scene(tuple(items)), model={"panels": signs})


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

#: passive conductors inside the vessel [m]: two outboard plates
_PASSIVE = ((0.97, 0.25, 0.99, 0.5), (0.97, -0.5, 0.99, -0.25))


def _rect(r0, z0, r1, z1, offset, style, role):
    return Polyline.of(_cm([(r0, z0), (r1, z0), (r1, z1), (r0, z1)], offset), style, role=role, closed=True)


def _machine_items(offset, *, faded: bool = False) -> List:
    r0, r1, z0, z1 = _VESSEL
    style = "mesh hidden" if faded else "machine"
    items: List = [_rect(r0, z0, r1, z1, offset, style, "vessel")]
    tip = _LIMITER
    items.append(Polyline.of(_cm([(r0, tip[1] - 0.05), tip, (r0, tip[1] + 0.05)], offset),
                             style if faded else "section fill", role="limiter", closed=True))
    for p in _PASSIVE:
        items.append(_rect(*p, offset, style if faded else "region conductor", "passive_structure"))
    if not faded:
        items += _coil_items(flux_model("diverted")["coils"], offset)
    return items


def _leader(point_m, offset, text_at, text, role, anchor="west") -> List:
    p = _cm(point_m, offset)
    return [Polyline.of([tuple(p), text_at], "leader line", role=f"leader:{role}"),
            Label(text_at, text, "small label", anchor=anchor, role=role)]


def machine_and_equilibrium_geometry(*, labels: bool = True) -> Diagram:
    r"""Machine geometry against equilibrium geometry: built objects against found ones.

    Left, what construction fixes: the symmetry axis $R = 0$, the vessel wall,
    the limiter, the poloidal-field coils and the passive conductors. Right,
    what each equilibrium decides inside that machine: the magnetic axis,
    nested closed surfaces, the LCFS (here the separatrix through an X-point),
    the scrape-off layer and the private-flux region. The machine is redrawn
    faint on the right only for reference. Same toy flux as ``flux_model``.
    """
    labels = _check_labels(labels)
    model = flux_model("diverted")
    left, right = (0.0, 0.0), (11.0, 0.0)
    boundary = lcfs(model)
    items: List = [Polyline.of([(0, -4.6), (0, 4.6)], "approx", role="symmetry_axis")]
    items += _machine_items(left)
    items += _machine_items(right, faded=True)
    surfaces = _surfaces(model, right)
    items += surfaces
    items.append(Polyline.of(_cm(boundary, right), "separatrix", role="separatrix", closed=True))
    items.append(Marker(tuple(_cm(model["x_point"], right)), "x", "xpoint", role="x_point"))
    items.append(Marker(tuple(_cm(model["axis"], right)), "o", "opoint", role="axis"))
    # a point on an outboard open flux line, a little below the midplane
    outboard = [p for it in surfaces if it.role == "open_flux" for p in it.points
                if p[0] > right[0] + 5.0 * (model["axis"][0] + 0.2)]
    sol_cm = min(outboard, key=lambda p: abs(p[1] + 1.2))
    if labels:
        xl, xr = 6.0, right[0] + 6.0
        items += [Label((2.6, 5.4), "machine geometry: fixed by construction", "label", anchor="south",
                        role="title"),
                  Label((right[0] + 2.6, 5.4), "equilibrium geometry: found for each plasma", "label",
                        anchor="south", role="title"),
                  Label((0.1, 4.6), "symmetry axis $R = 0$", "small label", anchor="south west",
                        role="symmetry_axis")]
        items += _leader((1.05, 0.65), left, (xl, 3.6), "vessel wall", "vessel")
        items += _leader(_LIMITER, left, (xl - 4.7, -1.4), "limiter", "limiter", anchor="east")
        items += _leader((0.99, 0.38), left, (xl, 1.6), "passive conductors", "passive_structure")
        items += _leader((1.18, -0.6), left, (xl, -2.6), "PF coils", "coil")
        items += _leader(model["axis"], right, (xr, 1.2), "magnetic axis", "axis")
        items += _leader((model["axis"][0] + 0.08, 0.12), right, (xr, 2.4), "closed flux surfaces",
                         "closed_surface")
        items += _leader(model["x_point"], right, (xr, -2.2), "X-point", "x_point")
        items += _leader((float(boundary[:, 0].max()), 0.0), right, (xr, 0.0), "LCFS = separatrix", "separatrix")
        items += [Polyline.of([sol_cm, (xr, -1.1)], "leader line", role="leader:sol"),
                  Label((xr, -1.1), "scrape-off layer (open)", "small label", anchor="west", role="sol")]
        items += _leader((model["x_point"][0], model["x_point"][1] - 0.18), right, (xr, -3.4),
                         "private-flux region", "private_flux")
        items.append(_note("The machine is the same for every shot; the right-hand objects move with each "
                           "equilibrium. Toy flux (\\texttt{flux\\_model}), not a machine", 8.5, -5.0))
    return Diagram("machine_and_equilibrium_geometry", Scene(tuple(items)),
                   model={"axis": model["axis"], "x_point": model["x_point"], "machine": ("vessel", "limiter",
                          "coil", "passive_structure", "symmetry_axis"),
                          "equilibrium": ("axis", "closed_surface", "separatrix", "x_point")})


# ---------------------------------------------------------------------------
# mesh
# ---------------------------------------------------------------------------


def structured_rz_grid(*, n_r: int = 13, n_z: int = 21, labels: bool = True) -> Diagram:
    r"""A structured $(R, Z)$ grid: rectangular computational domain, curved plasma boundary.

    The ``n_r`` $\times$ ``n_z`` nodes span the dashed computational
    rectangle, which contains the physical domain (vessel) and the coils. The
    LCFS cuts through cells: a rectangular grid is not aligned with the
    flux surfaces, so the boundary lies between nodes (filled: nodes inside
    the plasma). Same toy flux as ``flux_model``.
    """
    labels = _check_labels(labels)
    if not (isinstance(n_r, int) and isinstance(n_z, int) and 4 <= n_r <= 65 and 4 <= n_z <= 129):
        raise ValueError(f"n_r must be 4-65 and n_z 4-129 integers, not {n_r!r}, {n_z!r}")
    model = flux_model("diverted")
    boundary = lcfs(model)
    o = (0.0, 0.0)
    r = np.linspace(_GRID_R[0], _GRID_R[-1], n_r)
    z = np.linspace(_GRID_Z[0], _GRID_Z[-1], n_z)
    items: List = [Polyline.of(_cm(boundary, o), "region plasma", role="plasma", closed=True)]
    for x in r:
        items.append(Polyline.of(_cm([(x, z[0]), (x, z[-1])], o), "mesh", role="grid"))
    for y in z:
        items.append(Polyline.of(_cm([(r[0], y), (r[-1], y)], o), "mesh", role="grid"))
    items.append(_rect(r[0], z[0], r[-1], z[-1], o, "approx", "computational_boundary"))
    items += _machine_items(o)
    items.append(Polyline.of(_cm(boundary, o), "lcfs", role="lcfs", closed=True))
    inside = [(x, y) for x in r for y in z if _encloses(boundary, (x, y))]
    if not inside:
        raise ValueError(f"a {n_r} x {n_z} grid puts no node inside the plasma; refine it")
    items += [Marker(tuple(_cm(p, o)), ".", "grid node", role="plasma_node") for p in inside]
    if labels:
        xl = _cm((r[-1], 0), o)[0] + 0.9
        items += _leader((r[-1], z[-1]), o, (xl, 4.6), "computational domain: a rectangle in $(R, Z)$",
                         "computational_boundary")
        items += _leader((1.05, 0.3), o, (xl, 2.6), "physical domain: inside the vessel", "vessel")
        items += _leader((float(boundary[:, 0].max()), 0.0), o, (xl, 0.6),
                         "plasma boundary between nodes, not on them", "lcfs")
        items += _leader(inside[len(inside) // 2], o, (xl, -1.4), "nodes inside the plasma", "plasma_node")
        items.append(Label((xl, -3.2), f"$\\Delta R = {1e3 * (r[1] - r[0]):.0f}$ mm, "
                           f"$\\Delta Z = {1e3 * (z[1] - z[0]):.0f}$ mm, uniform", "small label", anchor="west",
                           role="spacing"))
        items.append(_note("Resolution is the same everywhere; boundary-following needs a mapped or "
                           "unstructured mesh", 6.0, -5.6))
    return Diagram("structured_rz_grid", Scene(tuple(items)),
                   model={"n_r": n_r, "n_z": n_z, "dr": float(r[1] - r[0]), "dz": float(z[1] - z[0]),
                          "n_inside": len(inside)})


#: target edge length of the unstructured mesh by region [m]
MESH_SPACING = {"plasma": 0.04, "conductor": 0.03, "vacuum": 0.11}
_WALL = 0.03  # vessel wall thickness [m]


def _resample(line: np.ndarray, h: float) -> np.ndarray:
    seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(line, axis=0).T))]
    s = np.linspace(0.0, seg[-1], max(4, int(round(seg[-1] / h))) + 1)[:-1]
    return np.stack([np.interp(s, seg, line[:, 0]), np.interp(s, seg, line[:, 1])], -1)


def _rect_line(r0, z0, r1, z1) -> np.ndarray:
    return np.array([(r0, z0), (r1, z0), (r1, z1), (r0, z1), (r0, z0)], dtype=float)


def _lattice(r0, z0, r1, z1, h, rng) -> np.ndarray:
    rr, zz = np.meshgrid(np.arange(r0 + h / 2, r1, h), np.arange(z0 + h / 2, z1, h), indexing="ij")
    pts = np.stack([rr.ravel(), zz.ravel()], -1)
    return pts + rng.uniform(-0.18 * h, 0.18 * h, pts.shape)


def _region_of(point, boundary, coils) -> str:
    r, z = point
    if _encloses(boundary, point):
        return "plasma"
    for rc, zc, _ in coils:
        if abs(r - rc) <= 0.04 and abs(z - zc) <= 0.04:
            return "conductor"
    v0, v1, w0, w1 = _VESSEL
    in_outer = v0 - _WALL <= r <= v1 + _WALL and w0 - _WALL <= z <= w1 + _WALL
    in_inner = v0 < r < v1 and w0 < z < w1
    return "conductor" if in_outer and not in_inner else "vacuum"


@lru_cache(maxsize=None)
def unstructured_mesh() -> dict:
    """A Delaunay mesh of the computational box whose node spacing follows :data:`MESH_SPACING` by region.

    Not constrained: triangles are classed by their centroid, so a few at a
    region edge straddle it. Deterministic: a fixed seed jitters every node, so
    no four nodes are cocircular and the triangulation has no ties to break.
    The arrays are read-only (the result is cached).
    """
    from scipy.spatial import Delaunay

    model = flux_model("diverted")
    boundary = lcfs(model)
    coils = model["coils"]
    rng = np.random.default_rng(1101)
    h = MESH_SPACING
    r0, r1, z0, z1 = float(_GRID_R[0]), float(_GRID_R[-1]), float(_GRID_Z[0]), float(_GRID_Z[-1])
    v0, v1, w0, w1 = _VESSEL
    parts = [_resample(_rect_line(r0, z0, r1, z1), h["vacuum"]),
             _resample(boundary, 0.8 * h["plasma"]),
             _resample(_rect_line(v0, w0, v1, w1), h["conductor"]),
             _resample(_rect_line(v0 - _WALL, w0 - _WALL, v1 + _WALL, w1 + _WALL), h["conductor"])]
    parts += [_resample(_rect_line(rc - 0.04, zc - 0.04, rc + 0.04, zc + 0.04), h["conductor"]) for rc, zc, _ in coils]
    lattice = np.vstack([_lattice(r0, z0, r1, z1, h["vacuum"], rng),
                         _lattice(boundary[:, 0].min(), boundary[:, 1].min(), boundary[:, 0].max(),
                                  boundary[:, 1].max(), h["plasma"], rng)])
    keep = []
    for p in lattice:
        region = _region_of(p, boundary, coils)
        if region == "conductor":
            continue
        near = min(float(np.min(np.hypot(*(part - p).T))) for part in parts)
        if (region == "plasma" and near > 0.6 * h["plasma"]) or (region == "vacuum" and near > 0.6 * h["vacuum"]
                                                                  and not _encloses(boundary, p)):
            if region == "vacuum" or _encloses(boundary, p):
                keep.append(p)
    nodes = np.vstack(parts + [np.array(keep)])
    # break exact cocircular ties (square outlines): qhull resolves a tie differently across platforms
    nodes = np.unique(np.round(nodes + rng.uniform(-0.04, 0.04, nodes.shape) * h["conductor"], 6), axis=0)
    tri = Delaunay(nodes)
    regions = [_region_of(nodes[t].mean(axis=0), boundary, coils) for t in tri.simplices]
    edges = set()
    for t in tri.simplices:
        for i, j in ((0, 1), (1, 2), (0, 2)):
            edges.add((min(t[i], t[j]), max(t[i], t[j])))
    lengths: Dict[str, List[float]] = {k: [] for k in h}
    for t, region in zip(tri.simplices, regions):
        p = nodes[t]
        lengths[region] += [float(np.hypot(*(p[i] - p[j]))) for i, j in ((0, 1), (1, 2), (0, 2))]
    triangles = tri.simplices.copy()
    for array in (nodes, triangles):
        array.setflags(write=False)
    return {"nodes": nodes, "triangles": triangles, "regions": tuple(regions), "edges": tuple(sorted(edges)),
            "median_edge": {k: float(np.median(v)) for k, v in lengths.items() if v},
            "count": {k: regions.count(k) for k in h}}


def geometry_to_mesh(*, labels: bool = True) -> Diagram:
    r"""Geometry, then regions, then mesh: the numerical workflow, not one code's.

    Left, the physical geometry: wall, coils, plasma boundary. Middle, the
    regions a solver distinguishes: plasma, vacuum and conductors (vessel
    wall, coils). Right, an unstructured triangular mesh of the
    computational box whose resolution follows the region
    (:data:`MESH_SPACING`: fine in the plasma and conductors, coarse in the
    vacuum) -- from :func:`unstructured_mesh`.
    """
    labels = _check_labels(labels)
    model = flux_model("diverted")
    boundary = lcfs(model)
    mesh = unstructured_mesh()
    dx = 9.5
    panels = [(0.0, 0.0), (dx, 0.0), (2 * dx, 0.0)]
    r0, r1, z0, z1 = float(_GRID_R[0]), float(_GRID_R[-1]), float(_GRID_Z[0]), float(_GRID_Z[-1])
    v0, v1, w0, w1 = _VESSEL
    items: List = []
    # geometry
    o = panels[0]
    items += [_rect(v0, w0, v1, w1, o, "machine", "vessel"),
              Polyline.of(_cm(boundary, o), "lcfs", role="lcfs", closed=True)]
    items += _coil_items(model["coils"], o)
    # regions
    o = panels[1]
    items += [_rect(r0, z0, r1, z1, o, "region vacuum", "region:vacuum"),
              _rect(v0 - _WALL, w0 - _WALL, v1 + _WALL, w1 + _WALL, o, "region conductor", "region:conductor"),
              _rect(v0, w0, v1, w1, o, "region vacuum", "region:vacuum"),
              Polyline.of(_cm(boundary, o), "region plasma", role="region:plasma", closed=True)]
    items += [_rect(rc - 0.04, zc - 0.04, rc + 0.04, zc + 0.04, o, "region conductor", "region:conductor")
              for rc, zc, _ in model["coils"]]
    items.append(_rect(r0, z0, r1, z1, o, "approx", "computational_boundary"))
    # mesh
    o = panels[2]
    for t, region in zip(mesh["triangles"], mesh["regions"]):
        if region != "vacuum":
            items.append(Polyline.of(_cm(mesh["nodes"][t], o), f"region {region}", role=f"cell:{region}",
                                     closed=True))
    nodes_cm = _cm(mesh["nodes"], o)
    for i, j in mesh["edges"]:
        items.append(Polyline.of([tuple(nodes_cm[i]), tuple(nodes_cm[j])], "mesh", role="edge"))
    top = 5.0 * z1 + 0.6
    for a, b, text in ((panels[0], panels[1], "define regions"), (panels[1], panels[2], "generate mesh")):
        xa, xb = a[0] + 5.0 * r1 + 0.25, b[0] + 5.0 * r0 - 0.25
        items.append(Arrow((xa, 0.0), (xb, 0.0), "connector", role="flow"))
        if labels:
            items.append(Label((0.5 * (xa + xb), 0.25), text, "small label", anchor="south", role="flow"))
    if labels:
        for o, text in zip(panels, ("physical geometry", "computational regions", "discretized mesh")):
            items.append(Label((o[0] + 2.5 * (r0 + r1), top), text, "label", anchor="south", role="title"))
        o = panels[1]
        items += [Label(tuple(_cm(model["axis"], o)), "plasma", "small label", role="region:plasma"),
                  Label(tuple(_cm((0.75, 0.62), o)), "vacuum", "small label", role="region:vacuum"),
                  Polyline.of([tuple(_cm((v1 + _WALL / 2, -0.5), o)), tuple(_cm((r1 + 0.05, -0.75), o))],
                              "leader line", role="leader:conductor"),
                  Label(tuple(_cm((r1 + 0.06, -0.75), o)), "conductors", "small label", anchor="west",
                        role="region:conductor")]
        m = mesh["median_edge"]
        items.append(_note(f"Median edge: plasma {1e3 * m['plasma']:.0f} mm, conductors "
                           f"{1e3 * m['conductor']:.0f} mm, vacuum {1e3 * m['vacuum']:.0f} mm. "
                           "Unconstrained Delaunay, cells classed by centroid", panels[1][0] + 3.0, 5.0 * z0 - 0.6))
    return Diagram("geometry_to_mesh", Scene(tuple(items)),
                   model={"median_edge": mesh["median_edge"], "count": mesh["count"], "spacing": dict(MESH_SPACING)})


#: the mapped toy surface: $R_0$, $a$, $\kappa$, $\delta$ [m, m, -, -]
_MAP_SHAPE = (1.7, 1.0, 1.6, 0.35)


def logical_to_physical(xi, eta):
    """$(\\xi, \\eta) \\mapsto (R, Z)$: $r = a\\xi$, $\\theta = 2\\pi\\eta$ on :data:`_MAP_SHAPE` Miller surfaces."""
    R0, a, kappa, delta = _MAP_SHAPE
    r = a * np.asarray(xi, dtype=float)
    return miller_surface(r, 2.0 * math.pi * np.asarray(eta, dtype=float), R0, kappa, delta * np.asarray(xi))


def logical_to_physical_mapping(*, n_xi: int = 6, n_eta: int = 16, cell: Tuple[int, int] = (3, 2),
                                labels: bool = True) -> Diagram:
    r"""A regular logical mesh $(\xi, \eta)$ mapped onto curved flux-aligned coordinates in $(R, Z)$.

    Left, the unit square, ``n_xi`` $\times$ ``n_eta`` cells. Right, its image
    under :func:`logical_to_physical`: lines of constant $\xi$ become nested
    flux surfaces ($r = a\xi$ on Miller surfaces), lines of constant $\eta$
    poloidal rays ($\theta = 2\pi\eta$). The shaded cell is the same cell in
    both. $\eta = 0$ and $\eta = 1$ map to one ray (periodic), and $\xi = 0$
    collapses to the magnetic axis -- the coordinate singularity every
    flux-aligned mesh has.
    """
    labels = _check_labels(labels)
    if not (isinstance(n_xi, int) and isinstance(n_eta, int) and 2 <= n_xi <= 20 and 4 <= n_eta <= 64):
        raise ValueError(f"n_xi must be 2-20 and n_eta 4-64 integers, not {n_xi!r}, {n_eta!r}")
    if not (isinstance(cell, tuple) and len(cell) == 2):
        raise ValueError(f"cell must be an (i, j) pair, not {cell!r}")
    i, j = cell
    if not (0 <= i < n_xi and 0 <= j < n_eta):
        raise ValueError(f"cell must index the logical mesh, not {cell!r}")
    L, sx, sy = (0.0, 0.0), 4.0, 6.0  # logical square drawn 4 cm x 6 cm
    phys, ps = (10.5, 0.0), 2.0  # physical panel origin and cm per m

    def logical_cm(xi, eta):
        return np.stack([L[0] + sx * np.asarray(xi), L[1] - 3.0 + sy * np.asarray(eta)], -1)

    def physical_cm(xi, eta):
        R, Z = logical_to_physical(xi, eta)
        return np.stack([phys[0] + ps * (R - _MAP_SHAPE[0]), phys[1] + ps * Z], -1)

    xs, es = np.linspace(0, 1, n_xi + 1), np.linspace(0, 1, n_eta + 1)
    fine = np.linspace(0, 1, 121)
    cxi, ceta = (xs[i], xs[i + 1]), (es[j], es[j + 1])
    outline_xi = np.r_[np.linspace(*cxi, 20), np.full(20, cxi[1]), np.linspace(cxi[1], cxi[0], 20), np.full(20, cxi[0])]
    outline_eta = np.r_[np.full(20, ceta[0]), np.linspace(*ceta, 20), np.full(20, ceta[1]),
                        np.linspace(ceta[1], ceta[0], 20)]
    items: List = [Polyline.of(logical_cm(outline_xi, outline_eta), "logical cell", role="cell", closed=True),
                   Polyline.of(physical_cm(outline_xi, outline_eta), "logical cell", role="cell", closed=True)]
    for x in xs:
        items.append(Polyline.of(logical_cm(np.full(2, x), [0, 1]), "mesh", role="logical:xi"))
        if x > 0:
            items.append(Polyline.of(physical_cm(np.full_like(fine, x), fine),
                                     "lcfs" if x == 1 else "surface", role="physical:xi", closed=True))
    for e in es:
        items.append(Polyline.of(logical_cm([0, 1], np.full(2, e)), "mesh", role="logical:eta"))
        if e < 1:
            items.append(Polyline.of(physical_cm(fine, np.full_like(fine, e)), "mesh", role="physical:eta"))
    items.append(Marker(tuple(physical_cm(0.0, 0.0)), "o", "opoint", role="axis"))
    items.append(Arrow((L[0] + sx + 0.9, 0.0), (phys[0] - ps * 1.2 - 0.6, 0.0), "connector", role="mapping"))
    if labels:
        items += [Label((L[0] + 0.5 * sx, -3.4), "$\\xi$", "label", anchor="north", role="logical:xi"),
                  Label((L[0] - 0.4, 0.0), "$\\eta$", "label", anchor="east", role="logical:eta"),
                  Label((L[0] + 0.5 * sx, 3.6), "logical space: regular", "label", anchor="south", role="title"),
                  Label((phys[0], 3.6), "physical space $(R, Z)$: curved", "label", anchor="south", role="title"),
                  Label((0.5 * (L[0] + sx + phys[0] - ps * 1.2), 0.3), "$(\\xi, \\eta) \\mapsto (R, Z)$",
                        "small label", anchor="south", role="mapping"),
                  Label((phys[0] + ps * 1.25, -2.9), "$\\xi$ = const: flux surface", "small label", anchor="west",
                        role="physical:xi"),
                  Label((phys[0] + ps * 1.25, -3.5), "$\\eta$ = const: poloidal ray", "small label", anchor="west",
                        role="physical:eta"),
                  Label(tuple(physical_cm(0.0, 0.0) + [0.15, -0.1]), "$\\xi = 0$: axis", "small label",
                        anchor="north west", role="axis"),
                  _note("$\\eta = 0$ and $\\eta = 1$ are one ray (periodic); the $\\xi = 0$ edge collapses to a "
                        "point (coordinate singularity). Miller surfaces, $r = a\\xi$, $\\theta = 2\\pi\\eta$",
                        6.0, -4.3)]
    return Diagram("logical_to_physical_mapping", Scene(tuple(items)),
                   model={"shape": dict(zip(("R0", "a", "kappa", "delta"), _MAP_SHAPE)), "cell": cell,
                          "n_xi": n_xi, "n_eta": n_eta})


# ---------------------------------------------------------------------------
# mapping
# ---------------------------------------------------------------------------


def _profile(psi_n):
    """A schematic temperature profile, a function of $\\psi_N$ only."""
    return (1.0 - np.clip(psi_n, 0.0, 1.0)) ** 1.5


def physical_to_flux_mapping(*, n_points: int = 11, labels: bool = True) -> Diagram:
    r"""Measurements at $(R, Z)$ become a profile in $\psi_N$: two halves of a chord, one curve.

    Left, ``n_points`` measurement positions on a horizontal chord through the
    magnetic axis of the toy equilibrium. Middle, a profile that depends on
    $\psi_N$ only, against $R$: in- and outboard halves differ (the axis is
    shifted). Right, the same values against
    $\psi_N = (\psi_\mathrm{axis} - \psi)/(\psi_\mathrm{axis} - \psi_b)$ from
    the equilibrium: the halves fall on one curve. $\rho_{\mathrm{tor},N} =
    \sqrt{\Phi/\Phi_b}$ relabels $\psi_N$ again, through $q(\psi)$.
    """
    labels = _check_labels(labels)
    if not (isinstance(n_points, int) and 4 <= n_points <= 40):
        raise ValueError(f"n_points must be an integer from 4 to 40, not {n_points!r}")
    from scipy.interpolate import RectBivariateSpline

    model = flux_model("diverted")
    boundary = lcfs(model)
    spline = RectBivariateSpline(_GRID_R, _GRID_Z, model["psi"])
    z_chord = model["axis"][1]
    on_chord = boundary[np.argsort(np.abs(boundary[:, 1] - z_chord))[:40]]
    r_in = float(on_chord[on_chord[:, 0] < model["axis"][0], 0].max())
    r_out = float(on_chord[on_chord[:, 0] > model["axis"][0], 0].min())
    R = np.linspace(r_in, r_out, n_points + 2)[1:-1]
    psi = spline(R, np.full_like(R, z_chord), grid=False)
    psi_n = (model["psi_axis"] - psi) / (model["psi_axis"] - model["psi_boundary"])
    value = _profile(psi_n)
    inboard = R < model["axis"][0]
    o = (0.0, 0.0)
    items: List = _surfaces(model, o, inside_only=True)
    items += [Polyline.of(_cm(boundary, o), "lcfs", role="lcfs", closed=True),
              Marker(tuple(_cm(model["axis"], o)), "o", "opoint", role="axis"),
              Polyline.of(_cm([(r_in - 0.05, z_chord), (r_out + 0.05, z_chord)], o), "diagnostic chord",
                          role="chord")]
    items += [Marker(tuple(_cm((r, z_chord), o)), "o", "diagnostic inboard" if inner else "diagnostic point",
                     role="measurement") for r, inner in zip(R, inboard)]
    charts = []
    for k, (x, x_range, x_label) in enumerate(((R, (r_in, r_out), "$R$ [m]"), (psi_n, (0.0, 1.0), "$\\psi_N$"))):
        chart = Chart(x_range=x_range, y_range=(0.0, 1.1))
        offset, scale = (7.5 + 7.5 * k, -1.6), 0.55
        if k == 1:
            chart.curves["profile"] = np.stack([np.linspace(0, 1, 60), _profile(np.linspace(0, 1, 60))], -1)
        scene = render_chart(chart, x_label=x_label, y_label="$T$ (a.u.)",
                             curve_styles={"profile": "approx"} if k == 1 else {}, region_text={})
        items += list(scene.transformed(scale=scale, offset=offset).items)
        for xv, yv, inner in zip(x, value, inboard):
            p = chart.to_cm(np.array([xv, yv]))
            items.append(Marker((float(scale * p[0] + offset[0]), float(scale * p[1] + offset[1])), "o",
                                "diagnostic inboard" if inner else "diagnostic point", role=f"chart{k}:measurement"))
        charts.append(offset)
    flow = ["measurement at $(R, Z)$", "equilibrium $\\psi(R, Z)$", "$\\psi_N$",
            "$\\rho_{\\mathrm{tor},N} = \\sqrt{\\Phi/\\Phi_b}$", "radial profile coordinate"]
    boxes = [box(1.7 + 3.6 * n, -4.6, 3.1, 0.9, text, role=f"flow:{n}", latex=True) for n, text in enumerate(flow)]
    for b in boxes:
        items += list(b.items)
    items += [connector(a, b, role="flow") for a, b in zip(boxes, boxes[1:])]
    if labels:
        items += [Label((5.0 * model["axis"][0], 2.6), "measurements on a chord", "label",
                        anchor="south", role="title"),
                  Label((charts[0][0] + 2.5, 2.6), "against $R$: two halves", "label", anchor="south",
                        role="title"),
                  Label((charts[1][0] + 2.5, 2.6), "against $\\psi_N$: one profile", "label", anchor="south",
                        role="title"),
                  Marker((0.9, -2.9), "o", "diagnostic inboard", role="legend:inboard"),
                  Label((1.1, -2.9), "inboard", "small label", anchor="west", role="legend:inboard"),
                  Marker((3.1, -2.9), "o", "diagnostic point", role="legend:outboard"),
                  Label((3.3, -2.9), "outboard", "small label", anchor="west", role="legend:outboard")]
    return Diagram("physical_to_flux_mapping", Scene(tuple(items)),
                   model={"R": R, "psi_n": psi_n, "value": value, "inboard": inboard, "z_chord": z_chord})
