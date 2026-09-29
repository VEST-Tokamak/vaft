"""Field-aligned coordinates, flux tubes, shear and the ballooning eigenfunction (#1075, part 2).

``field_aligned_basis``
    on the unrolled flux surface, the field lines are the lines of constant
    $\\alpha$; $x = \\psi$ and $y = \\alpha$ are constant along them and only
    $z = \\theta$ moves, so $\\mathbf B\\cdot\\nabla = (\\mathbf B\\cdot\\nabla z)\\,\\partial_z$;
``flux_tube_patch``
    from a tokamak flux surface to one field line, to its thin neighbourhood,
    to the local $(x, y, z)$ box of flux-tube and sheared-slab models;
``magnetic_shear_field_aligned``
    the phase fronts of a field-aligned mode in the local $(x, y)$ plane at
    successive $z$: shear rotates them, $k_x = k_y\\hat s z$;
``ballooning_eigenfunction``
    the most unstable localised eigenmode of the $s$-$\\alpha$ equation on the
    extended angle, against the non-localised top mode of a stable surface.

The physics is :mod:`vaft.formula.stability` (``field_line_label``,
``ballooning_radial_wavenumber``, ``s_alpha_ballooning_eigenmode``) on the
circular $s$-$\\alpha$ model.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from vaft.formula.stability import ballooning_radial_wavenumber, s_alpha_ballooning_eigenmode

from ._chart import Chart, render_chart
from ._concept import box, connector
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: unstable and stable (s, alpha) of the eigenfunction figure: first-stability boundary at s = 1 lies near 0.6
UNSTABLE, STABLE = (1.0, 1.2), (1.0, 0.3)


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def field_aligned_basis(q: float = 2.5, *, labels: bool = True) -> Diagram:
    r"""Why a field-aligned system is simple: two coordinates are constant along the field, one follows it.

    The flux surface unrolled onto $(\phi, \theta) \in [0, 2\pi)^2$; its field
    lines are the lines of constant $\alpha = \phi - q\theta$ (``field_line_label``),
    here eight of them. At one point: $\mathbf B$ along the line, the
    components $\partial_i\alpha = (1, -q)$ across it, $\nabla\psi$ out of
    the page. The right angle is drawn in $(\phi, \theta)$ coordinate space:
    $\mathbf B\cdot\nabla\alpha = B^\phi\partial_\phi\alpha + B^\theta\partial_\theta\alpha
    = 0$ pairs contravariant with covariant components and holds with any
    metric, while the true gradient $g^{ij}\partial_j\alpha$ is not at right
    angles on the page. It is the in-surface part at $\theta = 0$ only:
    $\nabla\alpha = \nabla\phi - q\nabla\theta - \theta q'\nabla\psi$ has a
    radial part that grows secularly along the line -- the magnetic shear of
    ``magnetic_shear_field_aligned``.
    With $x = \psi$, $y = \alpha$, $z = \theta$: $\mathbf B\cdot\nabla x = 0$,
    $\mathbf B\cdot\nabla y = 0$, $\mathbf B\cdot\nabla z \ne 0$, so the parallel
    derivative is one partial derivative.
    """
    labels = _check_labels(labels)
    if not q > 0:
        raise ValueError(f"q must be positive, not {q!r}")
    size = 5.0
    sc = size / (2.0 * math.pi)
    items: List = [Polyline.of([(0, 0), (size, 0), (size, size), (0, size)], "inset frame", role="surface",
                               closed=True)]
    theta = np.linspace(0.0, 2.0 * math.pi, 400)
    for k in range(8):
        alpha0 = 2.0 * math.pi * k / 8
        # alpha = phi - q theta = alpha0  ->  phi = alpha0 + q theta, wrapped into the square
        phi = alpha0 + q * theta
        wrapped = np.mod(phi, 2.0 * math.pi)
        breaks = np.where(np.diff(wrapped) < 0)[0] + 1
        for seg_phi, seg_theta in zip(np.split(wrapped, breaks), np.split(theta, breaks)):
            if len(seg_phi) > 1:
                items.append(Polyline.of(np.stack([seg_phi * sc, seg_theta * sc], -1), "surface",
                                         role="field_line"))
    # at a point: B^i along (q, 1) in (phi, theta); the covariant d_i alpha = (1, -q): B^i d_i alpha = 0
    P = np.array([0.5 * size, 0.42 * size])
    b = np.array([q, 1.0]) / math.hypot(q, 1.0)
    g = np.array([1.0, -q]) / math.hypot(1.0, q)
    items += [Marker(tuple(P), "o", "opoint", role="point"),
              Arrow(tuple(P), tuple(P + 1.3 * b), "vector", role="B"),
              Arrow(tuple(P), tuple(P + 1.1 * g), "drift ion", role="grad_alpha"),
              Marker(tuple(P + np.array([-0.55, 0.45])), "o", "xpoint", role="grad_psi")]
    if labels:
        items += [Label(tuple(P + 1.35 * b), "$\\mathbf B$, $z = \\theta$", "small label", anchor="south west",
                        role="B"),
                  Label(tuple(P + 1.15 * g), "$\\nabla\\alpha$, $y$", "small label", anchor="north west",
                        role="grad_alpha"),
                  Label(tuple(P + np.array([-0.7, 0.45])), "$\\nabla\\psi$ out, $x$", "small label", anchor="east",
                        role="grad_psi"),
                  Label((size / 2, -0.15), "$\\phi$", "small label", anchor="north", role="axes"),
                  Label((-0.15, size / 2), "$\\theta$", "small label", anchor="east", role="axes"),
                  Label((size + 0.5, size - 0.3),
                        "\\begin{tabular}{l}$\\mathbf B\\cdot\\nabla x = 0$\\\\$\\mathbf B\\cdot\\nabla y = 0$\\\\"
                        "$\\mathbf B\\cdot\\nabla z \\neq 0$\\\\[3pt]$\\mathbf B\\cdot\\nabla = "
                        "(\\mathbf B\\cdot\\nabla z)\\,\\partial_z$\\end{tabular}", "small label", anchor="north west",
                        role="conditions"),
                  Label((size / 2, -0.75), f"one flux surface unrolled, $q = {q:g}$: lines of constant "
                        "$\\alpha = \\phi - q\\theta$ are the field lines", "note", anchor="north", role="note")]
    if labels:
        items.append(Label((size / 2, -1.25), "$\\nabla\\alpha = \\nabla\\phi - q\\nabla\\theta - \\theta q'\\nabla\\psi$: "
                           "the last term grows along the line (shear)", "note", anchor="north", role="secular"))
    return Diagram("field_aligned_basis", Scene(tuple(items)),
                   model={"q": q, "B_direction": tuple(b), "grad_alpha_direction": tuple(g)})


def flux_tube_patch(*, labels: bool = True) -> Diagram:
    r"""From a tokamak flux surface to a local flux tube: one surface, one line, its neighbourhood, a box.

    A concept sequence: a flux surface in the poloidal cross-section; one of
    its field lines; the thin radial and binormal neighbourhood that follows
    it; and the local box with $x$ radial, $y$ binormal, $z$ along
    $\mathbf B$ -- the frame of local gyrokinetic flux-tube codes and of the
    sheared slab. The box's ends are sheared by the magnetic shear, so they
    are joined by a shifted, not a plain periodic, condition (twist and
    shift; not derived here).
    """
    labels = _check_labels(labels)
    items: List = []
    # 1. cross-section: one flux surface and a point on it
    t = np.linspace(0.0, 2.0 * math.pi, 121)
    c1 = np.array([1.3, 1.3])
    items.append(Polyline.of(c1 + 1.1 * np.stack([np.cos(t), 1.3 * np.sin(t)], -1), "lcfs", role="flux_surface",
                             closed=True))
    items.append(Marker(tuple(c1 + np.array([1.1, 0.0])), "o", "opoint", role="point"))
    # 2. the surface unrolled with one field line
    x2 = 3.6
    items.append(Polyline.of([(x2, 0.0), (x2 + 2.6, 0.0), (x2 + 2.6, 2.6), (x2, 2.6)], "inset frame",
                             role="unrolled_surface", closed=True))
    line = np.array([[x2, 0.3], [x2 + 2.6, 2.3]])
    items.append(Polyline.of(line, "field line", role="field_line"))
    # 3. the thin neighbourhood around the line
    x3 = 7.0
    items.append(Polyline.of([(x3, 0.3), (x3 + 2.6, 2.1), (x3 + 2.6, 2.5), (x3, 0.7)], "layer", role="neighbourhood",
                             closed=True))
    items.append(Polyline.of([(x3, 0.5), (x3 + 2.6, 2.3)], "field line", role="field_line"))
    # 4. the local box in oblique projection: an x-y end face, elongated along z; the far face is
    #    sheared (its x position shifts with y) relative to the near one
    x4 = 10.6
    o = np.array([x4, 0.2])
    ex, ey, ez = np.array([0.8, 0.0]), np.array([0.0, 0.8]), np.array([1.9, 1.1])
    shear_shift = 0.35 * ex
    near = [o, o + ex, o + ex + ey, o + ey]
    far = [o + ez, o + ez + ex, o + ez + ex + ey + shear_shift, o + ez + ey + shear_shift]
    items.append(Polyline.of(near, "surface", role="flux_tube_end", closed=True))
    items.append(Polyline.of(far, "surface", role="flux_tube_end", closed=True))
    for a, b in zip(near, far):
        items.append(Polyline.of([a, b], "surface", role="flux_tube"))
    items += [Arrow(tuple(o), tuple(o + 1.2 * ex), "vector", role="x_axis"),
              Arrow(tuple(o), tuple(o + 1.25 * ey), "vector", role="y_axis"),
              Arrow(tuple(o + 0.5 * (ex + ey)), tuple(o + 0.5 * (ex + ey) + 0.75 * ez), "drift ion",
                    role="z_axis")]
    for a, b in ((c1 + np.array([1.35, 0.0]), (x2 - 0.15, 1.3)), ((x2 + 2.75, 1.3), (x3 - 0.15, 1.3)),
                 ((x3 + 2.75, 1.3), (x4 - 0.15, 1.2))):
        items.append(Arrow(tuple(a), tuple(b), "connector", role="step"))
    if labels:
        items += [Label((c1[0], -0.3), "a flux surface", "small label", anchor="north", role="flux_surface"),
                  Label((x2 + 1.3, -0.3), "one field line", "small label", anchor="north", role="field_line"),
                  Label((x3 + 1.3, -0.3), "its thin neighbourhood", "small label", anchor="north",
                        role="neighbourhood"),
                  Label((x4 + 1.4, -0.3), "local box: $x, y, z$", "small label", anchor="north", role="flux_tube"),
                  Label(tuple(o + 1.25 * ex), "$x$", "small label", anchor="north west", role="x_axis"),
                  Label(tuple(o + 1.3 * ey), "$y$", "small label", anchor="south east", role="y_axis"),
                  Label(tuple(o + ez + ex + 0.05 * ey), "$z \\parallel \\mathbf B$", "small label",
                        anchor="west", role="z_axis"),
                  Label((6.9, -1.0), "Flux-tube and sheared-slab frame; the ends are sheared, joined by a shifted "
                        "(twist-and-shift) condition", "note", anchor="north", role="note")]
    return Diagram("flux_tube_patch", Scene(tuple(items)), model={"steps": ("surface", "field_line",
                                                                            "neighbourhood", "flux_tube")})


def magnetic_shear_field_aligned(*, shear: float = 1.0, labels: bool = True) -> Diagram:
    r"""Magnetic shear rotates the phase fronts of a field-aligned mode as it is followed along the line.

    Five local $(x, y)$ planes at $z = \theta = -\pi, -\pi/2, 0, \pi/2, \pi$
    with the fronts of a mode of fixed $k_y$: the radial wavenumber is
    ``ballooning_radial_wavenumber`` ($\alpha = 0$, $\theta_0 = 0$),
    $k_x = k_y\hat s\theta$, so the fronts tilt by $\arctan(\hat s\theta)$ -- the
    sheared-slab $k_x(z) = k_{x0} + k_y\hat s z$ seen in the tokamak. The
    binormal period $2\pi/k_y$ is the same in every plane (the fronts cross
    each vertical line at the same spacing), so the spacing across the fronts
    shrinks as $1/\sqrt{1 + \hat s^2\theta^2}$: $k_\perp$ grows along the line,
    the $(1 + \Lambda^2)$ of line bending and inertia.
    """
    labels = _check_labels(labels)
    k_y, size, pitch, wavelength_y = 1.0, 2.0, 2.8, 0.6
    zs = [-math.pi, -0.5 * math.pi, 0.0, 0.5 * math.pi, math.pi]
    items: List = []
    kxs = []
    for i, z in enumerate(zs):
        k_x = float(ballooning_radial_wavenumber(k_y, shear, z))
        kxs.append(k_x)
        x0 = i * pitch
        items.append(Polyline.of([(x0, 0), (x0 + size, 0), (x0 + size, size), (x0, size)], "inset frame",
                                 role="plane", closed=True))
        # fronts k_x x + k_y y = const, one per binormal wavelength along y, clipped to the square
        kk = math.hypot(k_x, k_y)
        d = np.array([k_y, -k_x]) / kk  # along a front, normal to (k_x, k_y)
        c = np.array([x0 + size / 2, size / 2])
        reach = 0.5 * size * (1.0 + abs(k_x / k_y))  # y-span of the fronts that cross the square
        n_front = int(math.ceil(reach / wavelength_y))
        for j in range(-n_front, n_front + 1):
            p0 = c + np.array([0.0, j * wavelength_y])
            seg = np.array([p0 - 2 * size * d, p0 + 2 * size * d])
            clipped = _clip_segment(seg, (x0, 0.0, x0 + size, size))
            if clipped is not None:
                items.append(Polyline.of(clipped, "surface", role=f"front_{i}"))
        if labels:
            items.append(Label((x0 + size / 2, -0.15), f"$\\theta = {_pi_text(z)}$", "small label", anchor="north",
                               role="plane"))
            items.append(Label((x0 + size / 2, size + 0.1), f"$k_x/k_y = {k_x / k_y:.1f}$", "small label",
                               anchor="south", role="kx"))
    if labels:
        items += [Label((-0.15, size / 2), "$y$", "small label", anchor="east", role="axes"),
                  Label((0.2, -0.65), "$x$ right, $y$ up in each plane", "small label", anchor="north west",
                        role="axes"),
                  Label((2 * pitch + size / 2, -1.2), f"$\\hat s = {shear:g}$: $k_x = k_y\\hat s\\theta$ "
                        "(ballooning\\_radial\\_wavenumber), the sheared slab along the field line", "note",
                        anchor="north", role="note")]
    return Diagram("magnetic_shear_field_aligned", Scene(tuple(items)),
                   model={"shear": shear, "theta": tuple(zs), "k_x": tuple(kxs), "k_y": k_y,
                          "wavelength_y": wavelength_y})


def _pi_text(z: float) -> str:
    k = round(2.0 * z / math.pi)  # in units of pi/2
    return {0: "0", 1: "\\pi/2", -1: "-\\pi/2", 2: "\\pi", -2: "-\\pi"}.get(k, f"{k}\\pi/2")


def _clip_segment(seg: np.ndarray, box_):
    """Liang-Barsky clip of a segment to an axis-aligned box; None if outside."""
    x0, y0, x1, y1 = box_
    p, q = seg
    d = q - p
    t0, t1 = 0.0, 1.0
    for pk, qk in ((-d[0], p[0] - x0), (d[0], x1 - p[0]), (-d[1], p[1] - y0), (d[1], y1 - p[1])):
        if pk == 0:
            if qk < 0:
                return None
            continue
        r = qk / pk
        if pk < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
    if t0 >= t1:
        return None
    return np.array([p + t0 * d, p + t1 * d])


def ballooning_eigenfunction(*, labels: bool = True) -> Diagram:
    r"""A ballooning mode: localised on the extended angle where the curvature is bad.

    ``s_alpha_ballooning_eigenmode`` on $[-6\pi, 6\pi]$ with vanishing ends:
    at $(s, \alpha) = (1, 1.2)$, beyond the first stability boundary, the
    most unstable mode ($\hat\gamma^2 > 0$) peaks at $\theta = 0$ -- the
    outboard midplane, bad curvature -- and decays within a few poloidal
    transits; at $(1, 0.3)$, stable, the top eigenvector is the discretised
    continuum and fills the interval. Ticks at $\theta = 0, \pm 2\pi, \ldots$
    are the outboard midplane on successive transits, $\pm\pi, \pm 3\pi, \ldots$
    the inboard one. The extended-angle localisation is the ballooning
    boundary condition; flux-tube codes instead join the sheared ends.
    """
    labels = _check_labels(labels)
    g_u, th, F_u = s_alpha_ballooning_eigenmode(*UNSTABLE)
    g_s, _, F_s = s_alpha_ballooning_eigenmode(*STABLE)
    x = th / math.pi
    chart = Chart(x_range=(float(x[0]), float(x[-1])), y_range=(-0.1, 1.15))
    chart.curves["unstable"] = np.stack([x, F_u], -1)
    chart.curves["stable"] = np.stack([x, F_s], -1)
    chart.labels.update({"unstable": (3.8, 0.85), "stable": (-4.2, 0.6)})
    chart.parameters.update({"growth_rate_squared_unstable": g_u, "growth_rate_squared_stable": g_s,
                             "unstable": UNSTABLE, "stable": STABLE})
    ticks = [float(k) for k in range(-6, 7, 2)]
    scene = render_chart(
        chart, x_label="extended angle $\\theta/\\pi$ along the field line", y_label="$F(\\theta)$",
        curve_styles={"stable": "approx", "unstable": "boundary"},
        region_text={"unstable": f"\\small unstable $(1, 1.2)$: $\\hat\\gamma^2 = {g_u:.2f}$",
                     "stable": "\\small stable $(1, 0.3)$"} if labels else {},
        x_ticks=ticks, x_tick_text=[f"${int(t)}$" for t in ticks], y_ticks=[0.0, 0.5, 1.0],
        note=("Even ticks: the outboard midplane (bad curvature) on each transit; the unstable mode balloons "
              "there and decays" if labels else ""),
    )
    return Diagram("ballooning_eigenfunction", scene, model=chart)
