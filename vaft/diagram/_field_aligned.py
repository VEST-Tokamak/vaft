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


# ---------------------------------------------------------------------------
# #1075 remainder: transits, boundary conditions, the X-point limit
# ---------------------------------------------------------------------------


def _amplitude_text(a: float) -> str:
    if abs(a) >= 0.01:
        return f"${a:.2f}$"
    mantissa, exponent = f"{a:.1e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def ballooning_transit_map(*, transits: int = 2, labels: bool = True) -> Diagram:
    r"""Where the extended angle is in the cross-section: one poloidal circle per transit.

    Top, the unstable eigenfunction of ``ballooning_eigenfunction``
    ($(s, \alpha) = (1, 1.2)$) on $|\theta| \le (2\,\mathrm{transits} + 1)\pi$.
    Below, one poloidal cross-section per transit $k$, centred under
    $\theta = 2\pi k$: the field line crosses the outboard midplane (bad
    curvature, filled) at $\theta = 2\pi k$ and the inboard one (good
    curvature, open) at $2\pi k \pm \pi$. The number by each outboard point is
    $F(2\pi k)$: the same geometric point, visited again on every transit,
    carries less of the mode each time -- why the mode "balloons" on the
    outboard side of one transit.
    """
    labels = _check_labels(labels)
    if not (isinstance(transits, int) and not isinstance(transits, bool) and 1 <= transits <= 3):
        raise ValueError(f"transits must be 1, 2 or 3, not {transits!r}")
    _, th, F = s_alpha_ballooning_eigenmode(*UNSTABLE)
    F = F / np.max(np.abs(F))
    half = (2 * transits + 1)
    keep = np.abs(th) <= half * math.pi
    x = th[keep] / math.pi
    chart = Chart(x_range=(-float(half), float(half)), y_range=(-0.1, 1.15))
    chart.curves["unstable"] = np.stack([x, F[keep]], -1)
    ticks = [float(k) for k in range(-half, half + 1)]
    scene = render_chart(
        chart, x_label="extended angle $\\theta/\\pi$", y_label="$F(\\theta)$",
        curve_styles={"unstable": "boundary"}, region_text={},
        x_ticks=ticks, x_tick_text=[f"${int(t)}$" for t in ticks], y_ticks=[0.0, 0.5, 1.0])
    items: List = list(scene.items)
    ks = list(range(-transits, transits + 1))
    amplitude = {k: float(np.interp(2.0 * k * math.pi, th, F)) for k in ks}
    rad, y0 = 0.62, -3.0
    for k in ks:
        cx = float(chart.to_cm(np.array([2.0 * k, 0.0]))[0])
        t = np.linspace(0.0, 2.0 * math.pi, 73)
        items.append(Polyline.of(np.stack([cx + rad * np.cos(t), y0 + rad * np.sin(t)], -1), "lcfs",
                                 role="cross_section", closed=True))
        items.append(Marker((cx + rad, y0), "o", "star", role="outboard"))
        items.append(Polyline.of([(cx - rad - 0.08, y0 - 0.08), (cx - rad + 0.08, y0 + 0.08)], "surface",
                                 role="inboard"))
        if labels:
            items += [Label((cx, y0 - rad - 0.1), f"$k = {k}$", "small label", anchor="north", role="transit"),
                      Label((cx, y0 - rad - 0.65), _amplitude_text(amplitude[k]), "small label", anchor="north",
                            role="amplitude")]
    if labels:
        items += [Label((chart.to_cm(np.array([0.0, 1.0]))[0] + 0.3, 6.9),
                        "unstable $(s, \\alpha) = (1, 1.2)$", "small label", anchor="west", role="title"),
                  Label((-0.3, y0 - rad - 0.65), "$F(2\\pi k)$", "small label", anchor="north east",
                        role="amplitude"),
                  Label((4.5, y0 - 2.1), "filled: outboard midplane (bad curvature) at $\\theta = 2\\pi k$, "
                        "one circle per poloidal transit $k$; tick: inboard at $2\\pi k \\pm \\pi$", "note", anchor="north",
                        role="note")]
    return Diagram("ballooning_transit_map", Scene(tuple(items)),
                   model={"transits": tuple(ks), "amplitude": amplitude, "unstable": UNSTABLE})


def ballooning_boundary_conditions(*, shear: float = 1.0, labels: bool = True) -> Diagram:
    r"""Two ways to close the field line: decay on the covering space, or a sheared parallel join.

    Left, the ballooning representation: the eigenfunction lives on the
    extended angle $\theta \in (-\infty, \infty)$, the covering space of the
    periodic poloidal angle, and the boundary condition is decay,
    $F \to 0$ as $|\theta| \to \infty$ (the unstable mode of
    ``ballooning_eigenfunction``). Right, a flux-tube box of finite parallel
    length: its two ends are the same poloidal position, joined after
    following the line once round. Magnetic shear connects the two pictures:
    a mode with $k_y$ has $k_x = k_y\hat s\theta$ (``ballooning_radial_wavenumber``),
    so the end it rejoins has a shifted $k_x$ -- the twist-and-shift
    condition, named here and not derived.
    """
    labels = _check_labels(labels)
    _, th, F = s_alpha_ballooning_eigenmode(*UNSTABLE)
    F = F / np.max(np.abs(F))
    chart = Chart(x_range=(-6.0, 6.0), y_range=(-0.1, 1.15))
    chart.curves["unstable"] = np.stack([th / math.pi, F], -1)
    ticks = [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
    scene = render_chart(chart, x_label="extended angle $\\theta/\\pi$", y_label="$F(\\theta)$",
                         curve_styles={"unstable": "boundary"}, region_text={}, x_ticks=ticks,
                         x_tick_text=[f"${int(t)}$" for t in ticks], y_ticks=[0.0, 1.0])
    items: List = list(scene.transformed(scale=0.7).items)
    for side in (-1, 1):
        x = 0.7 * float(chart.to_cm(np.array([5.0 * side, 0.0]))[0])
        items.append(Arrow((x, 0.75), (x + side * 0.9, 0.75), "connector", role="decay"))
    # the flux tube: a box along z, both ends drawn, joined by a curved return with a shifted end
    bx, by, length, width = 9.6, 0.6, 5.0, 1.6
    items += [Polyline.of([(bx, by), (bx + length, by), (bx + length, by + width), (bx, by + width)], "inset frame",
                          role="flux_tube", closed=True)]
    for f in (0.25, 0.5, 0.75):
        y = by + f * width
        items.append(Polyline.of([(bx, y), (bx + length, y + 0.25 * shear * (f - 0.5))], "field line",
                                 role="field_line"))
    t = np.linspace(0.0, math.pi, 50)
    arc = np.stack([bx + length / 2 + (length / 2 + 0.25) * np.cos(t), by + width + 0.2 + 1.1 * np.sin(t)], -1)
    items.append(Polyline.of(arc, "angle arc", role="parallel_join"))
    k_x = float(ballooning_radial_wavenumber(1.0, shear, 2.0 * math.pi))
    if labels:
        items += [Label((0.7 * 4.5, 0.7 * 6.5 + 0.6), "ballooning representation", "label", anchor="south",
                        role="title"),
                  Label((bx + length / 2, 4.3), "flux tube (local)", "label", anchor="south", role="title"),
                  Label((0.7 * 4.5, 0.7 * 6.5 + 0.1), "decay on the covering space: $F \\to 0$ as "
                        "$|\\theta| \\to \\infty$", "small label", anchor="south", role="decay"),
                  Label((bx + length / 2, by + width + 1.4), "ends joined after one poloidal turn",
                        "small label", anchor="south", role="parallel_join"),
                  Label((bx, by - 0.15), "$z = -\\pi$", "small label", anchor="north", role="ends"),
                  Label((bx + length, by - 0.15), "$z = +\\pi$", "small label", anchor="north", role="ends"),
                  Label((bx + length / 2, by - 0.75),
                        f"shear: $k_x = k_y\\hat s\\theta$, so after one turn $k_x/k_y = {k_x:.1f}$",
                        "small label", anchor="north", role="shear"),
                  Label((bx + length / 2, by - 1.35), "rejoining with shifted $k_x$: twist-and-shift "
                        "(not derived here)", "small label", anchor="north", role="twist_and_shift"),
                  Label((7.4, -1.9), f"$\\hat s = {shear:g}$; the eigenfunction is the unstable "
                        "$s$-$\\alpha$ mode at $(1, 1.2)$", "note", anchor="north", role="note")]
    return Diagram("ballooning_boundary_conditions", Scene(tuple(items)),
                   model={"shear": shear, "kx_after_one_turn": k_x, "unstable": UNSTABLE})


def _straight_field_line_points(model: dict, psi_n: float, n_theta: int):
    """Points of equal straight-field-line angle on the closed surface $\\psi_N$, and $\\oint dl/(R|\\nabla\\psi|)$.

    With $F$ constant (toy), $d\\theta^*/dl \\propto 1/(R^2B_p) = 1/(R|\\nabla\\psi|)$; the loop
    integral is proportional to $q$. Starts at the outboard midplane, runs counter-clockwise.
    """
    from scipy.interpolate import RectBivariateSpline

    from ._gs_equilibrium import _GRID_R, _GRID_Z, _encloses, _lines

    spline = RectBivariateSpline(_GRID_R, _GRID_Z, model["psi"])
    level = model["psi_axis"] - psi_n * (model["psi_axis"] - model["psi_boundary"])
    line = next(c for c in _lines(model["psi"], level) if _encloses(c, model["axis"]))
    line = line[:-1]
    x = line[:, 0] - model["axis"][0]
    y = line[:, 1] - model["axis"][1]
    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:  # make it counter-clockwise
        line = line[::-1]
        x, y = x[::-1], y[::-1]
    start = int(np.argmin(np.abs(y) + 10.0 * (x < 0)))
    line = np.roll(line, -start, axis=0)
    closed = np.vstack([line, line[:1]])
    mid = 0.5 * (closed[1:] + closed[:-1])
    dl = np.hypot(*np.diff(closed, axis=0).T)
    grad = np.hypot(spline(mid[:, 0], mid[:, 1], dx=1, grid=False), spline(mid[:, 0], mid[:, 1], dy=1, grid=False))
    w = dl / (mid[:, 0] * grad)
    cum = np.r_[0.0, np.cumsum(w)]
    targets = np.linspace(0.0, cum[-1], n_theta, endpoint=False)
    pts = np.stack([np.interp(targets, cum, closed[:, 0]), np.interp(targets, cum, closed[:, 1])], -1)
    return pts, float(cum[-1])


#: surfaces of the X-point figure: core to just inside the separatrix
_XPOINT_SURFACES = (0.15, 0.3, 0.45, 0.6, 0.75, 0.87, 0.95, 0.985, 0.996, 0.999)


def field_aligned_xpoint_limitation(*, n_theta: int = 24, labels: bool = True) -> Diagram:
    r"""Why field-aligned and flux coordinates fail at the X-point, on the diverted toy equilibrium.

    Left, lines of constant straight-field-line angle $\theta^*$ across the
    closed surfaces of ``flux_model("diverted")`` ($n_\theta$ equal steps;
    with $F$ constant $d\theta^*/dl \propto 1/(R^2B_p)$). In the core they
    are evenly spread; towards the separatrix, where $B_p \to 0$ at the
    X-point, $\theta^*$ is spent almost entirely near the X-point, so the
    coordinate lines crowd into it and the cells elsewhere stretch -- a
    strongly distorted metric. Outside the separatrix the field lines are
    open (SOL, private flux): there is no poloidal angle at all, and
    X-point-adapted or divertor coordinates are used instead. Right,
    $q \propto \oint dl/(R^2B_p)$ relative to its value at $\psi_N = 0.5$,
    diverging logarithmically as $\psi_N \to 1$ (cf. ``sfl_coordinate_validity``).
    """
    labels = _check_labels(labels)
    if not (isinstance(n_theta, int) and not isinstance(n_theta, bool) and 8 <= n_theta <= 64):
        raise ValueError(f"n_theta must be an integer from 8 to 64, not {n_theta!r}")
    from ._gs_equilibrium import _cm, _surfaces, flux_model, lcfs

    model = flux_model("diverted")
    o = (0.0, 0.0)
    points, loops = [], []
    for psi_n in _XPOINT_SURFACES:
        pts, loop = _straight_field_line_points(model, psi_n, n_theta)
        points.append(pts)
        loops.append(loop)
    _, loop_half = _straight_field_line_points(model, 0.5, n_theta)
    items: List = [Polyline.of(_cm(p, o), "surface", role="flux_surface", closed=True) for p in points]
    items += [it for it in _surfaces(model, o) if it.role == "open_flux"]
    stack = np.stack(points)  # surface, theta, (R, Z)
    for j in range(n_theta):
        items.append(Polyline.of(_cm(stack[:, j, :], o), "mesh", role="theta_line"))
    boundary = lcfs(model)
    items += [Polyline.of(_cm(boundary, o), "separatrix", role="separatrix", closed=True),
              Marker(tuple(_cm(model["x_point"], o)), "x", "xpoint", role="x_point"),
              Marker(tuple(_cm(model["axis"], o)), "o", "opoint", role="axis")]
    q_rel = np.array(loops) / loop_half
    chart = Chart(x_range=(0.0, 1.0), y_range=(0.0, float(math.ceil(q_rel[-1] + 0.5))))
    chart.curves["q"] = np.stack([np.array(_XPOINT_SURFACES), q_rel], -1)
    q_scene = render_chart(chart, x_label="$\\psi_N$", y_label="$q / q(0.5)$", curve_styles={"q": "boundary"},
                           region_text={}, x_ticks=[0.0, 0.5, 1.0],
                           y_ticks=[float(v) for v in range(0, int(chart.y_range[1]) + 1)])
    offset, scale = (7.6, 0.2), 0.55
    items += list(q_scene.transformed(scale=scale, offset=offset).items)
    # the fraction of theta* steps within 0.15 m of the X-point on the outermost surface
    near = float(np.mean(np.hypot(*(points[-1] - np.array(model["x_point"])).T) < 0.15))
    if labels:
        xl = 5.6
        items += [Label((3.3, 4.6), "closed surfaces: $\\theta^*$ lines", "label", anchor="south", role="title"),
                  Polyline.of([tuple(_cm(points[1][n_theta // 2], o)), (0.3, 3.6)], "leader line",
                              role="leader:core"),
                  Label((0.25, 3.6), "core: evenly spread", "small label", anchor="east", role="core"),
                  Polyline.of([tuple(_cm(model["x_point"], o) + [0.15, 0.0]), (xl, -2.6)], "leader line",
                              role="leader:x_point"),
                  Label((xl, -2.6), f"$B_p \\to 0$: {100 * near:.0f}\\% of $\\theta^*$ within 15 cm",
                        "small label", anchor="west", role="x_point"),
                  Label((xl, -3.2), "open lines: no $\\theta$; X-point-adapted coordinates", "small label",
                        anchor="west", role="open_flux"),
                  Label((offset[0] + scale * 4.5, offset[1] + scale * 6.5 + 0.9), "$q$ diverges at the separatrix",
                        "label", anchor="south", role="title"),
                  Label((5.0, -4.6), "Toy flux (\\texttt{flux\\_model}), $F$ constant: $d\\theta^*/dl \\propto "
                        "1/(R^2B_p)$", "note", anchor="north", role="note")]
    return Diagram("field_aligned_xpoint_limitation", Scene(tuple(items)),
                   model={"psi_n": _XPOINT_SURFACES, "q_relative": tuple(float(v) for v in q_rel),
                          "fraction_near_x_point": near, "n_theta": n_theta})
