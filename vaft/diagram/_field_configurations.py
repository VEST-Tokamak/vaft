"""Canonical slab field configurations and reconnection topology (#1063).

``slab_field_configuration``
    the field direction on stacked $x$ = const sheets: uniform, sheared
    (``sheared_slab_field``), reversed (``harris_sheet_field``) and reversed
    with a guide field -- shear and reversal kept apart;
``current_sheet``
    the reversing field in the plane normal to the current, field lines at
    equal flux spacing, the sheet, its normal, thickness and $J_z$;
``harris_sheet``
    $B_y(x)$ and $J_z(x)$ of ``harris_sheet_field`` and
    ``harris_sheet_current_density`` on one chart;
``x_point``
    the current-free null of ``x_point_flux``: four branches, separatrices;
``magnetic_reconnection``
    the model-neutral picture: inflow, outflow, diffusion region, upstream
    and reconnected field about a stretched X-point;
``island_formation``
    ``slab_perturbed_flux`` with growing tearing amplitude: straight sheared
    field, then X- and O-points, then an island.

All in the slab frame of ``vaft.formula.geometry``: $x$ the sheet normal
(radial), $y$ the reconnecting (binormal) direction, $z$ the current and
guide-field direction; drawn with $x$ up and $y$ across, so $z$ points into
the page -- except ``slab_field_configuration``, which draws $x$ up, $z$ across
and $y$ out of the page, obliquely.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from vaft.formula.constants import MU0
from vaft.formula.geometry import harris_sheet_current_density, harris_sheet_field, sheared_slab_field, x_point_flux
from vaft.formula.stability import slab_perturbed_flux

from ._chart import CHART_HEIGHT, CHART_WIDTH, Chart, render_chart
from ._equations import formula_equation
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene
from ._slab_parity import _contours

KINDS = ("uniform", "sheared", "reversed", "guide")
#: sheet half-thickness a and asymptotic field B0 of the drawn Harris sheet; guide field B_g
_A, _B0, _BG = 1.0, 1.0, 0.8
#: shear length of the drawn sheared slab, in units of a
_LS = 3.0


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _note(text: str, x: float, y: float) -> Label:
    return Label((x, y), text, "note", anchor="north", role="note")


def _field(kind: str, x: np.ndarray) -> np.ndarray:
    """$(B_y, B_z)$ of each configuration at the sheet positions ``x`` (units of $a$)."""
    x = np.asarray(x, dtype=float)
    if kind == "uniform":
        return np.stack([np.zeros_like(x), np.full_like(x, _B0)], -1)
    if kind == "sheared":
        return sheared_slab_field(x, _B0, _LS)[..., 1:]
    b_y = harris_sheet_field(x, _B0, _A)
    b_z = np.full_like(x, _BG if kind == "guide" else 0.0)
    return np.stack([b_y, b_z], -1)


# ---------------------------------------------------------------------------
# stacked sheets
# ---------------------------------------------------------------------------

#: sheet size in (z, y) [cm] and the oblique projection of y
_LZ, _LY = 6.0, 3.0
#: y points out of the page (x up, z right, right-handed), drawn receding down-left
_OBL = (-0.5 * math.cos(math.radians(35.0)), -0.5 * math.sin(math.radians(35.0)))
_SPACING = 1.45


def _screen(x_level: float, y, z) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    return np.stack([z + _OBL[0] * y, x_level * _SPACING + _OBL[1] * y], -1)


def _clip_line(p, d, n: int = 2) -> np.ndarray | None:
    """The segment of the line $p + td$ inside $[0, L_y]\\times[0, L_z]$, in (y, z)."""
    lo, hi = -1e9, 1e9
    for k, L in ((0, _LY), (1, _LZ)):
        if abs(d[k]) < 1e-12:
            if not 0.0 <= p[k] <= L:
                return None
            continue
        t0, t1 = (0.0 - p[k]) / d[k], (L - p[k]) / d[k]
        lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
    if hi - lo < 0.3:
        return None
    t = np.linspace(lo, hi, n)
    return np.asarray(p)[None, :] + t[:, None] * np.asarray(d)[None, :]


def slab_field_configuration(kind: str = "sheared", *, labels: bool = True) -> Diagram:
    r"""Field direction on stacked $x$ = const sheets: uniform, sheared, reversed, guide-field.

    Five sheets at $x/a = -1, -\tfrac12, 0, \tfrac12, 1$, each carrying field
    lines along its $(B_y, B_z)$ and an arrow whose length is $|\mathbf B|$:
    ``uniform`` $\mathbf B = B_0\hat{\mathbf z}$; ``sheared``
    ``sheared_slab_field`` ($L_s = 3a$), the direction rotates while
    $|\mathbf B| = B_0$ to first order in $x/L_s$ (5 % more at $x = a$); ``reversed`` ``harris_sheet_field``, the
    direction flips and $|\mathbf B| = 0$ at $x = 0$; ``guide`` the same plus
    $B_g\hat{\mathbf z}$ ($B_g = 0.8B_0$), rotating through the sheet without a
    null. Shear is a rotation of the field; reversal is a change of sign of
    one component -- the current sheet of ``current_sheet``.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    labels = _check_labels(labels)
    levels = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    fields = _field(kind, levels)
    items: List = []
    for x_level, (b_y, b_z) in zip(levels, fields):
        corners = _screen(x_level, [0.0, 0.0, _LY, _LY], [0.0, _LZ, _LZ, 0.0])
        items.append(Polyline.of(corners, "concept band", role=f"sheet:{x_level:g}", closed=True))
        items.append(Polyline.of(corners, "surface", role=f"sheet:{x_level:g}", closed=True))
        mag = math.hypot(b_y, b_z)
        if mag < 1e-9:
            if labels:
                items.append(Label(tuple(_screen(x_level, 0.5 * _LY, 0.5 * _LZ)), "$\\mathbf{B} = 0$",
                                   "small label", role="null"))
            continue
        d = np.array([b_y, b_z]) / mag
        normal = np.array([d[1], -d[0]])
        for off in (-0.9, 0.0, 0.9):
            seg = _clip_line(np.array([0.5 * _LY, 0.5 * _LZ]) + off * normal, d)
            if seg is not None:
                items.append(Polyline.of(_screen(x_level, seg[:, 0], seg[:, 1]), "orbit electron",
                                         role=f"field_line:{x_level:g}"))
        centre = np.array([0.5 * _LY, 0.5 * _LZ])
        half = 1.1 * mag / _B0
        a, b = centre - half * d, centre + half * d
        items.append(Arrow(tuple(_screen(x_level, a[0], a[1])), tuple(_screen(x_level, b[0], b[1])),
                           "field vector", role=f"B:{x_level:g}"))
    if labels:
        for x_level in levels:
            text = {1.0: "$x = a$", -1.0: "$x = -a$", 0.0: "$x = 0$"}.get(float(x_level), f"$x = {x_level:g}a$")
            items.append(Label(tuple(_screen(x_level, 0.5 * _LY, _LZ) + np.array([0.25, 0.0])), text, "small label",
                               anchor="west", role="x_level"))
        # the triad: x up, z along the sheets, y receding
        o = np.array([-1.2 + _OBL[0] * _LY, -1.0 * _SPACING + 0.9])
        items += [Arrow(tuple(o), tuple(o + [0.0, 0.9]), "frame axis arrow", role="triad"),
                  Arrow(tuple(o), tuple(o + [0.9, 0.0]), "frame axis arrow", role="triad"),
                  Arrow(tuple(o), tuple(o + 1.6 * np.array(_OBL)), "frame axis arrow", role="triad"),
                  Label(tuple(o + [0.0, 0.95]), "$x$", "small label", anchor="south", role="triad"),
                  Label(tuple(o + [0.95, 0.0]), "$z$", "small label", anchor="west", role="triad"),
                  Label(tuple(o + 1.7 * np.array(_OBL)), "$y$", "small label", anchor="north east", role="triad")]
        title = {"uniform": "uniform field", "sheared": "magnetic shear: direction rotates, $|\\mathbf{B}| \\simeq B_0$",
                 "reversed": "field reversal: $B_y(-x) = -B_y(x)$, null at $x = 0$",
                 "guide": "reversal with a guide field: rotation, no null"}[kind]
        top = 1.0 * _SPACING + max(0.0, _OBL[1] * _LY)
        items.append(Label((0.5 * (_LZ + _OBL[0] * _LY), top + 0.35), title, "label", anchor="south",
                           role="title"))
        eq = {"uniform": "\\mathbf{B} = B_0\\hat{\\mathbf{z}}",
              "sheared": formula_equation(sheared_slab_field),
              "reversed": formula_equation(harris_sheet_field),
              "guide": formula_equation(harris_sheet_field) + ",\\quad B_z = B_g"}[kind]
        bottom = -1.0 * _SPACING + min(0.0, _OBL[1] * _LY) - 0.4
        items.append(Label((0.5 * (_LZ + _OBL[0] * _LY), bottom), f"$\\displaystyle {eq}$", "formula box",
                           anchor="north", role="equations"))
        note = {"uniform": "Baseline for waves and perturbations; arrows: $\\mathbf{B}$ on each sheet",
                "sheared": f"$L_s = {_LS:g}a$; the rotation is the shear, $k_\\parallel(x) = k_yx/L_s$",
                "reversed": "$B_z = 0$; arrow length $|\\mathbf{B}|$; the current sits where $B_y$ changes sign",
                "guide": f"$B_g = {_BG:g}B_0$: $|\\mathbf{{B}}| \\ge B_g$, the field rotates by "
                         f"${2 * math.degrees(math.atan(_B0 * math.tanh(1.0) / _BG)):.0f}^\\circ$ over $|x| \\le a$"}[kind]
        items.append(_note(note, 0.5 * (_LZ + _OBL[0] * _LY), bottom - 1.2))
    return Diagram(f"slab_field_configuration_{kind}", Scene(tuple(items)),
                   model={"kind": kind, "x": levels, "field": fields})


# ---------------------------------------------------------------------------
# current sheet
# ---------------------------------------------------------------------------


def current_sheet(guide_field: bool = False, *, labels: bool = True) -> Diagram:
    r"""A current sheet seen along the current: reversing field, sheet, normal, thickness, $J_z$.

    Field lines of ``harris_sheet_field`` at equal steps of the flux function
    $\psi(x) = B_0a\ln\cosh(x/a)$ ($\mathbf B_\perp = \hat{\mathbf z}\times\nabla\psi$,
    $\psi = -A_z$), so their spacing shows $|B_y|$: dense
    outside, none at the centre. $x$ up is the sheet normal, $y$ across the
    reconnecting direction, $z$ into the page, so the current
    $J_z = \mu_0^{-1}dB_y/dx$ (``harris_sheet_current_density``) of $B_0 > 0$
    is $\otimes$. ``guide_field`` adds $B_g\hat{\mathbf z}$, also $\otimes$ and
    uniform: it leaves this in-plane picture and $J_z$ unchanged and removes
    the null ($|\mathbf B| \ge B_g$).
    """
    if not isinstance(guide_field, bool):
        raise ValueError(f"guide_field must be True or False, not {guide_field!r}")
    labels = _check_labels(labels)
    W, sy = 10.0, 1.0  # sheet length [cm], cm per a
    x_max = 3.0
    flux = lambda x: _B0 * _A * np.log(np.cosh(x / _A))  # noqa: E731 -- psi = -A_z of the Harris field
    steps = np.linspace(0.0, float(flux(x_max)), 8)[1:]
    xs = [_A * math.acosh(math.exp(s / (_B0 * _A))) for s in steps]
    items: List = []
    items.append(Polyline.of([(0.0, -_A * sy), (W, -_A * sy), (W, _A * sy), (0.0, _A * sy)], "layer",
                             role="sheet", closed=True))
    lines = []
    for sign in (1.0, -1.0):
        for x in xs:
            y_cm = sign * x * sy
            items.append(Polyline.of([(0.0, y_cm), (W, y_cm)], "orbit electron", role="field_line"))
            b_y = float(harris_sheet_field(sign * x, _B0, _A))
            mid = 0.5 * W
            direction = 1.0 if b_y > 0 else -1.0
            items.append(Arrow((mid - 0.25 * direction, y_cm), (mid + 0.25 * direction, y_cm), "field arrow",
                               role="field_direction"))
            lines.append((sign * x, b_y))
    for y in np.linspace(0.8, W - 0.8, 6):
        items.append(Label((float(y), 0.0), "$\\otimes$", "legend symbol", role="current"))
    if guide_field:  # B_g along z, into the page like J_z but everywhere: small crossed markers upstream
        for y in np.linspace(1.4, W - 1.4, 4):
            for x in (1.9, -1.9):
                items.append(Label((float(y), x * sy), "$\\otimes$", "charge small", role="guide_field"))
    J0 = float(harris_sheet_current_density(0.0, _B0, _A))
    if labels:
        right = W + 0.3
        items += [
            Arrow((right + 0.4, 0.0), (right + 0.4, 1.3), "vector", role="normal"),
            Label((right + 0.5, 1.3), "$\\hat{\\mathbf{n}} = \\hat{\\mathbf{x}}$", "small label", anchor="west",
                  role="normal"),
            Arrow((-0.4, -_A * sy), (-0.4, _A * sy), "width arrow", role="thickness", both=True),
            Label((-0.55, 0.0), "$2a$", "small label", anchor="east", role="thickness"),
            Label((right, x_max * sy), "$B_y > 0$", "small label", anchor="west", role="upstream"),
            Label((right, -x_max * sy), "$B_y < 0$", "small label", anchor="west", role="upstream"),
            Label((0.5 * W, -x_max * sy - 0.35), "$\\otimes\\ J_z$ in the sheet ($z$ into the page); "
                  "line spacing $\\propto 1/|B_y|$", "small label", anchor="north", role="legend"),
            Label((0.5 * W, x_max * sy + 0.35), "current sheet: $\\mu_0 J_z = dB_y/dx$", "label", anchor="south",
                  role="title"),
        ]
        if guide_field:
            items.append(Label((0.5 * W, -x_max * sy - 0.9), f"small $\\otimes$: guide field $B_g = {_BG:g}B_0$ everywhere: "
                               "$|\\mathbf{B}| = (B_y^2 + B_g^2)^{1/2} \\ge B_g$, no null; $J_z$ unchanged",
                               "small label", anchor="north", role="guide"))
        items.append(Label((0.5 * W, -x_max * sy - (1.55 if guide_field else 1.0)),
                           f"$\\displaystyle {formula_equation(harris_sheet_current_density)}$", "formula box",
                           anchor="north", role="equations"))
    name = "current_sheet_guide_field" if guide_field else "current_sheet"
    return Diagram(name, Scene(tuple(items)), model={"field_lines": lines, "J0": J0, "guide_field": guide_field,
                                                    "B_g": _BG if guide_field else 0.0})


def harris_sheet(*, labels: bool = True) -> Diagram:
    r"""$B_y(x)$ and $J_z(x)$ of the Harris sheet on one chart.

    ``harris_sheet_field`` normalised to $B_0$ and
    ``harris_sheet_current_density`` normalised to $B_0/(\mu_0a)$ over
    $|x| \le 3a$: the reversal of $B_y$ and the localized current are the same
    layer, $J_z = \mu_0^{-1}dB_y/dx$.
    """
    labels = _check_labels(labels)
    x = np.linspace(-3.0, 3.0, 301)
    b = harris_sheet_field(x * _A, _B0, _A) / _B0
    j = harris_sheet_current_density(x * _A, _B0, _A) / (_B0 / (MU0 * _A))
    chart = Chart(x_range=(-3.0, 3.0), y_range=(-1.25, 1.25))
    chart.curves.update({"B_y": np.stack([x, b], -1), "J_z": np.stack([x, j], -1),
                         "zero": np.array([[-3.0, 0.0], [3.0, 0.0]])})
    chart.parameters.update({"B0": _B0, "a": _A})
    scene = render_chart(chart, x_label="$x/a$", y_label="normalised",
                         curve_styles={"zero": "approx", "B_y": "component imag", "J_z": "component real"},
                         region_text={}, x_ticks=(-1.0, 0.0, 1.0), x_tick_text=("$-1$", "$0$", "$1$"),
                         y_ticks=(-1.0, 0.0, 1.0), y_tick_text=("$-1$", "$0$", "$1$"))
    items = list(scene.items)
    if labels:
        items += [
            Label((CHART_WIDTH + 0.9, CHART_HEIGHT - 0.1), "blue: $B_y/B_0$", "small label", anchor="north west",
                  role="legend"),
            Label((CHART_WIDTH + 0.9, CHART_HEIGHT - 0.6), "red: $\\mu_0aJ_z/B_0$", "small label",
                  anchor="north west", role="legend"),
            Label((CHART_WIDTH / 2, -1.45), f"$\\displaystyle {formula_equation(harris_sheet_field)}$",
                  "formula box", anchor="north", role="equations"),
            Label((CHART_WIDTH / 2, -2.45), f"$\\displaystyle {formula_equation(harris_sheet_current_density)}$",
                  "formula box", anchor="north", role="equations"),
            _note("Field reversal and localized current are one layer; the half-thickness $a$ sets both",
                  CHART_WIDTH / 2, -3.75),
        ]
    return Diagram("harris_sheet", Scene(tuple(items)), model=chart)


# ---------------------------------------------------------------------------
# X-point and reconnection
# ---------------------------------------------------------------------------


def _flux_lines(half_y: float, half_x: float, stretch: float, levels):
    """Contours of ``x_point_flux`` over $|y| \\le$ half_y, $|x| \\le$ half_x, $y$ scaled by ``stretch``; (y, x) points."""
    ys = np.linspace(-half_y, half_y, 401)
    xs = np.linspace(-half_x, half_x, 241)
    Y, X = np.meshgrid(ys, xs)
    Z = x_point_flux(X, Y / stretch, 1.0)
    return _contours(Z, xs, ys, levels)


def _direction_arrow(line: np.ndarray, stretch: float, style: str, role: str, at=None) -> Arrow:
    """An arrowhead on a field line, along $\\mathbf B_\\perp = B'(y, x)$ (screen: ($B_y$, $B_x$)).

    ``at = (axis, value)`` puts it where screen coordinate ``axis`` (0 across, 1 up) is nearest ``value``;
    otherwise at the middle of the line.
    """
    i = (len(line) - 1) // 2 if at is None else int(np.argmin(np.abs(line[:, at[0]] - at[1])))
    i = min(max(i, 1), len(line) - 2)
    p = line[i]
    y, x = p[0] / stretch, p[1]
    b_screen = np.array([x, y / stretch])  # (B_y, B_x) with the y-stretch of the drawing
    t = line[min(i + 1, len(line) - 1)] - line[max(i - 1, 0)]
    if np.dot(t, b_screen) < 0:
        t = -t
    t = t / np.linalg.norm(t)
    return Arrow(tuple(p - 0.2 * t), tuple(p + 0.2 * t), style, role=role)


def x_point(*, labels: bool = True) -> Diagram:
    r"""The current-free magnetic X-point of ``x_point_flux``: four branches and two separatrices.

    Contours of $\psi = \tfrac12B'(x^2 - y^2)$ with $x$ up and $y$ across;
    arrows along $\mathbf B_\perp = B'(y, x)$. The separatrices $y = \pm x$
    ($\psi = 0$) meet at right angles at the null and divide the plane into
    four families: the upper and lower ($\psi > 0$) field lines have $B_y$ of
    opposite sign, as across a current sheet; the left and right ($\psi < 0$)
    ones $B_x$. Geometry only: no flow is implied (see
    ``magnetic_reconnection``).
    """
    labels = _check_labels(labels)
    L = 2.6
    levels = [s * v for v in (0.25, 0.9, 1.8, 2.9) for s in (1.0, -1.0)]
    items: List = []
    lines = _flux_lines(L, L, 1.0, levels)
    for line in lines:
        items.append(Polyline.of(line, "orbit electron", role="field_line"))
        items.append(_direction_arrow(line, 1.0, "field arrow", "field_direction"))
    for s in (1.0, -1.0):
        items.append(Polyline.of([(-L, -s * L), (L, s * L)], "separatrix", role="separatrix"))
    items.append(Marker((0.0, 0.0), "x", "xpoint", role="x_point"))
    if labels:
        items += [
            Label((0.15, 0.2), "X", "small label", anchor="south west", role="x_point"),
            Label((L + 0.1, L), "separatrix $\\psi = 0$", "small label", anchor="west", role="separatrix"),
            Label((0.0, L + 0.2), "$B_y > 0$", "small label", anchor="south", role="branch"),
            Label((0.0, -L - 0.2), "$B_y < 0$", "small label", anchor="north", role="branch"),
            Label((L + 0.1, 0.0), "$B_x > 0$", "small label", anchor="west", role="branch"),
            Label((-L - 0.1, 0.0), "$B_x < 0$", "small label", anchor="east", role="branch"),
            Label((-L - 0.2, -L), "$x$ up, $y$ across", "small label", anchor="south east", role="axes"),
            Label((0.0, -L - 0.85), f"$\\displaystyle {formula_equation(x_point_flux)}$", "formula box",
                  anchor="north", role="equations"),
            _note("$\\nabla^2\\psi = 0$: no current at the null; a current $J_z$ there closes the angle "
                  "towards a sheet", 0.0, -L - 2.1),
        ]
    return Diagram("x_point", Scene(tuple(items)), model={"levels": levels, "lines": lines})


def magnetic_reconnection(*, labels: bool = True) -> Diagram:
    r"""Model-neutral 2-D reconnection: inflow, outflow, diffusion region, upstream and reconnected field.

    Field lines are contours of ``x_point_flux`` with $y$ stretched by 2.5,
    the X-point opened into a short current sheet; the diffusion region
    (shaded) and the flows are drawn, not solved. Upstream field lines
    ($\psi > 0$, dark) carry $\pm B_y$ in from above and below; reconnected
    field lines ($\psi < 0$, blue) leave to the sides with the outflow jets.
    No Sweet--Parker, Petschek, Hall or kinetic assumption is made; a guide
    field would be $\otimes$ along $z$ and leave this in-plane picture.
    """
    labels = _check_labels(labels)
    stretch, Ly, Lx = 2.5, 5.2, 2.4
    up = [0.12, 0.45, 1.0, 1.8, 2.8]
    down = [-v for v in (0.12, 0.45, 1.0, 1.8, 2.8)]
    items: List = []
    sheet = (1.3, 0.22)
    items.append(Polyline.of([(-sheet[0], -sheet[1]), (sheet[0], -sheet[1]), (sheet[0], sheet[1]),
                              (-sheet[0], sheet[1])], "layer", role="diffusion_region", closed=True))
    upstream = _flux_lines(Ly, Lx, stretch, up)
    reconnected = _flux_lines(Ly, Lx, stretch, down)
    for line in upstream:
        items.append(Polyline.of(line, "orbit electron", role="upstream_field"))
        items.append(_direction_arrow(line, stretch, "field arrow", "field_direction", (0, -2.6)))
    for line in reconnected:
        items.append(Polyline.of(line, "orbit ion", role="reconnected_field"))
        side = 1.0 if line[:, 0].mean() > 0 else -1.0
        items.append(_direction_arrow(line, stretch, "field arrow", "field_direction", (1, 1.0 * side)))
    for s in (1.0, -1.0):
        items.append(Polyline.of([(-Ly, -s * Ly / stretch), (Ly, s * Ly / stretch)], "approx",
                                 role="separatrix"))
    items.append(Marker((0.0, 0.0), "x", "xpoint", role="x_point"))
    items += [Arrow((0.0, Lx + 0.9), (0.0, Lx - 0.2), "exb", role="inflow"),
              Arrow((0.0, -Lx - 0.9), (0.0, -Lx + 0.2), "exb", role="inflow"),
              Arrow((1.6, 0.0), (3.4, 0.0), "exb", role="outflow"),
              Arrow((-1.6, 0.0), (-3.4, 0.0), "exb", role="outflow")]
    if labels:
        items += [
            Label((0.2, Lx + 0.6), "inflow", "small label", anchor="west", role="inflow"),
            Label((0.2, -Lx - 0.6), "inflow", "small label", anchor="west", role="inflow"),
            Label((Ly + 0.1, 0.0), "outflow jet", "small label", anchor="west", role="outflow"),
            Label((-Ly - 0.1, 0.0), "outflow jet", "small label", anchor="east", role="outflow"),
            Label((-Ly, Lx + 0.35), "upstream field, $+B_y$", "small label", anchor="south west",
                  role="upstream_field"),
            Label((-Ly, -Lx - 0.35), "upstream field, $-B_y$", "small label", anchor="north west",
                  role="upstream_field"),
            Label((0.15, 0.12), "X", "small label", anchor="south west", role="x_point"),
            Label((Ly + 0.6, 1.0), "reconnected field (blue)", "small label", anchor="west",
                  role="reconnected_field"),
            Label((Ly + 0.1, Ly / stretch), "separatrix (dashed)", "small label", anchor="west", role="separatrix"),
            Label((0.0, -Lx - 1.3), "shaded: diffusion region, inside the current sheet ($\\otimes J_z$ into "
                  "the page); X: the X-point",
                  "small label", anchor="north", role="diffusion_region"),
            Label((0.0, -Lx - 1.85), f"$\\displaystyle {formula_equation(x_point_flux)}$, $y \\to y/{stretch:g}$",
                  "formula box", anchor="north", role="equations"),
            _note("Model-neutral: no Sweet--Parker, Petschek, Hall or kinetic assumption; field lines from the "
                  "stretched X-point flux, regions and flows drawn", 0.0, -Lx - 3.0),
        ]
    return Diagram("magnetic_reconnection", Scene(tuple(items)),
                   model={"stretch": stretch, "upstream": upstream, "reconnected": reconnected})


# ---------------------------------------------------------------------------
# island formation
# ---------------------------------------------------------------------------

#: tearing amplitudes psi_0/B' of the three stages (island widths 0, 0.28, 0.6 in units of the panel half-height)
_STAGES = (0.0, 0.005, 0.0225)


def island_formation(*, labels: bool = True) -> Diagram:
    r"""From a sheared field to a magnetic island: ``slab_perturbed_flux`` with growing tearing amplitude.

    Three panels of constant-$\Psi$ contours over two wavelengths:
    $\psi_0 = 0$, the straight sheared field lines with the resonant surface
    $x = 0$ ($k_\parallel = 0$); a small $\psi_0$, where the field lines
    reconnect at X-points and close around O-points; a larger $\psi_0$, the
    island of full width $w = 4\sqrt{\psi_0/B'}$ bounded by its separatrix.
    The dynamics that grow $\psi_0$ -- $\Delta'$, the layer, Rutherford -- are
    ``delta_prime``, ``slab_parity`` and ``tearing_layer_matching``.
    """
    labels = _check_labels(labels)
    k_y, shear = 1.0, 1.0
    wavelength = 2.0 * math.pi / k_y
    W, Hh, gap = 5.0, 1.6, 1.3  # panel width, half-height [cm], gap
    # 301 samples: grid nodes land at multiples of W/300 cm, never on a 4-decimal rounding tie (321 would put
    # them at k/64 cm, where last-bit differences between platforms flip the printed coordinate)
    ys = np.linspace(0.0, 2.0 * wavelength, 301)
    xs = np.linspace(-1.0, 1.0, 157)  # no contour level falls on a grid row (161 put d = 0.95 on one)
    Y, X = np.meshgrid(ys, xs)
    items: List = []
    widths, separatrices = [], []
    for i, amp in enumerate(_STAGES):
        x0 = i * (W + gap)

        def cm(pts, x0=x0):
            pts = np.asarray(pts, dtype=float)
            return np.stack([x0 + pts[..., 0] / (2.0 * wavelength) * W, pts[..., 1] * Hh], -1)

        Z = slab_perturbed_flux(X, Y, shear, amp, k_y)
        levels = [0.5 * shear * d * d for d in np.linspace(0.12, 0.95, 8)]
        for line in _contours(Z, xs, ys, levels):
            items.append(Polyline.of(cm(line), "surface", role=f"flux_surface:{i}"))
        items.append(Polyline.of(cm([[0.0, 0.0], [2.0 * wavelength, 0.0]]), "rational", role=f"resonant_surface:{i}"))
        width = 4.0 * math.sqrt(amp / shear)
        widths.append(width)
        if amp > 0.0:
            # a hair inside the saddle value: exactly at it the X-points sit on grid nodes and a last-bit
            # difference in cos(k_y y) decides how contourpy joins the lines, platform by platform
            lines = _contours(Z, xs, ys, [amp * (1.0 - 1e-6)])
            separatrices.append(lines)
            for line in lines:
                items.append(Polyline.of(cm(line), "separatrix", role=f"separatrix:{i}"))
            for yy in (0.0, wavelength, 2.0 * wavelength):
                items.append(Marker(tuple(cm((yy, 0.0))), "x", "xpoint", role=f"x_point:{i}"))
            for yy in (0.5 * wavelength, 1.5 * wavelength):
                items.append(Marker(tuple(cm((yy, 0.0))), "o", "opoint", role=f"o_point:{i}"))
        if i:
            items.append(Arrow((x0 - gap + 0.25, 0.0), (x0 - 0.25, 0.0), "connector", role="stage"))
        if labels:
            title = ("$\\psi_0 = 0$: sheared field", "small $\\psi_0$: X- and O-points",
                     "island, $w = 4\\sqrt{\\psi_0/B_s'}$")[i]
            items.append(Label((x0 + 0.5 * W, Hh + 0.2), title, "small label", anchor="south", role="title"))
        if labels and i == 2:
            yo = cm((1.5 * wavelength, 0.0))
            items += [Arrow((float(yo[0]) + 0.3, -0.5 * width * Hh), (float(yo[0]) + 0.3, 0.5 * width * Hh),
                            "width arrow", role="width", both=True),
                      Label((float(yo[0]) + 0.4, 0.12), "$w$", "small label", anchor="west", role="width")]
    if labels:
        items += [
            Label((-0.15, 0.0), "$x = 0$ ($r_s$)", "small label", anchor="east", role="resonant_surface"),
            Label((1.5 * W + gap, -Hh - 0.3), f"$\\displaystyle {formula_equation(slab_perturbed_flux)}$",
                  "formula box", anchor="north", role="equations"),
            _note("Tearing form $\\Psi_T$ drawn. Resonant surface $k_\\parallel = 0$ locally, $q(r_s) = m/n$ "
                  "globally; the growth ($\\Delta'$, layer, Rutherford) is not drawn", 1.5 * W + gap, -Hh - 1.5),
        ]
    return Diagram("island_formation", Scene(tuple(items)), model={"amplitudes": _STAGES, "widths": widths,
                                                                 "separatrices": separatrices})
