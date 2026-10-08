"""Slab resonant layers: tearing and twisting parity, harmonic coupling (#1071).

``slab_parity``
    constant-$\\Psi$ contours of ``slab_perturbed_flux``: tearing parity opens an
    island with O- and X-points and reconnects flux across $x = 0$; twisting
    parity keeps $x = 0$ a flux surface and displaces its neighbours;
``slab_parity_comparison``
    the two side by side, with what distinguishes them;
``poloidal_harmonic_coupling``
    a $\\cos\\theta$ (toroidicity) and a $\\cos 2\\theta$ (elongation) equilibrium
    coefficient turn one harmonic $m$ into $m \\pm 1$ and $m \\pm 2$ at fixed $n$;
``resonant_layer_matching``
    several rational surfaces, each with a tearing and a twisting channel,
    coupled through the outer region -- the matrix the multi-surface codes solve.

The mapping of $(m, n)$ to the slab's $(k_y, k_z)$ is ``mode_number_mapping``.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from vaft.formula.stability import slab_perturbed_flux

from ._chart import CHART_HEIGHT, CHART_WIDTH, Chart, render_chart
from ._concept import band, box, connector
from ._equations import formula_equation
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

PARITIES = ("tearing", "twisting")
#: the slab: shear, wavenumber, amplitudes (tearing island width 0.6; twisting displacement 0.12)
_SHEAR, _KY = 1.0, 1.0
_AMPLITUDE = {"tearing": 0.0225, "twisting": 0.12}
_X_HALF = 1.0
#: drawing size of one parity panel [cm]
_W, _H = 9.0, 5.0


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _contours(Z, xs, ys, levels) -> List[np.ndarray]:
    """Contour lines of ``Z`` (rows over ``xs``, columns over ``ys``), each as (y, x) points."""
    from contourpy import LineType, contour_generator

    gen = contour_generator(x=ys, y=xs, z=Z, line_type=LineType.Separate)
    out = []
    for level in levels:
        out += [np.asarray(line) for line in gen.lines(level) if len(line) > 2]
    return out


def _panel(parity: str, labels: bool, x_off: float = 0.0) -> dict:
    wavelength = 2.0 * math.pi / _KY
    ys = np.linspace(0.0, 2.0 * wavelength, 401)
    # 197 rows: with 201 the outermost level d = 0.95 lay exactly on a grid row, where last-bit differences
    # between platforms make the contour zigzag between cells
    xs = np.linspace(-_X_HALF, _X_HALF, 197)
    Y, X = np.meshgrid(ys, xs)
    amp = _AMPLITUDE[parity]
    Z = slab_perturbed_flux(X, Y, _SHEAR, amp, _KY, parity=parity)
    if parity == "twisting":
        # complete the square with the O(psi1^2) term the linear flux drops: the surfaces, the
        # rational one included, are then rigidly displaced by xi = -psi1 cos(k_y y)/B' with no
        # spurious cells at x = 0
        Z = Z + 0.5 * (amp * np.cos(_KY * Y)) ** 2 / _SHEAR

    def cm(pts):
        pts = np.asarray(pts, dtype=float)
        return np.stack([x_off + pts[..., 0] / (2.0 * wavelength) * _W, pts[..., 1] / _X_HALF * 0.5 * _H], -1)

    # levels evenly spaced in distance from the rational surface (Psi ~ x^2), so the layer is resolved
    levels = [0.5 * _SHEAR * d * d for d in np.linspace(0.06, 0.95, 12)]
    items: List = []
    for line in _contours(Z, xs, ys, levels):
        items.append(Polyline.of(cm(line), "surface", role="flux_surface"))
    if parity == "tearing":
        rational = np.array([[0.0, 0.0], [2.0 * wavelength, 0.0]])
    else:  # the rational surface moves with its neighbours
        rational = np.stack([ys, -amp * np.cos(_KY * ys) / _SHEAR], -1)
    items.append(Polyline.of(cm(rational), "rational", role="rational_surface"))
    info = {"amplitude": amp, "rational_surface": rational, "levels": levels, "grid": (ys, xs, Z)}
    if parity == "tearing":
        sep_level = amp  # the X-points' value, Psi = psi_0 at x = 0, cos = 1
        # a hair inside the saddle value: exactly at it the X-points are grid nodes and last-bit
        # differences decide how the contour joins there, platform by platform
        for line in _contours(Z, xs, ys, [sep_level * (1.0 - 1e-6)]):
            items.append(Polyline.of(cm(line), "separatrix", role="separatrix"))
        x_points = [(0.0, 0.0), (wavelength, 0.0), (2.0 * wavelength, 0.0)]
        o_points = [(0.5 * wavelength, 0.0), (1.5 * wavelength, 0.0)]
        for p in x_points:
            items.append(Marker(tuple(cm(p)), "x", "xpoint", role="x_point"))
        for p in o_points:
            items.append(Marker(tuple(cm(p)), "o", "opoint", role="o_point"))
        width = 4.0 * math.sqrt(amp / _SHEAR)
        info.update({"x_points": x_points, "o_points": o_points, "width": width, "separatrix_level": sep_level})
    # delta B_x = -dPsi/dy on x = 0, drawn along the rational surface
    arrows = []
    for yy in np.linspace(0.25, 1.75, 7) * wavelength:
        dPsi_dy = (slab_perturbed_flux(0.0, yy + 1e-6, _SHEAR, amp, _KY, parity)
                   - slab_perturbed_flux(0.0, yy - 1e-6, _SHEAR, amp, _KY, parity)) / 2e-6
        bx = -dPsi_dy
        arrows.append((yy, bx))
        if abs(bx) > 1e-9:
            start = cm((yy, 0.0))
            items.append(Arrow(tuple(start), (float(start[0]), float(start[1]) + 12.0 * bx), "drift", role="delta_Bx"))
    info["delta_Bx_on_x0"] = arrows
    if labels:
        title = {"tearing": "tearing parity: $\\tilde\\psi$ even, island",
                 "twisting": "twisting parity: $\\tilde\\psi$ odd, rigid displacement"}[parity]
        items += [
            Label((x_off + 0.5 * _W, 0.5 * _H + 0.25), title, "label", anchor="south", role="title"),
            Label((x_off - 0.15, 0.0), "$x = 0$", "small label", anchor="east", role="rational_surface"),
            Label((x_off + 0.5 * _W, -0.5 * _H - 0.15), "$y$", "label", anchor="north", role="axes"),
        ]
        bx_text = ("$\\delta B_x(0) \\ne 0$: flux crosses the rational surface" if parity == "tearing"
                   else "$\\delta B_x = 0$ at the layer ($k_\\parallel = 0$): no reconnection; all surfaces move together")
        items.append(Label((x_off + 0.5 * _W, -0.5 * _H - 0.7), bx_text, "small label", anchor="north",
                           role="delta_Bx"))
    return {"items": items, "info": info}


def slab_parity(parity: str = "tearing", *, labels: bool = True) -> Diagram:
    r"""Tearing or twisting parity at a rational surface, as contours of the helical flux.

    Contours of $\Psi$ from ``slab_perturbed_flux`` over two wavelengths: for
    tearing parity the island of width $4\sqrt{\psi_0/B_s'}$ with its O- and
    X-points and separatrix, and $\delta B_x \ne 0$ on $x = 0$ (arrows); for
    twisting parity $\Psi_W$ completed with its $O(\psi_1^2)$ term, so every
    surface -- the rational one included -- is displaced together by
    $\xi = -\psi_1\cos k_yy/B_s'$, with no normal field at the layer and no
    reconnection.
    """
    if parity not in PARITIES:
        raise ValueError(f"parity must be one of {PARITIES}, not {parity!r}")
    labels = _check_labels(labels)
    panel = _panel(parity, labels)
    items = list(panel["items"])
    if labels:
        items += [
            Label((0.5 * _W, -0.5 * _H - 1.35), f"$\\displaystyle {formula_equation(slab_perturbed_flux)}$",
                  "formula box", anchor="north", role="equations"),
            Label((0.5 * _W, -0.5 * _H - 2.6), "Sheared slab, $B_y = B_s'x$; radial $x$ up, binormal $y$ across"
                  + ("; twisting drawn with its $O(\\psi_1^2)$ completion" if parity == "twisting" else ""),
                  "note", anchor="north", role="note"),
        ]
    return Diagram(f"slab_parity_{parity}", Scene(tuple(items)), model={"parity": parity, **panel["info"]})


def slab_parity_comparison(*, labels: bool = True) -> Diagram:
    r"""Tearing and twisting parity side by side, with what tells them apart.

    The two panels of ``slab_parity``, and under them the defining
    properties: the parity of $\tilde\psi$ and of the stream function, the
    normal field on the rational surface, and whether flux reconnects. The
    parity is a property of the local layer response; the harmonic index $m$
    of ``poloidal_harmonic_coupling`` is a different index.
    """
    labels = _check_labels(labels)
    items: List = []
    info = {}
    for i, parity in enumerate(PARITIES):
        panel = _panel(parity, labels, x_off=i * (_W + 1.5))
        items += panel["items"]
        info[parity] = panel["info"]
    if labels:
        rows = [("perturbation", "$\\psi_0\\cos k_yy$", "$\\psi_1x\\cos k_yy$"),
                ("$\\tilde\\psi(-x)$", "$+\\tilde\\psi(x)$", "$-\\tilde\\psi(x)$"),
                ("$\\tilde\\phi(-x)$", "$-\\tilde\\phi(x)$", "$+\\tilde\\phi(x)$"),
                ("$\\delta B_x(0)$", "$\\ne 0$", "$= 0$"),
                ("displacement", "odd, $\\propto 1/x$", "even, rigid"),
                ("topology", "island, reconnected flux", "no reconnection")]
        y0 = -0.5 * _H - 1.6
        for j, (name, t, w) in enumerate(rows):
            y = y0 - 0.5 * j
            items += [Label((-0.2, y), name, "small label", anchor="east", role="table"),
                      Label((0.5 * _W, y), t, "small label", anchor="center", role="table"),
                      Label((_W + 1.5 + 0.5 * _W, y), w, "small label", anchor="center", role="table")]
        items.append(Label((_W + 0.75, y0 - 3.3), "T and W are two layer responses of the same $(m, n)$, not "
                           "different mode numbers", "note", anchor="north", role="note"))
    return Diagram("slab_parity_comparison", Scene(tuple(items)), model=info)


# ---------------------------------------------------------------------------
# harmonic coupling
# ---------------------------------------------------------------------------


def harmonic_coupling_spectrum(m: int, c1: float, c2: float, n_theta: int = 256) -> dict:
    """The poloidal spectrum of $C(\\theta)e^{im\\theta}$, $C = 1 + c_1\\cos\\theta + c_2\\cos 2\\theta$, by FFT."""
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    f = (1.0 + c1 * np.cos(theta) + c2 * np.cos(2.0 * theta)) * np.exp(1j * m * theta)
    spec = np.fft.fft(f) / n_theta
    ms = np.fft.fftfreq(n_theta, d=1.0 / n_theta).astype(int)
    return {int(k): complex(v) for k, v in zip(ms, spec) if abs(v) > 1e-12}


def poloidal_harmonic_coupling(m: int = 3, *, labels: bool = True) -> Diagram:
    r"""Toroidicity couples $m$ to $m \pm 1$, elongation to $m \pm 2$, at fixed $n$.

    A single harmonic $e^{im\theta}$ multiplied by an equilibrium coefficient
    $C(\theta) = 1 + c_1\cos\theta + c_2\cos 2\theta$ (schematic $c_1 = 0.4$ for
    toroidicity and the Shafranov shift, $c_2 = 0.25$ for elongation) has the
    spectrum below, computed by FFT: $\cos\theta\,e^{im\theta} =
    \tfrac12[e^{i(m+1)\theta} + e^{i(m-1)\theta}]$, and $\cos 2\theta$ likewise
    with $m \pm 2$. Real shaped equilibria have more harmonics and couple more.
    """
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or not 1 <= m <= 12:
        raise ValueError(f"m must be an integer from 1 to 12, not {m!r}")
    labels = _check_labels(labels)
    c1, c2 = 0.4, 0.25
    spectrum = harmonic_coupling_spectrum(m, c1, c2)
    ms = sorted(spectrum)
    chart = Chart(x_range=(m - 3.0, m + 3.0), y_range=(0.0, 1.2))
    source = {m: "self", m - 1: "cos1", m + 1: "cos1", m - 2: "cos2", m + 2: "cos2"}
    style = {"self": "component real", "cos1": "component imag", "cos2": "orbit electron"}
    for k in ms:
        chart.curves[f"bar_{k}"] = np.array([[k, 0.0], [k, abs(spectrum[k])]])
    chart.parameters.update({"m": m, "c1": c1, "c2": c2, "spectrum": spectrum})
    ticks = tuple(float(k) for k in range(m - 2, m + 3))
    tick_text = ("$m-2$", "$m-1$", f"$m = {m}$", "$m+1$", "$m+2$")
    scene = render_chart(chart, x_label="poloidal harmonic", y_label="amplitude",
                         curve_styles={f"bar_{k}": style[source[k]] for k in ms}, region_text={},
                         x_ticks=ticks, x_tick_text=tick_text, y_ticks=(0.0, 0.5, 1.0))
    items = [Polyline(it.points, it.style, f"harmonic:{source[int(it.role.split('_')[1])]}", it.closed)
             if isinstance(it, Polyline) and it.role.startswith("bar_") else it for it in scene.items]
    if labels:
        items += [
            Label((CHART_WIDTH + 0.9, CHART_HEIGHT - 0.1), "red: the driven $m$", "small label", anchor="north west",
                  role="legend"),
            Label((CHART_WIDTH + 0.9, CHART_HEIGHT - 0.6), f"blue: $m \\pm 1$, $c_1/2 = {c1 / 2:g}$ (toroidicity)",
                  "small label", anchor="north west", role="legend"),
            Label((CHART_WIDTH + 0.9, CHART_HEIGHT - 1.1), f"thin dark: $m \\pm 2$, $c_2/2 = {c2 / 2:g}$ (elongation)",
                  "small label", anchor="north west", role="legend"),
            Label((CHART_WIDTH / 2, -1.45), "$\\cos\\theta\\,e^{im\\theta} = \\tfrac12\\left[e^{i(m+1)\\theta} + "
                  "e^{i(m-1)\\theta}\\right],\\quad \\cos 2\\theta\\,e^{im\\theta} = \\tfrac12\\left[e^{i(m+2)\\theta} + "
                  "e^{i(m-2)\\theta}\\right]$", "formula box", anchor="north", role="equations"),
            Label((CHART_WIDTH / 2, -2.6), "Same $n$ throughout; $m$ is the harmonic index, not the T/W parity. "
                  "$C(\\theta)$ schematic", "note", anchor="north", role="note"),
        ]
    return Diagram("poloidal_harmonic_coupling", Scene(tuple(items)), model=chart)


# ---------------------------------------------------------------------------
# multi-surface matching
# ---------------------------------------------------------------------------


def resonant_layer_matching(*, labels: bool = True) -> Diagram:
    r"""Several rational surfaces, each with a tearing and a twisting channel, matched through the outer region.

    For one $n$, rational surfaces $q = m/n$ ($m = 2, 3, 4$ at $n = 1$) each
    carry a tearing-parity (T) and a twisting-parity (W) layer response. The
    ideal outer region couples all of them -- through the poloidal-harmonic
    coupling of the equilibrium -- into one $2N \times 2N$ matching matrix
    (the multi-surface generalisation of $\Delta'$ that RDCON/STRIDE compute),
    while the layer physics is solved surface by surface (SLAYER-type models
    for the tearing response).
    """
    labels = _check_labels(labels)
    items: List = []
    items += band(-1.2, 17.2, -4.1, 1.6, "one toroidal mode number $n$", role="band")
    outer = box(8.0, 0.4, 16.0, 1.2, "ideal outer region: every surface coupled through the equilibrium"
                "\\\\ $\\det[D'_\\mathrm{outer} - D_\\mathrm{layer}] = 0$ on a $2N\\times2N$ matrix", role="outer",
                latex=True)
    items += list(outer.items)
    surfaces = []
    for i, m in enumerate((2, 3, 4)):
        x = 2.5 + 5.5 * i
        head = box(x, -1.4, 4.6, 0.9, f"$(n, m) = (1, {m})$ at $r_{{s,{i + 1}}}$", role=f"surface:{m}", latex=True)
        t = box(x - 1.2, -3.0, 2.1, 1.1, "T\\\\ tearing", role=f"layer:{m}:tearing", latex=True)
        w = box(x + 1.2, -3.0, 2.1, 1.1, "W\\\\ twisting", role=f"layer:{m}:twisting", latex=True)
        items += list(head.items) + list(t.items) + list(w.items)
        items += [connector(outer, head, role=f"edge:outer->{m}"), connector(head, t, role=f"edge:{m}->T"),
                  connector(head, w, role=f"edge:{m}->W")]
        surfaces.append(m)
    if labels:
        items.append(Label((8.0, -4.5), "Each layer labelled $(n, m, r_s, \\mathrm{parity})$; the outer region couples "
                           "all (RDCON/STRIDE); layer models give T (e.g. SLAYER) and W responses", "note",
                           anchor="north", role="note"))
    return Diagram("resonant_layer_matching", Scene(tuple(items)), model={"surfaces": tuple(surfaces)})
