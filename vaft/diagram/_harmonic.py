"""3-D perturbation harmonics: what a complex Fourier coefficient means (#1088).

Five single-concept diagrams, independent of any equilibrium, shot or solver
output:

``normal_field_component``
    $\\delta B_n = \\delta\\mathbf B\\cdot\\hat{\\mathbf n}$: the part of a
    perturbation that crosses a magnetic surface;
``complex_harmonic``
    one harmonic as a complex coefficient -- real and imaginary parts are the
    two quadratures of one pattern, not two fields;
``toroidal_harmonic_phase``
    moving the toroidal origin by $\\Delta\\phi$ turns the coefficient by
    $-n\\Delta\\phi$: amplitude invariant, quadratures not;
``harmonic_real_space_projection``
    the real field the coefficient stands for, on the unwrapped
    $(\\phi, \\theta)$ plane, from :func:`vaft.formula.stability.helical_harmonic`;
``complex_field_superposition``
    external and plasma-response fields add as complex numbers, so the total
    depends on both amplitude and phase.

The phase convention is ``helical_phase``'s, $\\xi = m\\theta - n\\phi$, which is
also how ``vaft.code.gpec`` pairs its stored real/imaginary columns
(``real + 1j * imag``). Amplitudes and phases here are schematic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from vaft.formula.stability import helical_harmonic, helical_phase

from ._chart import Chart, clip, render_chart
from ._equations import formula_equation
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: drawn length of a unit phasor [cm]
_UNIT = 2.6
#: toroidal shift drawn by toroidal_harmonic_phase [rad]
_DELTA_PHI = math.pi / 4
#: the largest n whose rotation -n * _DELTA_PHI stays within one turn
_MAX_N = 7
#: schematic plasma responses relative to a unit external field on the real axis
_CASES = {
    "screening": 0.75 * np.exp(1j * (math.pi - 0.3)),
    "amplification": 0.8 * np.exp(1j * 0.45),
    "phase_shift": 0.9 * np.exp(1j * 2.0 * math.pi / 3.0),
}


@dataclass(eq=False)
class HarmonicFigure:
    """What a harmonic diagram shows: vectors in drawing units and the complex numbers behind them."""

    vectors: Dict[str, np.ndarray] = field(default_factory=dict)
    values: Dict[str, complex] = field(default_factory=dict)
    parameters: Dict[str, float] = field(default_factory=dict)


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _finite(name: str, value, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not {value!r}")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, not {value!r}") from None
    if not math.isfinite(value) or (positive and value <= 0.0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite, not {value!r}")
    return value


def _mode(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer mode number, not {value!r}")
    return int(value)


def _xy(z: complex, unit: float = _UNIT, origin=(0.0, 0.0)):
    return (origin[0] + unit * z.real, origin[1] + unit * z.imag)


def _arc(radius: float, start: float, stop: float, origin=(0.0, 0.0), n: int = 48) -> np.ndarray:
    a = np.linspace(start, stop, n)
    return np.stack([origin[0] + radius * np.cos(a), origin[1] + radius * np.sin(a)], axis=-1)


def _complex_axes(extent: float, origin=(0.0, 0.0), labels: bool = True) -> List:
    x0, y0 = origin
    items: List = [
        Arrow((x0 - extent, y0), (x0 + extent, y0), "chart axis", role="real_axis"),
        Arrow((x0, y0 - extent), (x0, y0 + extent), "chart axis", role="imag_axis"),
    ]
    if labels:
        items += [Label((x0 + extent + 0.15, y0), "Re", "label", anchor="west", role="real_axis"),
                  Label((x0, y0 + extent + 0.15), "Im", "label", anchor="south", role="imag_axis")]
    return items


def _quadratures(z: complex, role: str, unit: float = _UNIT, origin=(0.0, 0.0)) -> List:
    """Dashed drops from a phasor tip to both axes."""
    tip = _xy(z, unit, origin)
    return [Polyline.of([tip, (tip[0], origin[1])], "approx", role=role),
            Polyline.of([tip, (origin[0], tip[1])], "approx", role=role)]


def _equation_box(text: str, x: float, y: float) -> Label:
    return Label((x, y), text, "formula box", anchor="north", role="equations")


def _note(text: str, x: float, y: float) -> Label:
    return Label((x, y), text, "note", anchor="north", role="note")


# ---------------------------------------------------------------------------
# normal component
# ---------------------------------------------------------------------------


def normal_field_component(*, labels: bool = True) -> Diagram:
    r"""$\delta B_n = \delta\mathbf B\cdot\hat{\mathbf n}$: the part of a perturbation that crosses a surface.

    A perturbation $\delta\mathbf B$ at a point of a (schematic) magnetic
    surface splits into $\delta B_n\hat{\mathbf n}$ along the unit normal and
    $\delta\mathbf B_t$ in the surface. Only the normal part moves field
    lines off the surface; no code-specific normalisation is implied.
    """
    labels = _check_labels(labels)
    centre, radius = np.array([0.0, -6.0]), 7.0
    surface = _arc(radius, math.radians(55), math.radians(125), origin=tuple(centre), n=81)
    angle = math.radians(72)
    normal = np.array([math.cos(angle), math.sin(angle)])
    tangent = np.array([math.sin(angle), -math.cos(angle)])  # along the surface, to the right
    point = centre + radius * normal
    dB = 1.7 * normal + 2.3 * tangent  # [cm], schematic
    normal_part = float(dB @ normal) * normal
    tangential_part = dB - normal_part
    q_angle = math.radians(104)
    q_normal = np.array([math.cos(q_angle), math.sin(q_angle)])
    q_point = centre + radius * q_normal
    P = tuple(point)
    items: List = [
        Polyline.of(surface, "lcfs", role="surface"),
        Arrow(tuple(q_point), tuple(q_point + 1.1 * q_normal), "vector", role="surface_normal"),
        Polyline.of([point + normal_part, point + dB], "approx", role="construction"),
        Polyline.of([point + tangential_part, point + dB], "approx", role="construction"),
        Arrow(P, tuple(point + normal_part), "drift", role="normal_component"),
        Arrow(P, tuple(point + tangential_part), "drift ion", role="tangential_component"),
        Arrow(P, tuple(point + dB), "field vector", role="perturbation"),
    ]
    # right-angle mark between the normal and the surface
    s = 0.28
    items.append(Polyline.of([point + s * tangent, point + s * (tangent + normal), point + s * normal],
                             "surface", role="construction"))
    if labels:
        items += [
            Label(tuple(q_point + 1.25 * q_normal), "$\\hat{\\mathbf n}$", "label", anchor="south",
                  role="surface_normal"),
            Label(tuple(point + dB + 0.12 * (normal + tangent)), "$\\delta\\mathbf B$", "label",
                  anchor="south west", role="perturbation"),
            Label(tuple(point + normal_part - 0.15 * tangent), "$\\delta B_n\\,\\hat{\\mathbf n}$", "label",
                  anchor="east", role="normal_component"),
            Label(tuple(point + tangential_part - 0.2 * normal), "$\\delta\\mathbf B_t$", "label",
                  anchor="north", role="tangential_component"),
            Label(tuple(surface[-10] + np.array([0.1, -0.3])), "magnetic surface", "small label",
                  anchor="north west", role="surface"),
            _equation_box("$\\delta\\mathbf B = \\delta B_n\\,\\hat{\\mathbf n} + \\delta\\mathbf B_t,"
                          "\\qquad \\delta B_n = \\delta\\mathbf B\\cdot\\hat{\\mathbf n}$", 0.0, -0.9),
            _note("$\\delta B_n$ crosses the surface; $\\delta\\mathbf B_t$ lies in it", 0.0, -2.0),
        ]
    model = HarmonicFigure(
        vectors={"point": point, "normal": normal, "tangent": tangent, "perturbation": dB,
                 "normal_component": normal_part, "tangential_component": tangential_part,
                 "surface": surface, "surface_centre": centre},
        parameters={"surface_radius": radius, "delta_B_n": float(dB @ normal)},
    )
    return Diagram("normal_field_component", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# one complex harmonic
# ---------------------------------------------------------------------------


def _wrapped(angle: float) -> float:
    """``angle`` in (-pi, pi]."""
    wrapped = math.atan2(math.sin(angle), math.cos(angle))
    return math.pi if wrapped == -math.pi else wrapped


def complex_harmonic(amplitude: float = 1.0, phase: float = math.pi / 3, *, labels: bool = True) -> Diagram:
    r"""One harmonic as a complex coefficient $\hat b = b_R + i\,b_I = A\,e^{i\alpha}$.

    The phasor's projections on the axes are $b_R = A\cos\alpha$ and
    $b_I = A\sin\alpha$, the cosine and sine quadratures of a single pattern
    (``vaft.code.gpec`` stores exactly this pair as ``i = 0, 1``); they are not
    two magnetic fields. ``amplitude`` sets the label only: the phasor is drawn
    at a fixed length.
    """
    A = _finite("amplitude", amplitude, positive=True)
    alpha = _wrapped(_finite("phase", phase))
    labels = _check_labels(labels)
    b = A * complex(math.cos(alpha), math.sin(alpha))
    unit = _UNIT / A
    tip = _xy(b, unit)
    items: List = _complex_axes(1.35 * _UNIT, labels=labels)
    items += [Polyline.of(_arc(_UNIT, 0.0, 2.0 * math.pi, n=121), "surface", role="amplitude", closed=True)]
    items += _quadratures(b, "construction", unit)
    items += [
        Polyline.of([(0.0, 0.0), (tip[0], 0.0)], "component real", role="real_component"),
        Polyline.of([(0.0, 0.0), (0.0, tip[1])], "component imag", role="imag_component"),
        Arrow((0.0, 0.0), tip, "phasor", role="phasor"),
        Marker(tip, "o", "opoint", role="phasor"),
    ]
    if abs(alpha) > 1e-9:
        items.append(Polyline.of(_arc(0.7, 0.0, alpha), "angle arc", role="phase"))
    if labels:
        mid = 0.5 * np.array(tip)
        normal = np.array([-math.sin(alpha), math.cos(alpha)])
        items += [
            Label(tuple(np.array(tip) * 1.08 + 0.1 * normal), "$\\hat b = A\\,e^{i\\alpha}$", "label",
                  anchor="south west" if tip[0] >= 0 else "south east", role="phasor"),
            Label(tuple(mid + 0.3 * normal), "$A$", "label", anchor="center", role="amplitude"),
            Label(tuple(0.95 * np.array([math.cos(alpha / 2), math.sin(alpha / 2)])), "$\\alpha$", "label",
                  anchor="center", role="phase"),
            Label((tip[0], -0.2 if tip[1] >= 0 else 0.2), "$b_R = A\\cos\\alpha$", "label",
                  anchor="north" if tip[1] >= 0 else "south", role="real_component"),
            Label((-0.2 if tip[0] >= 0 else 0.2, tip[1]), "$b_I = A\\sin\\alpha$", "label",
                  anchor="east" if tip[0] >= 0 else "west", role="imag_component"),
            _equation_box("$\\hat b = b_R + i\\,b_I = A\\,e^{i\\alpha}, \\qquad A = |\\hat b|,"
                          "\\quad \\alpha = \\arg\\hat b$", 0.0, -1.35 * _UNIT - 0.7),
            _note("$b_R$ and $b_I$ are two quadratures of one harmonic, not two fields",
                  0.0, -1.35 * _UNIT - 1.8),
        ]
    model = HarmonicFigure(
        vectors={"phasor": np.array(tip)},
        values={"b_hat": b},
        parameters={"amplitude": A, "phase": alpha, "real": b.real, "imag": b.imag, "cm_per_unit": unit},
    )
    return Diagram("complex_harmonic", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# toroidal phase
# ---------------------------------------------------------------------------


def toroidal_harmonic_phase(n: int = 1, *, labels: bool = True) -> Diagram:
    r"""The toroidal mode number is the rate at which the phase winds around the torus.

    Moving the toroidal origin by $\Delta\phi$ multiplies the coefficient by
    $e^{i\xi}$ with $\xi$ from ``helical_phase`` at $\theta = 0$, i.e. turns it
    by $-n\Delta\phi$ ($\Delta\phi = \pi/4$ here, so $n \le 7$ stays within one
    turn). The amplitude is unchanged; $b_R$ and $b_I$ are not, so they are
    meaningful only with a stated phase origin.
    """
    n = _mode("n", n)
    if n > _MAX_N:
        raise ValueError(f"n must be at most {_MAX_N} so the drawn rotation stays within one turn, not {n}")
    labels = _check_labels(labels)
    alpha0 = math.pi / 6
    b0 = complex(math.cos(alpha0), math.sin(alpha0))
    rotation = float(helical_phase(0.0, _DELTA_PHI, 1, n))  # = -n * delta_phi
    b1 = b0 * complex(math.cos(rotation), math.sin(rotation))
    items: List = _complex_axes(1.35 * _UNIT, labels=labels)
    items += [Polyline.of(_arc(_UNIT, 0.0, 2.0 * math.pi, n=121), "surface", role="amplitude", closed=True)]
    items += _quadratures(b0, "construction", _UNIT) + _quadratures(b1, "construction", _UNIT)
    items += [
        Arrow((0.0, 0.0), _xy(b0), "phasor", role="phase_zero"),
        Arrow((0.0, 0.0), _xy(b1), "phasor alt", role="phase_shifted"),
        Polyline.of(_arc(0.38 * _UNIT, alpha0, alpha0 + rotation, n=64), "angle arc", role="phasor"),
    ]
    # the toroidal angle, seen from above
    c, r = (2.45 * _UNIT, 0.5 * _UNIT), 0.7 * _UNIT
    items += [
        Polyline.of(_arc(r, 0.0, 2.0 * math.pi, origin=c, n=121), "lcfs", role="toroidal_angle", closed=True),
        Polyline.of([c, (c[0] + r, c[1])], "surface", role="toroidal_angle"),
        Polyline.of([c, (c[0] + r * math.cos(_DELTA_PHI), c[1] + r * math.sin(_DELTA_PHI))], "surface",
                    role="toroidal_angle"),
        Polyline.of(_arc(0.45 * r, 0.0, _DELTA_PHI, origin=c), "angle arc", role="toroidal_angle"),
    ]
    if labels:
        mid = alpha0 + rotation / 2
        items += [
            Label(tuple(1.08 * np.array(_xy(b0))), "$\\hat b(\\phi_0)$", "label", anchor="south west",
                  role="phase_zero"),
            Label(tuple(1.1 * np.array(_xy(b1))), "$\\hat b(\\phi_0 + \\Delta\\phi)$", "label",
                  anchor="north west" if b1.imag < 0 else "west", role="phase_shifted"),
            Label((0.68 * _UNIT * math.cos(mid), 0.68 * _UNIT * math.sin(mid)), "$-n\\Delta\\phi$", "label",
                  anchor="center", role="phasor"),
            Label((c[0] + 0.72 * r * math.cos(_DELTA_PHI / 2), c[1] + 0.72 * r * math.sin(_DELTA_PHI / 2)),
                  "$\\Delta\\phi$", "small label", anchor="center", role="toroidal_angle"),
            Label((c[0], c[1] + r + 0.15), "top view", "small label", anchor="south", role="toroidal_angle"),
            _equation_box(f"$\\displaystyle {formula_equation(helical_phase)}, \\qquad "
                          f"\\hat b(\\phi_0 + \\Delta\\phi) = \\hat b(\\phi_0)\\,e^{{-in\\Delta\\phi}},"
                          f"\\quad n = {n}$", 0.8 * _UNIT, -1.35 * _UNIT - 0.7),
            _note("$|\\hat b|$ is unchanged; $b_R$ and $b_I$ depend on the phase origin",
                  0.8 * _UNIT, -1.35 * _UNIT - 1.8),
        ]
    model = HarmonicFigure(
        vectors={"phase_zero": np.array(_xy(b0)), "phase_shifted": np.array(_xy(b1))},
        values={"phase_zero": b0, "phase_shifted": b1},
        parameters={"n": n, "delta_phi": _DELTA_PHI, "rotation": rotation},
    )
    return Diagram("toroidal_harmonic_phase", Scene(tuple(items)), model=model)


# ---------------------------------------------------------------------------
# reconstruction in real space
# ---------------------------------------------------------------------------


def _level_lines(m: int, n: int, offset: float) -> List[np.ndarray]:
    """Lines $m\\theta - n\\phi = $ ``offset`` $+ 2\\pi k$ across $[0, 2\\pi]^2$ in $(\\phi, \\theta)$."""
    phi = np.linspace(0.0, 2.0 * math.pi, 241)
    lines = []
    for k in range(-n - 1, m + 2):
        theta = (n * phi + offset + 2.0 * math.pi * k) / m
        lines.append(np.stack([phi, theta], axis=-1))
    return lines


def harmonic_real_space_projection(m: int = 2, n: int = 1, phase: float = math.pi / 3, *,
                                   labels: bool = True) -> Diagram:
    r"""From a complex coefficient to the real field it stands for, $\mathrm{Re}[\hat b\,e^{i(m\theta - n\phi)}]$.

    On the unwrapped $(\phi, \theta)$ plane the field of one harmonic with
    $\hat b = e^{i\alpha}$ is a set of parallel stripes of slope $n/m$; crests
    ($\delta b = +|\hat b|$) lie on $\xi = -\alpha$ and troughs on
    $\xi = \pi - \alpha$, evaluated by ``helical_harmonic``. Changing
    $\alpha$ slides the pattern; the complex number is its amplitude and
    position, the stripes are the physical field.
    """
    m, n = _mode("m", m), _mode("n", n)
    alpha = _wrapped(_finite("phase", phase))
    labels = _check_labels(labels)
    b = complex(math.cos(alpha), math.sin(alpha))
    two_pi = 2.0 * math.pi
    chart = Chart(x_range=(0.0, two_pi), y_range=(0.0, two_pi))
    crests = [line for line in _level_lines(m, n, -alpha) if clip(line, chart)]
    troughs = [line for line in _level_lines(m, n, math.pi - alpha) if clip(line, chart)]
    for i, line in enumerate(crests):
        chart.curves[f"crest_{i}"] = line
    for i, line in enumerate(troughs):
        chart.curves[f"trough_{i}"] = line
    # a crest point inside the box, checked against the formula
    sample = next(p for line in crests for p in line[len(line) // 2::7] if 0.3 < p[1] < two_pi - 0.3)
    chart.points["sample"] = (float(sample[0]), float(sample[1]))
    chart.parameters.update({"m": m, "n": n, "phase": alpha,
                             "sample_value": helical_harmonic(b, sample[1], sample[0], m, n)})
    ticks = (0.0, math.pi, two_pi)
    tick_text = ("$0$", "$\\pi$", "$2\\pi$")
    scene = render_chart(
        chart, x_label="$\\phi$", y_label="$\\theta$",
        curve_styles={**{f"crest_{i}": "crest" for i in range(len(crests))},
                      **{f"trough_{i}": "trough" for i in range(len(troughs))}},
        region_text={}, x_ticks=ticks, y_ticks=ticks, x_tick_text=tick_text, y_tick_text=tick_text,
    )
    # the curves carry numbered roles; give them the family role the tests and readers use
    items = [
        Polyline(it.points, it.style, "spatial_pattern", it.closed) if isinstance(it, Polyline)
        and it.role.startswith(("crest_", "trough_")) else it
        for it in scene.items
    ]
    items.append(Marker(tuple(chart.to_cm(np.array(chart.points["sample"]))), "o", "opoint",
                        role="real_projection"))
    # the coefficient, on its own complex plane to the left
    origin, unit = (-4.3, 3.6), 1.25
    items += _complex_axes(1.6, origin=origin, labels=labels)
    items += [Polyline.of(_arc(unit, 0.0, 2.0 * math.pi, origin=origin, n=97), "surface",
                          role="complex_coefficient", closed=True),
              Arrow(origin, _xy(b, unit, origin), "phasor", role="complex_coefficient"),
              Arrow((origin[0] + 0.3, origin[1] - 2.1), (-0.7, origin[1] - 2.1), "match", role="phase_factor")]
    if labels:
        items += [
            Label((origin[0], origin[1] + 2.45), f"$\\hat b = e^{{i\\alpha}},\\ m/n = {m}/{n}$", "label",
                  anchor="south", role="complex_coefficient"),
            Label((0.5 * (origin[0] - 0.4), origin[1] - 2.2), "$\\mathrm{Re}[\\,\\cdot\\,e^{i(m\\theta - n\\phi)}]$",
                  "small label", anchor="north", role="phase_factor"),
            _equation_box(f"$\\displaystyle {formula_equation(helical_harmonic)}$", 2.0, -1.45),
            _note("blue: crests $\\delta b = +|\\hat b|$; red: troughs $\\delta b = -|\\hat b|$", 2.0, -2.75),
        ]
    return Diagram("harmonic_real_space_projection", Scene(tuple(items)), model=chart)


# ---------------------------------------------------------------------------
# superposition
# ---------------------------------------------------------------------------


def complex_field_superposition(case: str = "screening", *, labels: bool = True) -> Diagram:
    r"""External and plasma-response fields add as complex numbers.

    $\hat b_\mathrm{total} = \hat b_\mathrm{external} + \hat b_\mathrm{plasma}$
    is a vector sum in the complex plane: a response opposing the external
    field screens it (``"screening"``), one nearly in phase amplifies it
    (``"amplification"``), and in general both size and phase change
    (``"phase_shift"``). The values are schematic, not a response calculation.
    """
    if case not in _CASES:
        raise ValueError(f"case must be one of {tuple(_CASES)}, not {case!r}")
    labels = _check_labels(labels)
    external = complex(1.0, 0.0)
    plasma = complex(_CASES[case])
    total = external + plasma
    unit = 2.4
    extent = 1.1 * unit * max(abs(external), abs(total), 1.0) + 0.4
    items: List = _complex_axes(extent, labels=labels)
    items += [
        Polyline.of(_arc(unit * abs(external), 0.0, 2.0 * math.pi, n=121), "surface", role="external",
                    closed=True),
        Arrow((0.0, 0.0), _xy(external, unit), "phasor", role="external"),
        Arrow(_xy(external, unit), _xy(total, unit), "phasor alt", role="plasma"),
        Arrow((0.0, 0.0), _xy(total, unit), "phasor total", role="total"),
    ]
    if labels:
        ext_tip = np.array(_xy(external, unit))
        mid_plasma = 0.5 * (ext_tip + np.array(_xy(total, unit)))
        tot = np.array(_xy(total, unit))
        items += [
            Label(tuple(0.5 * ext_tip + np.array([0.0, -0.25])), "external", "label", anchor="north",
                  role="external"),
            Label(tuple(mid_plasma + np.array([0.25, 0.1])), "plasma", "label", anchor="west", role="plasma"),
            Label(tuple(0.55 * tot + np.array([-0.2, 0.25])), "total", "label", anchor="south east",
                  role="total"),
            _equation_box("$\\hat b_\\mathrm{total} = \\hat b_\\mathrm{external} + \\hat b_\\mathrm{plasma}$",
                          0.0, -extent - 0.6),
            _note(f"{case.replace('_', ' ')}: $|\\hat b_\\mathrm{{total}}|/|\\hat b_\\mathrm{{external}}|"
                  f" = {abs(total) / abs(external):.2f}$, phase shift "
                  f"${math.degrees(math.atan2(total.imag, total.real)):.0f}^\\circ$", 0.0, -extent - 1.6),
        ]
    model = HarmonicFigure(values={"external": external, "plasma": plasma, "total": total},
                           parameters={"case": case, "cm_per_unit": unit})
    return Diagram("complex_field_superposition", Scene(tuple(items)), model=model)
