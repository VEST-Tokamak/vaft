"""Linear ideal-MHD waves in a uniform plasma: polarization and the branch family (#1063).

``shear_alfven_wave``
    field-line bending: displacement, $\\delta\\mathbf v_\\perp$ and
    $\\delta\\mathbf B_\\perp$ perpendicular to the $\\mathbf k$-$\\mathbf B_0$
    plane, field lines equally spaced so $|\\mathbf B|$ is unchanged to first
    order (``shear_alfven_frequency``);
``fast_magnetosonic_wave``
    the $\\mathbf k \\perp \\mathbf B_0$ fast wave: field lines bunch and
    spread, $\\delta B_\\parallel \\ne 0$, the compressional Alfven wave when
    $c_s \\ll v_A$ (``magnetosonic_phase_speeds``);
``mhd_wave_family``
    the Friedrichs polar diagram of the three branches and what restores
    each.

These are the uniform-slab starting point; the Alfven continuum and toroidal
eigenmodes (TAE, EAE) build on them and are not computed here.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from vaft.formula.stability import magnetosonic_phase_speeds, shear_alfven_frequency

from ._equations import formula_equation
from ._render import Diagram
from ._scene import Arrow, Label, Polyline, Scene

#: drawn plasma: Alfven speed, sound speed (beta-like ratio c_s/v_A = 0.6), unit B0
_VA, _CS, _B0 = 1.0, 0.6, 1.0


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _note(text: str, x: float, y: float) -> Label:
    return Label((x, y), text, "note", anchor="north", role="note")


def shear_alfven_wave(*, labels: bool = True) -> Diagram:
    r"""Shear Alfven polarization: field-line bending without compression.

    $\mathbf B_0 = B_0\hat{\mathbf z}$ across the page, the displacement
    $\xi_y = \xi_0\cos(k_\parallel z - \omega t)$ up, at $t = 0$, over one and
    a half wavelengths. Every field line is displaced alike, so the spacing
    -- $|\mathbf B|$ -- is unchanged to first order and the restoring force is
    tension alone. Red: $\delta v_y = \partial_t\xi_y$; blue:
    $\delta B_y = B_0\partial_z\xi_y = -(B_0/v_A)\,\delta v_y$ for the wave
    travelling along $+\mathbf B_0$, with $\omega$ from
    ``shear_alfven_frequency``. Both are along $\hat{\mathbf y}$, normal to the
    $\mathbf k$-$\mathbf B_0$ plane; $\omega$ does not depend on $k_\perp$.
    """
    labels = _check_labels(labels)
    L, n_lines, dy = 12.0, 5, 0.9  # page width [cm], field lines, spacing [cm]
    wavelength = L / 1.5
    k = 2.0 * math.pi / wavelength
    omega = float(shear_alfven_frequency(k, _VA))
    xi0 = 0.35
    z = np.linspace(0.0, L, 361)
    items: List = []
    lines = []
    for j in range(n_lines):
        y0 = (j - (n_lines - 1) / 2) * dy
        y = y0 + xi0 * np.cos(k * z)
        lines.append(np.stack([z, y], -1))
        items.append(Polyline.of(lines[-1], "field line", role="field_line"))
    zs = wavelength / 8 + np.arange(8) * wavelength / 4  # off the nodes of sin(kz)
    zs = zs[zs < L - 0.2]
    scale_v, scale_b = 0.9 / (omega * xi0), 0.9 / (_B0 * k * xi0)
    y_top = (n_lines - 1) / 2 * dy
    samples = []
    for zz in zs:
        dv = omega * xi0 * math.sin(k * zz)  # d/dt of xi0 cos(kz - wt) at t = 0
        dB = -_B0 * k * xi0 * math.sin(k * zz)
        base_v = y_top + xi0 * math.cos(k * zz)
        base_b = -y_top + xi0 * math.cos(k * zz)
        samples.append((zz, dv, dB))
        if abs(dv) > 1e-9:
            items.append(Arrow((zz, base_v), (zz, base_v + scale_v * dv), "drift", role="delta_v"))
            items.append(Arrow((zz, base_b), (zz, base_b + scale_b * dB), "drift ion", role="delta_B"))
    base = -(n_lines + 1) / 2 * dy - 0.4
    items += [Arrow((0.0, base), (2.0, base), "vector", role="B0"),
              Arrow((L - 2.0, base), (L, base), "exb", role="k")]
    if labels:
        items += [
            Label((2.1, base), "$\\mathbf{B}_0 = B_0\\hat{\\mathbf{z}}$", "small label", anchor="west", role="B0"),
            Label((L - 2.1, base), "$k_\\parallel$", "small label", anchor="east", role="k"),
            Label((L + 0.2, y_top + 0.3), "red: $\\delta v_y$", "small label", anchor="west", role="legend"),
            Label((L + 0.2, -y_top - 0.3), "blue: $\\delta B_y = -(B_0/v_A)\\,\\delta v_y$", "small label",
                  anchor="west", role="legend"),
            Label((L + 0.2, 0.0), "$y$ up, $z$ across, $x$ into the page;\\\\ $\\mathbf{k}$ in the $x$-$z$ plane",
                  "small label,align=left", anchor="west", role="axes"),
            Label((0.5 * L, y_top + 1.4), "shear Alfv\\'en wave: field-line bending, $|\\mathbf{B}|$ unchanged",
                  "label", anchor="south", role="title"),
            Label((0.5 * L, base - 0.5), f"$\\displaystyle {formula_equation(shear_alfven_frequency)}$",
                  "formula box", anchor="north", role="equations"),
            _note("Equal spacing: no first-order compression; restoring force is tension. Inhomogeneous "
                  "$v_A(r)$, $k_\\parallel(r)$ turn this into the Alfv\\'en continuum", 0.5 * L, base - 1.6),
        ]
    return Diagram("shear_alfven_wave", Scene(tuple(items)),
                   model={"k": k, "omega": omega, "xi0": xi0, "lines": lines, "samples": samples})


def fast_magnetosonic_wave(*, labels: bool = True) -> Diagram:
    r"""Fast magnetosonic polarization at $\mathbf k \perp \mathbf B_0$: compression and rarefaction.

    $\mathbf B_0$ across the page, $\mathbf k = k\hat{\mathbf x}$ up, the
    displacement $\xi_x = \xi_0\sin(kx - \omega t)$ at $t = 0$ along
    $\mathbf k$: field lines bunch where $\partial_x\xi_x < 0$ and spread
    where it is positive, $\delta B_z = -B_0\partial_x\xi_x$. Magnetic and
    thermal pressure rise together, the restoring force of the fast wave,
    whose speed here is ``magnetosonic_phase_speeds`` at $\theta = 90^\circ$,
    $\sqrt{v_A^2 + c_s^2}$; for $c_s \ll v_A$ it is the compressional Alfven
    wave $\omega \simeq kv_A$. At other angles the fast and slow branches mix
    compression with bending (``mhd_wave_family``).
    """
    labels = _check_labels(labels)
    L, n_lines, dx = 10.0, 15, 0.42  # page width [cm], field lines, unperturbed spacing [cm]
    height = (n_lines - 1) * dx
    k = 2.0 * math.pi / height
    xi0 = 0.2
    fast, slow = magnetosonic_phase_speeds(0.5 * math.pi, _VA, _CS)
    x0s = (np.arange(n_lines) - (n_lines - 1) / 2) * dx
    xs = x0s + xi0 * np.sin(k * x0s)
    items: List = []
    # compressed (dxi/dx < 0) and rarefied bands, from delta B_z = -B0 dxi/dx
    grid = np.linspace(x0s[0], x0s[-1], 721)
    dBz = -_B0 * k * xi0 * np.cos(k * grid)
    for sign, style, role in ((1.0, "layer", "compressed"), (-1.0, "concept band", "rarefied")):
        inside = sign * dBz > 0.5 * _B0 * k * xi0
        edges = np.flatnonzero(np.diff(inside.astype(int)))
        runs = np.split(np.arange(len(grid)), edges + 1)
        for run in runs:
            if inside[run[0]]:
                a, b = grid[run[0]], grid[run[-1]]
                items.append(Polyline.of([(0.0, a), (L, a), (L, b), (0.0, b)], style, role=role, closed=True))
    for x in xs:
        items.append(Polyline.of([(0.0, x), (L, x)], "field line", role="field_line"))
    for x in x0s[::2]:
        v = -xi0 * math.cos(k * x)  # d/dt of xi0 sin(kx - wt) at t = 0, per unit omega
        if abs(v) > 1e-3:
            items.append(Arrow((-0.5, x), (-0.5, x + 3.0 * v), "drift", role="delta_v"))
    items += [Arrow((L + 0.6, x0s[0]), (L + 0.6, x0s[0] + 1.8), "exb", role="k"),
              Arrow((0.0, x0s[0] - 0.7), (2.0, x0s[0] - 0.7), "vector", role="B0")]
    if labels:
        items += [
            Label((L + 0.75, x0s[0] + 0.9), "$\\mathbf{k} = k\\hat{\\mathbf{x}}$", "small label", anchor="west",
                  role="k"),
            Label((2.1, x0s[0] - 0.7), "$\\mathbf{B}_0$", "small label", anchor="west", role="B0"),
            Label((-0.7, 0.0), "$\\delta v_x$", "small label", anchor="east", role="delta_v"),
            Label((L + 0.2, x0s[-1]), "shaded red: compressed, $\\delta B_z > 0$\\\\ grey: rarefied, "
                  "$\\delta B_z < 0$", "small label,align=left", anchor="north west", role="legend"),
            Label((0.5 * L, x0s[-1] + 0.5), "fast magnetosonic wave, $\\mathbf{k} \\perp \\mathbf{B}_0$: "
                  "field compression", "label", anchor="south", role="title"),
            Label((0.5 * L, x0s[0] - 1.2),
                  "$\\displaystyle \\delta B_z = -B_0\\,\\partial_x\\xi_x,\\qquad v_f(90^\\circ) = "
                  f"\\sqrt{{v_A^2 + c_s^2}} = {float(fast):.2f}\\,v_A$", "formula box", anchor="north",
                  role="equations"),
            _note(f"$c_s = {_CS:g}v_A$; compressional Alfv\\'en wave in the limit $c_s \\ll v_A$ only; "
                  "the slow wave vanishes at $90^\\circ$", 0.5 * L, x0s[0] - 2.3),
        ]
    return Diagram("fast_magnetosonic_wave", Scene(tuple(items)),
                   model={"k": k, "xi0": xi0, "positions": xs, "v_fast": float(fast), "v_slow": float(slow)})


def mhd_wave_family(*, labels: bool = True) -> Diagram:
    r"""The Friedrichs diagram: phase speed against the angle to $\mathbf B_0$, three branches.

    Polar curves of the phase speed at angle $\theta$ between $\mathbf k$ and
    $\mathbf B_0$ (horizontal) for $c_s = 0.6v_A$: fast and slow from
    ``magnetosonic_phase_speeds``, shear Alfven $v_A|\cos\theta|$ from
    ``shear_alfven_frequency``$/k$ -- two circles through the origin. At every
    angle $v_s \le v_A|\cos\theta| \le v_f$. Beside it, what restores each
    branch, whether it compresses, and its polarization.
    """
    labels = _check_labels(labels)
    theta = np.linspace(0.0, 2.0 * math.pi, 721)
    fast, slow = magnetosonic_phase_speeds(theta, _VA, _CS)
    alfven = shear_alfven_frequency(np.cos(theta), _VA)  # omega/k with k = 1
    S = 3.2  # cm per v_A
    curves = {"fast": fast, "alfven": alfven, "slow": slow}
    styles = {"fast": "component real", "alfven": "component imag", "slow": "orbit electron"}
    items: List = [Arrow((-1.25 * S, 0.0), (1.35 * S, 0.0), "frame axis arrow", role="axes"),
                   Arrow((0.0, -1.25 * S), (0.0, 1.3 * S), "frame axis arrow", role="axes")]
    for name, v in curves.items():
        items.append(Polyline.of(np.stack([S * v * np.cos(theta), S * v * np.sin(theta)], -1), styles[name],
                                 role=name, closed=True))
    if labels:
        t = math.radians(50.0)
        f50, s50 = magnetosonic_phase_speeds(t, _VA, _CS)
        items += [
            Label((1.36 * S, 0.0), "$\\mathbf{B}_0$", "small label", anchor="west", role="axes"),
            Label((0.1, 1.3 * S), "$\\perp\\mathbf{B}_0$", "small label", anchor="west", role="axes"),
            Label((S * float(f50) * math.cos(t) + 0.15, S * float(f50) * math.sin(t) + 0.1), "fast",
                  "small label", anchor="south west", role="fast"),
            Label((S * 0.5 * _VA + 0.1, S * 0.55 * _VA), "shear Alfv\\'en", "small label", anchor="south west",
                  role="alfven"),
            Label((S * float(s50) * math.cos(t) + 0.05, S * float(s50) * math.sin(t) - 0.05), "slow",
                  "small label", anchor="north west", role="slow"),
            Label((0.0, -1.3 * S - 0.3), f"phase speed $\\omega/k$ at angle $\\theta$ to $\\mathbf{{B}}_0$; "
                  f"$c_s = {_CS:g}v_A$", "small label", anchor="north", role="axes"),
        ]
        rows = [("branch", "restoring force", "compression", "polarization"),
                ("shear Alfv\\'en", "tension", "none (ideal)", "$\\perp$ $\\mathbf{k}$-$\\mathbf{B}_0$ plane"),
                ("fast", "$B^2$ + thermal, in phase", "yes", "in $\\mathbf{k}$-$\\mathbf{B}_0$ plane"),
                ("slow", "$B^2$ + thermal, antiphase", "yes", "in $\\mathbf{k}$-$\\mathbf{B}_0$ plane")]
        x0, cols = 1.6 * S, (0.0, 2.6, 7.2, 9.6)
        for j, row in enumerate(rows):
            for c, text in zip(cols, row):
                items.append(Label((x0 + c, 1.2 - 0.6 * j), text, "small label", anchor="west",
                                   role="table:head" if j == 0 else "table"))
        items.append(Polyline.of([(x0, 0.9), (x0 + 12.6, 0.9)], "surface", role="table"))
        items.append(Label((x0 + 6.3, -1.5), f"$\\displaystyle {formula_equation(magnetosonic_phase_speeds)}$",
                           "formula box", anchor="north", role="equations"))
        items.append(_note("Uniform ideal MHD; inhomogeneity gives the Alfv\\'en continuum, toroidal coupling "
                           "the TAE/EAE gaps", x0 + 6.3, -3.0))
    return Diagram("mhd_wave_family", Scene(tuple(items)),
                   model={"theta": theta, "fast": fast, "alfven": alfven, "slow": slow, "c_s": _CS, "v_A": _VA})
