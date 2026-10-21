"""Canonical turbulence concepts: drift waves, cross-phase, saturation, shear, zonal flows, scales (#1112).

``drift_wave_transport_cycle``
    a gradient's free energy, the drift frequency, the instability drive and
    kinetic response, growth (if $\\gamma > 0$), density and potential
    fluctuations, their $E\\times B$ velocity and the radial flux; fast
    nonlinear saturation at fixed gradients kept apart from the slow profile
    relaxation;
``cross_phase_transport``
    a density and a potential fluctuation at three cross-phases, with the
    signed flux of ``coherent_particle_flux_from_amplitudes_phase``: none in
    phase, outward at $+\\pi/2$, inward at $-\\pi/2$;
``mixing_length_saturation``
    the phenomenological mixing-length scaling $D \\sim \\gamma/k_\\perp^2$ from
    balancing $\\gamma$ against an estimated rate $k_\\perp\\tilde v$ -- not a
    predictive saturation rule;
``exb_shear_decorrelation``
    one eddy without and with a sheared $E\\times B$ flow: tilted, stretched,
    shorter along a fixed-$y$ cut (geometry), followed by the separate
    statistical decorrelation $k_x(t) = k_x(0) - Sk_yt$;
``zonal_flow_feedback``
    turbulence, Reynolds stress, zonal flow, shear, and the shear acting back
    on the turbulence -- self-regulation without a coefficient model;
``turbulence_scale_hierarchy``
    ion- and electron-scale turbulence bands against the equilibrium lengths
    of one illustrative state.

The flux, velocity, rate and stress relations drawn are ``vaft.formula.turbulence``
definitions; the wavenumber advection $k_x(t)$ and the scale bands are stated
conventions. None of these figures is a turbulence model or a prediction.
Geometry is slab: $x$ radial (outward), $y$ binormal, $\\mathbf B = B\\hat z$.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from vaft.formula.turbulence import (
    coherent_particle_flux_from_amplitudes_phase,
    diamagnetic_frequency_from_k_v_star,
    exb_shearing_rate_from_velocity_gradient,
    exb_velocity_from_potential_gradient,
    mixing_length_diffusivity_from_gamma_k_perp,
    reynolds_stress_from_velocity_fluctuations,
    turbulent_particle_flux_from_fluctuations,
    zonal_flow_drive_from_reynolds_stress_gradient,
)

from ._chart import CHART_HEIGHT, CHART_WIDTH, Chart, render_chart
from ._concept import box, connector
from ._equations import formula_equation
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _eq(function) -> str:
    return f"$\\displaystyle {formula_equation(function)}$"


def drift_wave_transport_cycle(*, labels: bool = True) -> Diagram:
    r"""From an equilibrium gradient to a turbulent flux, and the slow profile feedback.

    Top row: density and temperature gradients ($a/L_n$, $a/L_T$) are a
    free-energy source; the diamagnetic frequency
    $\omega_* = \mathbf k\cdot\mathbf v_*$ (``diamagnetic_frequency_from_k_v_star``)
    sets the drift-wave frequency scale but is not the growth rate, and a
    single $\omega_*$ does not carry the energy-dependent $\omega_*^T$ that
    drives ITG and ETG; whether a mode grows at all is decided by the
    instability drive and kinetic response (curvature, magnetic shear,
    trapped particles, collisions, resonances), so $\gamma > 0$ is not
    implied by the gradient. Bottom row: the fluctuations $\tilde n$,
    $\tilde\phi$ drift at $\tilde{\mathbf v}_E = \mathbf B\times\nabla\tilde\phi/B^2$
    (``exb_velocity_from_potential_gradient``) and their correlation, not
    their mere presence, gives the flux $\Gamma = \langle\tilde n\tilde v_r\rangle$
    (``turbulent_particle_flux_from_fluctuations``). Two feedbacks are kept
    apart: nonlinear saturation (mode coupling, zonal flows,
    ``zonal_flow_feedback``) at fixed gradients, as in a flux-tube
    simulation, and the dashed profile relaxation on the slower transport
    time scale, set by the particle and energy balance with the sources.
    """
    labels = _check_labels(labels)
    w, h = 3.9, 1.35
    nodes = {
        "gradient": (0.0, 3.2, "gradients $a/L_n$, $a/L_T$\\\\free energy"),
        "drift": (4.5, 3.2, f"drift frequency\\\\{_eq(diamagnetic_frequency_from_k_v_star)}"),
        "drive": (9.0, 3.2, "instability drive and kinetic response"),
        "growth": (13.5, 3.2, "linear growth $\\gamma$\\\\(if $\\gamma > 0$)"),
        "fluctuations": (13.5, 0.0, "fluctuations\\\\$\\tilde n$, $\\tilde\\phi$"),
        "exb_velocity": (9.0, 0.0, f"{_eq(exb_velocity_from_potential_gradient)}"),
        "radial_flux": (4.5, 0.0, f"radial flux\\\\{_eq(turbulent_particle_flux_from_fluctuations)}"),
    }
    boxes = {k: box(x, y, w, h, tx if labels else "", latex=True, role=k) for k, (x, y, tx) in nodes.items()}
    items: List = [it for b in boxes.values() for it in b.items]
    chain = ("gradient", "drift", "drive", "growth", "fluctuations", "exb_velocity", "radial_flux")
    items += [connector(boxes[a], boxes[b], role="cycle") for a, b in zip(chain, chain[1:])]
    # fast saturation acts on the fluctuations at fixed gradients: a leader to that step
    items.append(Polyline.of([(15.55, 1.6), (13.6, 1.6)], "leader line", role="saturation"))
    # the slow feedback: the flux relaxes the profile that drives it
    items.append(connector(boxes["radial_flux"], boxes["gradient"], style="connector feedback", role="relaxation"))
    if labels:
        items += [Label((4.9, 1.6), "profile relaxation\\\\(slow, transport time scale)", "small label,align=left",
                        anchor="west", role="relaxation"),
                  Label((15.6, 1.6), "nonlinear saturation\\\\at fixed gradients\\\\"
                        "(fast; mode coupling,\\\\zonal flows)",
                        "small label,align=left", anchor="west", role="saturation"),
                  Label((6.75, -1.3), "Slab, $\\mathbf B = B\\hat z$. A gradient is a possible free-energy source, "
                        "not a guarantee of growth; $\\omega_*$ is a frequency scale, not $\\gamma$", "note",
                        anchor="north", role="note")]
    return Diagram("drift_wave_transport_cycle", Scene(tuple(items)),
                   model={"cycle": list(chain), "feedback": ("radial_flux", "gradient"),
                          "feedback_time_scale": "transport (slow)", "saturation": "nonlinear, at fixed gradients",
                          "equations": {"drift": formula_equation(diamagnetic_frequency_from_k_v_star),
                                        "exb_velocity": formula_equation(exb_velocity_from_potential_gradient),
                                        "radial_flux": formula_equation(turbulent_particle_flux_from_fluctuations)}})


#: the three representative cross-phases, theta_nphi = alpha_phi - alpha_n [rad]
CROSS_PHASES = (("zero", 0.0), ("plus", 0.5 * math.pi), ("minus", -0.5 * math.pi))


def cross_phase_transport(*, labels: bool = True) -> Diagram:
    r"""Density and potential fluctuations at three cross-phases, and the signed flux each carries.

    One Fourier mode over a wavelength, $\tilde n \propto \cos k_yy$ and
    $\tilde\phi \propto \cos(k_yy + \theta_{n\phi})$, for
    $\theta_{n\phi} = 0, +\pi/2, -\pi/2$. Under each, the period-averaged flux
    of ``coherent_particle_flux_from_amplitudes_phase``,
    $\Gamma_k = k_y|\tilde n||\tilde\phi|\sin\theta_{n\phi}/(2B)$: zero in
    phase, outward ($+x$) at $+\pi/2$, inward at $-\pi/2$. Convention:
    $x$ radial, $y$ binormal, $\mathbf B = B\hat z$, real cosine fields,
    $\theta_{n\phi} = \alpha_\phi - \alpha_n$, and $\mathbf B$ into the page with
    $x$ up and $y$ to the right. One Fourier component: the turbulent flux sums
    many, and a zero particle flux does not imply a zero heat flux. The
    model checks each flux against a direct average of
    $\tilde n\,\tilde v_x$ with $\tilde v_x$ from
    ``exb_velocity_from_potential_gradient``.
    """
    labels = _check_labels(labels)
    k_y, b = 2.0 * math.pi, 1.0
    y = np.linspace(0.0, 1.0, 241)
    width, height, gap = 4.2, 1.4, 1.2
    items: List = []
    fluxes = {}
    peak = coherent_particle_flux_from_amplitudes_phase(1.0, 1.0, k_y, b, 0.5 * math.pi)
    for i, (key, theta) in enumerate(CROSS_PHASES):
        x0 = i * (width + gap)
        n = np.cos(k_y * y)
        phi = np.cos(k_y * y + theta)
        grad = np.zeros((y.size, 3))
        grad[:, 1] = -k_y * np.sin(k_y * y + theta)
        v_x = exb_velocity_from_potential_gradient(grad, [0.0, 0.0, b])[:, 0]
        direct = float(turbulent_particle_flux_from_fluctuations(n[:-1], v_x[:-1]))
        gamma = float(coherent_particle_flux_from_amplitudes_phase(1.0, 1.0, k_y, b, theta))
        if abs(direct - gamma) > 1e-9 * peak:
            raise AssertionError(f"cross-phase flux {gamma} disagrees with the direct average {direct}")
        fluxes[key] = gamma
        items += [Polyline.of(np.stack([x0 + width * y, 0.45 * height * n], -1), "boundary",
                              role="fluctuation_density"),
                  Polyline.of(np.stack([x0 + width * y, 0.45 * height * phi], -1), "approx",
                              role="fluctuation_potential"),
                  Polyline.of([(x0, 0.0), (x0 + width, 0.0)], "connector line", role="axes")]
        cx, ay = x0 + 0.5 * width, -1.6
        length = 0.9 * gamma / peak
        if abs(length) < 1e-6:
            items.append(Marker((cx, ay), "o", "opoint", role="radial_flux"))
        else:  # the flux is along x (radial), drawn up the page
            items.append(Arrow((cx, ay - 0.5 * length), (cx, ay + 0.5 * length), "exb", role="radial_flux"))
        if labels:
            phase = {"zero": "0", "plus": "+\\pi/2", "minus": "-\\pi/2"}[key]
            sign = {"zero": "$\\Gamma_k = 0$", "plus": "$\\Gamma_k > 0$ (outward)",
                    "minus": "$\\Gamma_k < 0$ (inward)"}[key]
            items += [Label((cx, 0.5 * height + 0.25), f"$\\theta_{{n\\phi}} = {phase}$", "small label",
                            anchor="south", role="phase"),
                      Label((cx + 0.3, ay), sign, "small label", anchor="west", role="radial_flux")]
    total = 3 * width + 2 * gap
    items += [Arrow((-1.4, -2.1), (-0.6, -2.1), "chart axis", role="axes"),
              Arrow((-1.4, -2.1), (-1.4, -1.3), "chart axis", role="axes")]
    if labels:
        items += [Label((-0.55, -2.1), "$y$", "small label", anchor="west", role="axes"),
                  Label((-1.4, -1.25), "$x$ (outward)", "small label", anchor="south", role="axes"),
                  Label((-0.3, 0.0), "amplitude\\\\against $y$", "small label,align=right", anchor="east",
                        role="axes"),
                  Label((total + 0.2, 0.3), "$\\tilde n$", "small label", anchor="west", role="fluctuation_density"),
                  Label((total + 0.2, -0.3), "$\\tilde\\phi$ (dashed)", "small label", anchor="west",
                        role="fluctuation_potential"),
                  Label((0.5 * total, -2.55), _eq(coherent_particle_flux_from_amplitudes_phase), "small label",
                        anchor="north", role="equation"),
                  Label((0.5 * total, -3.55), "Slab: $x$ radial (outward, up), $y$ binormal (right), "
                        "$\\mathbf B = B\\hat z$ into the page; real cosine fields, "
                        "$\\theta_{n\\phi} = \\alpha_\\phi - \\alpha_n$, $k_y/B > 0$. One Fourier component; "
                        "zero particle flux does not mean zero heat flux.\\\\Changing every convention consistently "
                        "changes the intermediate signs, not the physical flux.", "note,align=center", anchor="north",
                        role="note")]
    return Diagram("cross_phase_transport", Scene(tuple(items)),
                   model={"cross_phases": dict(CROSS_PHASES), "fluxes": fluxes, "k_y": k_y, "B": b,
                          "convention": "x radial outward (up), y binormal (right), B along +z into the page, "
                                        "real cos fields, "
                                        "theta = alpha_phi - alpha_n"})


def mixing_length_saturation(*, labels: bool = True) -> Diagram:
    r"""Phenomenological mixing-length scaling: growth balanced by an estimated decorrelation rate.

    Left: against the velocity amplitude $\tilde v$ of an eddy of wavenumber
    $k_\perp$, the linear growth rate $\gamma$ (flat) and the nonlinear
    decorrelation rate $\tau_\mathrm{nl}^{-1} \sim k_\perp\tilde v$ (rising).
    They cross at $\tilde v_\mathrm{sat} = \gamma/k_\perp$, and a step
    $1/k_\perp$ per $\tau_\mathrm{nl}$ gives
    ``mixing_length_diffusivity_from_gamma_k_perp``,
    $D_\mathrm{ML} \sim \gamma/k_\perp^2$. Right: the same argument as a
    chain. A dimensional scaling estimate, not a predictive saturation rule:
    one mode's $E\times B$ velocity does not advect its own potential
    ($\tilde{\mathbf v}_E\cdot\nabla\tilde\phi = 0$), so saturation needs
    coupling between modes, and is set by zonal flows, stable-mode coupling,
    spectral transfer and nonlinear phases. Quasilinear transport models
    keep the linear phases but need a separate saturation rule (Staebler,
    Bourdelle, Citrin and Waltz, Nucl. Fusion 64 (2024) 103001), and
    $D_\mathrm{ML}$ is not the particle diffusivity or a heat diffusivity of
    any one channel.
    """
    labels = _check_labels(labels)
    gamma, k_perp = 1.0, 1.0
    v_sat = gamma / k_perp
    d_ml = mixing_length_diffusivity_from_gamma_k_perp(gamma, k_perp)
    chart = Chart(x_range=(0.0, 2.0), y_range=(0.0, 2.0))
    v = np.linspace(0.0, 2.0, 101)
    chart.curves.update({"growth": np.stack([v, np.full_like(v, gamma)], -1),
                         "decorrelation": np.stack([v, k_perp * v], -1)})
    chart.points["saturation"] = (v_sat, gamma)
    chart.labels.update({"growth": (0.45, 1.2), "decorrelation": (1.45, 1.75), "grows": (0.35, 0.72),
                         "decays": (1.55, 0.35)})
    scene = render_chart(chart, x_label="eddy amplitude $\\tilde v$", y_label="rate",
                         curve_styles={"growth": "boundary", "decorrelation": "approx"},
                         region_text={"growth": "$\\gamma$", "decorrelation": "$k_\\perp\\tilde v$",
                                      "grows": "\\small grows", "decays": "\\small decorrelates"}
                         if labels else {},
                         x_ticks=[v_sat], x_tick_text=["$\\gamma/k_\\perp$"], y_ticks=[gamma],
                         y_tick_text=["$\\gamma$"])
    scale = 0.85
    items: List = list(scene.transformed(scale=scale).items)
    items.append(Marker(tuple(float(c) for c in chart.to_cm(np.array(chart.points["saturation"])) * scale), "o",
                        "opoint", role="saturation"))
    x = CHART_WIDTH * scale + 4.4
    steps = (("growth", "linear growth $\\gamma$"),
             ("amplitude", "eddy amplitude grows"),
             ("decorrelation", "decorrelation $\\tau_\\mathrm{nl}^{-1} \\sim k_\\perp\\tilde v$"),
             ("balance", "saturation $\\gamma \\sim \\tau_\\mathrm{nl}^{-1}$"),
             ("diffusivity", "phenomenological scaling " + _eq(mixing_length_diffusivity_from_gamma_k_perp)))
    chain = [box(x, CHART_HEIGHT * scale + 0.6 - 1.45 * i, 6.8, 1.2 if k == "diffusivity" else 0.8,
                 t if labels else "", latex=True, role=k)
             for i, (k, t) in enumerate(steps)]
    for b_ in chain:
        items += list(b_.items)
    items += [connector(a, b_, role="saturation_chain") for a, b_ in zip(chain, chain[1:])]
    if labels:
        items.append(Label((0.5 * (CHART_WIDTH * scale + x + 2.6), -1.55),
                           "Not a predictive saturation rule: one mode's "
                           "$\\tilde{\\mathbf v}_E\\cdot\\nabla\\tilde\\phi = 0$, "
                           "so saturation needs coupling between modes (zonal flows, stable modes, spectral "
                           "transfer).\\\\Quasilinear models need a separate saturation rule (Staebler, Bourdelle, "
                           "Citrin and Waltz, Nucl. Fusion 64 (2024) 103001); $D_\\mathrm{ML}$ is not $D_n$ or "
                           "$\\chi$ of any channel", "note,align=center", anchor="north", role="note"))
    return Diagram("mixing_length_saturation", Scene(tuple(items)),
                   model={"gamma": gamma, "k_perp": k_perp, "saturated_amplitude": v_sat, "diffusivity": d_ml,
                          "steps": [k for k, _ in steps]})


def _eddy(center, radius, shear, levels=(0.35, 0.7, 1.0)):
    """Closed contours of a round eddy after the shear map $y \\to y + s\\,x$ about its centre."""
    t = np.linspace(0.0, 2.0 * np.pi, 121)
    out = []
    for level in levels:
        x = level * radius * np.cos(t)
        y = level * radius * np.sin(t) + shear * x
        out.append(np.stack([center[0] + x, center[1] + y], -1))
    return out


def exb_shear_decorrelation(*, shear: float = 1.6, labels: bool = True) -> Diagram:
    r"""An eddy without and with a sheared $E\times B$ flow: tilted, stretched, shorter along a radial cut.

    Left: a round eddy in a uniform flow. Right: the same eddy after the
    differential $E\times B$ motion $v_y = S x$ has acted for a time $t$
    (``shear`` $= St$), the map $y \to y + St\,x$: it is tilted and stretched,
    and along a radial cut through its centre (at fixed $y$) it spans
    $2a/\sqrt{1 + (St)^2}$ instead of $2a$, although its total radial extent is
    unchanged. $\gamma_E = |S|$ is ``exb_shearing_rate_from_velocity_gradient``;
    the figure's ``shear`` is the dimensionless strain $St$. Geometry
    only: real $E_r$ and shearing-rate profiles are plots. The fixed-$y$ cut
    is not the statistical radial correlation length; the decorrelation is a
    separate, following step, drawn as such: the shear advects the
    wavenumber, $k_x(t) = k_x(0) - Sk_yt$, changing $k_\perp$ and the mode
    coupling (Biglari, Diamond and Terry, Phys. Fluids B 2 (1990) 1).
    $\gamma_E$ above $\gamma_\mathrm{lin}$ is a rule of thumb, not a universal
    threshold.
    """
    labels = _check_labels(labels)
    if not shear > 0.0:
        raise ValueError(f"shear must be positive, not {shear!r}")
    a = 1.0
    strain = exb_shearing_rate_from_velocity_gradient(shear)  # |S| t for the drawn strain S t
    items: List = []
    cut = {}
    for i, (key, s) in enumerate((("unsheared", 0.0), ("sheared", shear))):
        cx = 1.6 + i * 6.2
        for contour in _eddy((cx, 0.0), a, s):
            items.append(Polyline.of(contour, "surface", role="eddy" if s == 0 else "sheared_eddy", closed=True))
        half = a / math.sqrt(1.0 + s * s)
        cut[key] = 2.0 * half
        items.append(Arrow((cx - half, 0.0), (cx + half, 0.0), "width arrow", role="radial_cut",
                           both=True))
        # the flow profile: arrows along y whose length grows with x (sheared) or not
        for xx in np.linspace(-1.2, 1.2, 5):
            length = 0.5 if s == 0 else 0.55 * s * xx / 1.2
            if abs(length) > 1e-9:
                items.append(Arrow((cx + xx, -2.2 - 0.5 * length), (cx + xx, -2.2 + 0.5 * length), "exb",
                                   role="exb_velocity" if s == 0 else "shear"))
        if labels:
            title = "uniform flow" if s == 0 else f"sheared flow, $St = {s:g}$ (eddy frame)"
            items += [Label((cx, 1.9), title, "small label", anchor="south", role="title"),
                      Label((cx + (1.05 if s else 0.0), -0.95 if s else -1.15),
                            f"$L_\\mathrm{{cut}} = {cut[key]:.2g}\\,a$ at fixed $y$", "small label",
                            anchor="west",
                            role="radial_cut")]
    items.append(Arrow((4.2, 0.0), (5.2, 0.0), "connector", role="shear_map"))
    items.append(Arrow((-0.2, -3.0), (0.6, -3.0), "chart axis", role="axes"))
    # the statistical step is separate from the geometry: the shear advects the wavenumber
    step_text = ("decorrelation (statistical):\\\\$k_x(t) = k_x(0) - Sk_yt$\\\\"
                 "changes $k_\\perp$ and mode coupling")
    step = box(15.8, 0.0, 4.4, 2.4, step_text if labels else "", latex=True, role="decorrelation_step")
    items += [*step.items, Arrow((10.3, 0.0), (13.45, 0.0), "connector", role="decorrelation_step")]
    if labels:
        items += [Label((4.7, 0.2), "\\small $v_y = Sx$", "small label", anchor="south", role="shear_map"),
                  Label((-0.4, -2.2), "$v_y(x)$", "small label", anchor="east", role="flow_profile"),
                  Label((0.65, -3.0), "$x$ (radial)", "small label", anchor="west", role="axes"),
                  Label((6.8, -3.3), f"{_eq(exb_shearing_rate_from_velocity_gradient)} $= |S|$. The fixed-$y$ cut "
                        "$2a/\\sqrt{1 + (St)^2}$ is geometry, not the statistical correlation length;\\\\"
                        "$\\gamma_E$ above $\\gamma_\\mathrm{lin}$ is a rule of thumb, not a universal threshold "
                        "(Biglari, Diamond and Terry, Phys. Fluids B 2 (1990) 1)", "note,align=center", anchor="north",
                        role="note")]
    return Diagram("exb_shear_decorrelation", Scene(tuple(items)),
                   model={"shear": shear, "strain": strain, "radial_cut": cut, "eddy_radius": a})


def zonal_flow_feedback(*, labels: bool = True) -> Diagram:
    r"""Turbulence drives zonal flows through the Reynolds stress, and their shear acts back on it.

    The loop: a profile gradient drives turbulence; its tilted eddies carry
    the Reynolds stress $R_{r\theta} = \langle\tilde v_r\tilde v_\theta\rangle$
    (``reynolds_stress_from_velocity_fluctuations``); its radial gradient
    accelerates the zonal flow,
    $\partial_tV_\mathrm{ZF} = -\partial_rR_{r\theta} - \mu V_\mathrm{ZF}$
    (``zonal_flow_drive_from_reynolds_stress_gradient``, damping $\mu$ a
    stated choice) -- a stress alone drives nothing, its divergence
    $\partial_rR_{r\theta} \neq 0$ does; the zonal flow (the $n = 0$, $m = 0$,
    radially varying potential) and its $E\times B$ shear then change the
    turbulence through mode coupling and decorrelation -- the bar, which
    often but not always suppresses it. Local, electrostatic, slab with
    constant density: a tokamak adds curvature and metric factors, GAMs,
    collisional damping and Maxwell stress. This is the fast flow feedback;
    the slow profile feedback is ``drift_wave_transport_cycle``'s
    relaxation. No coefficient model is assumed (Diamond, Itoh, Itoh and
    Hahm, Plasma Phys. Control. Fusion 47 (2005) R35).
    """
    labels = _check_labels(labels)
    w, h = 4.4, 1.3
    nodes = {
        "gradient": (0.0, 3.0, "profile gradient\\\\free energy"),
        "turbulence": (5.4, 3.0, "turbulence\\\\$\\tilde n$, $\\tilde\\phi$, tilted eddies"),
        "reynolds_stress": (10.8, 3.0, f"Reynolds stress\\\\{_eq(reynolds_stress_from_velocity_fluctuations)}"),
        "zonal_flow": (10.8, -0.3, "zonal flow\\\\$" + formula_equation(zonal_flow_drive_from_reynolds_stress_gradient)
                       + "$"),
        "shear": (4.6, 0.0, "$E\\times B$ shear $\\gamma_E$"),
    }
    boxes = {k: box(x, y, w + (1.4 if k == "zonal_flow" else 0.0), h + (0.9 if k == "zonal_flow" else 0.0),
                    t if labels else "", latex=True, role=k) for k, (x, y, t) in nodes.items()}
    items: List = [it for b in boxes.values() for it in b.items]
    chain = ("gradient", "turbulence", "reynolds_stress", "zonal_flow", "shear")
    items += [connector(boxes[a], boxes[b], role="feedback_loop") for a, b in zip(chain, chain[1:])]
    # inhibition: shear decorrelates the turbulence (a bar instead of a head)
    start = boxes["shear"].boundary_point(boxes["turbulence"].center)
    end = boxes["turbulence"].boundary_point(boxes["shear"].center)
    d = np.subtract(end, start) / np.hypot(*np.subtract(end, start))
    s, e = np.add(start, 0.08 * d), np.subtract(end, 0.14 * d)
    bar = 0.3 * np.array([-d[1], d[0]])  # perpendicular to the line: an inhibition bar
    items += [Polyline.of([s, e], "connector line", role="suppression"),
              Polyline.of([e - bar, e + bar], "connector line", role="suppression")]
    if labels:
        items += [Label((4.7, 1.5), "mode coupling, decorrelation\\\\(often suppresses; mode-dependent)",
                        "small label,align=right", anchor="east", role="suppression"),
                  Label((11.0, 1.4), "drives only if $\\partial_r R_{r\\theta} \\neq 0$", "small label", anchor="west",
                        role="stress_divergence"),
                  Label((13.9, -0.3), "damping $\\mu$\\\\(model choice)", "small label,align=left", anchor="west",
                        role="damping"),
                  Label((6.5, -1.9), "Fast flow feedback, no coefficient model; slab, constant density. A tokamak adds "
                        "curvature, GAMs, collisional damping and Maxwell stress", "note", anchor="north",
                        role="note")]
    return Diagram("zonal_flow_feedback", Scene(tuple(items)),
                   model={"loop": list(chain), "suppression": ("shear", "turbulence"),
                          "equations": {"reynolds_stress": formula_equation(reynolds_stress_from_velocity_fluctuations),
                                        "zonal_flow": formula_equation(
                                            zonal_flow_drive_from_reynolds_stress_gradient)}})


#: representative perpendicular-wavenumber bands, k_perp * rho for the stated rho (conventions, not thresholds)
TURBULENCE_BANDS = {
    "ion_scale": ("rho_i", (0.1, 1.0), "ITG / TEM\\\\$k_\\perp\\rho_i \\sim 0.1$--$1$"),
    "tem_extent": ("rho_s", (0.1, 5.0), "TEM can reach $k_\\perp\\rho_s \\sim 5$ in some regimes (example)"),
    "electron_scale": ("rho_e", (0.1, 1.0), "ETG\\\\$k_\\perp\\rho_e \\sim 0.1$--$1$"),
}


def turbulence_scale_hierarchy(*, labels: bool = True) -> Diagram:
    r"""Ion- and electron-scale turbulence against the equilibrium lengths of one illustrative state.

    A logarithmic length axis with $\rho_e$, $\rho_i$, $\rho_s$, an
    illustrative gradient length $a/2$, $a$ and $R_0$ from
    ``vaft.diagram._orderings.lengths`` (the state of
    ``spatial_scale_orderings``). Above it, representative perpendicular
    scale bands $1/k_\perp$: ion-scale ITG/TEM at
    $k_\perp\rho_i \sim 0.1$--$1$, electron-scale ETG at
    $k_\perp\rho_e \sim 0.1$--$1$, and a dashed TEM extension towards
    $k_\perp\rho_s \sim 5$, because TEM is not confined to one scale. The
    bands are conventions of :data:`TURBULENCE_BANDS`, not stability
    thresholds; the separation $\rho_e \ll \rho_i \ll L$ is the state's, and
    every band stays below the minor radius. The state is hydrogen with
    $v_t = \sqrt{T/m}$ (deuterium makes $\rho_i$ and $\rho_s$ larger by
    $\sqrt2$); the length is $1/k_\perp$, not the wavelength $2\pi/k_\perp$; and
    the literature's $k_y\rho_s$ ranges are taken as $k_\perp$ with $k_x = 0$.
    TEM can overlap the electron scale, so the extension is one regime's
    example, not a bound.
    """
    from ._orderings import ILLUSTRATIVE_STATE, lengths

    labels = _check_labels(labels)
    values = {k: v[1] for k, v in lengths().items()}
    x0, decade = -6.0, 2.0   # log10 origin [m], cm per decade
    span = 7
    items: List = [Arrow((0.0, 0.0), (span * decade + 0.4, 0.0), "chart axis", role="axes")]
    for k in range(span + 1):
        x = k * decade
        items += [Polyline.of([(x, -0.1), (x, 0.1)], "connector line", role="ticks"),
                  Label((x, -0.2), f"$10^{{{int(x0) + k}}}$", "ticklabel", anchor="north", role="ticks")]

    def at(length):
        return (math.log10(length) - x0) * decade

    marks = (("rho_e", "$\\rho_e$"), ("rho_i", "$\\rho_i$"), ("rho_s", "$\\rho_s$"), ("gradient", "$L = a/2$"),
             ("minor", "$a$"), ("major", "$R_0$"))
    for i, (key, symbol) in enumerate(marks):
        x = at(values[key])
        items.append(Marker((x, 0.0), "o" if key.startswith("rho") else "x",
                            "opoint" if key.startswith("rho") else "xpoint", role=f"length:{key}"))
        if labels:
            items.append(Label((x, -0.95 - 0.45 * (i % 3)), symbol, "small label", anchor="north",
                               role=f"length:{key}"))
    bands = {}
    rows = {"ion_scale": 0.7, "tem_extent": 2.6, "electron_scale": 0.7}
    for key, (rho, (k_lo, k_hi), text) in TURBULENCE_BANDS.items():
        lo, hi = values[rho] / k_hi, values[rho] / k_lo   # perpendicular scale 1/k_perp
        if hi >= values["minor"]:
            raise AssertionError(f"the {key} band reaches the minor radius")
        bands[key] = (lo, hi)
        y = rows[key]
        style = "approx" if key == "tem_extent" else "boundary"
        items.append(Polyline.of([(at(lo), y - 0.25), (at(hi), y - 0.25), (at(hi), y + 0.25), (at(lo), y + 0.25)],
                                 style, role=key, closed=True))
        if labels:
            items.append(Label((0.5 * (at(lo) + at(hi)), y + 0.3), text, "small label,align=center", anchor="south",
                               role=key))
    eq_lo = values["gradient"]
    items.append(Polyline.of([(at(eq_lo), 3.6), (at(values["major"]), 3.6)], "boundary", role="equilibrium_scale"))
    if labels:
        s = ILLUSTRATIVE_STATE
        items += [Label((0.5 * (at(eq_lo) + at(values["major"])), 3.75), "equilibrium scales", "small label",
                        anchor="south", role="equilibrium_scale"),
                  Label((span * decade + 0.5, -0.2), "length [m]", "xlabel", anchor="north west", role="axes"),
                  Label((0.5 * span * decade, -2.3),
                        f"Lengths $1/k_\\perp$ (not the wavelength $2\\pi/k_\\perp$) of an illustrative hydrogen state "
                        f"($B = {s['B']:g}$ T, $T_e = {s['T_e']:g}$ eV, $T_i = {s['T_i']:g}$ eV; "
                        "$v_t = \\sqrt{T/m}$, deuterium gives $\\sqrt2$ larger $\\rho_i$, $\\rho_s$).\\\\"
                        "Bands are conventions, not thresholds, in $k_\\perp\\rho_i$ (ITG/TEM), "
                        "$k_\\perp\\rho_e$ (ETG) "
                        "and $k_\\perp\\rho_s$ (TEM extension, the literature's $k_y\\rho_s$ with $k_x = 0$); "
                        "that extension is one regime's example, not a bound.",
                        "note,align=center", anchor="north", role="note")]
    return Diagram("turbulence_scale_hierarchy", Scene(tuple(items)),
                   model={"lengths": values, "bands": bands,
                          "band_conventions": {k: (v[0], v[1]) for k, v in TURBULENCE_BANDS.items()}})
