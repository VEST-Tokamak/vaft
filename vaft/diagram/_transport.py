"""Canonical transport concepts: random walk, flux and gradient, reference scales (#1112).

``random_walk_diffusion``
    one correlated radial motion seen three ways -- sample trajectories, the
    mean-square displacement (ballistic, then diffusive) and the velocity
    autocorrelation -- and the single diffusivity both definitions give,
    from ``vaft.formula.transport``;
``transport_flux_gradient``
    gradients driving particle, heat and momentum fluxes, with the diffusive
    and convective parts kept apart, and why one flux at one gradient does
    not fix $D$ and $V$ separately;
``transport_reference_scales``
    classical, neoclassical, Bohm and gyro-Bohm sorted by what they are --
    collisional transport, an empirical reference, a turbulence normalisation
    -- not by size.

The banana / plateau / Pfirsch--Schlueter regimes are
``neoclassical_collisionality`` (#1111), which already states its
collisionality convention.
"""

from __future__ import annotations

from typing import List

import numpy as np

from vaft.formula.transport import (
    bohm_diffusivity_from_T_B,
    classical_diffusivity_from_nu_rho,
    conductive_heat_flux_from_n_chi_grad_T,
    diffusivity_from_mean_square_displacement,
    diffusivity_from_velocity_autocorrelation,
    gyrobohm_diffusivity_from_rho_s_c_s_L_ref,
    particle_flux_from_D_grad_n_V_n,
)

from ._chart import CHART_HEIGHT, CHART_WIDTH, Chart, render_chart
from ._concept import box, connector
from ._equations import formula_equation
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: seed of the sample trajectories, fixed so the reference SVG is deterministic
_SEED = 1112


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def _msd(t, variance, tau_c):
    """Mean-square displacement of an exponentially correlated velocity (Ornstein--Uhlenbeck)."""
    return 2.0 * variance * tau_c * (t - tau_c * (1.0 - np.exp(-t / tau_c)))


def random_walk_diffusion(*, labels: bool = True) -> Diagram:
    r"""From microscopic radial motion to one diffusivity: the Einstein and Taylor--Green--Kubo views.

    A radial velocity with variance $\langle v_r^2\rangle$ and an exponential
    autocorrelation of time $\tau_c$ (an Ornstein--Uhlenbeck process, in units
    $\langle v_r^2\rangle = \tau_c = 1$). Left: sample trajectories from a fixed
    seed inside the $\pm\sqrt{\langle(\Delta r)^2\rangle}$ envelope. Middle: the
    mean-square displacement, ballistic ($\propto t^2$) for $t \ll \tau_c$ and
    diffusive ($\propto 2Dt$) for $t \gg \tau_c$, so
    ``diffusivity_from_mean_square_displacement`` applies only to the late
    slope. Right: the autocorrelation, whose area is
    ``diffusivity_from_velocity_autocorrelation``. Both give
    $D = \langle v_r^2\rangle\tau_c$; a process with long memory or non-local
    steps has no single $D$.
    """
    labels = _check_labels(labels)
    variance, tau_c, t_end = 1.0, 1.0, 12.0
    lag = np.linspace(0.0, 30.0 * tau_c, 3001)
    d_tgk = diffusivity_from_velocity_autocorrelation(lag, variance * np.exp(-lag / tau_c))
    d_msd = diffusivity_from_mean_square_displacement(2.0 * variance * tau_c)  # the late (diffusive) slope

    # sample trajectories: exact OU update of the velocity, integrated position
    rng = np.random.default_rng(_SEED)
    dt = 0.02
    t = np.arange(0.0, t_end + dt / 2, dt)
    decay = np.exp(-dt / tau_c)
    v = rng.normal(0.0, np.sqrt(variance), 5)
    paths = np.zeros((t.size, 5))
    for k in range(1, t.size):
        v = decay * v + np.sqrt(variance * (1.0 - decay ** 2)) * rng.normal(size=5)
        paths[k] = paths[k - 1] + v * dt
    envelope = np.sqrt(_msd(t, variance, tau_c))
    scale = 0.5
    items: List = []

    a = Chart(x_range=(0.0, t_end), y_range=(-8.0, 8.0))
    a.curves.update({f"trajectory_{k}": np.stack([t, paths[:, k]], -1) for k in range(paths.shape[1])})
    a.curves.update({"envelope_upper": np.stack([t, envelope], -1), "envelope_lower": np.stack([t, -envelope], -1)})
    sa = render_chart(a, x_label="$t/\\tau_c$", y_label="$\\Delta r$",
                      curve_styles={**{f"trajectory_{k}": "passing" for k in range(paths.shape[1])},
                                    "envelope_upper": "approx", "envelope_lower": "approx"},
                      region_text={}, x_ticks=[0.0, 6.0, 12.0], x_tick_text=["0", "6", "12"])

    b = Chart(x_range=(0.0, t_end), y_range=(0.0, 24.0))
    late = t[t >= tau_c]
    b.curves.update({"msd": np.stack([t, _msd(t, variance, tau_c)], -1),
                     "diffusive": np.stack([late, 2.0 * d_msd * (late - tau_c)], -1),
                     "ballistic": np.stack([t[t <= 4.0], variance * t[t <= 4.0] ** 2], -1)})
    b.labels.update({"ballistic": (2.2, 15.0), "diffusive": (8.2, 7.0)})
    sb = render_chart(b, x_label="$t/\\tau_c$", y_label="$\\langle(\\Delta r)^2\\rangle$",
                      curve_styles={"msd": "boundary", "diffusive": "approx", "ballistic": "slope plus"},
                      region_text={"ballistic": "$\\propto t^2$", "diffusive": "$\\propto 2Dt$"} if labels else {},
                      x_ticks=[0.0, 6.0, 12.0], x_tick_text=["0", "6", "12"])

    c = Chart(x_range=(0.0, 5.0), y_range=(0.0, 1.2))
    tc = np.linspace(0.0, 5.0, 201)
    corr = variance * np.exp(-tc / tau_c)
    c.curves["autocorrelation"] = np.stack([tc, corr], -1)
    c.labels["area"] = (3.3, 0.55)
    sc = render_chart(c, x_label="$\\tau/\\tau_c$", y_label="$\\langle v_r(0)v_r(\\tau)\\rangle$",
                      curve_styles={"autocorrelation": "boundary"},
                      region_text={"area": "\\small shaded area $= D$"} if labels else {},
                      x_ticks=[0.0, 1.0, 5.0], x_tick_text=["0", "1", "5"])
    area = c.to_cm(np.concatenate([np.stack([tc, corr], -1), [[5.0, 0.0], [0.0, 0.0]]]))

    gap = CHART_WIDTH * scale + 2.2
    for k, s in enumerate((sa, sb, sc)):
        offset = (k * gap, 0.0)
        if k == 2:
            items.append(Polyline.of(area * scale + np.array(offset), "concept band", role="correlation_area",
                                     closed=True))
        items += list(s.transformed(scale=scale, offset=offset).items)
    h = CHART_HEIGHT * scale
    d_box = box(gap * 1.5 + CHART_WIDTH * scale / 2 - gap / 2, -2.6, 5.6, 1.0,
                f"$D = \\langle v_r^2\\rangle\\tau_c = {d_tgk:.3g}$" if labels else "", latex=True, role="diffusivity")
    items += list(d_box.items)
    from_b = (gap + CHART_WIDTH * scale / 2, -1.25)
    from_c = (2 * gap + CHART_WIDTH * scale / 2, -1.25)
    items += [Arrow(from_b, d_box.boundary_point(from_b), "connector", role="einstein"),
              Arrow(from_c, d_box.boundary_point(from_c), "connector", role="taylor_green_kubo"),
              Arrow((CHART_WIDTH * scale / 2 + 2.2, h + 0.55), (gap + CHART_WIDTH * scale / 2 - 2.5, h + 0.55),
                    "connector", role="ensemble_average")]
    if labels:
        eq = {"einstein": formula_equation(diffusivity_from_mean_square_displacement),
              "tgk": formula_equation(diffusivity_from_velocity_autocorrelation)}
        items += [Label((CHART_WIDTH * scale / 2, h + 0.55), "microscopic radial motion", "small label",
                        role="trajectories"),
                  Label((gap + CHART_WIDTH * scale / 2, h + 0.55), "mean-square displacement", "small label",
                        role="msd"),
                  Label((2 * gap + CHART_WIDTH * scale / 2, h + 0.55), "velocity autocorrelation", "small label",
                        role="autocorrelation"),
                  Label((from_b[0] - 0.2, -1.6), f"$\\displaystyle {eq['einstein']}$, $d = 1$, late slope",
                        "small label", anchor="east", role="einstein"),
                  Label((from_c[0] + 0.2, -1.6), f"$\\displaystyle {eq['tgk']}$", "small label", anchor="west",
                        role="taylor_green_kubo"),
                  Label((gap * 1.5 + CHART_WIDTH * scale / 2 - gap / 2, -3.55),
                        "Diffusive only for $t \\gg \\tau_c$: a process with long memory or non-local steps "
                        "has no single $D$", "note", anchor="north", role="note")]
    return Diagram("random_walk_diffusion", Scene(tuple(items)),
                   model={"variance": variance, "correlation_time": tau_c, "seed": _SEED,
                          "diffusivity_einstein": d_msd, "diffusivity_taylor_green_kubo": d_tgk})


#: the three channels: (key, gradient, flux, diffusive term, convective term), LaTeX
_CHANNELS = (
    ("particle", "$\\partial_r n$", "particle flux $\\Gamma$", "diffusion $-D\\,\\partial_r n$",
     "convection $nV$"),
    ("heat", "$\\partial_r T$", "heat flux $q$", "conduction $-n\\chi\\,\\partial_r T$",
     "heat pinch $nT\\,V_T$"),
    ("momentum", "$\\partial_r v_\\phi$", "momentum flux $\\Pi_\\phi$",
     "viscosity $-mn\\chi_\\phi\\,\\partial_r v_\\phi$", "pinch $mn v_\\phi V_\\phi$ + residual stress"),
)


def transport_flux_gradient(*, labels: bool = True) -> Diagram:
    r"""Gradients drive fluxes; the diffusive and convective parts are different terms, and a flux alone does not separate them.

    Left: density, temperature and toroidal-rotation gradients each drive a
    flux through a diffusive (or conductive, viscous) term proportional to
    the gradient and a convective (pinch) term that is not; the momentum
    flux also carries a residual stress independent of both. The particle
    and heat definitions are ``particle_flux_from_D_grad_n_V_n`` and
    ``conductive_heat_flux_from_n_chi_grad_T``; the momentum row is per unit
    volume in the toroidal velocity $v_\phi$, not angular momentum. Right:
    one measured $\Gamma$ at one $\partial_r n$ is the line
    $V = (\Gamma + D\,\partial_r n)/n$ in the $(D, V)$ plane -- every point on
    it reproduces the flux (checked here against the formula), so
    separating $D$ from $V$ needs transient or modulated data.
    """
    labels = _check_labels(labels)
    items: List = []
    row_y = {"particle": 4.4, "heat": 2.2, "momentum": 0.0}
    for key, grad, flux, diffusive, convective in _CHANNELS:
        y = row_y[key]
        g = box(0.0, y, 2.0, 0.9, grad if labels else "", latex=True, role="gradient")
        dif = box(3.6, y + 0.45, 4.2, 0.75, diffusive if labels else "", latex=True, role="diffusive_term")
        con = box(3.6, y - 0.45, 4.2, 0.75, convective if labels else "", latex=True, role="convective_term")
        f = box(7.6, y, 2.6, 0.9, flux if labels else "", latex=True, role="radial_flux")
        items += [*g.items, *dif.items, *con.items, *f.items, connector(g, dif, role="drives"),
                  connector(dif, f, role="sums"), connector(con, f, role="sums")]
        if labels:
            items.append(Label((-1.15, y), key, "small label", anchor="east", role="channel"))

    # the inverse problem: one flux, one gradient, a line of (D, V)
    n, grad_n, gamma = 1.0e19, -2.0e19, 1.0e19
    d = np.linspace(0.0, 2.0, 41)
    v = (gamma + d * grad_n) / n
    reproduced = np.asarray(particle_flux_from_D_grad_n_V_n(d, grad_n, n, v))
    if not np.allclose(reproduced, gamma, rtol=1e-12):
        raise AssertionError("the (D, V) line does not reproduce the flux")
    chart = Chart(x_range=(0.0, 2.0), y_range=(-4.0, 1.5))
    chart.curves["degenerate_line"] = np.stack([d, v], -1)
    chart.labels.update({"line": (1.0, -3.35)})
    s = render_chart(chart, x_label="$D$ [m$^2$/s]", y_label="$V$ [m/s]",
                     curve_styles={"degenerate_line": "boundary"},
                     region_text={"line": "\\small one $\\Gamma$: a line of $(D, V)$"} if labels else {},
                     x_ticks=[0.0, 1.0, 2.0], x_tick_text=["0", "1", "2"], y_ticks=[-4.0, -2.0, 0.0],
                     y_tick_text=["$-4$", "$-2$", "0"])
    offset = (11.4, 0.2)
    items += list(s.transformed(scale=0.5, offset=offset).items)
    for dv in ((0.3, (gamma + 0.3 * grad_n) / n), (1.5, (gamma + 1.5 * grad_n) / n)):
        at = chart.to_cm(np.array(dv)) * 0.5 + np.array(offset)
        items.append(Marker(tuple(float(c) for c in at), "o", "opoint", role="candidate_pair"))
    if labels:
        items += [Label((11.4 + CHART_WIDTH * 0.25, 4.6),
                        "measured $\\Gamma \\neq$ unique $(D, V)$", "small label", role="inverse_problem"),
                  Label((11.4 + CHART_WIDTH * 0.25, 4.05),
                        "$\\Gamma = 10^{19}$, $\\partial_r n = -2\\times 10^{19}$, $n = 10^{19}$ (SI)",
                        "note", role="inverse_problem"),
                  Label((5.5, -1.1), "Forward definitions; signs positive outward, gradients along $r$ in metres",
                        "note", anchor="north", role="note")]
    return Diagram("transport_flux_gradient", Scene(tuple(items)),
                   model={"channels": [c[0] for c in _CHANNELS], "inverse_example": {"n": n, "grad_n": grad_n,
                                                                                    "flux": gamma},
                          "particle_flux": formula_equation(particle_flux_from_D_grad_n_V_n),
                          "heat_flux": formula_equation(conductive_heat_flux_from_n_chi_grad_T)})


def transport_reference_scales(*, labels: bool = True) -> Diagram:
    r"""Transport reference scales sorted by role: collisional, empirical, turbulence normalisation.

    A taxonomy, not a ladder: classical $\nu\rho^2$ and neoclassical
    transport are collisional, Bohm $C_BT/(|q|B)$ is an empirical reference,
    and gyro-Bohm $\rho_s^2c_s/L_\mathrm{ref}$ is the unit gyrokinetic fluxes
    are normalised by. Their sizes are not universally ordered -- which is
    largest depends on collisionality, $\rho_*$ and the regime -- so the
    figure draws no magnitude axis. The equations are the
    ``vaft.formula.transport`` definitions.
    """
    labels = _check_labels(labels)
    root = box(6.0, 5.2, 4.6, 0.85, "Transport reference scales" if labels else "", role="root")
    groups = {
        "collisional": box(1.8, 3.5, 3.4, 0.8, "collisional" if labels else "", role="group_collisional"),
        "empirical": box(6.0, 3.5, 3.4, 0.8, "empirical reference" if labels else "", role="group_empirical"),
        "turbulence": box(10.6, 3.5, 4.8, 0.8, "turbulence normalisation" if labels else "",
                          role="group_turbulence"),
    }
    eq = {name: formula_equation(fn) for name, fn in (
        ("classical", classical_diffusivity_from_nu_rho), ("bohm", bohm_diffusivity_from_T_B),
        ("gyrobohm", gyrobohm_diffusivity_from_rho_s_c_s_L_ref))}
    leaves = {
        "classical": (box(0.0, 1.4, 3.4, 1.3, f"classical\\\\${eq['classical']}$" if labels else "", latex=True,
                          role="classical"), "collisional"),
        "neoclassical": (box(3.6, 1.4, 3.4, 1.3, "neoclassical\\\\{\\small banana, plateau, P--S}" if labels else "",
                             latex=True, role="neoclassical"), "collisional"),
        "bohm": (box(7.2, 1.4, 3.0, 1.3, f"Bohm\\\\${eq['bohm']}$" if labels else "", latex=True, role="bohm"),
                 "empirical"),
        "gyrobohm": (box(10.6, 1.4, 3.2, 1.3, f"gyro-Bohm\\\\${eq['gyrobohm']}$" if labels else "", latex=True,
                         role="gyrobohm"), "turbulence"),
    }
    items: List = list(root.items)
    for name, g in groups.items():
        items += [*g.items, connector(root, g, role="taxonomy")]
    for name, (leaf, parent) in leaves.items():
        items += [*leaf.items, connector(groups[parent], leaf, role="taxonomy")]
    if labels:
        items += [Label((1.8, 0.45), "collisions decorrelate the orbit", "note", anchor="north", role="classical"),
                  Label((10.6, 0.45), "$\\chi_\\mathrm{gB} = \\rho_*\\,\\rho_s c_s$ with "
                        "$\\rho_* = \\rho_s/L_\\mathrm{ref}$", "note", anchor="north", role="gyrobohm"),
                  Label((5.9, -0.3), "Roles, not a ranking: no universal ordering of their magnitudes",
                        "note", anchor="north", role="note")]
    return Diagram("transport_reference_scales", Scene(tuple(items)),
                   model={"groups": {"collisional": ["classical", "neoclassical"], "empirical": ["bohm"],
                                     "turbulence": ["gyrobohm"]},
                          "equations": eq, "magnitude_axis": None})
