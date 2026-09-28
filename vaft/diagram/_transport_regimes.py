"""Collisionality regimes: axisymmetric neoclassical and non-axisymmetric NTV (#1111).

``neoclassical_collisionality``
    banana, plateau and Pfirsch--Schlueter along $\\hat\\nu = qR\\nu/v$, with the
    boundaries from ``vaft.formula.neoclassical.neoclassical_regime_boundaries``;
``ntv_collisionality``
    Shaing's NTV branches (1/nu, nu-sqrt(nu), superbanana-plateau, nu), non-resonant and
    resonant drawn apart -- a different axis and a different physics from the first;
``ntv_precession_regimes``
    collisionality against $E\\times B$ precession: the superbanana-plateau line
    $\\omega_d = 0$ from ``vaft.formula.ntv.ntv_precession_frequency``.

The charts are logarithmic: the model stores log10 values, and the tick
labels say so. Asymptotic orderings, not phase boundaries -- real transport
coefficients cross over smoothly, and every diagram says it is schematic
where its lines are orderings.
"""

from __future__ import annotations

import numpy as np

from vaft.formula.neoclassical import neoclassical_regime_boundaries
from vaft.formula.ntv import ntv_precession_frequency

from ._chart import Chart, render_chart
from ._render import Diagram


def _decades(lo: int, hi: int):
    ticks = [float(k) for k in range(lo, hi + 1)]
    return ticks, [f"$10^{{{k}}}$" for k in range(lo, hi + 1)]


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def neoclassical_collisionality(*, epsilon: float = 0.1, labels: bool = True) -> Diagram:
    r"""Banana, plateau and Pfirsch--Schl\"uter: the axisymmetric neoclassical regimes.

    The horizontal axis is $\hat\nu = qR_0\nu/v$ (``collisions_per_transit``),
    not a $\nu_*$ convention; the boundaries $\hat\nu = \epsilon^{3/2}$ and
    $\hat\nu = 1$ come from ``neoclassical_regime_boundaries``. The vertical
    axis is the diffusivity in units of the plateau value, with the three
    textbook asymptotes $\hat\nu/\epsilon^{3/2}$, $1$ and $\hat\nu$ (order-unity
    coefficients set to one), which meet exactly at those boundaries. Lower
    $\epsilon$ widens the plateau.
    """
    labels = _check_labels(labels)
    if not 0.0 < epsilon < 1.0:
        raise ValueError(f"epsilon must lie in (0, 1), not {epsilon!r}")
    banana_plateau, plateau_ps = neoclassical_regime_boundaries(epsilon)
    xb, xp = float(np.log10(banana_plateau)), float(np.log10(plateau_ps))
    x = np.linspace(-4.0, 2.0, 301)
    nu = 10.0 ** x
    banana, plateau, ps = nu / banana_plateau, np.ones_like(nu), nu / plateau_ps
    total = np.minimum(banana, np.maximum(plateau, ps))
    chart = Chart(x_range=(-4.0, 2.0), y_range=(-2.5, 2.5))
    chart.curves.update({
        "diffusivity": np.stack([x, np.log10(total)], axis=-1),
        "banana_asymptote": np.stack([x, np.log10(banana)], axis=-1),
        "pfirsch_schlueter_asymptote": np.stack([x, np.log10(ps)], axis=-1),
        "banana_plateau": np.array([[xb, -2.5], [xb, 2.5]]),
        "plateau_pfirsch_schlueter": np.array([[xp, -2.5], [xp, 2.5]]),
    })
    chart.labels.update({"banana": ((-4.0 + xb) / 2, 1.6), "plateau": ((xb + xp) / 2, 0.4),
                         "pfirsch_schlueter": ((xp + 2.0) / 2, -1.6)})
    chart.parameters.update({"epsilon": epsilon, "banana_plateau": banana_plateau,
                             "plateau_pfirsch_schlueter": plateau_ps})
    x_ticks, x_text = _decades(-4, 2)
    y_ticks, y_text = _decades(-2, 2)
    scene = render_chart(
        chart,
        x_label="$\\hat\\nu = qR_0\\nu/v$",
        y_label="$D/D_\\mathrm{plateau}$",
        curve_styles={"banana_asymptote": "approx", "pfirsch_schlueter_asymptote": "approx",
                      "banana_plateau": "approx", "plateau_pfirsch_schlueter": "approx", "diffusivity": "boundary"},
        region_text={"banana": "banana", "plateau": "plateau",
                     "pfirsch_schlueter": "\\begin{tabular}{c}Pfirsch--\\\\Schl\\\"uter\\end{tabular}"}
        if labels else {},
        x_ticks=x_ticks, x_tick_text=x_text, y_ticks=y_ticks, y_tick_text=y_text,
        note=(f"Asymptotic orderings at $\\epsilon = {epsilon:g}$: boundaries $\\hat\\nu = \\epsilon^{{3/2}}$ "
              "and 1, coefficients set to one" if labels else ""),
    )
    return Diagram("neoclassical_collisionality", scene, model=chart)


#: Shaing's asymptotic flux exponents, $\Gamma \propto \nu^p$ in each NTV regime
NTV_EXPONENTS = {"1/nu": -1.0, "nu-sqrt(nu)": 0.5, "superbanana_plateau": 0.0, "nu": 1.0}


def ntv_collisionality(*, labels: bool = True) -> Diagram:
    r"""Shaing's NTV regimes: the non-resonant and resonant branches of the flux against collisionality.

    Schematic in both axes: only the slopes are physics -- the asymptotic
    exponents of ``NTV_EXPONENTS`` -- and the breakpoints are placed for
    legibility. Non-resonant ($|\omega_d| \gg$ the resonance width): $1/\nu$
    while $\nu_\mathrm{eff} > |\omega_d|$, then the $\nu$--$\sqrt\nu$ boundary
    layer, then $\nu$. Resonant ($\omega_d \to 0$): $1/\nu$ saturates into the
    $\nu$-independent superbanana plateau, then $\nu$. These regimes are not
    banana/plateau/Pfirsch--Schl\"uter: the ordering parameter is the
    precession $\omega_d$, not the transit or bounce frequency.
    """
    labels = _check_labels(labels)
    p = NTV_EXPONENTS
    x = np.linspace(-4.0, 2.0, 301)
    # non-resonant: 1/nu above nu_eff = |omega_d| (x = 0), sqrt(nu) down to x = -2, nu below
    nonres = np.where(x >= 0.0, p["1/nu"] * x,
                      np.where(x >= -2.0, p["nu-sqrt(nu)"] * x, p["nu-sqrt(nu)"] * -2.0 + p["nu"] * (x + 2.0)))
    # resonant: 1/nu down to x = -1, plateau to x = -3, nu below
    res = np.where(x >= -1.0, p["1/nu"] * x,
                   np.where(x >= -3.0, p["1/nu"] * -1.0 + p["superbanana_plateau"] * (x + 1.0),
                            1.0 + p["nu"] * (x + 3.0)))
    chart = Chart(x_range=(-4.0, 2.0), y_range=(-2.5, 2.5))
    chart.curves.update({"non_resonant": np.stack([x, nonres], axis=-1), "resonant": np.stack([x, res], axis=-1),
                         "precession_ordering": np.array([[0.0, -2.5], [0.0, 2.0]])})
    chart.labels.update({"ordering": (0.0, 2.3), "one_over_nu": (1.0, 0.2), "sqrt_nu": (-1.0, -1.05),
                         "sbp": (-2.0, 1.35),
                         "nu_nonres": (-3.1, -1.75), "nu_res": (-3.55, 0.05)})
    chart.parameters.update({f"exponent {k}": v for k, v in p.items()})
    scene = render_chart(
        chart,
        x_label="$\\nu_\\mathrm{eff}$ (log, schematic)",
        y_label="NTV flux (schematic)",
        curve_styles={"precession_ordering": "approx", "non_resonant": "boundary", "resonant": "boundary"},
        region_text={"ordering": "$\\nu_\\mathrm{eff} \\sim |\\omega_d|$", "one_over_nu": "$1/\\nu$", "sqrt_nu": "$\\nu$--$\\sqrt\\nu$",
                     "sbp": "superbanana plateau", "nu_nonres": "$\\nu$", "nu_res": "$\\nu$"} if labels else {},
        note=("Log--log, slopes only: resonant $\\omega_d \\to 0$ (upper), non-resonant (lower). "
              "Not banana/plateau/PS" if labels else ""),
    )
    return Diagram("ntv_collisionality", scene, model=chart)


def ntv_precession_regimes(*, omega_magnetic: float = 1.0, labels: bool = True) -> Diagram:
    r"""Collisionality alone does not fix the NTV regime: the precession plane.

    Axes $\nu_\mathrm{eff}/\omega_B$ (log) and $\omega_E/\omega_B$ for a
    magnetic precession $\omega_B$ = ``omega_magnetic``. The superbanana-plateau
    line is where ``ntv_precession_frequency`` vanishes; the V around it is
    the ordering $\nu_\mathrm{eff} = |\omega_d|$ from the same formula,
    separating $1/\nu$ (collisions faster than precession, inside) from
    $\nu$--$\sqrt\nu$ (outside). That ordering is schematic -- the
    coefficient is set to one.
    """
    labels = _check_labels(labels)
    if not (np.isfinite(omega_magnetic) and omega_magnetic != 0.0):
        raise ValueError(f"omega_magnetic must be finite and non-zero, not {omega_magnetic!r}")
    wb = float(omega_magnetic)
    # the resonance: omega_d = omega_E + omega_B is linear in omega_E; locate its zero from the formula
    slope = ntv_precession_frequency(1.0, 0.0)
    y_res = -ntv_precession_frequency(0.0, wb) / (slope * wb)
    y = np.linspace(y_res - 2.0, y_res + 2.0, 401)
    omega_d = ntv_precession_frequency(y * wb, wb)
    with np.errstate(divide="ignore"):
        x_order = np.log10(np.abs(omega_d / wb))
    upper, lower = y > y_res, y < y_res
    chart = Chart(x_range=(-3.0, 1.0), y_range=(y_res - 2.0, y_res + 2.0))
    chart.curves.update({
        "resonance": np.array([[-3.0, y_res], [1.0, y_res]]),
        "ordering_upper": np.stack([x_order[upper], y[upper]], axis=-1),
        "ordering_lower": np.stack([x_order[lower], y[lower]], axis=-1),
    })
    chart.labels.update({"one_over_nu": (0.45, y_res + 0.5), "nonres_upper": (-1.7, y_res + 1.4),
                         "nonres_lower": (-1.7, y_res - 1.3), "sbp": (-1.8, y_res + 0.62)})
    chart.parameters.update({"omega_magnetic": wb, "omega_exb_at_resonance": y_res * wb})
    x_ticks, x_text = _decades(-3, 1)
    y_ticks = [float(v) for v in np.arange(np.ceil(y_res - 2.0), np.floor(y_res + 2.0) + 1.0)]
    scene = render_chart(
        chart,
        x_label="$\\nu_\\mathrm{eff}/|\\omega_B|$",
        y_label="$\\omega_E/\\omega_B$",
        curve_styles={"ordering_upper": "approx", "ordering_lower": "approx", "resonance": "boundary"},
        region_text={"one_over_nu": "$1/\\nu$", "nonres_upper": "$\\nu$--$\\sqrt\\nu$",
                     "nonres_lower": "$\\nu$--$\\sqrt\\nu$",
                     "sbp": "\\begin{tabular}{c}superbanana plateau\\\\$\\omega_d = \\omega_E + \\omega_B = 0$\\end{tabular}"} if labels else {},
        x_ticks=x_ticks, x_tick_text=x_text, y_ticks=y_ticks,
        note=("Schematic: dashed is the ordering $\\nu_\\mathrm{eff} = |\\omega_d|$; "
              "$\\omega_E$ is the $E\\times B$ frequency, not $\\omega_\\phi$" if labels else ""),
    )
    return Diagram("ntv_precession_regimes", scene, model=chart)
