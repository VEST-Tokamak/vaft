"""Stability and operational-space diagrams: textbook 2-D projections.

Each diagram is a chart -- two axes, the boundaries that divide the plane,
and a label per region -- with the rule the island diagrams follow: a
boundary that physics defines is computed by :mod:`vaft.formula`, never
drawn by hand.

* ``s_alpha_ballooning``: first and second stability from the ballooning
  equation (``s_alpha_marginal_alpha``).
* ``hugill``: the Greenwald line from ``greenwald_density`` and
  ``q_cyl_from_B_R_epsilon_kappa_I``, and the low-q limit.
* ``troyon``: the beta limit from ``beta_N_from_beta_a_B0_Ip`` and the low-q
  cutoff from ``q_cyl_from_B_R_epsilon_kappa_I``.
* ``peeling_ballooning``: a *schematic* -- the edge-stability boundary has no
  closed form -- built from two linear stability margins joined by a smooth
  maximum. The figure says so, and its corner (the star) is computed where
  the two margins are equal rather than placed.

Every diagram's ``Diagram.model`` is a :class:`Chart` holding the curves in
data coordinates, so tests check the physics without reading the drawing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Tuple

import numpy as np

from vaft.formula.equilibrium import q_cyl_from_B_R_epsilon_kappa_I
from vaft.formula.stability import (
    beta_N_from_beta_a_B0_Ip,
    greenwald_density,
    s_alpha_marginal_alpha,
)

from ._chart import Chart, nice_ticks as _nice_ticks, render_chart as _render_chart
from ._render import Diagram

# ---------------------------------------------------------------------------
# Peeling-ballooning (schematic)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeelingBallooningModel:
    """Two linear edge-stability margins joined by a smooth maximum.

    Peeling: unstable when ``J > j0 + j_slope * alpha`` (the current limit
    rises with the pressure gradient). Ballooning: unstable when
    ``alpha > a0 + a_slope * J`` (current eases ballooning). Both margins
    are in units of ``width``; the stability function is
    ``M = tau * log(exp(P / tau) + exp(B / tau))`` and the boundary is
    ``M = 0``, which rounds the corner where the two limits meet.
    """

    j0: float = 0.45
    j_slope: float = 0.35
    a0: float = 0.55
    a_slope: float = 0.28
    width: float = 0.1
    tau: float = 0.8

    def peeling(self, alpha, J):
        return (np.asarray(J) - self.j0 - self.j_slope * np.asarray(alpha)) / self.width

    def ballooning(self, alpha, J):
        return (np.asarray(alpha) - self.a0 - self.a_slope * np.asarray(J)) / self.width

    def margin(self, alpha, J):
        return self.tau * np.logaddexp(self.peeling(alpha, J) / self.tau, self.ballooning(alpha, J) / self.tau)

    def corner(self) -> Tuple[float, float]:
        """The boundary point where the peeling and ballooning margins are equal.

        On ``M = 0`` with ``P = B`` both equal ``-tau ln 2``; two linear
        equations in ``(alpha, J)``.
        """
        c = -self.tau * math.log(2.0) * self.width
        # J - j_slope alpha = j0 + c ;  alpha - a_slope J = a0 + c
        A = np.array([[-self.j_slope, 1.0], [1.0, -self.a_slope]])
        alpha, J = np.linalg.solve(A, [self.j0 + c, self.a0 + c])
        return float(alpha), float(J)

    def boundary(self, n: int = 241) -> np.ndarray:
        """``M = 0`` traced along rays from the origin, from the alpha axis to the J axis."""
        angles = np.linspace(0.0, np.pi / 2, n)
        radius = np.linspace(0.0, 2.0, 4001)
        out = []
        for ang in angles:
            a, j = radius * np.cos(ang), radius * np.sin(ang)
            m = self.margin(a, j)
            k = int(np.argmax(m > 0.0))
            lo, hi = radius[k - 1], radius[k]
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if self.margin(mid * np.cos(ang), mid * np.sin(ang)) > 0.0:
                    hi = mid
                else:
                    lo = mid
            r = 0.5 * (lo + hi)
            out.append((r * np.cos(ang), r * np.sin(ang)))
        return np.array(out)


def peeling_ballooning(*, labels: bool = True) -> Diagram:
    r"""Schematic edge-stability (peeling-ballooning) diagram.

    Axes are the pedestal's peak normalised pressure gradient
    $\alpha_\mathrm{max}$ and peak bootstrap (edge) current density
    $J_{B,\mathrm{max}}$, in arbitrary units: the boundary's shape is
    illustrative, not computed from an equilibrium. Its corner, where the
    peeling and ballooning limits meet (the star), is where ELM-limited
    pedestals typically sit; it is computed from the model, not placed.
    """
    model = PeelingBallooningModel()
    chart = Chart(x_range=(0.0, 1.25), y_range=(0.0, 1.0))
    chart.curves["boundary"] = model.boundary()
    chart.points["corner"] = model.corner()
    chart.labels.update({"peeling": (0.4, 0.86), "stable": (0.3, 0.28), "ballooning": (1.0, 0.28)})
    chart.parameters.update({k: getattr(model, k) for k in ("j0", "j_slope", "a0", "a_slope", "width", "tau")})
    scene = _render_chart(
        chart,
        x_label="$\\alpha_\\mathrm{max}$",
        y_label="$J_{B,\\mathrm{max}}$",
        curve_styles={"boundary": "boundary"},
        region_text={"peeling": "Peeling unstable", "stable": "Stable",
                     "ballooning": "\\begin{tabular}{l}Ballooning\\\\unstable\\end{tabular}"} if labels else {},
        note="schematic: boundary shape illustrative (Connor et al. 1998; Snyder et al. 2002)" if labels else "",
        star="corner",
    )
    return Diagram("peeling_ballooning", scene, model=chart)


# ---------------------------------------------------------------------------
# s-alpha ballooning
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _s_alpha_boundaries(s_min: float, s_max: float, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.linspace(s_min, s_max, n)
    first, second = s_alpha_marginal_alpha(s)
    return s, first, second


def s_alpha_ballooning(*, s_max: float = 1.5, alpha_max: float = 3.5, labels: bool = True) -> Diagram:
    r"""First and second ballooning stability in the circular $s$-$\alpha$ model.

    Both boundaries come from ``vaft.formula.s_alpha_marginal_alpha``; the
    dashed line is the $\alpha = 0.6\,s$ approximation
    ``ballooning_stability_criterion`` uses.
    """
    if not 0.1 < s_max <= 2.5 or not 0.5 < alpha_max <= 6.0:
        raise ValueError("s_max must lie in (0.1, 2.5] and alpha_max in (0.5, 6]")
    s, first, second = _s_alpha_boundaries(0.05, float(s_max), 36)
    ok = np.isfinite(first)
    chart = Chart(x_range=(0.0, float(alpha_max)), y_range=(0.0, float(s_max)))
    # two open branches: below s ~ 0.05 the model's boundaries are not resolved
    chart.curves["first"] = np.stack([first[ok], s[ok]], axis=-1)
    second_ok = ok & np.isfinite(second)
    chart.curves["second"] = np.stack([second[second_ok], s[second_ok]], axis=-1)
    chart.curves["approximation"] = np.array([[0.0, 0.0], [0.6 * s_max, s_max]])
    # labels are placed from the computed boundaries, so they stay in their
    # regions whatever the axis ranges; a region off the chart gets none
    def at(row, curve):
        return float(np.interp(row, s[ok], curve[ok]))

    s_first, s_unstable, s_second = 0.85 * s_max, 0.6 * s_max, max(0.25 * s_max, 0.1)
    a1 = at(s_first, first)
    chart.labels["first"] = (0.45 * a1, s_first)
    a1, a2 = at(s_unstable, first), at(s_unstable, np.where(np.isfinite(second), second, alpha_max))
    if a1 < 0.9 * alpha_max:
        chart.labels["unstable"] = (0.5 * (a1 + min(a2, alpha_max)), s_unstable)
    a2 = at(s_second, np.where(np.isfinite(second), second, np.inf))
    if a2 < 0.75 * alpha_max:
        chart.labels["second"] = (0.5 * (a2 + alpha_max), s_second)
    chart.parameters.update({"s_min": 0.05, "s_max": float(s_max), "alpha_max": float(alpha_max)})
    scene = _render_chart(
        chart,
        x_label="$\\alpha$",
        y_label="$s$",
        curve_styles={"approximation": "approx", "first": "boundary", "second": "boundary"},
        region_text={k: v for k, v in {
            "first": "\\begin{tabular}{c}First\\\\stable\\end{tabular}", "unstable": "Unstable",
            "second": "\\begin{tabular}{c}Second\\\\stable\\end{tabular}"}.items() if k in chart.labels} if labels else {},
        x_ticks=_nice_ticks(alpha_max + 1e-9), y_ticks=_nice_ticks(s_max + 1e-9),
        note="circular $s$-$\\alpha$ model (Connor, Hastie \\& Taylor 1978); dashed: $\\alpha = 0.6\\,s$" if labels else "",
    )
    return Diagram("s_alpha_ballooning", scene, model=chart)


# ---------------------------------------------------------------------------
# Hugill and Troyon operational space
# ---------------------------------------------------------------------------

#: nominal machine used to evaluate the formulas; both boundaries depend
#: only on the elongation (Hugill) or on aspect ratio and elongation (Troyon)
_R0, _B0 = 1.0, 1.0
#: any aspect ratio gives the same Hugill line; this one only sizes the nominal machine
_HUGILL_ASPECT_RATIO = 3.0


def _validate_shape(aspect_ratio: float, elongation: float, q_limit: float) -> None:
    if not aspect_ratio > 1.0:
        raise ValueError(f"aspect_ratio must exceed 1, not {aspect_ratio!r}")
    if not elongation > 0.0:
        raise ValueError(f"elongation must be positive, not {elongation!r}")
    if not q_limit > 0.0:
        raise ValueError(f"q_limit must be positive, not {q_limit!r}")


def _current_at_q(elongation: float, aspect_ratio: float, q_limit: float) -> float:
    """Plasma current [MA] at which the nominal machine's q_cyl equals ``q_limit``.

    ``q_cyl_from_B_R_epsilon_kappa_I`` is inversely proportional to the
    current, so one evaluation at 1 MA fixes it exactly.
    """
    q_at_1MA = q_cyl_from_B_R_epsilon_kappa_I(_B0, _R0, 1.0 / aspect_ratio, elongation, 1e6)
    return float(q_at_1MA) / q_limit


def hugill(*, elongation: float = 1.0, q_limit: float = 2.0, labels: bool = True) -> Diagram:
    r"""Hugill diagram: $1/q_\mathrm{cyl}$ against the Murakami parameter $\bar n R/B$.

    The density limit is the Greenwald density written in these
    coordinates: along a current scan, ``greenwald_density`` and
    ``q_cyl_from_B_R_epsilon_kappa_I`` give a straight line through the
    origin whose slope, $50\kappa_a/\pi$, depends only on the (area)
    elongation -- the minor radius, major radius and field cancel, so there is
    no machine-size parameter. The low-q limit is $q_\mathrm{cyl} = q_\mathrm{limit}$.
    """
    _validate_shape(_HUGILL_ASPECT_RATIO, elongation, q_limit)
    a = _R0 / _HUGILL_ASPECT_RATIO
    y_max = 1.4 / q_limit
    I_q = _current_at_q(elongation, _HUGILL_ASPECT_RATIO, q_limit)
    I_MA = np.linspace(0.0, 1.4 * I_q, 201)[1:]  # up to 1/q = y_max
    q = q_cyl_from_B_R_epsilon_kappa_I(_B0, _R0, 1.0 / _HUGILL_ASPECT_RATIO, elongation, I_MA * 1e6)
    murakami = greenwald_density(I_MA, a) * _R0 / _B0  # [1e19 m^-2 T^-1]
    x_q = float(greenwald_density(I_q, a) * _R0 / _B0)
    x_max = 1.45 * x_q
    chart = Chart(x_range=(0.0, x_max), y_range=(0.0, y_max))
    chart.curves["greenwald"] = np.concatenate([[[0.0, 0.0]], np.stack([murakami, 1.0 / q], axis=-1)])
    chart.curves["low_q"] = np.array([[0.0, 1.0 / q_limit], [x_max, 1.0 / q_limit]])
    chart.labels.update({
        "accessible": (0.3 * x_q, 0.72 / q_limit),
        "density": (1.2 * x_q, 0.45 / q_limit),
        "low_q": (0.33 * x_max, 1.2 / q_limit),
    })
    chart.parameters.update({"elongation": elongation, "q_limit": q_limit, "murakami_at_q_limit": x_q})
    scene = _render_chart(
        chart,
        x_label="$\\bar n_e R/B_T\\ [10^{19}\\,\\mathrm{m^{-2}\\,T^{-1}}]$",
        y_label="$1/q_\\mathrm{cyl}$",
        curve_styles={"greenwald": "boundary", "low_q": "boundary"},
        region_text={"accessible": "Accessible",
                     "density": "\\begin{tabular}{c}Density limit\\\\($\\bar n_e > n_G$)\\end{tabular}",
                     "low_q": f"Low-$q$ limit ($q_\\mathrm{{cyl}} < {q_limit:g}$)"} if labels else {},
        x_ticks=_nice_ticks(x_max),
        y_ticks=_nice_ticks(y_max),
        note=f"Greenwald limit in Murakami coordinates, $\\kappa_a = {elongation:g}$" if labels else "",
    )
    return Diagram("hugill", scene, model=chart)


def troyon(*, beta_N_max: float = 2.8, aspect_ratio: float = 3.0, elongation: float = 1.7,
           q_limit: float = 2.0, labels: bool = True) -> Diagram:
    r"""Troyon diagram: toroidal beta against the normalised current $I_p/(aB_T)$.

    The beta limit is the line on which ``beta_N_from_beta_a_B0_Ip`` equals
    ``beta_N_max``; the low-q cutoff is the current at which
    ``q_cyl_from_B_R_epsilon_kappa_I`` reaches ``q_limit``, at
    $I_p/(aB_T) = 5\varepsilon\kappa_a/q_\mathrm{limit}$.
    """
    _validate_shape(aspect_ratio, elongation, q_limit)
    if not beta_N_max > 0.0:
        raise ValueError(f"beta_N_max must be positive, not {beta_N_max!r}")
    a = _R0 / aspect_ratio
    x_q = _current_at_q(elongation, aspect_ratio, q_limit) / (a * _B0)
    x = np.linspace(0.0, 1.35 * x_q, 101)
    # beta_N is linear in beta: one evaluation at beta = 1 % sets the slope of the limit line
    beta_limit = beta_N_max / beta_N_from_beta_a_B0_Ip(1.0, a, _B0, np.maximum(x, 1e-12) * a * _B0)
    beta_limit = np.where(x > 0, beta_limit, 0.0)
    y_max = 1.3 * beta_N_max * x_q
    chart = Chart(x_range=(0.0, float(x[-1])), y_range=(0.0, float(y_max)))
    chart.curves["beta_limit"] = np.stack([x, beta_limit], axis=-1)
    chart.curves["low_q"] = np.array([[x_q, 0.0], [x_q, y_max]])
    chart.labels.update({
        "stable": (0.6 * x_q, 0.25 * beta_N_max * x_q),
        "beta": (0.35 * x_q, 0.9 * beta_N_max * x_q),
        "low_q": (1.18 * x_q, 0.45 * beta_N_max * x_q),
    })
    chart.parameters.update({"beta_N_max": beta_N_max, "aspect_ratio": aspect_ratio, "elongation": elongation,
                             "q_limit": q_limit, "current_at_q_limit": x_q})
    scene = _render_chart(
        chart,
        x_label="$I_p/(aB_T)\\ [\\mathrm{MA\\,m^{-1}\\,T^{-1}}]$",
        y_label="$\\beta_T\\ [\\%]$",
        curve_styles={"beta_limit": "boundary", "low_q": "boundary"},
        region_text={"stable": "Stable",
                     "beta": f"$\\beta_N > {beta_N_max:g}$",
                     "low_q": "\\begin{tabular}{c}Low-$q$\\\\limit\\end{tabular}"} if labels else {},
        x_ticks=_nice_ticks(float(x[-1])),
        y_ticks=_nice_ticks(float(y_max)),
        note=(f"Troyon limit $\\beta_N = {beta_N_max:g}$; $q_\\mathrm{{cyl}} = {q_limit:g}$ at "
              f"$R_0/a = {aspect_ratio:g}$, $\\kappa_a = {elongation:g}$") if labels else "",
    )
    return Diagram("troyon", scene, model=chart)


_BUILDERS: Dict[str, Callable[..., Diagram]] = {
    "peeling_ballooning": peeling_ballooning,
    "s_alpha_ballooning": s_alpha_ballooning,
    "hugill": hugill,
    "troyon": troyon,
}
