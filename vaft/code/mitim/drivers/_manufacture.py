"""Manufactured-source construction shared by the PORTALS closed-loop drivers (#1588 C1).

Staged next to a driver in its run directory by :func:`vaft.code.mitim.runner.run_mitim_driver`
and imported there as ``_manufacture``; NumPy and SciPy only, so VAFT's tests import it
too.
"""

__all__ = ["manufacture_converged", "manufacture_error", "manufactured_source"]


def manufactured_source(rho, r, volp, rho_k, nodes):
    """Power density q [MW/m^3] whose volume integral passes through ``nodes`` [MW] at ``rho_k``.

    The cumulative power P is interpolated in s = rho^2, not in rho. Near the axis
    dV ~ r dr, so a finite source needs P ~ rho^2. A monotone interpolant in rho
    through (0, 0) generally has dP/drho != 0 at the axis, and then
    q = dP/drho * drho/dr / volp diverges like 1/volp there. A single node always does
    this: the interpolant is a straight line. Several nodes do it whenever their
    shape keeps the endpoint slope nonzero. PORTALS integrates the targets on its own
    coarse grid, where that spike becomes a 12-21x overestimate. In s the slope at the
    axis is dP/drho = 2 rho dP/ds = 0, so q stays finite for any node set (#1588 C1).
    Signed nodes (a net sink) are valid. Only grid points with volp > 0 get a value.
    """
    import numpy as np
    from scipy.interpolate import PchipInterpolator

    rho, r, volp = (np.asarray(x, dtype=float) for x in (rho, r, volp))
    rho_k, nodes = np.asarray(rho_k, dtype=float), np.asarray(nodes, dtype=float)
    power = PchipInterpolator(np.concatenate(([0.0], rho_k ** 2)), np.concatenate(([0.0], nodes)),
                              extrapolate=True)
    inside = volp > 0
    q = np.zeros_like(r)
    q[inside] = (power.derivative()(rho[inside] ** 2) * 2.0 * rho[inside]
                 * np.gradient(rho, r)[inside] / volp[inside])
    return q


def manufacture_error(got, required):
    """Largest node error of the targets, relative to the largest |required| flux.

    Normalising by the largest flux rather than node by node keeps a legitimate
    near-zero (or sign-changing) flux at one node from reading as divergence.
    """
    import numpy as np

    got, required = np.asarray(got, dtype=float), np.asarray(required, dtype=float)
    scale = float(np.max(np.abs(required)))
    if not np.all(np.isfinite(got)) or not scale > 0:
        return float("inf")
    return float(np.max(np.abs(got - required)) / scale)


def manufacture_converged(history, tolerance=1e-4):
    """Whether the node iteration reached ``tolerance`` (finite, last entry)."""
    import math

    return bool(history) and math.isfinite(history[-1]) and history[-1] <= tolerance
