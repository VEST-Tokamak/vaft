"""Toroidal current-density moments of an axisymmetric equilibrium (#943).

Implementation module; :mod:`vaft.process.equilibrium` is the stable public
import location, the same arrangement as ``_equilibrium_parametric``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.constants import mu_0 as MU0

from vaft.data.cocos import cocos_spec
from vaft.data.equilibrium import CurrentMomentRepresentation, DerivationProvenance


def _grid(j_tor: Any, r: Any, z: Any, weights: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=float).reshape(-1); z = np.asarray(z, dtype=float).reshape(-1)
    j = np.asarray(j_tor, dtype=float)
    if j.shape != (r.size, z.size):
        raise ValueError(f"j_tor must be indexed (R, Z) with shape {(r.size, z.size)}, got {j.shape}")
    if r.size < 2 or z.size < 2:
        raise ValueError("the grid needs at least two points along each axis")
    area = np.gradient(r)[:, None]*np.gradient(z)[None, :]
    w = np.ones_like(j) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != j.shape:
        raise ValueError("weights must have the shape of j_tor")
    # Outside the weighted region the density is irrelevant, and may be NaN
    # (an ODS writes NaN beyond the LCFS); it must not poison the sums.  Inside
    # it a NaN is missing current, and zeroing it would bias every moment.
    missing = (w > 0) & ~np.isfinite(j)
    if np.any(missing):
        raise ValueError(
            f"j_tor is not finite in {int(np.sum(missing))} cells with nonzero weight; "
            "fill them (e.g. partial LCFS cells an ODS leaves NaN) or zero their weights"
        )
    element = np.where(w > 0, np.where(np.isfinite(j), j, 0.0)*w*area, 0.0)
    rm, zm = np.meshgrid(r, z, indexing="ij")
    return element, rm, zm, w


def current_centroid(j_tor: Any, r: Any, z: Any, *, weights: Any = None) -> tuple[float, float, float]:
    """Total toroidal current and its centroid, from a gridded current density.

    The zeroth and first current moments: ``I_p = integral J_phi dA`` and
    ``(R_c, Z_c) = (1/I_p) integral (R, Z) J_phi dA``.  A single filament at
    the centroid carrying ``I_p`` is the lowest-order representation of the
    distribution.

    Parameters
    ----------
    j_tor : array_like
        Toroidal current density, indexed ``(R, Z)`` [A/m^2].
    r : array_like
        Major-radius grid axis [m].
    z : array_like
        Height grid axis [m].
    weights : array_like, optional
        Fraction of each cell to integrate, such as
        :func:`fractional_cell_weights_from_boundary` gives; all ones when not
        given [-].

    Returns
    -------
    tuple of float
        ``(I_p, R_c, Z_c)`` in ampere, metre and metre [-].

    Raises
    ------
    ValueError
        The shapes disagree, or the integrated current is zero, which leaves
        the centroid undefined.

    Convention
    ----------
    The area element is the cell area ``dR dZ`` times its weight, not
    ``2 pi R dR dZ``: these are moments of the current through the poloidal
    cross-section, whose zeroth moment is the plasma current.  The sign of
    ``I_p`` is that of ``j_tor``; the centroid does not depend on it.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] Definition of the current moments; see issue #943.
    """
    element, rm, zm, _ = _grid(j_tor, r, z, weights)
    total = float(np.sum(element))
    if total == 0.0 or not np.isfinite(total):
        raise ValueError("the integrated current is zero or not finite; the centroid is undefined")
    return total, float(np.sum(rm*element)/total), float(np.sum(zm*element)/total)


def current_moment(
    j_tor: Any, r: Any, z: Any, p: int, q: int, *, weights: Any = None,
    center: tuple[float, float] | None = None, normalize: bool = True,
) -> float:
    """One moment ``integral (R - R_0)**p (Z - Z_0)**q J_phi dA`` of a gridded current density.

    Parameters
    ----------
    j_tor : array_like
        Toroidal current density, indexed ``(R, Z)`` [A/m^2].
    r : array_like
        Major-radius grid axis [m].
    z : array_like
        Height grid axis [m].
    p : int
        Order in R, non-negative [-].
    q : int
        Order in Z, non-negative [-].
    weights : array_like, optional
        Fraction of each cell to integrate [-].
    center : tuple of float, optional
        Reference point ``(R_0, Z_0)``; the current centroid when not given,
        which makes the moment a central one [m].
    normalize : bool, optional
        Divide by the total current, giving m**(p+q) instead of A m**(p+q) [-].

    Returns
    -------
    float
        The moment, in m**(p+q) when normalized and A m**(p+q) otherwise [-].

    Raises
    ------
    ValueError
        A negative order, disagreeing shapes, or a zero total current where
        the centroid or the normalization needs it.

    Convention
    ----------
    ``center=None`` means *central*: about ``(R_c, Z_c)`` from
    :func:`current_centroid`, so the first central moments vanish.  Pass
    ``center=(0.0, 0.0)`` for raw moments about the origin of the (R, Z)
    plane.  The area element is ``dR dZ`` times the weight.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] Definition of the current moments; see issue #943.
    """
    p, q = int(p), int(q)
    if p < 0 or q < 0:
        raise ValueError("moment orders must be non-negative")
    element, rm, zm, _ = _grid(j_tor, r, z, weights)
    if center is None:
        _, r0, z0 = current_centroid(j_tor, r, z, weights=weights)
    else:
        r0, z0 = map(float, center)
    value = float(np.sum((rm - r0)**p*(zm - z0)**q*element))
    if not normalize:
        return value
    total = float(np.sum(element))
    if total == 0.0 or not np.isfinite(total):
        raise ValueError("the integrated current is zero or not finite; cannot normalize")
    return value/total


def current_covariance(j_tor: Any, r: Any, z: Any, *, weights: Any = None) -> np.ndarray:
    """Second central moment tensor of a gridded current density.

    ``[[mu_20, mu_11], [mu_11, mu_02]]`` about the current centroid,
    normalized by the total current.  Its eigenvalues are the squared
    principal widths of the current distribution and its eigenvectors their
    directions.

    Parameters
    ----------
    j_tor : array_like
        Toroidal current density, indexed ``(R, Z)`` [A/m^2].
    r : array_like
        Major-radius grid axis [m].
    z : array_like
        Height grid axis [m].
    weights : array_like, optional
        Fraction of each cell to integrate [-].

    Returns
    -------
    np.ndarray
        The symmetric 2 x 2 tensor [m^2].

    Raises
    ------
    ValueError
        As for :func:`current_moment`.

    Convention
    ----------
    Central and normalized, as :func:`current_moment` with its defaults.  A
    width of the *current*, not of the plasma boundary: a peaked current
    profile has a smaller covariance than its LCFS suggests.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] Definition of the current moments; see issue #943.
    """
    _, r0, z0 = current_centroid(j_tor, r, z, weights=weights)
    mu = {pq: current_moment(j_tor, r, z, *pq, weights=weights, center=(r0, z0)) for pq in ((2, 0), (1, 1), (0, 2))}
    return np.array([[mu[(2, 0)], mu[(1, 1)]], [mu[(1, 1)], mu[(0, 2)]]], dtype=float)


def _flux_function_current(eq: Any) -> np.ndarray:
    """J_phi on the record's grid from its p' and FF' profiles, in the record's COCOS.

    The convention is resolved as ``vaft.omas.update`` resolves it for an ODS:
    the 2*pi exponent from ``psi_per_radian`` (or the COCOS), ``sigma_Bp`` from
    the index or from candidates that agree on it, and the DD's +1 otherwise.
    """
    if eq.convention is None:
        raise ValueError("the equilibrium declares no COCOS convention, so the sign and 2*pi of J_phi are unknown; pass j_tor")
    if eq.pprime is None or eq.ffprime is None or eq.psi_1d is None:
        raise ValueError("the equilibrium has no p'/FF' profiles to derive J_phi from; pass j_tor")
    if eq.psi_axis is None or eq.psi_boundary is None or not (np.isfinite(eq.psi_axis) and np.isfinite(eq.psi_boundary)):
        raise ValueError("psi_axis and psi_boundary must be finite to map p'/FF' onto the grid")
    if eq.psi_axis == eq.psi_boundary:
        raise ValueError("psi_axis equals psi_boundary; p'/FF' cannot be mapped onto the grid")
    convention = eq.convention
    indices = [convention.cocos] if convention.cocos is not None else [c for c in (convention.candidates or ()) if c]
    specs = [cocos_spec(int(c)) for c in indices]
    if convention.psi_per_radian is not None:
        exp_bp = 0 if convention.psi_per_radian else 1
    elif specs and len({spec.exp_bp for spec in specs}) == 1:
        exp_bp = specs[0].exp_bp
    else:
        raise ValueError("the flux unit (Wb or Wb/rad) of this record is not identified, so J_phi is off by 2*pi either way; "
                         "declare the convention or pass j_tor")
    signs = {spec.sigma_bp for spec in specs}
    sigma_bp = signs.pop() if len(signs) == 1 else 1
    psi_n_1d = (np.asarray(eq.psi_1d, dtype=float) - eq.psi_axis)/(eq.psi_boundary - eq.psi_axis)
    order = np.argsort(psi_n_1d)
    psi_n = np.clip((np.asarray(eq.psi, dtype=float) - eq.psi_axis)/(eq.psi_boundary - eq.psi_axis), 0.0, 1.0)
    # anti-alias: not a time series -- flux-function profiles mapped onto the
    # 2-D grid by their normalized flux, as update_equilibrium_profiles_2d_j_tor does.
    pprime = np.interp(psi_n, psi_n_1d[order], np.asarray(eq.pprime, dtype=float)[order])
    ffprime = np.interp(psi_n, psi_n_1d[order], np.asarray(eq.ffprime, dtype=float)[order])
    rm = np.asarray(eq.r, dtype=float)[:, None]*np.ones((1, np.asarray(eq.z).size))
    return -sigma_bp*(2*np.pi)**exp_bp*(rm*pprime + ffprime/(MU0*rm))


def derive_current_moments(equilibrium: Any, *, max_order: int = 4, j_tor: Any = None) -> CurrentMomentRepresentation:
    """Reduce an equilibrium's toroidal current density to its low-order moments.

    Between a single filament and the full ``J_phi(R, Z)``: the total
    current, its centroid, and every normalized central moment up to
    *max_order*, integrated over the plasma with fractional LCFS cells.  Two
    equilibria on different grids can then be compared moment by moment.

    Parameters
    ----------
    equilibrium : EquilibriumData, ODS, GEQDSK or path
        Adapted through :func:`as_equilibrium`; needs a psi map and an LCFS [-].
    max_order : int, optional
        Highest total order ``p + q`` of the central moments, at least two [-].
    j_tor : array_like, optional
        A canonical toroidal current density on the record's grid, indexed
        ``(R, Z)``, such as an ODS ``profiles_2d.j_tor``; derived from the
        record's ``p'`` and ``FF'`` when not given [A/m^2].

    Returns
    -------
    CurrentMomentRepresentation
        Total current, centroid, the central moments, the order, where the
        current density came from, and the derivation record [-].

    Raises
    ------
    ValueError
        *max_order* below two, no psi map or LCFS, a record with neither a
        *j_tor* nor the convention and profiles to derive one, or zero
        integrated current.

    Processing steps
    ----------------
    1. Take *j_tor* as given, or evaluate the Grad-Shafranov current
       ``J_phi = -sigma_Bp (2 pi)**e_Bp (R p'(psi) + FF'(psi)/(mu0 R))`` from
       the record's flux-function profiles in its declared COCOS.
    2. Weight each cell by the fraction of it inside the LCFS.
    3. Integrate the total current and the centroid, then every central
       moment with ``2 <= p + q <= max_order``.

    Defaults
    --------
    ``max_order = 4`` is a numerical convenience: second order gives the
    widths and tilt, third the skewness and up-down asymmetry, fourth the
    first shape beyond them; higher orders grow more sensitive to the edge.

    Convention
    ----------
    The derived current density is the same expression, and the same resolved
    psi convention, as ``vaft.omas.update.update_equilibrium_profiles_2d_j_tor``
    uses for an ODS, resolved the same way (the 2*pi exponent from
    ``psi_per_radian``, ``sigma_Bp`` from agreeing candidates, else +1), so
    the two paths agree; it is never a ``dpsi/dR``
    pseudo-current.  Moments are central about the current centroid and
    normalized by the total current (m**(p+q)); the area element is
    ``dR dZ``, so the zeroth moment is the plasma current.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    An ODS's own ``profiles_2d.j_tor`` is not read automatically --
    :func:`as_equilibrium` does not carry it -- so pass it as *j_tor*; the
    derived density agrees with ``update_equilibrium_profiles_2d_j_tor`` to
    about 1e-10 inside the LCFS.  NaN cells of a supplied *j_tor* that the
    fractional weights still count are filled from ``p'``/``FF'`` and the
    source label says how many.  The integrated current is checked against
    the record's ``ip``, and a
    disagreement beyond five percent -- typically a factor 2*pi or -1 from a
    wrongly declared convention -- is warned about, not corrected.  Current
    outside the LCFS, such as a halo or a vessel current, is excluded
    by the weighting.  The Shafranov (magnetic) moments of #943 are a separate
    representation and are not computed here.

    Provenance
    ----------
    .. [1] Definition of the current moments; see issue #943.
    .. [2] O. Sauter and S. Yu. Medvedev, *Tokamak coordinate conventions:
       COCOS*, Comput. Phys. Commun. 184, 293 (2013), Eq. 12, for the sign and
       2*pi of the Grad-Shafranov current.
    """
    from vaft.process.equilibrium import as_equilibrium, fractional_cell_weights_from_boundary

    if int(max_order) < 2:
        raise ValueError("max_order must be at least 2")
    eq = as_equilibrium(equilibrium)
    if eq.psi is None or eq.r is None or eq.z is None:
        raise ValueError("the equilibrium carries no (R, Z) psi map")
    if eq.lcfs is None:
        raise ValueError("the equilibrium has no LCFS to bound the integration")
    weights = fractional_cell_weights_from_boundary(eq.r, eq.z, eq.lcfs.r, eq.lcfs.z)
    if j_tor is None:
        j = _flux_function_current(eq)
        source = "grad_shafranov_flux_functions"
    else:
        j = np.array(j_tor, dtype=float)
        source = "supplied"
        missing = (weights > 0) & ~np.isfinite(j)
        if np.any(missing):
            # An ODS's profiles_2d.j_tor is NaN wherever the cell *centre* is
            # outside the LCFS, which includes partial boundary cells.
            j[missing] = _flux_function_current(eq)[missing]
            source = f"supplied; {int(np.sum(missing))} partial LCFS cells filled from p'/FF'"
    total, r_c, z_c = current_centroid(j, eq.r, eq.z, weights=weights)
    if eq.ip not in (None, 0) and np.isfinite(eq.ip) and abs(total/eq.ip - 1.0) > 0.05:
        # A factor 2*pi or -1 here almost always means a wrongly declared
        # convention, not a bad integral.
        warnings.warn(
            f"integrated current {total:.6g} A differs from the record's ip {eq.ip:.6g} A by a factor "
            f"{total/eq.ip:.4g}; check the declared COCOS convention", stacklevel=2,
        )
    moments = {
        (p, n - p): current_moment(j, eq.r, eq.z, p, n - p, weights=weights, center=(r_c, z_c))
        for n in range(2, int(max_order) + 1) for p in range(n, -1, -1)
    }
    provenance = DerivationProvenance(
        "fractional-LCFS-cell quadrature of central current moments",
        source_type=str(eq.metadata.get("source_type", "native")),
        source_fields=("psi", "lcfs") + (("pprime", "ffprime") if j_tor is None else ("j_tor",)),
        source_time=eq.time, convention=eq.convention,
        notes=(f"current density: {source}", "moments central about the current centroid, normalized by I_p"),
    )
    return CurrentMomentRepresentation(total, r_c, z_c, moments, int(max_order), source, provenance)


__all__ = ["current_centroid", "current_covariance", "current_moment", "derive_current_moments"]
