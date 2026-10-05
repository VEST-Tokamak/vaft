r"""Impurity-mixture algebra: moments, target-Z_eff densities, the reduced pseudo-impurity and main-ion dilution.

A plasma of electrons, one main ion and any number of impurity species
(issue #1565, Stage A).  Each impurity species is an element in one stated
charge state with a relative particle density $w_i$; the functions here turn
that composition and a target plasma effective charge into absolute density
fractions, collapse the mixture onto one pseudo-impurity that keeps the
charge, $Z^2$ and mass moments, expand it back, and give the main-ion
fraction and dilution both from a resolved species list and from
$Z_\mathrm{eff}$ with an effective impurity charge.

The whole-plasma effective charge $\sum_s n_s Z_s^2 / n_e$ and the charge of
the reduced pseudo-impurity $Z_{I,\mathrm{eff}} = S_2/S_1$ are different
quantities and are never interchanged here.  The plasma $Z_\mathrm{eff}$ of
a resolved species list is :func:`vaft.formula.atomic.z_eff_from_n_s_Z_s`;
the issue's ``compute_zeff`` is that function, not a second copy.

Notation
--------
w_i       : relative impurity particle density, sum_i w_i = 1           [-]
Z_i       : charge state of impurity species i (never its atomic number) [-]
A_i       : mass of impurity species i                                   [u]
S_1, S_2  : sum_i w_i Z_i and sum_i w_i Z_i^2                             [-]
A_bar     : sum_i w_i A_i                                                [u]
Z_m       : main-ion charge (1 for hydrogenic)                           [-]
alpha     : total impurity density over n_e, sum_i n_i / n_e             [-]
Z_I,eff   : charge of the reduced pseudo-impurity, S_2 / S_1             [-]
f_main    : main-ion fraction n_main / n_e                               [-]
f_dil     : main-ion dilution 1 - f_main                                 [-]

Conventions
-----------
**The species axis is the last axis.**  Every species-indexed argument
(weights, charges, masses, densities) runs over species along its last
axis and broadcasts over the leading ones, so one call evaluates a whole
radial profile.  Weights must already sum to one: renormalising a user's
fractions is a provenance decision and lives in :mod:`vaft.process.impurity`,
which keeps the original input beside the normalised one.

**Quasi-neutrality closes every system.**  $n_e = Z_m n_\mathrm{main} +
\sum_i Z_i n_i$ and $Z_\mathrm{eff} n_e = Z_m^2 n_\mathrm{main} + \sum_i
Z_i^2 n_i$; no other ion is present.

References
----------
.. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
       Sec. 4.25 (impurities and the effective charge).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

__all__ = [
    "MixtureMoments",
    "ImpurityMixtureSolution",
    "EffectiveImpurity",
    "impurity_mixture_moments",
    "solve_impurity_mixture_for_target_zeff",
    "reduce_impurity_mixture",
    "expand_effective_impurity",
    "main_ion_fraction_from_zeff",
    "dilution_fraction_from_zeff",
    "main_ion_density_from_zeff",
    "dilution_fraction_from_species",
    "main_ion_density_from_species",
]

#: Relative tolerance on sum(w) = 1 and on the consistency checks below.
_RTOL = 1.0e-9


class MixtureMoments(NamedTuple):
    """Charge and mass moments of a normalised impurity mixture [-]."""

    S1: np.ndarray
    S2: np.ndarray
    A_bar: Optional[np.ndarray]


class ImpurityMixtureSolution(NamedTuple):
    """Absolute density fractions that realise a target effective charge [-]."""

    impurity_fractions: np.ndarray
    main_ion_fraction: np.ndarray
    alpha: np.ndarray


class EffectiveImpurity(NamedTuple):
    """One pseudo-impurity carrying a mixture's charge, Z^2 and mass moments [-]."""

    charge: np.ndarray
    density: np.ndarray
    mass: Optional[np.ndarray]


def _out(value):
    value = np.asarray(value, dtype=float)
    return float(value) if value.ndim == 0 else value


def _finite(value, name):
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _species(value, name, *, positive):
    arr = _finite(value, name)
    if arr.ndim == 0:
        raise ValueError(f"{name} needs a species axis (the last axis)")
    if arr.shape[-1] == 0:
        raise ValueError(f"{name} names no species")
    if positive and np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive")
    if not positive and np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return arr


def _weights(weights):
    w = _species(weights, "weights", positive=False)
    total = np.sum(w, axis=-1)
    if not np.allclose(total, 1.0, rtol=_RTOL, atol=_RTOL):
        raise ValueError(
            "weights must sum to one along the species axis; normalise them first "
            "(vaft.process.impurity keeps the original input in its provenance)"
        )
    return w


def _main_charge(main_ion_charge):
    z = float(main_ion_charge)
    if not np.isfinite(z) or z <= 0.0:
        raise ValueError(f"main_ion_charge must be positive; got {main_ion_charge!r}")
    return z


def impurity_mixture_moments(weights, charges, masses=None):
    r"""Charge and mass moments of a normalised impurity mixture.

    $$S_1 = \sum_i w_i Z_i,\qquad S_2 = \sum_i w_i Z_i^2,\qquad
      \bar A = \sum_i w_i A_i,\qquad \sum_i w_i = 1$$

    Parameters
    ----------
    weights : array-like
        Relative particle density $w_i$ of each impurity species along the
        last axis, non-negative and summing to one [-].
    charges : array-like
        Charge state $Z_i$ of each species, positive, broadcastable against
        ``weights`` [-].
    masses : array-like, optional
        Mass $A_i$ of each species, positive; omitted, $\bar A$ is ``None`` [u].

    Returns
    -------
    MixtureMoments
        ``S1`` and ``S2`` [-], and ``A_bar`` or ``None`` [u], each with the
        species axis removed [-].

    Raises
    ------
    ValueError
        Non-finite input, a negative weight, a non-positive charge or mass,
        or weights that do not sum to one.

    Convention
    ----------
    $Z_i$ is the ionic charge of the state the species is in, not the
    element's atomic number: carbon as C$^{4+}$ has $Z = 4$.  A partially
    ionised element contributes its charge-state-averaged $\langle Z\rangle$
    to $S_1$ and $\langle Z^2\rangle$ to $S_2$, not $\langle Z\rangle^2$.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 4.25.

    See Also
    --------
    solve_impurity_mixture_for_target_zeff
    reduce_impurity_mixture
    """
    w = _weights(weights)
    z = _species(charges, "charges", positive=True)
    w, z = np.broadcast_arrays(w, z)
    s1 = np.sum(w * z, axis=-1)
    s2 = np.sum(w * z**2, axis=-1)
    a_bar = None
    if masses is not None:
        a = _species(masses, "masses", positive=True)
        a_bar = _out(np.sum(np.broadcast_to(w, np.broadcast_shapes(w.shape, a.shape)) * a, axis=-1))
    return MixtureMoments(_out(s1), _out(s2), a_bar)


def solve_impurity_mixture_for_target_zeff(target_zeff, weights, charges, main_ion_charge=1.0):
    r"""Impurity and main-ion density fractions that give a target effective charge.

    $$\alpha = \frac{Z_\mathrm{eff} - Z_m}{S_2 - Z_m S_1},\qquad
      \frac{n_i}{n_e} = \alpha\,w_i,\qquad
      \frac{n_\mathrm{main}}{n_e} = \frac{1 - \alpha S_1}{Z_m}$$

    which for a hydrogenic main ion ($Z_m = 1$) is
    $\alpha = (Z_\mathrm{eff}-1)/(S_2-S_1)$.

    Parameters
    ----------
    target_zeff : float or array-like
        Plasma effective charge to realise, in $[Z_m,\,S_2/S_1]$ [-].
    weights : array-like
        Relative impurity particle densities $w_i$ along the last axis,
        summing to one [-].
    charges : array-like
        Charge state $Z_i$ of each impurity species, positive [-].
    main_ion_charge : float, optional
        Main-ion charge $Z_m$, default 1 [-].

    Returns
    -------
    ImpurityMixtureSolution
        ``impurity_fractions`` $n_i/n_e$ (species on the last axis),
        ``main_ion_fraction`` $n_\mathrm{main}/n_e$ and ``alpha``
        $=\sum_i n_i/n_e$ [-].

    Raises
    ------
    ValueError
        Invalid weights or charges, a mixture no more charged than the main
        ion ($S_2 \le Z_m S_1$), or a target outside $[Z_m,\,S_2/S_1]$,
        where some density would be negative.

    Assumptions
    -----------
    Quasi-neutrality and exactly these ions: one main ion of charge $Z_m$
    and the stated impurity species in the stated charge states.

    Physical interpretation
    -----------------------
    The ceiling $S_2/S_1$ is the plasma with no main ion left, which is
    exactly $Z_{I,\mathrm{eff}}$: no amount of this mixture can raise
    $Z_\mathrm{eff}$ past its own reduced charge.  For C$^{6+}$:O$^{8+}$ =
    1:1 at $Z_\mathrm{eff} = 2$, $S_1 = 7$, $S_2 = 50$ and
    $n_C/n_e = n_O/n_e = 1/86$.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 4.25.

    See Also
    --------
    vaft.formula.atomic.impurity_fraction_from_effective_charge
    reduce_impurity_mixture
    """
    zm = _main_charge(main_ion_charge)
    w = _weights(weights)
    z = _species(charges, "charges", positive=True)
    w, z = np.broadcast_arrays(w, z)
    s1 = np.sum(w * z, axis=-1)
    s2 = np.sum(w * z**2, axis=-1)
    denominator = s2 - zm * s1
    if np.any(denominator <= 0.0):
        raise ValueError(
            "the impurity mixture is no more charged than the main ion (S2 <= Z_m S1); "
            "it cannot raise Z_eff"
        )
    target = _finite(target_zeff, "target_zeff")
    ceiling = s2 / s1
    tol = _RTOL * np.maximum(1.0, ceiling)
    if np.any(target < zm - tol) or np.any(target > ceiling + tol):
        raise ValueError(
            f"target_zeff must lie in [Z_m, S2/S1] = [{zm:g}, {np.min(ceiling):g}]; "
            "outside it some density is negative"
        )
    alpha = (target - zm) / denominator
    fractions = alpha[..., None] * w if np.ndim(alpha) else alpha * w
    main = (1.0 - alpha * s1) / zm
    return ImpurityMixtureSolution(_out(fractions), _out(np.clip(main, 0.0, None)), _out(alpha))


def reduce_impurity_mixture(densities, charges, masses=None):
    r"""The single pseudo-impurity that keeps a mixture's charge, Z^2 and mass moments.

    $$Z_{I,\mathrm{eff}} = \frac{\sum_i n_i Z_i^2}{\sum_i n_i Z_i},\qquad
      n_{I,\mathrm{eff}} = \frac{(\sum_i n_i Z_i)^2}{\sum_i n_i Z_i^2},\qquad
      A_{I,\mathrm{eff}} = \frac{\sum_i n_i A_i}{n_{I,\mathrm{eff}}}$$

    so that $n_{I,\mathrm{eff}} Z_{I,\mathrm{eff}} = \sum n_i Z_i$,
    $n_{I,\mathrm{eff}} Z_{I,\mathrm{eff}}^2 = \sum n_i Z_i^2$ and
    $n_{I,\mathrm{eff}} A_{I,\mathrm{eff}} = \sum n_i A_i$.

    Parameters
    ----------
    densities : array-like
        Density of each impurity species along the last axis, non-negative,
        in any one unit (absolute, or relative to $n_e$) [m^-3].
    charges : array-like
        Charge state $Z_i$ of each species, positive [-].
    masses : array-like, optional
        Mass $A_i$ of each species, positive; omitted, ``mass`` is ``None`` [u].

    Returns
    -------
    EffectiveImpurity
        ``charge`` $Z_{I,\mathrm{eff}}$ [-], ``density`` $n_{I,\mathrm{eff}}$
        in the unit of ``densities`` [m^-3], and ``mass`` $A_{I,\mathrm{eff}}$
        or ``None`` [u].

    Raises
    ------
    ValueError
        Non-finite or negative densities, non-positive charges or masses, or a
        mixture of zero total density.

    Physical interpretation
    -----------------------
    Quasi-neutrality and $Z_\mathrm{eff}$ see only the first two moments, so
    an interface that takes one impurity reproduces both exactly with this
    pseudo-species; the mass moment keeps the impurity mass density.  For the
    VEST C$^{6+}$:O$^{8+}$ = 1:1 mixture at $Z_\mathrm{eff} = 2$ this is
    $Z_{I,\mathrm{eff}} = 50/7$, $n_{I,\mathrm{eff}}/n_e = 49/2150$ and
    $A_{I,\mathrm{eff}} = \bar A S_2/S_1^2 \approx 14.29$.

    Limitations
    -----------
    Higher moments are not kept: a model sensitive to $Z^3$ or to the
    individual masses (impurity transport, collision operators resolving
    each species) sees a different plasma.  Prefer the explicit species list
    wherever the downstream code accepts one.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 4.25.

    See Also
    --------
    expand_effective_impurity
    """
    n = _species(densities, "densities", positive=False)
    z = _species(charges, "charges", positive=True)
    n, z = np.broadcast_arrays(n, z)
    charge_density = np.sum(n * z, axis=-1)
    if np.any(charge_density <= 0.0):
        raise ValueError("a mixture of zero total density has no effective impurity")
    z2_density = np.sum(n * z**2, axis=-1)
    effective_charge = z2_density / charge_density
    effective_density = charge_density**2 / z2_density
    mass = None
    if masses is not None:
        a = _species(masses, "masses", positive=True)
        mass_density = np.sum(np.broadcast_to(n, np.broadcast_shapes(n.shape, a.shape)) * a, axis=-1)
        mass = _out(mass_density / effective_density)
    return EffectiveImpurity(_out(effective_charge), _out(effective_density), mass)


def expand_effective_impurity(density, charge, weights, charges, mass=None, masses=None):
    r"""Species densities of a known relative composition behind a pseudo-impurity.

    $$n_i = \frac{n_{I,\mathrm{eff}} Z_{I,\mathrm{eff}}}{S_1}\,w_i,
      \qquad\text{requiring}\quad Z_{I,\mathrm{eff}} = \frac{S_2}{S_1}$$

    the inverse of :func:`reduce_impurity_mixture` once the relative
    composition $w_i$ is retained.

    Parameters
    ----------
    density : float or array-like
        Pseudo-impurity density $n_{I,\mathrm{eff}}$, non-negative, in any
        unit [m^-3].
    charge : float or array-like
        Pseudo-impurity charge $Z_{I,\mathrm{eff}}$, which must equal
        $S_2/S_1$ of the composition [-].
    weights : array-like
        Relative particle densities $w_i$ along the last axis, summing to one [-].
    charges : array-like
        Charge state $Z_i$ of each species, positive [-].
    mass : float or array-like, optional
        Pseudo-impurity mass $A_{I,\mathrm{eff}}$; when given with ``masses``
        it must equal $\bar A S_2/S_1^2$ [u].
    masses : array-like, optional
        Mass $A_i$ of each species, positive [u].

    Returns
    -------
    np.ndarray
        Density $n_i$ of each species, species on the last axis, in the unit of
        ``density`` [m^-3].

    Raises
    ------
    ValueError
        Invalid input, a negative density, ``mass`` without ``masses``, or a
        pseudo-impurity whose charge (or mass) is not the one this composition
        reduces to -- then no species densities reproduce both its charge and
        $Z^2$ moments.

    Assumptions
    -----------
    The relative composition is the one the pseudo-impurity was reduced
    from.  A pseudo-impurity alone does not determine a mixture; the
    weights are the extra information the round trip needs.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 4.25.

    See Also
    --------
    reduce_impurity_mixture
    """
    n_eff = _finite(density, "density")
    if np.any(n_eff < 0.0):
        raise ValueError("density must be non-negative")
    z_eff = _finite(charge, "charge")
    moments = impurity_mixture_moments(weights, charges, masses)
    s1 = np.asarray(moments.S1, dtype=float)
    s2 = np.asarray(moments.S2, dtype=float)
    if not np.allclose(z_eff, s2 / s1, rtol=1.0e-6, atol=0.0):
        raise ValueError(
            "charge is not the effective charge S2/S1 of this composition; "
            "no species densities keep both its charge and Z^2 moments"
        )
    if mass is not None and moments.A_bar is None:
        raise ValueError("mass can only be checked against the species masses; pass masses too")
    if mass is not None:
        expected = np.asarray(moments.A_bar) * s2 / s1**2
        if not np.allclose(_finite(mass, "mass"), expected, rtol=1.0e-6, atol=0.0):
            raise ValueError("mass is not the effective mass A_bar S2/S1^2 of this composition")
    w = _weights(weights)
    scale = n_eff * z_eff / s1
    result = np.asarray(scale)[..., None] * w if np.ndim(scale) else scale * w
    return np.asarray(result, dtype=float)


def _impurity_charge(impurity_charge, main):
    z = _finite(impurity_charge, "impurity_charge")
    if np.any(z <= main):
        raise ValueError("impurity_charge must exceed the main-ion charge")
    return z


def main_ion_fraction_from_zeff(z_eff, impurity_charge, main_ion_charge=1.0):
    r"""Main-ion fraction of the electron density from Z_eff and one effective impurity charge.

    $$f_\mathrm{main} = \frac{n_\mathrm{main}}{n_e}
      = \frac{Z_I - Z_\mathrm{eff}}{Z_m\,(Z_I - Z_m)}
      \;\xrightarrow{Z_m = 1}\; \frac{Z_I - Z_\mathrm{eff}}{Z_I - 1}$$

    Parameters
    ----------
    z_eff : float or array-like
        Plasma effective charge, in $[Z_m, Z_I]$ [-].
    impurity_charge : float or array-like
        Charge $Z_I$ of the one (effective) impurity, e.g.
        $Z_{I,\mathrm{eff}}(\rho)$ of a reduced mixture, greater than $Z_m$ [-].
    main_ion_charge : float, optional
        Main-ion charge $Z_m$, default 1 [-].

    Returns
    -------
    float or np.ndarray
        $f_\mathrm{main}$ [-].

    Raises
    ------
    ValueError
        Non-finite input, $Z_I \le Z_m$, or $Z_\mathrm{eff}$ outside
        $[Z_m, Z_I]$.

    Physical interpretation
    -----------------------
    A $Z_\mathrm{eff}$ scan is not a dilution scan: the same $Z_\mathrm{eff}$
    removes more main ions when the impurity charge is lower.  At
    $Z_\mathrm{eff} = 2$, carbon ($Z_I = 6$) leaves $f_\mathrm{main} = 0.8$,
    the VEST C/O mixture ($Z_{I,\mathrm{eff}} = 50/7$) leaves $36/43 \approx 0.837$.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 4.25.

    See Also
    --------
    dilution_fraction_from_zeff
    main_ion_density_from_species
    """
    zm = _main_charge(main_ion_charge)
    zi = _impurity_charge(impurity_charge, zm)
    zeff = _finite(z_eff, "z_eff")
    if np.any(zeff < zm - _RTOL) or np.any(zeff > zi + _RTOL * zi):
        raise ValueError("z_eff must lie in [Z_m, Z_I]")
    return _out(np.clip((zi - zeff) / (zm * (zi - zm)), 0.0, 1.0 / zm))


def dilution_fraction_from_zeff(z_eff, impurity_charge, main_ion_charge=1.0):
    r"""Main-ion dilution from Z_eff and one effective impurity charge.

    $$f_\mathrm{dil} = 1 - \frac{n_\mathrm{main}}{n_e}
      \;\xrightarrow{Z_m = 1}\; \frac{Z_\mathrm{eff} - 1}{Z_I - 1}$$

    Parameters
    ----------
    z_eff : float or array-like
        Plasma effective charge, in $[Z_m, Z_I]$ [-].
    impurity_charge : float or array-like
        Charge $Z_I$ of the one (effective) impurity, greater than $Z_m$ [-].
    main_ion_charge : float, optional
        Main-ion charge $Z_m$, default 1 [-].

    Returns
    -------
    float or np.ndarray
        $f_\mathrm{dil}$ [-].

    Raises
    ------
    ValueError
        As :func:`main_ion_fraction_from_zeff`.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 4.25.

    See Also
    --------
    main_ion_fraction_from_zeff
    dilution_fraction_from_species
    """
    return _out(1.0 - np.asarray(main_ion_fraction_from_zeff(z_eff, impurity_charge, main_ion_charge)))


def main_ion_density_from_zeff(n_e, z_eff, impurity_charge, main_ion_charge=1.0):
    r"""Main-ion density from n_e, Z_eff and one effective impurity charge.

    $$n_\mathrm{main} = n_e\,\frac{Z_I - Z_\mathrm{eff}}{Z_m\,(Z_I - Z_m)}$$

    Parameters
    ----------
    n_e : float or array-like
        Electron density, finite and positive [m^-3].
    z_eff : float or array-like
        Plasma effective charge, in $[Z_m, Z_I]$ [-].
    impurity_charge : float or array-like
        Charge $Z_I$ of the one (effective) impurity, greater than $Z_m$ [-].
    main_ion_charge : float, optional
        Main-ion charge $Z_m$, default 1 [-].

    Returns
    -------
    float or np.ndarray
        $n_\mathrm{main}$ [m^-3].

    Raises
    ------
    ValueError
        A non-positive ``n_e``, or as :func:`main_ion_fraction_from_zeff`.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 4.25.

    See Also
    --------
    main_ion_density_from_species
    """
    ne = _finite(n_e, "n_e")
    if np.any(ne <= 0.0):
        raise ValueError("n_e must be positive")
    return _out(ne * np.asarray(main_ion_fraction_from_zeff(z_eff, impurity_charge, main_ion_charge)))


def dilution_fraction_from_species(n_e, impurity_densities, impurity_charges, main_ion_charge=1.0):
    r"""Main-ion dilution from an explicit impurity species list.

    $$f_\mathrm{dil} = 1 - \frac{n_\mathrm{main}}{n_e},\qquad
      n_\mathrm{main} = \frac{n_e - \sum_j Z_j n_j}{Z_m}
      \;\xrightarrow{Z_m = 1}\; f_\mathrm{dil} = \sum_j Z_j \frac{n_j}{n_e}$$

    Parameters
    ----------
    n_e : float or array-like
        Electron density, finite and positive [m^-3].
    impurity_densities : array-like
        Density of each impurity species along the last axis, non-negative [m^-3].
    impurity_charges : array-like
        Charge state of each species, positive [-].
    main_ion_charge : float, optional
        Main-ion charge $Z_m$, default 1 [-].

    Returns
    -------
    float or np.ndarray
        $f_\mathrm{dil}$ [-].

    Raises
    ------
    ValueError
        Invalid input, or impurities carrying more charge than there are
        electrons ($\sum_j Z_j n_j > n_e$).

    Convention
    ----------
    Preferred over :func:`dilution_fraction_from_zeff` whenever the species
    are resolved: it needs no effective charge and is exact for any mixture
    and any charge-state distribution (pass each charge state, or each
    element at its $\langle Z\rangle$ -- quasi-neutrality is linear in $Z$).

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 4.25.

    See Also
    --------
    main_ion_density_from_species
    """
    ne = _finite(n_e, "n_e")
    if np.any(ne <= 0.0):
        raise ValueError("n_e must be positive")
    main = np.asarray(main_ion_density_from_species(ne, impurity_densities, impurity_charges, main_ion_charge))
    return _out(1.0 - main / ne)


def main_ion_density_from_species(n_e, impurity_densities, impurity_charges, main_ion_charge=1.0):
    r"""Main-ion density that keeps a plasma with these impurities quasi-neutral.

    $$n_\mathrm{main} = \frac{n_e - \sum_j Z_j n_j}{Z_m}$$

    Parameters
    ----------
    n_e : float or array-like
        Electron density, finite and positive [m^-3].
    impurity_densities : array-like
        Density of each impurity species along the last axis, non-negative [m^-3].
    impurity_charges : array-like
        Charge state of each species, positive [-].
    main_ion_charge : float, optional
        Main-ion charge $Z_m$, default 1 [-].

    Returns
    -------
    float or np.ndarray
        $n_\mathrm{main}$ [m^-3].

    Raises
    ------
    ValueError
        Invalid input, or $\sum_j Z_j n_j > n_e$ (a negative main-ion density).

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 4.25.

    See Also
    --------
    dilution_fraction_from_species
    """
    zm = _main_charge(main_ion_charge)
    ne = _finite(n_e, "n_e")
    if np.any(ne <= 0.0):
        raise ValueError("n_e must be positive")
    n = _species(impurity_densities, "impurity_densities", positive=False)
    z = _species(impurity_charges, "impurity_charges", positive=True)
    n, z = np.broadcast_arrays(n, z)
    impurity_electrons = np.sum(n * z, axis=-1)
    main = (ne - impurity_electrons) / zm
    if np.any(main < -_RTOL * ne):
        raise ValueError("the impurities carry more charge than there are electrons")
    return _out(np.clip(main, 0.0, None))
