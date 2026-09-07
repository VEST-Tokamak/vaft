"""Kinetic profiles: from diagnostic points to a stored ``core_profiles`` slice.

The chain, and where each step's state changes::

    measured diagnostic points (R, Z, value)
    -> equilibrium-mapped coordinates: psi_norm, rho_pol_norm = sqrt(psi_norm),
       rho_tor_norm from the equilibrium's own Phi(psi)          [MappedPositions]
    -> coordinate-selected profile points (default rho_tor_norm)
    -> fitted callable in that coordinate                          [FittedProfile]
    -> equilibrium-grid profile, stored on grid.rho_tor_norm / psi /
       rho_pol_norm with the fit's provenance beside it           [core_profiles]

Everything here is machine-independent.  The radial coordinate to fit in,
the statistical Ti/Te ratio for a Thomson-only slice, and their provenance
are *arguments*; on VEST the pipeline
(:func:`vaft.code.efit.build_kinetic_core_profiles`) resolves them from
``vest.yaml`` through
:func:`vaft.machine_mapping.core_profiles.vest_core_profiles_policy`.  This
module holds no VEST number (issue #420).

Notation
--------
psi_norm      : normalized poloidal flux, (psi - psi_axis)/(psi_boundary - psi_axis)   [-]
rho_pol_norm  : sqrt(psi_norm)                                                          [-]
rho_tor_norm  : normalized toroidal-flux radius, sqrt(Phi/Phi_boundary), Phi = int q dpsi [-]
T_e, T_i      : electron, ion temperature                                              [eV]
n_e           : electron density                                                       [m^-3]
V_tor         : toroidal ion velocity                                                  [m/s]
p             : thermal pressure, e n_e (T_e + T_i)                                     [Pa]

Conventions
-----------
Three radial coordinates, never interchangeable: ``rho_tor_norm`` is not
``sqrt(psi_norm)`` except for a flat-``q`` cylinder.  A fit is a function
of exactly one of them and carries which; :func:`core_profiles` evaluates it
on the equilibrium grid expressed in that coordinate.  Parameter names that
still say ``rho`` (``rho_points``, the deprecated ``mapped_rho_position``)
predate the choice and mean "the fit coordinate".  Quasi-neutral single
main ion H+ with ``n_i = n_e``; all electrons thermal.

Provenance
----------
.. [FIT] :func:`vaft.formula.utils.fit_profile`, the 1-D fitting kernel every
   fitter here delegates to (polynomial, exponential, core-poly-edge-exp,
   linear, Gaussian process).
.. [TITE] ``vest.yaml`` ``diagnostics.core_profiles.ti_te_ratio``: the VEST
   statistical Ti/Te coefficient and its derivation record, resolved by
   :func:`vaft.machine_mapping.core_profiles.vest_core_profiles_policy`;
   :func:`fit_ti_te_ratio` is the estimator it was derived with.
"""

import os
import warnings
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
from uncertainties import unumpy

import vaft
from vaft.formula import fit_profile
from vaft.formula.utils import eped_tanh_bounds


__all__ = [
    "COORDINATES",
    "CoordinateUnavailableError",
    "FittedProfile",
    "LEGACY_COORDINATE",
    "MappedPositions",
    "NE_DYNAMIC_RANGE_MAX",
    "PEDESTAL_FALLBACK_PSI_NORM",
    "PEDESTAL_RESOLUTION_FACTOR",
    "PedestalTop",
    "PHYSICAL_EDGE_MAX",
    "PHYSICAL_PSIN_MAX",
    "TE_POSITIVE_EDGE_MAX",
    "TE_POSITIVE_PSIN_MAX",
    "core_profiles",
    "core_profiles_from_eq",
    "core_profiles_from_eq_ratio",
    "equilibrium_mapping_charge_exchange",
    "equilibrium_mapping_thomson_scattering",
    "export_electron_profile_txt",
    "fit_ti_te_ratio",
    "pedestal_top",
    "profile_fitting_charge_exchange",
    "profile_fitting_thomson_scattering",
]


#: The radial coordinates a profile can be fitted in, in order of preference.
#: ``rho_tor_norm`` is the IMAS convention and the default; ``rho_pol_norm``
#: is ``sqrt(psi_norm)``; ``psi_norm`` is what the mappers computed and every
#: fit consumed before the coordinate became a choice (issue #420).
COORDINATES = ("rho_tor_norm", "rho_pol_norm", "psi_norm")

#: The coordinate every fit used before issue #420, and the only meaning a
#: bare array of mapped positions can have.
LEGACY_COORDINATE = "psi_norm"


class CoordinateUnavailableError(ValueError):
    """The requested radial coordinate cannot be derived from this equilibrium.

    ``rho_tor_norm`` needs a finite, monotonic ``q`` profile on the flux grid;
    a legacy ``fluxSurfaces`` mapping carries none, and so does a GEQDSK whose
    ``QPSI`` is missing or unusable.  The message carries the reason recorded
    at mapping time, so the caller can choose ``rho_pol_norm`` knowingly rather
    than have ``psi_norm`` substituted in silence.
    """


@dataclass(frozen=True, eq=False)
class MappedPositions:
    """Diagnostic channel positions in every radial coordinate the equilibrium supports.

    One entry per channel, ``NaN`` for a channel outside the last closed flux
    surface.  ``psi_norm`` and ``rho_pol_norm`` are always present;
    ``rho_tor_norm`` is ``None`` when the equilibrium cannot supply it, and
    ``rho_tor_norm_unavailable`` then says why.  :meth:`select` is how a fitter
    asks for one coordinate and is refused with that reason.
    """

    psi_norm: np.ndarray
    rho_pol_norm: np.ndarray
    rho_tor_norm: np.ndarray | None
    rho_tor_norm_unavailable: str | None
    source: str

    @property
    def n_channels(self) -> int:
        return int(np.asarray(self.psi_norm).size)

    def available(self) -> tuple[str, ...]:
        """The coordinates this mapping can hand out, in preference order."""
        return tuple(c for c in COORDINATES if getattr(self, c) is not None)

    def select(self, coordinate: str) -> np.ndarray:
        """The channel positions in ``coordinate``, or a refusal saying why not."""
        if coordinate not in COORDINATES:
            raise ValueError(
                f"coordinate must be one of {COORDINATES}, got {coordinate!r}"
            )
        values = getattr(self, coordinate)
        if values is None:
            raise CoordinateUnavailableError(
                f"{coordinate} is not available from this equilibrium "
                f"({self.source}): {self.rho_tor_norm_unavailable}. "
                f"Available: {', '.join(self.available())}. Pass coordinate="
                f"{self.available()[0]!r} explicitly to fit in that coordinate."
            )
        return np.asarray(values, dtype=float).reshape(-1)


def _positions_for_fit(mapped, coordinate, *, legacy_name="mapped_rho_position"):
    """Resolve what a fitter was handed into ``(positions, coordinate)``.

    A :class:`MappedPositions` yields the requested coordinate.  A bare array
    is the pre-#420 calling convention: it was always ``psi_norm``, so it is
    accepted only when the caller says ``coordinate="psi_norm"`` -- making
    the meaning explicit -- and with a :class:`DeprecationWarning`; without
    that it is refused, because guessing would repeat the confusion this
    change removes.
    """
    if coordinate not in COORDINATES:
        raise ValueError(f"coordinate must be one of {COORDINATES}, got {coordinate!r}")
    if isinstance(mapped, MappedPositions):
        return mapped.select(coordinate), coordinate
    if mapped is None:
        raise ValueError("mapped positions are required")
    if coordinate != LEGACY_COORDINATE:
        raise TypeError(
            f"{legacy_name} is a bare array; the mappers now return MappedPositions, "
            f"which carries every coordinate. A bare array can only mean "
            f"{LEGACY_COORDINATE!r} (what the mappers always computed), so pass "
            f"coordinate={LEGACY_COORDINATE!r} to say so, or pass the MappedPositions "
            f"record to fit in {coordinate!r}."
        )
    warnings.warn(
        f"passing {legacy_name} as a bare array is deprecated; it is treated as "
        f"{LEGACY_COORDINATE}. Pass the MappedPositions record from "
        "equilibrium_mapping_* and select the coordinate explicitly.",
        DeprecationWarning,
        stacklevel=3,
    )
    return np.asarray(mapped, dtype=float).reshape(-1), coordinate


@dataclass(frozen=True, eq=False)
class FittedProfile:
    """A fitted 1-D profile that knows which radial coordinate it is a function of.

    Callable exactly like the plain function it wraps, so tuple-unpacking
    call sites keep working; ``coordinate`` is what :func:`core_profiles`
    reads to evaluate it on the matching equilibrium-grid array and to record
    the fit's provenance.  ``coefficients`` is ``None`` for methods without a
    closed form (Gaussian process, linear interpolation).
    """

    function: object
    coordinate: str
    method: str
    order: int | None
    coefficients: object = None
    span: tuple[float, float] | None = None

    def __call__(self, x):
        return self.function(x)

    def parameters_text(self) -> str:
        """The ``*_fit.parameters`` record: what produced this curve."""
        parts = [f"coordinate={self.coordinate}", f"method={self.method}"]
        if self.order is not None:
            parts.append(f"order={self.order}")
        if self.span is not None:
            parts.append(f"measured_span={self.span[0]:.4f}:{self.span[1]:.4f}")
        return "; ".join(parts)


def _fit_coordinate(fit, coordinate, name):
    """The coordinate a fit object is a function of, checked against ``coordinate``."""
    if isinstance(fit, FittedProfile):
        if coordinate is not None and fit.coordinate != coordinate:
            raise ValueError(
                f"{name} is a function of {fit.coordinate} but coordinate="
                f"{coordinate!r} was requested; a fit cannot be re-labelled"
            )
        return fit.coordinate
    if coordinate is None:
        raise TypeError(
            f"{name} is a plain callable; pass coordinate= to say which radial "
            "coordinate it is a function of (a FittedProfile carries its own)"
        )
    if coordinate not in COORDINATES:
        raise ValueError(f"coordinate must be one of {COORDINATES}, got {coordinate!r}")
    return coordinate


def _fit_parameters_text(fit, coordinate):
    return fit.parameters_text() if isinstance(fit, FittedProfile) else f"coordinate={coordinate}; method=external"


def fit_ti_te_ratio(te, ti, te_std=None, ti_std=None, max_iter=200, tol=1e-12):
    """Fit the proportionality coefficient ``alpha`` of ``Ti = alpha * Te``.

    Effective-variance weighted through-origin regression with errors in BOTH
    variables: minimizes ``sum_k (Ti_k - a*Te_k)^2 / (sTi_k^2 + a^2*sTe_k^2)``
    by iterating the weights.  This is the estimator the VEST statistical
    coefficient was derived with [TITE]_ (paired Thomson ``Te`` and fitted
    charge-exchange ``Ti`` at the same radial position, on the shots that carry
    both diagnostics); use it to re-derive the coefficient as more
    two-diagnostic shots become available.

    Parameters
    ----------
    te : array_like
        Electron temperature samples [eV].
    ti : array_like
        Ion temperature samples, paired with ``te`` [eV].
    te_std : array_like, optional
        1-sigma uncertainties of ``te``; ``None`` for zero [eV].
    ti_std : array_like, optional
        1-sigma uncertainties of ``ti``; ``None`` for unit weights, a plain
        least-squares fit [eV].
    max_iter : int, optional
        Iterations of the weight update [-].
    tol : float, optional
        Change in ``alpha`` below which the iteration stops [-].

    Returns
    -------
    dict
        ``alpha`` the fitted ratio; ``alpha_se`` its formal standard error,
        scaled by ``sqrt(chi2_red)`` when ``chi2_red > 1``; ``alpha_scatter``
        the error-weighted scatter of the per-point ratios, the predictive
        per-point sigma; ``chi2_red``; ``n_points`` the pairs used [-].

    Raises
    ------
    ValueError
        Mismatched lengths, or fewer than two valid pairs.

    Processing steps
    ----------------
    1. Drop pairs with a non-finite value, a non-finite or negative sigma, or
       ``Te <= 0``.
    2. Start from the unweighted through-origin estimate.
    3. Iterate ``w = 1 / (sTi^2 + a^2 sTe^2)``, ``a = sum(w Te Ti)/sum(w Te^2)``
       to convergence.
    4. Standard error, reduced chi-square, and the weighted scatter of
       ``Ti/Te``.

    Defaults
    --------
    ``max_iter = 200`` and ``tol = 1e-12`` are numerical conveniences.  A
    uniformly zero ``ti_std`` is replaced by unit weights, a numerical
    convenience that keeps the effective variance finite.

    Assumptions
    -----------
    ``Ti`` is proportional to ``Te`` with no offset, across the whole sample.

    Applicability
    -------------
    Machine-independent.  The VEST value it produced (0.17, sigma 0.08) lives in
    ``vest.yaml``, not here.

    Limitations
    -----------
    The derivation script cited by the VEST record, ``ids_test/fit_ti_te_ratio.py``,
    is not in the repository; the value is reproducible from this estimator and
    the three shots named there, but the script that did it is not tracked.

    Provenance
    ----------
    .. [1] The VEST Ti/Te policy record [TITE]_: shots 48224, 48226, 48233 at
       299-301 ms; pressure-matching estimator on the fitted 129-point profiles,
       cross-checked at 0.170 +/- 0.010 by this regression.
    """
    te = np.asarray(te, dtype=float).reshape(-1)
    ti = np.asarray(ti, dtype=float).reshape(-1)
    if te.shape != ti.shape:
        raise ValueError("te and ti must have the same length")
    ste = (np.zeros_like(te) if te_std is None
           else np.asarray(te_std, dtype=float).reshape(-1))
    sti = (np.ones_like(ti) if ti_std is None
           else np.asarray(ti_std, dtype=float).reshape(-1))
    if ste.shape != te.shape or sti.shape != ti.shape:
        raise ValueError("te_std/ti_std must match te/ti in length")

    ok = (np.isfinite(te) & np.isfinite(ti) & (te > 0)
          & np.isfinite(ste) & (ste >= 0) & np.isfinite(sti) & (sti >= 0))
    te, ti, ste, sti = te[ok], ti[ok], ste[ok], sti[ok]
    if te.size < 2:
        raise ValueError("need at least 2 valid (te, ti) pairs")
    # a zero effective variance is degenerate; floor sigma_ti like the fitters
    if np.all(sti == 0):
        sti = np.ones_like(ti)

    alpha = float(np.sum(ti * te) / np.sum(te * te))
    for _ in range(int(max_iter)):
        w = 1.0 / np.clip(sti**2 + alpha**2 * ste**2, 1e-300, None)
        alpha_new = float(np.sum(w * te * ti) / np.sum(w * te * te))
        if abs(alpha_new - alpha) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new

    w = 1.0 / np.clip(sti**2 + alpha**2 * ste**2, 1e-300, None)
    chi2_red = float(np.sum(w * (ti - alpha * te) ** 2) / max(te.size - 1, 1))
    alpha_se = float(np.sqrt(1.0 / np.sum(w * te**2))) * max(1.0, np.sqrt(chi2_red))

    ratios = ti / te
    sratios = np.sqrt((sti / te) ** 2 + (ti * ste / te**2) ** 2)
    sratios = np.where(sratios > 0, sratios, np.nanmedian(sratios[sratios > 0])
                       if np.any(sratios > 0) else 1.0)
    wr = 1.0 / sratios**2
    rmean = float(np.sum(wr * ratios) / np.sum(wr))
    alpha_scatter = float(np.sqrt(np.sum(wr * (ratios - rmean) ** 2) / np.sum(wr)))

    return {
        "alpha": alpha,
        "alpha_se": alpha_se,
        "alpha_scatter": alpha_scatter,
        "chi2_red": chi2_red,
        "n_points": int(te.size),
    }


def _has_flux_surfaces(geq) -> bool:
    """True for a legacy mapping that carries precomputed ``fluxSurfaces``.

    VAFT's own ``GEQDSK`` does not; it is read through the psi-grid branch.
    """
    try:
        return "fluxSurfaces" in geq and "levels" in geq["fluxSurfaces"]
    except Exception:
        return False


def _outermost_surface_path(geq):
    """Return a matplotlib Path around the outermost traced flux surface.

    Used to detect measurement points outside the plasma; the nearest-surface
    search would otherwise silently pin them to the edge psi_N level.
    """
    from matplotlib.path import Path as _MplPath

    levels = np.asarray(geq['fluxSurfaces']['levels'], dtype=float)
    outer = int(np.argmax(levels))
    R = np.asarray(geq['fluxSurfaces']['flux'][outer]['R'], dtype=float)
    Z = np.asarray(geq['fluxSurfaces']['flux'][outer]['Z'], dtype=float)
    return _MplPath(np.column_stack([R, Z]))


def _psi_norm_from_equilibrium_points(geq, r_points, z_points):
    """Map R/Z points to normalized poloidal flux; returns ``(psi_norm, source)``.

    Points outside the last closed flux surface are returned as NaN rather than
    pinned to the edge psi_N level.  ``source`` names which kind of equilibrium
    answered: ``legacy_flux_surfaces``, ``geqdsk`` or ``omas``.
    """
    # Compatibility with legacy precomputed flux-surface dictionaries. This is
    # keyed on the mapping actually carrying 'fluxSurfaces' rather than on a
    # bare except, so a genuine failure inside the nearest-surface search
    # surfaces instead of silently falling through to the psi-grid branch.
    if _has_flux_surfaces(geq):
        flux_levels = geq['fluxSurfaces']['levels']
        boundary = _outermost_surface_path(geq)
        mapped = []
        for r_dot, z_dot in zip(r_points, z_points):
            if not boundary.contains_point((float(r_dot), float(z_dot))):
                mapped.append(np.nan)
                continue
            min_dist = float('inf')
            closest_rho = None
            for i in range(len(geq['fluxSurfaces']['flux'])):
                R = np.asarray(geq['fluxSurfaces']['flux'][i]['R'], dtype=float)
                Z = np.asarray(geq['fluxSurfaces']['flux'][i]['Z'], dtype=float)
                dists = np.sqrt((R - r_dot) ** 2 + (Z - z_dot) ** 2)
                min_flux_dist = np.min(dists)
                if min_flux_dist < min_dist:
                    min_dist = min_flux_dist
                    closest_rho = flux_levels[i]
            mapped.append(closest_rho)
        return np.clip(np.asarray(mapped, dtype=float), 0.0, 1.0), "legacy_flux_surfaces"

    source = "geqdsk"
    try:
        nw = int(geq['NW'])
        nh = int(geq['NH'])
        r_grid = np.linspace(0.0, float(geq['RDIM']), nw) + float(geq['RLEFT'])
        z_grid = np.linspace(0.0, float(geq['ZDIM']), nh) - float(geq['ZDIM']) / 2.0 + float(geq['ZMID'])
        psi = np.asarray(geq['PSIRZ'], dtype=float).reshape(nw, nh)
        psi_axis = float(geq['SIMAG'])
        psi_boundary = float(geq['SIBRY'])
    except Exception:
        source = "omas"
        try:
            ts = geq['equilibrium.time_slice.0'] if 'equilibrium.time_slice.0' in geq else geq['equilibrium.time_slice'][0]
            prof2d = ts['profiles_2d.0']
            r_grid = np.asarray(prof2d['grid.dim1'], dtype=float)
            z_grid = np.asarray(prof2d['grid.dim2'], dtype=float)
            # The DD declares psi(:,:)'s coordinates as [grid.dim1, grid.dim2],
            # i.e. axis 0 = dim1 (R) and axis 1 = dim2 (Z). A shape-based
            # "is this secretly transposed?" heuristic is ambiguous whenever
            # dim1 and dim2 have the same length (VEST's EFIT/CHEASE grids
            # always are: 129x129, 513x513) and silently transposes psi that
            # was already written correctly. Trust the DD convention
            # unconditionally (see vaft.data.eqdsk.from_omas).
            psi = np.asarray(prof2d['psi'], dtype=float)
            psi_axis = float(ts['global_quantities.psi_axis'])
            psi_boundary = float(ts['global_quantities.psi_boundary'])
        except Exception as exc:
            raise ValueError("geq must be a VAFT GEQDSK, OMAS equilibrium ODS, or legacy fluxSurfaces mapping") from exc

    interp = RegularGridInterpolator((r_grid, z_grid), psi, bounds_error=False, fill_value=np.nan)
    points = np.column_stack([np.asarray(r_points, dtype=float), np.asarray(z_points, dtype=float)])
    psi_points = interp(points)
    rho = (psi_points - psi_axis) / (psi_boundary - psi_axis) if psi_boundary != psi_axis else np.zeros_like(psi_points)
    # Outside the LCFS (rho > 1) -> NaN rather than pinning to the edge psi_N level.
    rho = np.where(rho > 1.0, np.nan, rho)
    return np.clip(rho, 0.0, 1.0), source


def _coordinates_from_psi_norm(geq, psi_norm, source):
    """Every radial coordinate for channel ``psi_norm`` values, from the same equilibrium.

    ``rho_pol_norm = sqrt(psi_norm)`` by definition.  ``rho_tor_norm`` is
    interpolated from the equilibrium's own ``rho_tor_norm(psi_norm)`` table,
    derived by :func:`vaft.process._equilibrium_parametric.derive_radial_coordinates`
    from the ``q`` profile (``Phi = int q dpsi``); when that derivation cannot
    be made the record says why instead of substituting ``sqrt(psi_norm)``,
    which is a different coordinate.
    """
    psi_norm = np.asarray(psi_norm, dtype=float).reshape(-1)
    rho_pol = np.sqrt(np.clip(psi_norm, 0.0, 1.0))
    rho_pol = np.where(np.isfinite(psi_norm), rho_pol, np.nan)
    rho_tor = None
    reason = None
    if source == "legacy_flux_surfaces":
        reason = "a legacy fluxSurfaces mapping carries no q profile"
    else:
        try:
            from ._equilibrium_parametric import derive_radial_coordinates

            derived = derive_radial_coordinates(geq)
        except Exception as exc:  # noqa: BLE001 -- the reason is what matters
            reason = f"the equilibrium could not be adapted for rho_tor_norm ({exc})"
        else:
            table = derived["rho_tor_n"]
            if table.value is None:
                reason = str(getattr(table, "reason", None) or "rho_tor_norm could not be derived")
            else:
                x = np.asarray(derived["psi_n"].value, dtype=float).reshape(-1)
                y = np.asarray(table.value, dtype=float).reshape(-1)
                order = np.argsort(x)
                finite = np.isfinite(psi_norm)
                rho_tor = np.full(psi_norm.shape, np.nan)
                rho_tor[finite] = np.interp(psi_norm[finite], x[order], y[order])
    return MappedPositions(psi_norm, rho_pol, rho_tor, reason, source)


def equilibrium_mapping_thomson_scattering(ods, geq):
    """Map Thomson scattering channel positions into the equilibrium's radial coordinates.

    Parameters
    ----------
    ods : ODS
        Carries ``thomson_scattering.channel[:].position.r`` and ``.z`` [-].
    geq : GEQDSK, ODS or mapping
        A ``vaft.data.GEQDSK``, an OMAS equilibrium ODS (first time slice), or
        a legacy ``fluxSurfaces`` mapping [-].

    Returns
    -------
    MappedPositions
        ``psi_norm``, ``rho_pol_norm`` and, when the equilibrium carries a
        usable ``q``, ``rho_tor_norm`` for every channel; NaN outside the LCFS
        [-].

    Processing steps
    ----------------
    1. Interpolate ``psi(R, Z)`` at each channel and normalize to
       ``psi_norm = (psi - psi_axis)/(psi_boundary - psi_axis)``; a channel
       with ``psi_norm > 1`` is outside the plasma and becomes NaN.
    2. ``rho_pol_norm = sqrt(psi_norm)``.
    3. ``rho_tor_norm`` from the equilibrium's ``rho_tor_norm(psi_norm)``
       table (``Phi = int q dpsi``), interpolated at each channel; recorded
       as unavailable, with the reason, when ``q`` cannot supply it.

    Input semantics
    ---------------
    Measured: diagnostic-native ``(R, Z)`` channel positions.

    Output semantics
    ----------------
    Equilibrium-mapped, in three coordinates at once; the fitter selects one.

    Convention
    ----------
    ``psi_norm`` is normalized *poloidal* flux; ``rho_tor_norm`` is the
    normalized toroidal-flux radius and is **not** ``sqrt(psi_norm)`` except
    for a flat-``q`` cylinder.  Before issue #420 this function returned a
    bare ``psi_norm`` array under the name "rho".

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    ``rho_tor_norm`` is unavailable for a legacy ``fluxSurfaces`` mapping and
    for a GEQDSK or ODS whose ``q`` is missing, non-finite or gives a
    non-monotonic toroidal flux; a fit requested in that coordinate then
    raises :class:`CoordinateUnavailableError` rather than falling back.
    Only the first equilibrium time slice of an ODS is used.
    """
    r_t = ods['thomson_scattering.channel.:.position.r']
    z_t = ods['thomson_scattering.channel.:.position.z']
    psi_norm, source = _psi_norm_from_equilibrium_points(geq, r_t, z_t)
    return _coordinates_from_psi_norm(geq, psi_norm, source)


def profile_fitting_thomson_scattering(
    ods,
    time_ms,
    mapped_positions=None,
    Te_order=3,
    Ne_order=3,
    uncertainty_option=1,
    rho_points=100,
    fitting_function_te='polynomial',
    fitting_function_ne='polynomial',
    time_tolerance_ms=1.0,
    enforce_physical=True,
    *,
    coordinate="rho_tor_norm",
    mapped_rho_position=None,
    ):
    """Fit Thomson-scattering ``T_e`` and ``n_e`` profiles in a chosen radial coordinate.

    Extracts the channel values at the nearest Thomson time, drops unusable
    channels, and fits ``T_e(x)`` and ``n_e(x)`` on ``x in [0, 1]`` with the
    selected model, where ``x`` is the radial coordinate chosen from the mapping
    -- ``rho_tor_norm`` by default.  With ``enforce_physical`` a fit that goes
    non-positive or collapses inside the LCFS is retried at lower order.

    Parameters
    ----------
    ods : ODS
        Carries ``thomson_scattering.time`` and per-channel ``t_e`` / ``n_e``
        data with ``data_error_upper`` [-].
    time_ms : float
        Time to fit at; the nearest Thomson sample within ``time_tolerance_ms``
        [ms].
    mapped_positions : MappedPositions or np.ndarray, optional
        Channel positions from :func:`equilibrium_mapping_thomson_scattering`.  A
        bare array is the deprecated pre-#420 form, accepted only with
        ``coordinate="psi_norm"`` [-].
    Te_order : int, optional
        Polynomial order for ``T_e`` [-].
    Ne_order : int, optional
        Polynomial order for ``n_e`` [-].
    uncertainty_option : int, optional
        ``1`` weights the fit by the per-channel uncertainties [-].
    rho_points : int, optional
        Points of the uniform evaluation grid the sampled returns are on [-].
    fitting_function_te : str, optional
        ``'polynomial'``, ``'exponential'``, ``'core_poly_edge_exp'``,
        ``'linear'`` or ``'gp'`` for ``T_e`` [-].
    fitting_function_ne : str, optional
        The same choice for ``n_e`` [-].
    time_tolerance_ms : float, optional
        Largest distance to the nearest Thomson sample [ms].
    enforce_physical : bool, optional
        Retry at lower order until the profile is physical inside the LCFS [-].
    coordinate : str, optional
        Which coordinate of the mapping to fit in: ``rho_tor_norm``,
        ``rho_pol_norm`` or ``psi_norm`` [-].
    mapped_rho_position : MappedPositions or np.ndarray, optional
        Deprecated name for ``mapped_positions`` [-].

    Returns
    -------
    n_e_function : FittedProfile
        ``n_e(x)``, clipped non-negative, with its coordinate and method [m^-3].
    T_e_function : FittedProfile
        ``T_e(x)``, clipped non-negative [eV].
    coeffs_ne : np.ndarray or None
        Fit coefficients; ``None`` for the Gaussian-process and linear methods [-].
    coeffs_te : np.ndarray or None
        As ``coeffs_ne`` [-].
    n_e_rho : np.ndarray
        ``n_e`` sampled on ``rho_points`` uniform points of ``x`` [m^-3].
    T_e_rho : np.ndarray
        ``T_e`` sampled on the same grid [eV].

    Raises
    ------
    ValueError
        No Thomson sample within tolerance; an unknown ``coordinate``.
    TypeError
        A bare position array without ``coordinate="psi_norm"``.
    CoordinateUnavailableError
        The mapping cannot supply ``coordinate``.

    Processing steps
    ----------------
    1. Select the channel positions in ``coordinate`` from the mapping.
    2. Read ``T_e``, ``n_e`` and their uncertainties at the nearest time.
    3. Drop channels with a non-finite value, a non-finite or zero sigma, or an
       unmapped (NaN) position; floor the surviving sigmas at ``1e-3`` of the
       largest value.
    4. Fit ``T_e(x)`` with ``fitting_function_te``; with ``enforce_physical``,
       reduce the order until ``T_e > 0`` for ``x < PHYSICAL_EDGE_MAX``.
    5. Fit ``n_e(x)`` on ``n_e / 1e18``; with ``enforce_physical``, reduce the
       order until ``n_e > 0`` and its dynamic range inside the LCFS is below
       ``NE_DYNAMIC_RANGE_MAX``.
    6. Wrap both as :class:`FittedProfile` carrying ``coordinate``, method and
       the order actually used.

    Input semantics
    ---------------
    Measured, at diagnostic-native channels; equilibrium-mapped positions in
    the selected coordinate.

    Output semantics
    ----------------
    Fitted: a continuous function of the selected coordinate, plus its sampled
    form.  Not yet on an equilibrium grid -- that is :func:`core_profiles`.

    Defaults
    --------
    ``coordinate = "rho_tor_norm"`` is the IMAS convention; a VEST pipeline
    takes it from ``vest.yaml``.  ``Te_order = Ne_order = 3`` are legacy
    compatibility values.  ``uncertainty_option = 1`` and
    ``enforce_physical = True`` are validated-workflow defaults; the guard's
    ``PHYSICAL_EDGE_MAX = 0.98`` and ``NE_DYNAMIC_RANGE_MAX = 1e3`` are empirical
    estimates from shot 48224 at 299 ms, where a cubic extrapolated from five
    channels inside ``psi_N <= 0.27`` crossed zero at ``psi_N = 0.87`` and
    collapsed ``n_e`` by ``5e6``.  ``rho_points = 100`` and
    ``time_tolerance_ms = 1.0`` are numerical conveniences.

    Convention
    ----------
    The fit, its ``(1 - x)`` edge factor in the polynomial and exponential
    bases, and the physicality checks are all in the selected coordinate; a
    cubic in ``psi_norm`` is not a cubic in ``rho_tor_norm``, and the edge
    factor lands at a different physical radius.  Before issue #420 every fit
    was in ``psi_norm`` while the docstring said ``rho_tor_norm``.

    Assumptions
    -----------
    The profile is a smooth function of one radial coordinate: channel
    positions inside the LCFS are on flux surfaces and ``T_e``, ``n_e`` are
    flux functions.

    Applicability
    -------------
    Machine-independent.  The coordinate convention is the pipeline's to
    supply; for VEST it is resolved from ``vest.yaml``.

    Limitations
    -----------
    Thomson often covers only the inner part of the profile; every model
    extrapolates over the rest and the guard can only catch the gross failures.
    ``'gp'`` imports scikit-learn on first use (#426).  A slice with fewer
    channels than the order plus one cannot be fitted at that order and the
    guard steps down; with ``enforce_physical=False`` it raises from the
    kernel.

    Provenance
    ----------
    .. [1] The fitting kernel [FIT]_; the physicality guard and its constants
       from the 48224 failures (:data:`PHYSICAL_EDGE_MAX`).
    """
    mapped_positions = _legacy_keyword(mapped_positions, mapped_rho_position, "mapped_rho_position")
    rho_flat, coordinate = _positions_for_fit(mapped_positions, coordinate)

    # --- Extract Thomson data (nearest time within tolerance) ---
    times = np.asarray(ods['thomson_scattering.time'], dtype=float)
    target_s = time_ms / 1e3
    time_index = int(np.argmin(np.abs(times - target_s)))
    if abs(times[time_index] - target_s) > time_tolerance_ms / 1e3:
        raise ValueError(
            f"No Thomson time within {time_tolerance_ms} ms of {time_ms} ms "
            f"(nearest: {times[time_index] * 1e3:.3f} ms)"
        )

    num_channels = len(ods['thomson_scattering.channel'])
    t_e, n_e, t_e_std, n_e_std = [], [], [], []

    for i in range(num_channels):
        ch = ods['thomson_scattering.channel'][i]
        t_e.append(ch['t_e.data'][time_index])
        n_e.append(ch['n_e.data'][time_index])
        t_e_std.append(ch['t_e.data_error_upper'][time_index])
        n_e_std.append(ch['n_e.data_error_upper'][time_index])

    t_e = np.array(t_e, dtype=float)
    n_e = np.array(n_e, dtype=float)
    t_e_std = np.array(t_e_std, dtype=float)
    n_e_std = np.array(n_e_std, dtype=float)
    # --- drop invalid channels: non-finite values/sigmas, zero sigma, unmapped rho ---
    valid = (
        np.isfinite(rho_flat)
        & np.isfinite(t_e) & np.isfinite(n_e)
        & np.isfinite(t_e_std) & (t_e_std > 0)
        & np.isfinite(n_e_std) & (n_e_std > 0)
    )
    if not np.all(valid):
        print(
            f"[INFO] dropped {int(np.sum(~valid))} invalid TS channel(s) "
            f"at {time_ms:.3f} ms (non-finite value/sigma or unmapped position)"
        )
    t_e, n_e = t_e[valid], n_e[valid]
    t_e_std, n_e_std = t_e_std[valid], n_e_std[valid]
    rho = np.clip(rho_flat[valid].reshape(-1, 1), 0, 1)

    # relative sigma floors (in fit space) so tiny-but-valid sigmas cannot
    # dominate the weighted fit
    t_e_std = np.maximum(t_e_std, 1e-3 * np.max(np.abs(t_e)))

    # density normalization (floor applied AFTER normalization)
    n_e_scale = 1e18
    n_e_norm = n_e / n_e_scale
    n_e_std_norm = np.maximum(n_e_std / n_e_scale, 1e-3 * np.max(np.abs(n_e_norm)))
    rho_eval = np.linspace(0, 1, rho_points)

    # --- Te / Ne FITS ---
    te_anchor_strength = None
    te_anchor = None
    if te_anchor_strength is not None:
        te_anchor = (np.array([1.0]), np.array([0.0]), np.array([te_anchor_strength]))

    # Physicality guard grid (independent of rho_points so the check is stable).
    guard_grid = np.linspace(0.0, 1.0, 129)

    def _te_unphysical(fn):
        y = np.asarray(fn(guard_grid), dtype=float)
        inside = guard_grid < PHYSICAL_EDGE_MAX
        if not np.all(np.isfinite(y)):
            return "non-finite Te"
        if np.any(y[inside] <= 0.0):
            first = float(guard_grid[inside][np.argmax(y[inside] <= 0.0)])
            return f"Te<=0 inside the LCFS (first at {coordinate}={first:.2f})"
        return None

    T_e_rho, T_e_std, T_e_function_raw, coeffs_te, Te_order_used = (
        _fit_profile_until_physical(
            lambda o: fit_profile(
                rho, t_e, t_e_std, rho_eval,
                order=o,
                uncertainty_option=uncertainty_option,
                fitting_function=fitting_function_te,
                gp_anchor=te_anchor,
            ),
            Te_order, _te_unphysical, "Te", time_ms,
        )
        if enforce_physical else
        (*fit_profile(
            rho, t_e, t_e_std, rho_eval,
            order=Te_order, uncertainty_option=uncertainty_option,
            fitting_function=fitting_function_te, gp_anchor=te_anchor,
        ), Te_order)
    )

    T_e_rho = np.maximum(np.asarray(T_e_rho, dtype=float), 0.0)

    def T_e_function(rho_input):
        x = np.clip(np.asarray(rho_input, float), 0, 1)
        return np.maximum(T_e_function_raw(x), 0.0)

    ne_anchor = None
    if fitting_function_ne.lower() == 'gp':
        ne_typ = np.nanmedian(n_e_norm[n_e_norm > 0]) if np.any(n_e_norm > 0) else 1.0
        ne_anchor_sigma_norm = max(0.01 * ne_typ, 1e-4)
        ne_anchor = (np.array([1.0]), np.array([0.0]), np.array([ne_anchor_sigma_norm]))

    def _ne_unphysical(fn):
        y = np.asarray(fn(guard_grid), dtype=float)
        if not np.all(np.isfinite(y)):
            return "non-finite ne"
        inside = y[guard_grid < PHYSICAL_EDGE_MAX]
        lo, hi = float(np.min(inside)), float(np.max(inside))
        if lo <= 0.0:
            return "ne<=0 inside the LCFS"
        if hi / lo > NE_DYNAMIC_RANGE_MAX:
            return f"ne dynamic range {hi / lo:.1e} inside the LCFS"
        return None

    n_e_rho_norm, n_e_std_norm_fit, n_e_function_raw, coeffs_ne, Ne_order_used = (
        _fit_profile_until_physical(
            lambda o: fit_profile(
                rho, n_e_norm, n_e_std_norm, rho_eval,
                order=o,
                uncertainty_option=uncertainty_option,
                fitting_function=fitting_function_ne,
                gp_anchor=ne_anchor,
            ),
            Ne_order, _ne_unphysical, "ne", time_ms,
        )
        if enforce_physical else
        (*fit_profile(
            rho, n_e_norm, n_e_std_norm, rho_eval,
            order=Ne_order, uncertainty_option=uncertainty_option,
            fitting_function=fitting_function_ne, gp_anchor=ne_anchor,
        ), Ne_order)
    )

    n_e_rho = np.maximum(n_e_rho_norm, 0.0) * n_e_scale
    n_e_std = n_e_std_norm_fit * n_e_scale

    def n_e_function(rho_input):
        x = np.clip(np.asarray(rho_input, float), 0, 1)
        y_norm = n_e_function_raw(x)
        return np.maximum(y_norm, 0.0) * n_e_scale

    n_e_fit = FittedProfile(n_e_function, coordinate, str(fitting_function_ne), int(Ne_order_used), coeffs_ne)
    T_e_fit = FittedProfile(T_e_function, coordinate, str(fitting_function_te), int(Te_order_used), coeffs_te)
    return n_e_fit, T_e_fit, coeffs_ne, coeffs_te, n_e_rho, T_e_rho


def equilibrium_mapping_charge_exchange(ods, geq):
    """Map charge-exchange channel positions into the equilibrium's radial coordinates.

    The charge-exchange twin of :func:`equilibrium_mapping_thomson_scattering`:
    the same mapping, from ``charge_exchange.channel[:].position.(r, z)``.

    Parameters
    ----------
    ods : ODS
        Carries the channel positions, as ``.data`` leaves (what the VEST
        mapper writes) or as bare values [-].
    geq : GEQDSK, ODS or mapping
        A ``vaft.data.GEQDSK``, an OMAS equilibrium ODS (first time slice), or
        a legacy ``fluxSurfaces`` mapping [-].

    Returns
    -------
    MappedPositions
        ``psi_norm``, ``rho_pol_norm`` and, when derivable, ``rho_tor_norm``
        per channel; NaN outside the LCFS [-].

    Processing steps
    ----------------
    1. Read each channel's ``(R, Z)``, taking the nominal value of an
       uncertainty-carrying leaf and the first sample of a time series.
    2. As :func:`equilibrium_mapping_thomson_scattering`.

    Input semantics
    ---------------
    Measured: diagnostic-native ``(R, Z)`` channel positions.

    Output semantics
    ----------------
    Equilibrium-mapped, in three coordinates at once.

    Convention
    ----------
    As :func:`equilibrium_mapping_thomson_scattering`: ``psi_norm`` is
    normalized poloidal flux and ``rho_tor_norm`` is not its square root.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    As :func:`equilibrium_mapping_thomson_scattering`.  A channel whose
    position is a time series is mapped at its first sample only.
    """
    def _to_float_scalar(x):
        try:
            x = unumpy.nominal_values(x)
        except Exception:
            pass

        arr = np.asarray(x)
        if arr.size == 0:
            return float("nan")

        if arr.dtype.kind in {"U", "S", "O"}:
            try:
                arr = arr.astype(float)
            except Exception:
                return float(str(arr.reshape(-1)[0]))
        else:
            arr = arr.astype(float, copy=False)

        return float(arr.reshape(-1)[0])

    # Prefer OMAS `.data` leaves (what CES machine-mapping writes), fall back for compatibility.
    try:
        R_ce = ods["charge_exchange.channel.:.position.r.data"]
        Z_ce = ods["charge_exchange.channel.:.position.z.data"]
    except Exception:
        R_ce = ods["charge_exchange.channel.:.position.r"]
        Z_ce = ods["charge_exchange.channel.:.position.z"]

    r_vals = [_to_float_scalar(value) for value in R_ce]
    z_vals = [_to_float_scalar(value) for value in Z_ce]
    psi_norm, source = _psi_norm_from_equilibrium_points(geq, r_vals, z_vals)
    return _coordinates_from_psi_norm(geq, psi_norm, source)


#: A fitted electron profile is rejected when it goes non-positive, or when the
#: density spans more than :data:`NE_DYNAMIC_RANGE_MAX`, anywhere *inside* the
#: LCFS. Both signal a polynomial/exponential extrapolated far outside the
#: measured span -- e.g. shot 48224 @ 299 ms, where Thomson covers only
#: psi_N <= 0.27 and the quadratic crossed zero at psi_N = 0.87 with ne
#: collapsing by 5e6.
#:
#: The checks stop at :data:`PHYSICAL_EDGE_MAX` because the 'polynomial' and
#: 'exponential' bases carry a ``(1 - x)`` factor that drives the profile to
#: exactly 0 at ``x = 1`` by construction, in whichever radial coordinate
#: ``x`` the fit is made. That endpoint zero is intended, and including it
#: would reject every such fit (and make any dynamic-range ratio
#: meaningless). The value 0.98 was chosen with the fits in psi_N (issue
#: #420 made the coordinate a choice); it is a cut on the normalized radius
#: the fit is in, not a physical location.
PHYSICAL_EDGE_MAX = 0.98
TE_POSITIVE_EDGE_MAX = PHYSICAL_EDGE_MAX
PHYSICAL_PSIN_MAX = PHYSICAL_EDGE_MAX          # pre-#420 name, kept as an alias
TE_POSITIVE_PSIN_MAX = PHYSICAL_EDGE_MAX       # pre-#420 name, kept as an alias
NE_DYNAMIC_RANGE_MAX = 1.0e3


def _legacy_keyword(value, legacy_value, legacy_name):
    """Accept a renamed keyword for one cycle, with a warning."""
    if legacy_value is None:
        return value
    if value is not None:
        raise TypeError(f"pass either the new keyword or {legacy_name}, not both")
    warnings.warn(
        f"{legacy_name}= is deprecated; the parameter is now named without 'rho' "
        "because the coordinate is a choice, not always psi_N",
        DeprecationWarning,
        stacklevel=3,
    )
    return legacy_value


def _fit_profile_until_physical(fit_call, order, is_unphysical, label, time_ms,
                                min_order=1):
    """Call ``fit_call(order)`` reducing the order until the profile is physical.

    A low-order fit that stays physical is far more trustworthy than a
    high-order one that has to be extrapolated over most of the profile, so on
    rejection the order is reduced by one and the fit retried. If no order
    passes, the lowest-order attempt is returned with a warning (never raises --
    the caller still gets a usable profile).

    ``min_order`` goes down to 1 because order 1 is the guaranteed-physical
    fallback: with the ``(1 - psi_N)`` bases it is a single coefficient, i.e. a
    monotonically decreasing non-negative profile that cannot cross zero inside
    the LCFS. Callers that already request order 2 (the electron-only pipeline
    branch) would otherwise have no room to reduce at all.

    Args:
        fit_call: ``order -> (y_eval, y_std, function, coeffs)`` (a fit_profile call).
        order: starting (highest) order.
        is_unphysical: ``function -> reason str or None``.
        label, time_ms: for the log messages.
        min_order: lowest order to try.

    Returns:
        ``(y_eval, y_std, function, coeffs, order_used)``
    """
    lowest = None
    for candidate in range(int(order), int(min_order) - 1, -1):
        try:
            result = fit_call(candidate)
        except Exception as exc:  # noqa: BLE001 -- try a lower order before giving up
            print(f"[INFO] {label} fit order {candidate} failed at {time_ms:.3f} ms ({exc})")
            continue
        reason = is_unphysical(result[2])
        lowest = (result, candidate, reason)
        if reason is None:
            if candidate != order:
                print(
                    f"[INFO] {label} at {time_ms:.3f} ms: order {order} rejected, "
                    f"using order {candidate} (few/narrow measurement points)"
                )
            return (*result, candidate)
    if lowest is None:
        raise RuntimeError(f"{label} fit failed at every order at {time_ms:.3f} ms")
    result, candidate, reason = lowest
    print(
        f"[WARNING] {label} at {time_ms:.3f} ms: no order in "
        f"[{min_order}, {order}] gave a physical profile ({reason}); "
        f"keeping order {candidate}"
    )
    return (*result, candidate)


def _filter_channels_for_fit(values, sigmas, rho, label, time_ms):
    """Drop channels with non-finite values/rho; handle invalid sigmas.

    Channels whose sigma is non-finite or <= 0 are dropped, unless NO channel
    has a valid sigma (e.g. data stored without uncertainties) — then all
    value-valid channels are kept with a uniform relative sigma. Surviving
    sigmas are floored at 1e-3 * max|value| so a tiny-but-valid sigma cannot
    dominate the weighted fit.
    """
    values = np.asarray(values, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    rho = np.asarray(rho, dtype=float).reshape(-1)

    valid = np.isfinite(rho) & np.isfinite(values)
    sigma_ok = np.isfinite(sigmas) & (sigmas > 0)
    if np.any(valid & sigma_ok):
        valid &= sigma_ok
    else:
        sigmas = np.full_like(values, np.nan)  # replaced by the floor below

    dropped = int(np.sum(~valid))
    if dropped:
        print(
            f"[INFO] dropped {dropped} invalid {label} channel(s) at {time_ms:.3f} ms"
        )

    values, sigmas, rho = values[valid], sigmas[valid], rho[valid]
    floor = 1e-3 * float(np.max(np.abs(values))) if values.size else 1.0
    if not np.isfinite(floor) or floor <= 0:
        floor = 1.0
    sigmas = np.where(np.isfinite(sigmas) & (sigmas > 0), sigmas, floor)
    sigmas = np.maximum(sigmas, floor)
    return values, sigmas, rho


def _leaf_values_and_errors(node, time_index, clamp=False):
    """Return (nominal, sigma) scalars for an OMAS signal leaf at ``time_index``.

    OMAS (>=0.94.2) splits an assigned uarray immediately into ``<leaf>.data``
    (nominal) and ``<leaf>.data_error_upper``, so the re-read ``.data`` carries
    NO uncertainty and ``unumpy.std_devs(<leaf>.data)`` is ALL ZEROS. Read the
    stored ``.data_error_upper`` explicitly (mirroring the Thomson path) to
    recover the real per-channel sigma; fall back to any uncertainty still
    attached to ``.data``, else 0.

    ``node`` is the signal sub-node itself (e.g. ``ods[...ion.0.t_i]``).
    With ``clamp=True`` an out-of-range ``time_index`` is clamped to the last
    sample instead of raising (used for best-effort metadata extraction).
    """
    data = node['data']
    try:
        vals = np.asarray(unumpy.nominal_values(data), dtype=float)
    except Exception:
        vals = np.asarray(data, dtype=float)

    errs = None
    try:
        errs = np.asarray(unumpy.nominal_values(node['data_error_upper']), dtype=float)
    except Exception:
        try:
            errs = np.asarray(unumpy.std_devs(data), dtype=float)
        except Exception:
            errs = None
    if errs is None or errs.shape != vals.shape:
        errs = np.zeros_like(vals)

    if vals.ndim == 0:
        return float(vals), float(np.abs(errs))

    idx = int(time_index)
    if idx < 0 or idx >= vals.size:
        if clamp:
            idx = min(max(idx, 0), vals.size - 1)
        else:
            raise IndexError("charge_exchange signal .data shorter than time base")
    return float(vals[idx]), float(np.abs(errs[idx]))


def _sanitize_std(sigmas):
    """Replace 0 / NaN sigmas with the median of the valid ones.

    Even after :func:`_leaf_values_and_errors` recovers the real errors, a
    handful of channels can carry an exactly-zero or NaN sigma. A single ~0
    sigma explodes its ``1/sigma**2`` weight (~1e14) and lets one channel
    dominate the weighted fit. Substituting the median of the valid sigmas keeps
    every channel informative without a single-channel blow-up. If NO sigma is
    usable, fall back to a uniform sigma (an honestly unweighted fit).
    """
    sigmas = np.asarray(sigmas, dtype=float).copy()
    valid = np.isfinite(sigmas) & (sigmas > 0)
    if not np.any(valid):
        sigmas[:] = 1.0
        return sigmas
    med = float(np.median(sigmas[valid]))
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    sigmas[~valid] = med
    return sigmas


def profile_fitting_charge_exchange(
    ods,
    time_ms,
    mapped_positions=None,
    Ti_order=3,
    Vtor_order=3,
    uncertainty_option=1,
    rho_points=100,
    fitting_function_ti='polynomial',
    fitting_function_vtor='polynomial',
    ion_index=0,
    time_tolerance_ms=1.0,
    clamp_to_measured_span=True,
    *,
    coordinate="rho_tor_norm",
    mapped_rho_position=None,
):
    """Fit charge-exchange ``T_i`` and ``V_tor`` profiles in a chosen radial coordinate.

    The ion twin of :func:`profile_fitting_thomson_scattering`, reading
    ``charge_exchange.channel[:].ion[ion_index].t_i`` and ``velocity_tor``.  By
    default the fitted curves are held at their value at the innermost and
    outermost measured positions rather than extrapolated beyond them.

    Parameters
    ----------
    ods : ODS
        Carries ``charge_exchange.time`` and per-channel ion ``t_i`` and
        ``velocity_tor`` data with ``data_error_upper`` [-].
    time_ms : float
        Time to fit at [ms].
    mapped_positions : MappedPositions or np.ndarray, optional
        Channel positions from :func:`equilibrium_mapping_charge_exchange`; a
        bare array only with ``coordinate="psi_norm"`` [-].
    Ti_order : int, optional
        Polynomial order for ``T_i`` [-].
    Vtor_order : int, optional
        Polynomial order for ``V_tor`` [-].
    uncertainty_option : int, optional
        ``1`` weights the fit by the per-channel uncertainties [-].
    rho_points : int, optional
        Points of the uniform evaluation grid the sampled returns are on [-].
    fitting_function_ti : str, optional
        Model for ``T_i``, as the Thomson fitter's choices [-].
    fitting_function_vtor : str, optional
        Model for ``V_tor`` [-].
    ion_index : int, optional
        Which ion of each channel [-].
    time_tolerance_ms : float, optional
        Largest distance to the nearest charge-exchange sample [ms].
    clamp_to_measured_span : bool, optional
        Hold the fit at its end values outside the measured span [-].
    coordinate : str, optional
        Which coordinate of the mapping to fit in [-].
    mapped_rho_position : MappedPositions or np.ndarray, optional
        Deprecated name for ``mapped_positions`` [-].

    Returns
    -------
    Vtor_function : FittedProfile
        ``V_tor(x)`` [m/s].
    Ti_function : FittedProfile
        ``T_i(x)``, clipped non-negative [eV].
    coeffs_vtor : np.ndarray or None
        Fit coefficients, ``None`` for methods without them [-].
    coeffs_ti : np.ndarray or None
        As ``coeffs_vtor`` [-].
    Vtor_rho : np.ndarray
        ``V_tor`` sampled on ``rho_points`` uniform points of ``x`` [m/s].
    Ti_rho : np.ndarray
        ``T_i`` sampled on the same grid [eV].

    Raises
    ------
    ValueError
        No charge-exchange sample within tolerance; a non-1-D time axis; an
        unknown ``coordinate``.
    TypeError
        A bare position array without ``coordinate="psi_norm"``.
    CoordinateUnavailableError
        The mapping cannot supply ``coordinate``.

    Processing steps
    ----------------
    1. Select the channel positions in ``coordinate``.
    2. Read ``T_i`` and ``V_tor`` with their real per-channel sigmas from the
       ``data_error_upper`` leaves (OMAS strips the uncertainty from the re-read
       ``.data``).
    3. Drop channels with a non-finite value or position; replace zero or
       non-finite sigmas by the median valid sigma so no channel's weight
       explodes.
    4. Fit each with its model; clip ``T_i`` non-negative.
    5. With ``clamp_to_measured_span``, hold each fit at its value on the
       nearest measured end outside the span the channels cover.

    Input semantics
    ---------------
    Measured, at diagnostic-native channels; equilibrium-mapped positions in
    the selected coordinate.

    Output semantics
    ----------------
    Fitted: continuous functions of the selected coordinate, span-clamped.

    Defaults
    --------
    ``coordinate = "rho_tor_norm"`` is the IMAS convention.  ``Ti_order =
    Vtor_order = 3`` are legacy compatibility values; ``uncertainty_option = 1``
    and ``clamp_to_measured_span = True`` are validated-workflow defaults --
    the clamp was added after shot 48224 at 298 ms, where only 8 of 40 channels
    mapped inside the LCFS and the unclamped cubic gave ``T_i(axis) = 54`` eV
    against a 21 eV largest measurement.  ``ion_index = 0`` selects the first
    ion; ``rho_points = 100`` and ``time_tolerance_ms = 1.0`` are numerical
    conveniences.

    Convention
    ----------
    As :func:`profile_fitting_thomson_scattering`: the fit and its span are in
    the selected coordinate.  ``V_tor`` keeps the sign the diagnostic mapping
    gave it.

    Assumptions
    -----------
    ``T_i`` and ``V_tor`` are flux functions; the channel's ``(R, Z)`` is where
    the measurement is local.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    With the clamp off, a polynomial through a poorly covered span runs away
    toward the axis.  Every channel's ``ion[ion_index]`` must exist.

    Provenance
    ----------
    .. [1] The fitting kernel [FIT]_; the span clamp from the 48224 @ 298 ms
       failure, mirroring ``vaft.code.efit._ti_weighted_fit_psin``.
    """
    mapped_positions = _legacy_keyword(mapped_positions, mapped_rho_position, "mapped_rho_position")
    rho_flat, coordinate = _positions_for_fit(mapped_positions, coordinate)

    # --- time index ---
    times = np.asarray(ods['charge_exchange.time'], dtype=float)
    if times.ndim != 1:
        raise ValueError("charge_exchange.time must be 1D")

    target_s = time_ms / 1e3
    time_index = int(np.argmin(np.abs(times - target_s)))
    if abs(times[time_index] - target_s) > time_tolerance_ms / 1e3:
        raise ValueError(
            f"No charge_exchange time within {time_tolerance_ms} ms of {time_ms} ms "
            f"(nearest: {times[time_index] * 1e3:.3f} ms)"
        )

    num_channels = len(ods['charge_exchange.channel'])
    Ti, Vtor, Ti_std, Vtor_std = [], [], [], []

    for i in range(num_channels):
        ion = ods[f'charge_exchange.channel.{i}.ion.{ion_index}']

        # OMAS (>=0.94.2) stores an assigned uarray as `<leaf>.data` (nominal) +
        # `<leaf>.data_error_upper`; the re-read `.data` carries NO uncertainty,
        # so unumpy.std_devs(<leaf>.data) is ALL ZEROS. Read `.data_error_upper`
        # explicitly (mirrors the Thomson path) to recover the real sigma.
        ti_val, ti_err = _leaf_values_and_errors(ion['t_i'], time_index)
        v_val, v_err = _leaf_values_and_errors(ion['velocity_tor'], time_index)

        Ti.append(ti_val)
        Ti_std.append(ti_err)
        Vtor.append(v_val)
        Vtor_std.append(v_err)

    Ti, Ti_std, rho_ti = _filter_channels_for_fit(Ti, Ti_std, rho_flat, "CES T_i", time_ms)
    Vtor, Vtor_std, rho_v = _filter_channels_for_fit(Vtor, Vtor_std, rho_flat, "CES V_tor", time_ms)

    # Replace any residual 0/NaN sigma (from OMAS-split leaves) with the
    # valid-error median so a single near-zero sigma cannot blow up its
    # 1/sigma**2 weight and dominate the fit.
    Ti_std = _sanitize_std(Ti_std)
    Vtor_std = _sanitize_std(Vtor_std)

    rho = np.clip(rho_ti.reshape(-1, 1), 0.0, 1.0)
    rho_vtor = np.clip(rho_v.reshape(-1, 1), 0.0, 1.0)
    rho_eval = np.linspace(0.0, 1.0, rho_points)

    # --- Fit T_i(ρ) ---
    Ti_rho, Ti_std_fit, Ti_function_raw, coeffs_ti = fit_profile(
        rho,
        Ti,
        Ti_std,
        rho_eval,
        order=Ti_order,
        uncertainty_option=uncertainty_option,
        fitting_function=fitting_function_ti,
        gp_anchor=None,
    )

    Ti_rho = np.maximum(np.asarray(Ti_rho, dtype=float), 0.0)

    # No blind extrapolation beyond the psi_N actually covered by CX channels:
    # outside the measured span the fit is held at its value on the nearest
    # measured end. Same convention as efit._ti_weighted_fit_psin. Without
    # it a polynomial extrapolated inward from a poorly-covered slice can blow up
    # (48224 @ 298 ms: only 8/40 channels map inside the LCFS, innermost
    # psi_N = 0.23 -> Ti(axis) = 54 eV against a 21 eV largest measurement).
    ti_lo, ti_hi = float(np.min(rho_ti)), float(np.max(rho_ti))
    v_lo, v_hi = float(np.min(rho_v)), float(np.max(rho_v))

    def Ti_function(rho_input):
        x = np.clip(np.asarray(rho_input, float), 0.0, 1.0)
        if clamp_to_measured_span:
            x = np.clip(x, ti_lo, ti_hi)
        return np.maximum(Ti_function_raw(x), 0.0)

    # --- Fit V_tor(ρ) ---
    Vtor_rho, Vtor_std_fit, Vtor_function_raw, coeffs_vtor = fit_profile(
        rho_vtor,
        Vtor,
        Vtor_std,
        rho_eval,
        order=Vtor_order,
        uncertainty_option=uncertainty_option,
        fitting_function=fitting_function_vtor,
        gp_anchor=None,
    )

    def Vtor_function(rho_input):
        x = np.clip(np.asarray(rho_input, float), 0.0, 1.0)
        if clamp_to_measured_span:
            x = np.clip(x, v_lo, v_hi)
        return Vtor_function_raw(x)

    # Keep the sampled return values consistent with the public callables.
    # Several callers persist these arrays directly into core_profiles instead
    # of re-evaluating the functions, so returning the raw extrapolated fits
    # would bypass clamp_to_measured_span and reintroduce the pathology.
    Ti_rho = np.asarray(Ti_function(rho_eval), dtype=float)
    Vtor_rho = np.asarray(Vtor_function(rho_eval), dtype=float)

    ti_span = (ti_lo, ti_hi) if clamp_to_measured_span else None
    v_span = (v_lo, v_hi) if clamp_to_measured_span else None
    Vtor_fit = FittedProfile(Vtor_function, coordinate, str(fitting_function_vtor), int(Vtor_order), coeffs_vtor, v_span)
    Ti_fit = FittedProfile(Ti_function, coordinate, str(fitting_function_ti), int(Ti_order), coeffs_ti, ti_span)
    return Vtor_fit, Ti_fit, coeffs_vtor, coeffs_ti, Vtor_rho, Ti_rho
class _EquilibriumGrid(NamedTuple):
    """The 1-D flux grid a profile is stored on, and whether its rho is real.

    ``rho_tor_norm_trusted`` is ``False`` when the grid's ``rho_tor_norm`` is
    indistinguishable from the ``sqrt(psi_N)`` proxy that producers wrote
    before issue #276 and the real coordinate could not be re-derived from
    ``q``.  The array is still stored as the equilibrium's own description of
    itself, but a ``rho_tor_norm`` fit must not be evaluated on it.
    """

    rho_tor_norm: "np.ndarray"
    psi: "np.ndarray"
    psi_norm: "np.ndarray"
    rho_tor_norm_trusted: bool = True


def _grid_from_geq(geq):
    """Return the 1-D (rho_tor_norm, psi, psi_N) grid from a geqdsk, or None.

    Used so core_profiles can put profiles on the grid of the SAME equilibrium
    that was used for the psi_N mapping, without depending on (or mutating) the
    ODS ``equilibrium`` IDS -- important when the stored equilibrium carries only
    a time axis with empty ``profiles_1d`` (e.g. after an IMAS-version conversion).
    """
    if geq is None:
        return None
    try:
        # Only the 1D psi/rho grid is wanted here, and the derived-data pass
        # traces every flux surface -- seconds per slice, for arrays this
        # function discards.
        go = geq.to_omas(allow_derived_data=False)
        base = "equilibrium.time_slice.0.profiles_1d"
        rho = np.asarray(go[f"{base}.rho_tor_norm"], dtype=float)
        psi = np.asarray(go[f"{base}.psi"], dtype=float)
    except Exception:
        return None
    if rho.size < 2 or psi.size < 2 or psi[-1] == psi[0]:
        return None
    psi_n = (psi - psi[0]) / (psi[-1] - psi[0])
    return _resolve_grid_rho(rho, psi, psi_n, lambda: _rho_tor_from_equilibrium(geq, psi_n))


def _equilibrium_grid_at_time(ods, target_s, tol_s):
    """Return (rho_tor_norm, psi, psi_N) of the equilibrium slice at target_s, or None."""
    try:
        eq_times = np.asarray(ods['equilibrium.time'], dtype=float).reshape(-1)
    except Exception:
        return None
    if eq_times.size == 0:
        return None
    idx = int(np.argmin(np.abs(eq_times - target_s)))
    if abs(eq_times[idx] - target_s) > tol_s:
        return None
    try:
        rho = np.asarray(
            ods[f'equilibrium.time_slice.{idx}.profiles_1d.rho_tor_norm'], dtype=float
        )
        psi = np.asarray(
            ods[f'equilibrium.time_slice.{idx}.profiles_1d.psi'], dtype=float
        )
    except Exception:
        return None
    if rho.ndim != 1 or rho.shape != psi.shape or psi[-1] == psi[0]:
        return None
    psi_n = (psi - psi[0]) / (psi[-1] - psi[0])
    return _resolve_grid_rho(rho, psi, psi_n, lambda: _rho_tor_from_equilibrium(ods, psi_n, time_index=idx))


def _resolve_grid_rho(rho, psi, psi_n, derive):
    """Trust the grid's ``rho_tor_norm``, or re-derive it, or mark it untrusted.

    A producer before issue #276 wrote ``sqrt(psi_N)`` under ``rho_tor_norm``,
    and every packaged sample is in that state; evaluating a ``rho_tor_norm``
    fit on it would put the profile at the wrong radius.  When the array looks
    like that proxy the real coordinate is re-derived from ``q``; when it
    cannot be, the stored array is kept as the equilibrium's own description
    but flagged, and :func:`core_profiles` refuses the coordinate by name.
    """
    from vaft.data._derived import is_rho_pol_proxy

    if not is_rho_pol_proxy(rho, psi_n):
        return _EquilibriumGrid(rho, psi, psi_n, True)
    derived = derive()
    if derived is None:
        return _EquilibriumGrid(rho, psi, psi_n, False)
    return _EquilibriumGrid(derived, psi, psi_n, True)


def _rho_tor_from_equilibrium(source, psi_n_grid, *, time_index=0):
    """``rho_tor_norm`` on ``psi_n_grid``, derived from the equilibrium's own ``q``.

    ``None`` when the derivation is unavailable -- no ``q``, or a ``q`` that
    does not produce a monotonic toroidal flux -- so the caller can say so
    instead of substituting a different coordinate.
    """
    if source is None:
        return None
    try:
        from ._equilibrium_parametric import as_equilibrium, derive_radial_coordinates

        derived = derive_radial_coordinates(as_equilibrium(source, time_index=time_index))
        rho_tor = derived.get("rho_tor_n")
        psi_n = derived.get("psi_n")
        rho_tor_values = np.asarray(getattr(rho_tor, "value", rho_tor), dtype=float).reshape(-1)
        psi_n_values = np.asarray(getattr(psi_n, "value", psi_n), dtype=float).reshape(-1)
    except Exception:
        return None
    if rho_tor_values.size < 2 or rho_tor_values.shape != psi_n_values.shape:
        return None
    if not np.all(np.isfinite(rho_tor_values)) or not np.all(np.isfinite(psi_n_values)):
        return None
    return np.interp(np.asarray(psi_n_grid, dtype=float), psi_n_values, rho_tor_values)


def _channel_rho_tor_norm(mapped, positions, coordinate, eq_grid):
    """Channel positions as ``rho_tor_norm`` for the ``*_fit`` metadata, or ``None``.

    Straight from the mapping when it carries ``rho_tor_norm``; otherwise
    converted through the equilibrium grid's ``rho_tor_norm(psi_norm)`` table
    when there is one; otherwise unknown.
    """
    if isinstance(mapped, MappedPositions) and mapped.rho_tor_norm is not None:
        return np.asarray(mapped.rho_tor_norm, dtype=float).reshape(-1)
    if coordinate == "rho_tor_norm":
        return np.asarray(positions, dtype=float).reshape(-1)
    if eq_grid is not None and not eq_grid.rho_tor_norm_trusted:
        return None
    if eq_grid is None:
        return None
    rho_tor_grid, psi_n_grid = eq_grid.rho_tor_norm, eq_grid.psi_norm
    psi_n = positions if coordinate == "psi_norm" else np.asarray(positions, dtype=float) ** 2
    out = np.full(np.shape(positions), np.nan)
    finite = np.isfinite(psi_n)
    out[finite] = np.interp(np.clip(psi_n[finite], 0.0, 1.0), psi_n_grid, rho_tor_grid)
    return out


def _append_code_parameters(ods, ids, line, *, replace_key=None):
    """Append one provenance line to ``<ids>.code.parameters``, dropping a superseded one."""
    path = f"{ids}.code.parameters"
    existing = ""
    if ids in ods:
        try:
            existing = str(ods[path] or "")
        except Exception:
            existing = ""
    kept = [
        item for item in existing.splitlines()
        if item.strip() and not (replace_key and replace_key in item)
    ]
    ods[path] = "\n".join(kept + [line]) + "\n"


def core_profiles(
    ods,
    time_ms,
    mapped_positions=None,
    n_e_function=None,
    T_e_function=None,
    tol_ms=0.1,
    T_i_function=None,
    V_tor_function=None,
    ti_mapped_positions=None,
    rho_points=100,
    time_tolerance_ms=1.0,
    geq=None,
    ti_te_fallback=True,
    ti_te_ratio=None,
    *,
    coordinate=None,
    ti_te_ratio_record=None,
    mapped_rho_position=None,
    ti_mapped_rho_position=None,
):
    """Evaluate fitted kinetic profiles on the equilibrium grid and store them as a ``profiles_1d`` slice.

    The fits are functions of one radial coordinate -- the one they were made
    in, carried by :class:`FittedProfile` or named by ``coordinate`` for a
    plain callable -- and are evaluated on the equilibrium's grid expressed in
    that same coordinate, so a fit in ``rho_tor_norm`` is never evaluated at a
    ``psi_norm`` value.  What was fitted, in which coordinate, and where any
    assumed number came from is written beside the profile.

    Parameters
    ----------
    ods : ODS
        Mutated in place: gains or replaces the ``core_profiles.profiles_1d``
        slice at ``time_ms`` [-].
    time_ms : float
        Time of the slice [ms].
    mapped_positions : MappedPositions or np.ndarray, optional
        Thomson channel positions from :func:`equilibrium_mapping_thomson_scattering`;
        required with an electron fit.  A bare array is the deprecated
        pre-#420 form and is accepted only with ``coordinate="psi_norm"`` [-].
    n_e_function : FittedProfile or callable, optional
        Electron density fit [m^-3].
    T_e_function : FittedProfile or callable, optional
        Electron temperature fit [eV].
    tol_ms : float, optional
        An existing slice within this of ``time_ms`` is replaced [ms].
    T_i_function : FittedProfile or callable, optional
        Ion temperature fit from the charge-exchange fitter; when given, the
        ion block carries the real ``T_i`` and ``pressure_thermal`` is
        written [eV].
    V_tor_function : FittedProfile or callable, optional
        Toroidal ion velocity fit [m/s].
    ti_mapped_positions : MappedPositions or np.ndarray, optional
        Charge-exchange channel positions, for the ion fit metadata [-].
    rho_points : int, optional
        Size of the uniform fallback grid when no equilibrium slice matches [-].
    time_tolerance_ms : float, optional
        Tolerance for matching the Thomson and equilibrium times [ms].
    geq : GEQDSK, optional
        The equilibrium the mapping used; its 1-D grid is preferred over the
        ODS equilibrium so the profile lands on the same surfaces [-].
    ti_te_fallback : bool, optional
        With no ion fit, still write an ion temperature from ``T_e`` [-].
    ti_te_ratio : float, optional
        Coefficient for that fallback: ``T_i = ratio * T_e`` and
        ``pressure_thermal = e n_e (1 + ratio) T_e``; ``None`` keeps the
        legacy ``T_i = T_e`` and writes no pressure.  Resolved by the
        pipeline from the machine policy; this function holds no value [-].
    coordinate : str, optional
        Required when the fits are plain callables; must agree with every
        :class:`FittedProfile` given [-].
    ti_te_ratio_record : str, optional
        Provenance text for the ratio (value, status, source), stored beside
        the ion temperature; the pipeline passes the policy's own record [-].
    mapped_rho_position : MappedPositions or np.ndarray, optional
        Deprecated name for mapped_positions [-].
    ti_mapped_rho_position : MappedPositions or np.ndarray, optional
        Deprecated name for ti_mapped_positions [-].

    Returns
    -------
    ODS
        The same object, updated [-].

    Raises
    ------
    ValueError
        No electron and no ion fit; an electron fit without positions; fits in
        different coordinates; a ratio that is not finite and non-negative;
        no Thomson time within tolerance.
    TypeError
        A plain callable without ``coordinate``, or a bare position array
        without ``coordinate="psi_norm"``.
    CoordinateUnavailableError
        The mapping cannot supply the fit's coordinate.

    Processing steps
    ----------------
    1. Establish the fit coordinate from the fit objects (or ``coordinate``)
       and check they agree.
    2. Read the measured Thomson values at the nearest time.
    3. Take the equilibrium 1-D grid -- from ``geq``, else the ODS slice at
       this time -- as ``(rho_tor_norm, psi, psi_norm)``; express it in the
       fit coordinate; without one, use a uniform grid in that coordinate.
    4. Evaluate every fit on that grid; apply the ion fallback if needed.
    5. Replace any slice at the same time, write grid, electrons, ions,
       pressure, per-channel fit metadata and provenance.

    Input semantics
    ---------------
    Fitted: callables of one radial coordinate, from
    :func:`profile_fitting_thomson_scattering` /
    :func:`profile_fitting_charge_exchange`, plus the measured channel
    values and their equilibrium-mapped positions.

    Output semantics
    ----------------
    Stored, on the equilibrium grid: ``grid.rho_tor_norm``, ``grid.psi`` and
    ``grid.rho_pol_norm`` when an equilibrium slice exists, else only the
    coordinate the fit is actually in.  ``*_fit.measured`` are measured,
    ``*_fit.reconstructed`` are the fit at the measured points; a
    ratio-derived ion temperature is inferred and says so.

    Defaults
    --------
    ``ti_te_ratio = None`` is a legacy compatibility value: the pre-#420
    behaviour writes ``T_i = T_e`` and no pressure.  The VEST value is
    resolved by :func:`vaft.machine_mapping.core_profiles.vest_core_profiles_policy`
    and passed in by :func:`vaft.code.efit.build_kinetic_core_profiles`.
    ``tol_ms = 0.1`` and ``time_tolerance_ms = 1.0`` are numerical
    conveniences.

    Convention
    ----------
    The fit coordinate is one of ``rho_tor_norm``, ``rho_pol_norm``,
    ``psi_norm``; the equilibrium grid supplies all three (``rho_pol_norm =
    sqrt(psi_norm)``; ``rho_tor_norm`` from the equilibrium, never from
    ``psi_norm``).  Quasi-neutral single main ion H+, ``n_i = n_e``.
    ``pressure_thermal = e n_e (T_e + T_i)`` [Pa] with temperatures in eV.

    Assumptions
    -----------
    All electrons are thermal (an ohmic plasma); one hydrogenic main ion with
    no impurity dilution; the equilibrium at ``time_ms`` is the one the
    channels were mapped through.

    Applicability
    -------------
    Machine-independent.  Every machine-specific number -- the coordinate
    convention, the Ti/Te ratio, its provenance -- arrives as an argument;
    for VEST the pipeline resolves them from ``vest.yaml``.

    Limitations
    -----------
    Only the first equilibrium time slice of ``geq`` is used.  The
    charge-exchange ``*_fit`` metadata is best-effort and its failure is
    reported, not raised.  A slice produced before #420 has no
    ``*_fit.parameters``; that absence is the signature of a legacy product
    fitted in ``psi_norm``.

    Provenance
    ----------
    .. [1] Per-slice fit records in ``electrons.{density,temperature}_fit.parameters``
       and ``ion.0.temperature_fit.parameters``, and one line per slice in
       ``core_profiles.code.parameters``; issue #420.
    """
    mapped_positions = _legacy_keyword(mapped_positions, mapped_rho_position, "mapped_rho_position")
    ti_mapped_positions = _legacy_keyword(ti_mapped_positions, ti_mapped_rho_position, "ti_mapped_rho_position")
    e_J_per_eV = 1.602176634e-19
    target_s = time_ms / 1e3
    tol_s = time_tolerance_ms / 1e3

    # Write whatever diagnostic is available: electrons need a Thomson (ne/Te) fit,
    # ions need a charge_exchange (Ti[/Vtor]) fit. At least one is required.
    have_e = n_e_function is not None and T_e_function is not None
    have_i = T_i_function is not None
    if not have_e and not have_i:
        raise ValueError(
            "core_profiles needs at least an electron (n_e_function + T_e_function) "
            "or an ion (T_i_function) fit"
        )
    if have_e and mapped_positions is None:
        raise ValueError(
            "core_profiles: mapped_positions is required when an electron "
            "(n_e_function + T_e_function) fit is provided"
        )

    # --- the fit coordinate: every fit must be a function of the same one ---
    coord = None
    for fit, name in (
        (n_e_function, "n_e_function"),
        (T_e_function, "T_e_function"),
        (T_i_function, "T_i_function"),
        (V_tor_function, "V_tor_function"),
    ):
        if fit is None:
            continue
        this = _fit_coordinate(fit, coordinate, name)
        if coord is None:
            coord = this
        elif this != coord:
            raise ValueError(
                f"{name} is a function of {this} but the other fits are functions of {coord}"
            )

    # --- measured TS points (nearest time within tolerance) -- electron path only ---
    n_e_meas = T_e_meas = rho_meas = None
    if have_e:
        ts_times = np.asarray(ods['thomson_scattering.time'], dtype=float)
        t_idx = int(np.argmin(np.abs(ts_times - target_s)))
        if abs(ts_times[t_idx] - target_s) > tol_s:
            raise ValueError(
                f"No Thomson time within {time_tolerance_ms} ms of {time_ms} ms "
                f"(nearest: {ts_times[t_idx] * 1e3:.3f} ms)"
            )
        num_channels = len(ods['thomson_scattering.channel'])
        n_e_meas = np.array(
            [ods[f'thomson_scattering.channel.{i}.n_e.data'][t_idx] for i in range(num_channels)],
            dtype=float,
        )
        T_e_meas = np.array(
            [ods[f'thomson_scattering.channel.{i}.t_e.data'][t_idx] for i in range(num_channels)],
            dtype=float,
        )
        rho_meas, _ = _positions_for_fit(mapped_positions, coord, legacy_name="mapped_rho_position")

    # --- evaluation grid: prefer the mapping geqdsk, then the ODS equilibrium,
    #     else a uniform grid in the fit coordinate ---
    eq_grid = _grid_from_geq(geq)
    if eq_grid is None:
        eq_grid = _equilibrium_grid_at_time(ods, target_s, tol_s)
    if eq_grid is not None:
        rho_tor_grid, psi_grid, psi_n_grid = eq_grid[:3]
        if coord == "rho_tor_norm" and not eq_grid.rho_tor_norm_trusted:
            raise CoordinateUnavailableError(
                "the equilibrium slice at this time stores rho_tor_norm as a "
                "sqrt(psi_N) proxy and carries no usable q profile to re-derive "
                "it from, so a rho_tor_norm fit cannot be evaluated on this "
                "grid. Pass coordinate='rho_pol_norm' or 'psi_norm' to fit and "
                "store in a coordinate this equilibrium supports."
            )
        rho_pol_grid = np.sqrt(np.clip(psi_n_grid, 0.0, None))
        x_grid = {
            "rho_tor_norm": rho_tor_grid,
            "rho_pol_norm": rho_pol_grid,
            "psi_norm": psi_n_grid,
        }[coord]
    else:
        rho_tor_grid = psi_grid = psi_n_grid = rho_pol_grid = None
        x_grid = np.linspace(0.0, 1.0, rho_points)

    n_e_recon = np.asarray(n_e_function(x_grid), dtype=float) if have_e else None
    T_e_recon = np.asarray(T_e_function(x_grid), dtype=float) if have_e else None
    # Ti falls back to Te (or the statistical ratio*Te) only when electrons are
    # present but no ion fit was given.
    ratio_fallback = False
    if have_i:
        T_i_recon = np.asarray(T_i_function(x_grid), dtype=float)
    elif have_e and ti_te_fallback:
        if ti_te_ratio is not None:
            ratio = float(ti_te_ratio)
            if not np.isfinite(ratio) or ratio < 0.0:
                raise ValueError(
                    "ti_te_ratio must be finite and non-negative, "
                    f"got {ratio!r}"
                )
            T_i_recon = ratio * T_e_recon
            ratio_fallback = True
            print(
                f"[INFO] no ion fit at {time_ms:.3f} ms: statistical fallback "
                f"Ti = {ratio:.3f}*Te (kinetic pressure written)"
            )
        else:
            T_i_recon = T_e_recon
    else:
        T_i_recon = None
    V_tor_recon = (
        np.asarray(V_tor_function(x_grid), dtype=float)
        if V_tor_function is not None
        else None
    )

    # --- check for duplicate time entries ---
    existing_times = []
    if 'core_profiles.profiles_1d' in ods:
        n_profiles = len(ods['core_profiles.profiles_1d'])
        for i in range(n_profiles):
            try:
                t_existing = ods[f'core_profiles.profiles_1d.{i}.time']
                if abs(t_existing * 1000 - time_ms) < tol_ms:
                    existing_times.append(i)
            except Exception:
                continue

    # --- remove duplicates before writing ---
    for i in sorted(existing_times, reverse=True):
        ods.pop(f'core_profiles.profiles_1d.{i}')
        print(f"[INFO] Removed duplicate core_profile at {time_ms:.3f} ms (index {i})")

    # --- Determine next available index after removal ---
    next_idx = len(ods['core_profiles.profiles_1d']) if 'core_profiles.profiles_1d' in ods else 0
    base = f'core_profiles.profiles_1d.{next_idx}'

    ods[f'{base}.time'] = target_s
    if eq_grid is not None:
        ods[f'{base}.grid.rho_tor_norm'] = rho_tor_grid
        ods[f'{base}.grid.psi'] = psi_grid
        ods[f'{base}.grid.rho_pol_norm'] = rho_pol_grid
    elif coord == "rho_tor_norm":
        ods[f'{base}.grid.rho_tor_norm'] = x_grid
    elif coord == "rho_pol_norm":
        ods[f'{base}.grid.rho_pol_norm'] = x_grid
    else:
        # a psi_norm grid is stored as rho_pol_norm = sqrt(psi_norm); never as rho_tor_norm
        ods[f'{base}.grid.rho_pol_norm'] = np.sqrt(x_grid)

    # We assume all electrons are thermal electrons (because VEST is ohmically heated plasma)
    if have_e:
        ods[f'{base}.electrons.density_thermal'] = n_e_recon
        ods[f'{base}.electrons.density'] = n_e_recon
        ods[f'{base}.electrons.temperature'] = T_e_recon

    # single main ion H+ with n_i ~= n_e (quasi-neutrality, no impurity dilution).
    # Write the ion block only when there IS an ion fit, or when electrons are
    # present and the Ti=Te fallback is enabled. With ti_te_fallback=False a slice
    # with no ion measurement stays electron-only -- no phantom ni / Ti / velocity.
    write_ion = have_i or (have_e and ti_te_fallback)
    if write_ion:
        ods[f'{base}.ion.0.label'] = 'H+'
        ods[f'{base}.ion.0.z_ion'] = 1.0
        ods[f'{base}.ion.0.element.0.a'] = 1.00784
        ods[f'{base}.ion.0.element.0.z_n'] = 1.0
        ods[f'{base}.ion.0.element.0.atoms_n'] = 1
        if have_e:  # quasi-neutral main-ion density needs the electron density
            ods[f'{base}.ion.0.density_thermal'] = n_e_recon
            ods[f'{base}.ion.0.density'] = n_e_recon
        if T_i_recon is not None:
            ods[f'{base}.ion.0.temperature'] = T_i_recon
        if V_tor_recon is not None:
            ods[f'{base}.ion.0.velocity.toroidal'] = V_tor_recon
    # kinetic pressure needs ne and (Te, Ti): either a real ion fit or the
    # explicit statistical ratio fallback (Ti = ti_te_ratio*Te). The legacy
    # Ti=Te fallback intentionally writes NO pressure_thermal.
    if have_e and (have_i or ratio_fallback):
        ods[f'{base}.pressure_thermal'] = e_J_per_eV * n_e_recon * (T_e_recon + T_i_recon)

    # --- measurement/fit metadata (per IMAS DD, measured/reconstructed/rho all
    # have one entry per measurement point; reconstructed = fit AT those points) ---
    electron_record = ion_record = None
    if have_e:
        electron_record = f"ne[{_fit_parameters_text(n_e_function, coord)}] Te[{_fit_parameters_text(T_e_function, coord)}]"
        rho_meas_tor = _channel_rho_tor_norm(mapped_positions, rho_meas, coord, eq_grid)
        if rho_meas_tor is not None:
            finite_meas = np.isfinite(rho_meas) & np.isfinite(rho_meas_tor)
            x_meas = np.clip(rho_meas[finite_meas], 0, 1)
            fit_base_n = f'{base}.electrons.density_fit'
            fit_base_t = f'{base}.electrons.temperature_fit'
            ods[f'{fit_base_n}.rho_tor_norm'] = rho_meas_tor[finite_meas]
            ods[f'{fit_base_n}.measured'] = n_e_meas[finite_meas]
            ods[f'{fit_base_n}.reconstructed'] = np.asarray(n_e_function(x_meas), dtype=float)
            ods[f'{fit_base_n}.parameters'] = _fit_parameters_text(n_e_function, coord)
            ods[f'{fit_base_t}.rho_tor_norm'] = rho_meas_tor[finite_meas]
            ods[f'{fit_base_t}.measured'] = T_e_meas[finite_meas]
            ods[f'{fit_base_t}.reconstructed'] = np.asarray(T_e_function(x_meas), dtype=float)
            ods[f'{fit_base_t}.parameters'] = _fit_parameters_text(T_e_function, coord)

    if T_i_function is not None:
        ion_record = f"Ti[{_fit_parameters_text(T_i_function, coord)}]"
        if V_tor_function is not None:
            ion_record += f" Vtor[{_fit_parameters_text(V_tor_function, coord)}]"
        fit_base_ti = f'{base}.ion.0.temperature_fit'
        ods[f'{fit_base_ti}.parameters'] = _fit_parameters_text(T_i_function, coord)
        if ti_mapped_positions is not None:
            # Resolved outside the try: a refused coordinate is a caller error and
            # must not be downgraded to a "could not attach metadata" warning.
            ti_rho, _ = _positions_for_fit(
                ti_mapped_positions, coord, legacy_name="ti_mapped_rho_position"
            )
            try:
                ce_times = np.asarray(ods['charge_exchange.time'], dtype=float)
                ce_idx = int(np.argmin(np.abs(ce_times - target_s)))
                n_ce = len(ods['charge_exchange.channel'])
                ti_meas, ti_meas_err = [], []
                for i in range(n_ce):
                    # OMAS split the assigned uarray -> read .data + .data_error_upper
                    val, err = _leaf_values_and_errors(
                        ods[f'charge_exchange.channel.{i}.ion.0.t_i'],
                        ce_idx,
                        clamp=True,
                    )
                    ti_meas.append(val)
                    ti_meas_err.append(err)
                ti_meas = np.asarray(ti_meas, dtype=float)
                ti_meas_err = np.asarray(ti_meas_err, dtype=float)
                ti_rho_tor = _channel_rho_tor_norm(ti_mapped_positions, ti_rho, coord, eq_grid)
                if ti_rho_tor is not None:
                    finite_ti = np.isfinite(ti_rho) & np.isfinite(ti_rho_tor)
                    ods[f'{fit_base_ti}.rho_tor_norm'] = ti_rho_tor[finite_ti]
                    ods[f'{fit_base_ti}.measured'] = ti_meas[finite_ti]
                    ods[f'{fit_base_ti}.measured_error_upper'] = ti_meas_err[finite_ti]
                    ods[f'{fit_base_ti}.reconstructed'] = np.asarray(
                        T_i_function(np.clip(ti_rho[finite_ti], 0, 1)), dtype=float
                    )
            except Exception as exc:
                print(f"[WARN] could not attach ion temperature_fit metadata: {exc}")
    elif ratio_fallback:
        record = ti_te_ratio_record or f"ti_te_ratio={float(ti_te_ratio):g}; status=unspecified"
        ods[f'{base}.ion.0.temperature_fit.parameters'] = record
        ion_record = f"Ti[{record}]"
    elif write_ion and have_e:
        ods[f'{base}.ion.0.temperature_fit.parameters'] = "ti_te_ratio=1; status=assumed; source=legacy Ti=Te fallback"
        ion_record = "Ti[legacy Ti=Te]"

    # --- IDS bookkeeping: producer and per-slice provenance, then the time base ---
    if 'core_profiles.code.name' not in ods:
        ods['core_profiles.code.name'] = 'vaft.process.profile'
    _append_code_parameters(
        ods,
        'core_profiles',
        f"profiles_1d time={target_s:.6f} coordinate={coord} "
        f"grid={'equilibrium' if eq_grid is not None else 'uniform'} "
        f"electrons={electron_record or 'none'} ions={ion_record or 'none'}",
        replace_key=f"time={target_s:.6f} ",
    )
    ods['core_profiles.ids_properties.homogeneous_time'] = 1
    n_profiles = len(ods['core_profiles.profiles_1d'])
    ods['core_profiles.time'] = np.asarray(
        [float(ods[f'core_profiles.profiles_1d.{i}.time']) for i in range(n_profiles)]
    )
    print(f"[UPDATED] core_profile at {time_ms:.3f} ms (index {next_idx})")
    return ods


def core_profiles_from_eq(
    ods,
    Te0_eV,
    rho_fit=None,
    tol_ms=0.1,
    eq_time_index=0,
    ):
    """Build a synthetic ``core_profiles`` slice from the equilibrium pressure and an axis ``T_e``.

    Parameters
    ----------
    ods : ODS
        Carries the equilibrium slice; gains the ``core_profiles`` slice [-].
    Te0_eV : float
        Electron temperature on axis [eV].
    rho_fit : array_like, optional
        ``rho_tor_norm`` grid to write on; ``None`` for 100 uniform points [-].
    tol_ms : float, optional
        An existing slice within this of the equilibrium time is replaced [ms].
    eq_time_index : int, optional
        Which equilibrium time slice [-].

    Returns
    -------
    ODS
        The same object, updated [-].

    Raises
    ------
    ValueError
        Non-1-D or non-finite equilibrium profiles, or non-positive axis
        pressure.

    Processing steps
    ----------------
    1. Interpolate the equilibrium ``pressure(rho_tor_norm)`` onto ``rho_fit``.
    2. ``g = sqrt(p / p(0))``; ``T_e = Te0 g``; ``n_e = p(0) / (2 e Te0) g``.
    3. Write electrons and a quasi-neutral H+ ion with ``T_i = T_e``.

    Input semantics
    ---------------
    Reconstructed: the equilibrium pressure profile.

    Output semantics
    ----------------
    Synthetic: profiles consistent with that pressure under an assumed shape,
    not measured.

    Convention
    ----------
    ``p = 2 n_e T_e e`` [Pa]: ``T_i = T_e`` and ``n_i = n_e`` are absorbed in
    the factor 2.  The grid is the equilibrium's ``rho_tor_norm``, read directly.

    Assumptions
    -----------
    ``n_e`` and ``T_e`` share one shape, ``sqrt(p/p(0))``; no impurities.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Any split of the pressure between density and temperature is consistent
    with the equilibrium; this one is a convention, and ``Te0_eV`` fixes it.
    """
    e_J_per_eV = 1.602176634e-19

    time_ms = ods['equilibrium.time'][eq_time_index] * 1e3

    if rho_fit is None:
        rho_fit = np.linspace(0.0, 1.0, 100)
    else:
        rho_fit = np.asarray(rho_fit, dtype=float)

    rho_src_path = f"equilibrium.time_slice.{eq_time_index}.profiles_1d.rho_tor_norm"
    p_src_path   = f"equilibrium.time_slice.{eq_time_index}.profiles_1d.pressure"

    rho_src = np.asarray(ods[rho_src_path], dtype=float)
    p_src   = np.asarray(ods[p_src_path], dtype=float)  # Pa

    if rho_src.ndim != 1 or p_src.ndim != 1:
        raise ValueError("Expected 1D rho_tor_norm and 1D pressure at the selected time_slice.")

    order = np.argsort(rho_src)
    rho_src = rho_src[order]
    p_src = p_src[order]

    P_fit = np.interp(rho_fit, rho_src, p_src)

    if np.any(~np.isfinite(P_fit)):
        raise ValueError("Pressure profile contains NaN/Inf.")
    if P_fit[0] <= 0:
        raise ValueError("Pressure at rho=0 must be > 0 to define sqrt shape.")

    g = np.sqrt(np.clip(P_fit / P_fit[0], 0.0, None))

    Te = Te0_eV * g  # eV

    # ---- FIX: include eV->J ----
    ne0 = P_fit[0] / (2.0 * Te0_eV * e_J_per_eV)  # m^-3
    ne = ne0 * g

    if np.any(ne < 0) or np.any(Te < 0):
        raise ValueError("Generated ne/Te has negative values (check pressure and Te0_eV).")

    # ---- remove duplicates ----
    existing_idxs = []
    if "core_profiles.profiles_1d" in ods:
        for i in range(len(ods["core_profiles.profiles_1d"])):
            try:
                t_existing = ods[f"core_profiles.profiles_1d.{i}.time"]  # s
                if abs(t_existing * 1000.0 - time_ms) < tol_ms:
                    existing_idxs.append(i)
            except Exception:
                continue

    for i in sorted(existing_idxs, reverse=True):
        ods.pop(f"core_profiles.profiles_1d.{i}")
        print(f"[INFO] Removed duplicate core_profile at {time_ms:.3f} ms (index {i})")

    next_idx = len(ods["core_profiles.profiles_1d"]) if "core_profiles.profiles_1d" in ods else 0
    base = f"core_profiles.profiles_1d.{next_idx}"

    ods[f"{base}.time"] = time_ms / 1000.0
    ods[f"{base}.grid.rho_tor_norm"] = rho_fit.tolist()

    ods[f"{base}.electrons.density_thermal"] = ne.tolist()
    ods[f"{base}.electrons.density"] = ne.tolist()
    ods[f"{base}.electrons.temperature"] = Te.tolist()

    ods[f"{base}.ion.0.label"] = "H+"
    ods[f"{base}.ion.0.density_thermal"] = ne.tolist()
    ods[f"{base}.ion.0.density"] = ne.tolist()
    ods[f"{base}.ion.0.temperature"] = Te.tolist()

    print(f"[UPDATED] core_profile from eq pressure (Pa) at {time_ms:.3f} ms "
          f"(index {next_idx}), eq_time_slice={eq_time_index}")
    return ods

def core_profiles_from_eq_ratio(
    ods,
    C_ne_over_Te,   # density / temperature ratio
    rho_fit=None,
    tol_ms=0.1,
    eq_time_index=0,
    ):
    """Build a synthetic ``core_profiles`` slice from the equilibrium pressure and a fixed ``n_e/T_e``.

    Parameters
    ----------
    ods : ODS
        Carries the equilibrium slice; gains the ``core_profiles`` slice [-].
    C_ne_over_Te : float
        Density-to-temperature ratio held constant across the profile [m^-3/eV].
    rho_fit : array_like, optional
        ``rho_tor_norm`` grid to write on; ``None`` for 100 uniform points [-].
    tol_ms : float, optional
        An existing slice within this of the equilibrium time is replaced [ms].
    eq_time_index : int, optional
        Which equilibrium time slice [-].

    Returns
    -------
    ODS
        The same object, updated [-].

    Raises
    ------
    ValueError
        Non-positive axis pressure.

    Processing steps
    ----------------
    1. Interpolate the equilibrium ``pressure(rho_tor_norm)`` onto ``rho_fit``.
    2. ``f = p / p(0)``; ``T_e = sqrt(f)``, ``n_e = C sqrt(f)``, then scale both
       by ``sqrt(p(0) / (2 C e))`` so that ``p = 2 n_e T_e e``.
    3. Write electrons and a quasi-neutral H+ ion with ``T_i = T_e``.

    Input semantics
    ---------------
    Reconstructed: the equilibrium pressure profile.

    Output semantics
    ----------------
    Synthetic.

    Convention
    ----------
    ``p = 2 n_e T_e e`` [Pa] with ``T_i = T_e``; grid is the equilibrium's
    ``rho_tor_norm``.

    Assumptions
    -----------
    ``n_e / T_e`` is constant across the profile; no impurities.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A constant ratio is a modelling choice, not an observation.
    """

    e_J = 1.602176634e-19
    time_ms = ods['equilibrium.time'][eq_time_index] * 1e3

    if rho_fit is None:
        rho_fit = np.linspace(0, 1, 100)

    rho_src = np.asarray(
        ods[f'equilibrium.time_slice.{eq_time_index}.profiles_1d.rho_tor_norm']
    )
    p_src = np.asarray(
        ods[f'equilibrium.time_slice.{eq_time_index}.profiles_1d.pressure']
    )

    order = np.argsort(rho_src)
    rho_src, p_src = rho_src[order], p_src[order]

    P_fit = np.interp(rho_fit, rho_src, p_src)

    if P_fit[0] <= 0:
        raise ValueError("Invalid pressure profile")

    # --- shape ---
    f = P_fit / P_fit[0]

    # --- build profiles ---
    Te = np.sqrt(f)                 # eV (relative)
    ne = C_ne_over_Te * Te          # m^-3

    # --- scale to absolute pressure ---
    scale = P_fit[0] / (2 * C_ne_over_Te * e_J)
    Te *= np.sqrt(scale)
    ne *= np.sqrt(scale)

    # --- remove duplicates ---
    existing = []
    if 'core_profiles.profiles_1d' in ods:
        for i in range(len(ods['core_profiles.profiles_1d'])):
            t = ods[f'core_profiles.profiles_1d.{i}.time']
            if abs(t * 1000 - time_ms) < tol_ms:
                existing.append(i)
    for i in reversed(existing):
        ods.pop(f'core_profiles.profiles_1d.{i}')

    next_idx = len(ods['core_profiles.profiles_1d'])
    base = f'core_profiles.profiles_1d.{next_idx}'

    ods[f'{base}.time'] = time_ms / 1000
    ods[f'{base}.grid.rho_tor_norm'] = rho_fit.tolist()

    ods[f'{base}.electrons.density'] = ne.tolist()
    ods[f'{base}.electrons.density_thermal'] = ne.tolist()
    ods[f'{base}.electrons.temperature'] = Te.tolist()

    ods[f'{base}.ion.0.label'] = 'H+'
    ods[f'{base}.ion.0.density'] = ne.tolist()
    ods[f'{base}.ion.0.density_thermal'] = ne.tolist()
    ods[f'{base}.ion.0.temperature'] = Te.tolist()

    print(f"[UPDATED] core_profile from eq (ratio-fixed) at {time_ms:.2f} ms")
    return ods

def export_electron_profile_txt(
    n_e_function,
    T_e_function,
    n_e_coeff,
    T_e_coeff,
    rho_points=100,
    filename='electron_profiles.txt',
    ):
    """Write fitted electron profiles, sampled on a uniform grid, to a text file.

    Parameters
    ----------
    n_e_function : FittedProfile or callable
        ``n_e(x)`` [m^-3].
    T_e_function : FittedProfile or callable
        ``T_e(x)`` [eV].
    n_e_coeff : np.ndarray or None
        Fit coefficients; not written, kept for the legacy signature [-].
    T_e_coeff : np.ndarray or None
        As ``n_e_coeff`` [-].
    rho_points : int, optional
        Points of the uniform grid on ``[0, 1]`` [-].
    filename : str, optional
        Output path [-].

    Returns
    -------
    None
        The file is the result [-].

    Convention
    ----------
    The first column is labelled with the fit's coordinate when a
    :class:`FittedProfile` is given, else ``x``; a plain callable's coordinate
    is unknown here.

    Applicability
    -------------
    Machine-independent.
    """
    rho_eval = np.linspace(0, 1, rho_points)
    n_e_rho = n_e_function(rho_eval)
    T_e_rho = T_e_function(rho_eval)

    with open(filename, 'w', encoding='utf-8') as f:
        label = n_e_function.coordinate if isinstance(n_e_function, FittedProfile) else 'x'
        f.write(f'{label}, T_e [eV], n_e [m-3]\n')
        for rho, T_e, n_e in zip(rho_eval, T_e_rho, n_e_rho):
            f.write(f'{rho}, {T_e}, {n_e}\n')


#: Pedestal top used when a profile cannot support a fit (decision D-05).
#: A stated fallback, not a canonical boundary: the legacy code base carried
#: at least six different fixed windows, which is what made region metrics
#: incomparable between studies.
PEDESTAL_FALLBACK_PSI_NORM = 0.85

#: How far the fitted curve must rise above the residual scatter before a
#: pedestal counts as resolved.  Measured on a pure-noise profile the ratio is
#: about 1.3 and on a real pedestal a few hundred, so the exact value is not
#: delicate; three is comfortably between them.  It is not sufficient on its
#: own -- see :data:`PEDESTAL_MAX_WIDTH`.
PEDESTAL_RESOLUTION_FACTOR = 3.0

#: Widest tanh, in the fitted coordinate, that still describes a pedestal.
#: The model's own box allows 0.3, and a fit that comes back near it is a ramp
#: across most of the plasma rather than an edge feature: over the 171 fits of
#: the reference MAST campaign the widths separate into a physical group below
#: 0.11 and 25 fits between 0.20 and 0.29 whose "pedestal" spans half the
#: minor radius.  The legacy fitter bounded width at 0.2 for the same reason.
PEDESTAL_MAX_WIDTH = 0.15


@dataclass(frozen=True)
class PedestalTop:
    """Where the pedestal top is, and how that was decided.

    ``method`` is ``"eped_fit"`` or ``"fallback"``; ``reason`` says why the
    fallback was taken and is empty otherwise.  Carrying both is the point:
    a region reduction that does not record which one produced its boundary
    cannot be compared with another study's.
    """

    position: float
    method: str
    quantity: str
    coordinate: str
    width: float | None = None
    fit: "FittedProfile | None" = None
    reason: str = ""

    @property
    def from_fit(self) -> bool:
        """Whether a pedestal was actually resolved, rather than assumed."""
        return self.method == "eped_fit"

    @property
    def inner_edge(self) -> float | None:
        """The pedestal's inner knee, ``position - width/2``; ``None`` without a fit.

        :attr:`position` is the tanh's centre.  Which of the two a study calls
        "the pedestal top" differs, so both are available and neither is
        implied.
        """
        return None if self.width is None else self.position - 0.5 * self.width


def pedestal_top(
    x,
    values,
    *,
    quantity,
    value_std=None,
    coordinate=LEGACY_COORDINATE,
    fallback=PEDESTAL_FALLBACK_PSI_NORM,
    window=(0.4, 1.05),
    min_points=20,
    max_width=PEDESTAL_MAX_WIDTH,
):
    """Pedestal-top position of a profile, from an EPED-style fit.

    Fits the seven-parameter pedestal model
    (:func:`vaft.formula.fit_profile` with ``fitting_function='eped_tanh'``)
    over ``window`` and reports its ``x_ped``.  When the profile cannot
    support that fit the documented fallback is returned instead, with the
    reason recorded -- never silently.

    Parameters
    ----------
    x : array_like
        Radial coordinate of the profile, in ``coordinate`` [-].
    values : array_like
        Profile samples; any quantity with a pedestal [any].
    quantity : str
        What ``values`` is, recorded on the result.  Required: EPED defines
        the pedestal from total pressure, and a fit to a density or a
        temperature puts the top somewhere else, so the caller must say which
        one this boundary came from [n/a].
    value_std : array_like, optional
        Per-sample uncertainty, passed to the fit as weights [any].
    coordinate : str, optional
        Which radial coordinate ``x`` is, one of :data:`COORDINATES` [n/a].
    fallback : float, optional
        Position returned when the fit is not usable [-].
    window : tuple of float, optional
        ``(low, high)`` slice of ``x`` the fit sees; the core is excluded
        because the model's core term is only a shape, not a physical
        description [-].
    min_points : int, optional
        Fewest samples inside ``window`` that can support seven parameters [-].
    max_width : float, optional
        Widest tanh still counted as a pedestal, in ``coordinate`` [-].

    Returns
    -------
    PedestalTop
        Position, the method that produced it, and the fit when there was one [-].

    Raises
    ------
    ValueError
        ``x``, ``values`` or ``value_std`` differ in length, or ``coordinate``
        is not one of :data:`COORDINATES`.
    CoordinateUnavailableError
        The fit was not usable and ``coordinate`` is not ``psi_norm``, so the
        default fallback -- a psi_norm position -- would be wrong.

    Processing steps
    ----------------
    1. Restrict to ``window`` and drop non-finite samples.
    2. Fit ``eped_tanh`` there, weighted by ``value_std`` when given.
    3. Accept ``x_ped`` unless the fit is unusable -- too few points, a fit
       that did not converge, ``x_ped`` resting on a bound of the model's box,
       or a fitted curve that does not vary across the window by
       :data:`PEDESTAL_RESOLUTION_FACTOR` times the residual scatter, which
       means no pedestal was resolved.
    4. Otherwise return ``fallback`` and say which of those it was.

    Defaults
    --------
    ``fallback = 0.85`` is a validated-workflow default: the value decision
    D-05 names when profiles are missing or the fit is unreliable, chosen from
    the legacy windows rather than derived.  ``window = (0.4, 1.05)`` and
    ``min_points = 12`` are numerical convenience -- enough of the edge to
    resolve a pedestal, and more samples than the model has parameters.
    :data:`PEDESTAL_RESOLUTION_FACTOR` is an empirical estimate: the measured
    separation between a pure-noise fit and a real one.

    Convention
    ----------
    The position is in ``coordinate``, which defaults to ``psi_norm`` because
    that is what kinetic-profile files carry; a caller working in
    ``rho_tor_norm`` must say so, and the value is *not* converted.  ``x_ped``
    is the tanh's centre, i.e. the mid-point of the pedestal, not its knee.

    Applicability
    -------------
    Machine-independent.  Any profile with a pedestal, from any source: a
    kinetic-profile file, a fitted diagnostic profile, or a transport code.
    It takes arrays rather than an ODS so that a caller holding only a file
    can use it.

    Limitations
    -----------
    An L-mode profile has no pedestal to find and will take the fallback;
    that is the intended answer, not a failure.  The fit is unweighted in
    ``x``, so a grid that is much denser in the core than the edge biases it.

    Provenance
    ----------
    .. [D05] Migration decision D-05 (2026-09-06): no fixed canonical pedestal
       boundary; pedestal top from an EPED-style profile fit when profiles are
       available, fallback 0.85 otherwise, and every reduction records which
       method produced it.
    .. [EPED] The seven-parameter model and its parameter box are ported from
       the pedestal-fitting study in ``hsyun_GPEC``
       (``sample/transp_example/pedestal_fitting.ipynb``).  The older
       ``kinetic_analysis.ped_fitting`` modified tanh is deliberately not
       carried over.  Its slope term multiplies the tanh across the whole
       domain, so the curve runs away in both directions -- with that
       function's own defaults it reaches -3750 at x = 0.70, i.e. inside its
       own 0.7-1.0 fit window, and bounding the slope does not remove it.  It
       also cannot fit an all-negative profile such as a rotation: for a
       narrow-range one its parameter box inverts outright, and otherwise the
       seed falls outside the box and SciPy refuses it.
    """
    if coordinate not in COORDINATES:
        raise ValueError(f"coordinate must be one of {COORDINATES}, got {coordinate!r}")
    x = np.asarray(x, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    if x.shape != values.shape:
        raise ValueError(
            f"x and values must have the same length, got {x.size} and {values.size}"
        )

    def _fallback(reason):
        if coordinate != LEGACY_COORDINATE and fallback == PEDESTAL_FALLBACK_PSI_NORM:
            raise CoordinateUnavailableError(
                f"the fallback {PEDESTAL_FALLBACK_PSI_NORM} is a psi_norm position and "
                f"this profile is in {coordinate}; there is no pedestal to fit "
                f"({reason}), so pass an explicit fallback= in {coordinate} or "
                "supply the profile in psi_norm"
            )
        return PedestalTop(
            position=float(fallback),
            method="fallback",
            quantity=quantity,
            coordinate=coordinate,
            reason=reason,
        )

    low, high = window
    selected = np.isfinite(x) & np.isfinite(values) & (x >= low) & (x <= high)
    if int(np.count_nonzero(selected)) < min_points:
        return _fallback(
            f"{int(np.count_nonzero(selected))} finite samples in {window}, "
            f"fewer than min_points={min_points}"
        )

    x_fit, y_fit = x[selected], values[selected]
    std_fit = None
    if value_std is not None:
        std_fit = np.asarray(value_std, dtype=float).ravel()
        if std_fit.shape != x.shape:
            raise ValueError(
                f"value_std has {std_fit.size} samples but the profile has {x.size}"
            )
        std_fit = std_fit[selected]

    try:
        fitted, _, function, coefficients = fit_profile(
            x_fit, y_fit, std_fit, x_fit, fitting_function="eped_tanh"
        )
    except (RuntimeError, ValueError) as error:
        return _fallback(f"eped_tanh fit did not converge: {error}")

    x_ped, width = float(coefficients[3]), float(coefficients[4])
    lower, upper = eped_tanh_bounds(x_fit, y_fit)
    names = {1: "f_ped", 2: "f_sep", 3: "x_ped", 4: "width"}
    for index, name in names.items():
        span = upper[index] - lower[index]
        if min(coefficients[index] - lower[index], upper[index] - coefficients[index]) <= 1e-3 * span:
            return _fallback(f"{name} rests on a bound of the model's parameter box")

    if not (np.isfinite(x_ped) and np.isfinite(width)):
        return _fallback("the fit returned a non-finite x_ped or width")

    if width > max_width:
        return _fallback(
            f"the fitted tanh is {width:.4g} wide, more than max_width={max_width}: "
            "that is a ramp across the profile, not a pedestal"
        )

    low_fit, high_fit = float(x_fit.min()), float(x_fit.max())
    if not low_fit <= x_ped <= high_fit:
        return _fallback(
            f"x_ped={x_ped:.4g} lies outside the fitted data, which spans "
            f"[{low_fit:.4g}, {high_fit:.4g}]"
        )

    # How much the fitted curve actually varies where the data is -- not
    # ``f_ped - f_sep``, which is the tanh's asymptotic amplitude and can be
    # several times what the model says inside the window when the fit comes
    # back wide and shallow.  On pure noise the two differ by a factor of
    # three, and it is the parameter difference that looks like a pedestal.
    variation = float(np.ptp(fitted))
    residual = float(np.sqrt(np.mean((y_fit - fitted) ** 2)))
    if not (np.isfinite(variation) and np.isfinite(residual)):
        return _fallback("the fit produced non-finite values; no pedestal resolved")
    if variation <= PEDESTAL_RESOLUTION_FACTOR * residual:
        return _fallback(
            f"the fitted curve varies by {variation:.4g} across the window, not "
            f"{PEDESTAL_RESOLUTION_FACTOR}x the residual scatter {residual:.4g}: "
            "no pedestal resolved"
        )

    return PedestalTop(
        position=x_ped,
        method="eped_fit",
        quantity=quantity,
        coordinate=coordinate,
        width=width,
        fit=FittedProfile(
            function=function,
            coordinate=coordinate,
            method="eped_tanh",
            order=None,
            coefficients=coefficients,
            span=(float(x_fit.min()), float(x_fit.max())),
        ),
    )
