"""Atomic-radiation processing on OMAS data structures.

This module owns profile selection, time alignment, impurity-density
fallback and volume integration.  Numerical atomic physics remains in
:mod:`vaft.formula.atomic` and ADF11 file access in
:mod:`vaft.data.open_adas`.  Which impurity species to include and what
fraction to assume for them is machine policy, resolved by the pipeline
(:func:`vaft.omas.formula_wrapper.compute_power_balance` from ``vest.yaml``
on VEST) and passed in; this module holds no such number (issue #420).

Notation
--------
n_e       : electron density                                  [m^-3]
n_imp     : impurity density                                  [m^-3]
T_e       : electron temperature                              [eV]
L_Z       : line-radiation cooling coefficient of species Z   [W m^3]
epsilon   : line emissivity, n_e n_imp L_Z                    [W/m^3]
P_line    : radiated line power, int epsilon dV               [W]
V         : cumulative enclosed plasma volume                 [m^3]
rho_tor_norm : normalized toroidal-flux radius                [-]

Provenance
----------
.. [ADAS] OPEN-ADAS ADF11 tables, read through :mod:`vaft.data.open_adas`.
.. [FORMULA] :func:`vaft.formula.atomic.line_cooling_coefficient`, the
   interpolation of those tables in ``(n_e, T_e)``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
from numpy import ndarray
from omas import ODS
from scipy.interpolate import interp1d

from vaft.compat import trapz_compat
from vaft.data.open_adas import ADASDataError
from vaft.formula.atomic import line_cooling_coefficient
from vaft.spectroscopy import ATOMIC_NUMBERS, ELEMENT_NAMES


logger = logging.getLogger(__name__)

#: What each species' impurity density was taken from, per call.
IMPURITY_SOURCES = ("profile", "configured", "inferred_from_zeff", "none")

#: The VEST species and fractions this module used to default to. They are
#: machine policy, not atomic physics, and live in ``vest.yaml`` now
#: (``diagnostics.core_profiles.impurities``), resolved by the pipeline
#: through :func:`vaft.machine_mapping.core_profiles.vest_core_profiles_policy`
#: (issue #420). Reachable here for one cycle with a warning.
_RETIRED_DEFAULTS = {
    "DEFAULT_LINE_RADIATION_SPECIES": ("C", "O"),
    "DEFAULT_IMPURITY_FRACTIONS": {"C": 1.0e-2, "O": 1.0e-2},
}
_DEFAULT_TIME_MATCH_ATOL = 1.0e-6


def __getattr__(name: str):
    if name in _RETIRED_DEFAULTS:
        import warnings

        warnings.warn(
            f"vaft.process.atomic.{name} is VEST policy and no longer a default of "
            "this module; resolve it with "
            "vaft.machine_mapping.core_profiles.vest_core_profiles_policy(shot) and "
            "pass the result in (issue #420)",
            DeprecationWarning,
            stacklevel=2,
        )
        return _RETIRED_DEFAULTS[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
#: Shared with :mod:`vaft.spectroscopy`, which owns the element vocabulary so
#: that ``vaft.plot`` can resolve species without importing OMAS through this
#: module.  ``D`` and ``T`` stay distinct keys here because ADAS files are
#: named that way; :class:`vaft.spectroscopy.Species` records them as hydrogen
#: with a mass number instead.
_ATOMIC_NUMBERS = ATOMIC_NUMBERS
_ELEMENT_NAMES = ELEMENT_NAMES



def compute_time_match_atol(time_array: ndarray, base_atol: float = _DEFAULT_TIME_MATCH_ATOL) -> float:
    """An absolute time-matching tolerance adapted to the native time spacing.

    ``max(base_atol, 0.25 * min(diff(unique_finite_times)))``, which absorbs
    floating-point drift in a time coordinate without ever matching an
    adjacent physical slice.

    Parameters
    ----------
    time_array : np.ndarray
        The time coordinate to be matched against [s].
    base_atol : float, optional
        Floor on the tolerance [s].

    Returns
    -------
    float
        The tolerance; ``base_atol`` when fewer than two distinct finite times
        exist [s].

    Raises
    ------
    ValueError
        ``base_atol`` non-finite or negative.

    Defaults
    --------
    ``base_atol = 1e-6`` s and the factor ``0.25`` are numerical conveniences:
    a quarter of the smallest spacing can never reach the neighbouring slice.

    Applicability
    -------------
    Machine-independent.
    """

    if not np.isfinite(base_atol) or base_atol < 0.0:
        raise ValueError("base_atol must be finite and non-negative")

    arr = np.asarray(time_array, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float(base_atol)
    positive = np.diff(np.unique(np.sort(arr)))
    positive = positive[positive > 0.0]
    return float(max(base_atol, 0.25 * float(np.min(positive)))) if positive.size else float(base_atol)


def find_time_match_index(time_array: ndarray, target_time: float) -> Optional[int]:
    """Index of the time closest to ``target_time``, within the adaptive tolerance.

    Parameters
    ----------
    time_array : np.ndarray
        The time coordinate [s].
    target_time : float
        The time to match [s].

    Returns
    -------
    int or None
        Index of the closest time within :func:`compute_time_match_atol`;
        ``None`` for an empty array, a non-finite target, or no candidate within
        tolerance [-].

    Applicability
    -------------
    Machine-independent.
    """

    arr = np.asarray(time_array, dtype=float).reshape(-1)
    if arr.size == 0 or not np.isfinite(target_time):
        return None
    close = np.where(np.isclose(arr, float(target_time), rtol=0.0, atol=compute_time_match_atol(arr)))[0]
    if close.size == 0:
        return None
    return int(close[np.argmin(np.abs(arr[close] - float(target_time)))])


def normalize_atomic_symbol(label: Any) -> Optional[str]:
    """Normalize an element or ion label to a supported atomic symbol.

    Full English names and charge-decorated labels are accepted --
    ``carbon6+`` gives ``C``, ``neon`` gives ``Ne``.

    Parameters
    ----------
    label : str, bytes or None
        The label as found in an ODS ``ion.label`` or a species list [-].

    Returns
    -------
    str or None
        The symbol, or ``None`` for an unrecognized label so the caller decides
        whether absence is fatal [-].

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Only the elements in the module's table are recognized (H through W, the
    common tokamak species); isotopes D and T map to their own symbols.
    """

    if label is None:
        return None
    text = label.decode("utf-8", errors="ignore") if isinstance(label, bytes) else str(label)
    match = re.match(r"([A-Za-z]+)", text.strip())
    if not match:
        return None
    token = match.group(1)
    named = _ELEMENT_NAMES.get(token.lower())
    if named is not None:
        return named
    exact = token[0].upper() + token[1:].lower()
    if exact in _ATOMIC_NUMBERS:
        return exact
    return None


def _sanitize_rho_grid(rho: ndarray) -> Optional[ndarray]:
    """Return a sorted unique finite radial grid with at least two points."""
    values = np.asarray(rho, dtype=float).reshape(-1)
    values = np.unique(np.sort(values[np.isfinite(values)]))
    return values if values.size >= 2 else None


def _interp_profile_to_target(
    rho_src: ndarray,
    profile_src: ndarray,
    rho_target: ndarray,
) -> Optional[ndarray]:
    """Linearly map a finite 1D profile onto a target normalized-radius grid.

    Duplicate source coordinates are discarded and values outside the source
    interval use constant edge extrapolation. ``None`` indicates that a valid
    interpolation could not be constructed.
    """
    rho = np.asarray(rho_src, dtype=float).reshape(-1)
    profile = np.asarray(profile_src, dtype=float).reshape(-1)
    target = np.asarray(rho_target, dtype=float).reshape(-1)
    if rho.size != profile.size or rho.size < 2 or target.size == 0:
        return None
    finite = np.isfinite(rho) & np.isfinite(profile)
    if np.count_nonzero(finite) < 2:
        return None
    rho, profile = rho[finite], profile[finite]
    order = np.argsort(rho)
    rho, profile = rho[order], profile[order]
    unique = np.concatenate(([True], np.diff(rho) > 1.0e-10))
    rho, profile = rho[unique], profile[unique]
    if rho.size < 2:
        return None
    output = np.full(target.shape, np.nan, dtype=float)
    valid = np.isfinite(target)
    if np.any(valid):
        interpolation = interp1d(
            rho,
            profile,
            kind="linear",
            bounds_error=False,
            fill_value=(profile[0], profile[-1]),
        )
        output[valid] = interpolation(target[valid])
    return output


def _infer_impurity_fraction_from_zeff(z_eff: Optional[float], species: str) -> Optional[float]:
    r"""Infer ``n_imp / n_e`` from a single-impurity effective-charge model.

    For hydrogenic main ions,

    .. math:: Z_\mathrm{eff}=1+\frac{n_\mathrm{imp}}{n_e}Z(Z-1),

    hence ``n_imp / n_e = (Z_eff - 1) / (Z * (Z - 1))``. Negative inferred
    fractions are clipped to zero. ``None`` is returned when inference is not
    defined.
    """
    if z_eff is None or not np.isfinite(z_eff):
        return None
    atomic_number = _ATOMIC_NUMBERS.get(species)
    if atomic_number is None or atomic_number <= 1:
        return None
    return max((float(z_eff) - 1.0) / float(atomic_number * (atomic_number - 1)), 0.0)


def _impurity_fraction_profile(
    cp_slice: ODS,
    rho_cp: ndarray,
    rho_target: ndarray,
    ne_target: ndarray,
    species: str,
) -> Optional[ndarray]:
    """Extract and map an impurity fraction profile from ``core_profiles``.

    ``ion.density`` is preferred over ``ion.density_thermal``. The mapped ion
    density is divided by the mapped electron density, with invalid or negative
    ratios replaced by zero. ``None`` means that the requested ion profile was
    unavailable or unusable.
    """
    if "ion" not in cp_slice:
        return None
    for index in range(len(cp_slice["ion"])):
        ion = cp_slice["ion"][index]
        if normalize_atomic_symbol(ion["label"] if "label" in ion else None) != species:
            continue
        density_key = "density" if "density" in ion else ("density_thermal" if "density_thermal" in ion else None)
        if density_key is None:
            continue
        impurity = _interp_profile_to_target(rho_cp, np.asarray(ion[density_key], dtype=float), rho_target)
        if impurity is None:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = np.where(ne_target > 0.0, impurity / ne_target, 0.0)
        return np.clip(np.nan_to_num(fraction, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    return None


def integrate_emissivity_profile(
    emissivity_profile: ndarray,
    volume_profile: Optional[ndarray],
    total_volume: float,
) -> float:
    """Integrate an emissivity profile to a total radiated power.

    Parameters
    ----------
    emissivity_profile : np.ndarray
        Local emissivity on the radial grid [W/m^3].
    volume_profile : np.ndarray or None
        Cumulative enclosed volume on the same grid, or ``None`` [m^3].
    total_volume : float
        Total plasma volume, for the fallback [m^3].

    Returns
    -------
    float
        ``P = int epsilon dV``, or the fallback, or ``0`` [W].

    Processing steps
    ----------------
    1. If ``volume_profile`` is given, matches the emissivity's shape and has at
       least two finite points spanning a non-zero volume: trapezoidal
       integration of ``epsilon`` against ``V``, in volume order.
    2. Else, if ``total_volume`` is finite and positive:
       ``nanmean(epsilon) * total_volume``.
    3. Else ``0``.

    Convention
    ----------
    ``volume_profile`` is the *cumulative enclosed* volume ``V(rho)``, so
    ``dV`` is its difference; it is not a shell volume.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The two fallbacks are silent: a volume profile of the wrong length drops to
    the mean-times-volume estimate, and a missing volume to zero, with no
    warning from this function.
    """

    emissivity = np.asarray(emissivity_profile, dtype=float)
    finite_emissivity = np.isfinite(emissivity)
    if not np.any(finite_emissivity):
        return 0.0
    if volume_profile is not None:
        volume = np.asarray(volume_profile, dtype=float)
        if volume.shape == emissivity.shape:
            finite = finite_emissivity & np.isfinite(volume)
            if np.count_nonzero(finite) >= 2:
                volume_finite, emissivity_finite = volume[finite], emissivity[finite]
                order = np.argsort(volume_finite)
                volume_finite, emissivity_finite = volume_finite[order], emissivity_finite[order]
                if np.ptp(volume_finite) > 0.0:
                    return float(trapz_compat(emissivity_finite, x=volume_finite))
    if np.isfinite(total_volume) and total_volume > 0.0:
        return float(np.nanmean(emissivity[finite_emissivity]) * total_volume)
    return 0.0


def compute_line_radiation_power_series(
    ods: ODS,
    eq_indices: List[int],
    eq_times: ndarray,
    volume_series: ndarray,
    line_radiation_species: Optional[List[str]] = None,
    impurity_fractions: Optional[Dict[str, float]] = None,
    Z_eff: Optional[float] = None,
    *,
    return_assumptions: bool = False,
):
    """Line-radiation power for matched equilibrium slices, from the stored kinetic profiles.

    For each species and time slice the local emissivity is
    ``epsilon = n_e n_imp L_Z(n_e, T_e)`` and the power ``P = int epsilon dV``.

    Parameters
    ----------
    ods : ODS
        Carries ``core_profiles.profiles_1d`` and ``equilibrium.time_slice`` [-].
    eq_indices : list of int
        Equilibrium slice indices to evaluate [-].
    eq_times : array_like
        Time of each of those slices [s].
    volume_series : array_like
        Total plasma volume of each slice, for the integration fallback [m^3].
    line_radiation_species : list of str, optional
        Species to include; ``None`` for every species that has a configured
        fraction or an ion density profile in the ODS [-].
    impurity_fractions : dict of str to float, optional
        ``n_imp / n_e`` per species, used when the ODS carries no profile for
        it; ``None`` means no assumption [-].
    Z_eff : float, optional
        Effective charge, used only to infer a single species' fraction when
        nothing else is known [-].
    return_assumptions : bool, optional
        Also return what each species' density was taken from [-].

    Returns
    -------
    np.ndarray or tuple of (np.ndarray, dict)
        Non-negative line-radiation power per ``eq_indices`` entry, and with
        ``return_assumptions`` a dict of ``species``, per-species
        ``impurity_source`` (``profile``, ``configured``, ``inferred_from_zeff``
        or ``none``), the ``impurity_fraction`` used, and ``z_eff`` [W].

    Raises
    ------
    ValueError
        Mismatched ``eq_times`` / ``volume_series`` lengths, non-finite times,
        an unrecognized species label, or a negative or non-finite fraction.

    Processing steps
    ----------------
    1. Match each equilibrium time to a ``core_profiles`` slice
       (:func:`find_time_match_index`); an unmatched slice radiates nothing.
    2. Interpolate ``n_e``, ``T_e`` from the slice's ``rho_tor_norm`` onto the
       equilibrium's ``rho_tor_norm`` (constant edge extrapolation).
    3. For each species, take ``n_imp / n_e`` from, in order: the slice's own
       ion density profile; ``impurity_fractions``; a single-species inference
       from ``Z_eff``; else zero with one warning per species.
    4. ``L_Z`` from the ADAS tables [FORMULA]_; ``epsilon = L_Z n_e n_imp``.
    5. :func:`integrate_emissivity_profile` per species, summed; clipped at
       zero.

    Input semantics
    ---------------
    Stored kinetic profiles (fitted or synthetic) on their ``core_profiles``
    grid, plus the equilibrium grid and volume.

    Output semantics
    ----------------
    Synthetic: a modelled diagnostic-like quantity, not a measurement.

    Defaults
    --------
    ``impurity_fractions = None`` and ``line_radiation_species = None`` mean
    *no assumption*.  The VEST carbon and oxygen fractions that used to be
    this function's defaults are machine-specific settings, each an assumed
    value in ``vest.yaml`` resolved by the pipeline.  ``Z_eff = None``
    disables the single-species inference, a numerical convenience.

    Convention
    ----------
    The radial coordinate is ``rho_tor_norm`` throughout: the ``core_profiles``
    grid and the equilibrium ``profiles_1d`` both carry it and are interpolated
    against each other in it.  ``n_i`` for the impurity is ``fraction * n_e``
    where a fraction is used; ``Z_eff`` inference assumes hydrogenic main ions,
    ``n_imp/n_e = (Z_eff - 1)/(Z(Z - 1))``.

    Assumptions
    -----------
    Coronal equilibrium for ``L_Z``; one impurity species at a time with no
    mutual dilution; the ``core_profiles`` slice at the matched time describes
    the equilibrium slice.

    Applicability
    -------------
    Machine-independent.  Species and fractions arrive as arguments.

    Limitations
    -----------
    Every failure mode -- no ``core_profiles``, no matching time, no ``n_e``
    or ``T_e``, ADAS data unavailable, no fraction known -- yields zero for
    that slice or species with a logged warning rather than an error, to keep
    an offline power balance running; the returned assumptions are how a
    caller can tell.  A ``core_profiles`` slice stored on ``rho_pol_norm``
    only (no equilibrium at its time) has no ``rho_tor_norm`` and is skipped.

    Provenance
    ----------
    .. [1] ADAS tables [ADAS]_ through the cooling-coefficient kernel [FORMULA]_.
    """

    eq_indices = list(eq_indices)
    eq_times = np.asarray(eq_times, dtype=float).reshape(-1)
    volume_series = np.asarray(volume_series, dtype=float).reshape(-1)
    if eq_times.size != len(eq_indices):
        raise ValueError("eq_times must contain one value per eq_indices entry")
    if volume_series.size != len(eq_indices):
        raise ValueError("volume_series must contain one value per eq_indices entry")
    if not np.all(np.isfinite(eq_times)):
        raise ValueError("eq_times must contain only finite values")

    result = np.zeros(len(eq_indices), dtype=float)
    assumptions: Dict[str, Any] = {"species": [], "impurity_source": {}, "impurity_fraction": {}, "z_eff": Z_eff}

    def _done():
        return (result, assumptions) if return_assumptions else result

    if "core_profiles.profiles_1d" not in ods:
        logger.warning("core_profiles missing; line radiation set to zero.")
        return _done()

    profiles = ods["core_profiles.profiles_1d"]
    profile_times = np.asarray(
        [float(profiles[j]["time"]) if "time" in profiles[j] else float(j) for j in range(len(profiles))],
        dtype=float,
    )
    # No fraction means no assumption: a species without a profile, a configured
    # fraction or a Z_eff inference radiates nothing, with a warning. The VEST
    # C/O defaults that used to live here are policy, resolved by the pipeline.
    fraction_map: Dict[str, float] = {}
    if impurity_fractions:
        for label, value in impurity_fractions.items():
            species = normalize_atomic_symbol(label)
            if species is None:
                raise ValueError(f"Invalid impurity species label: {label!r}")
            fraction = float(value)
            if not np.isfinite(fraction) or fraction < 0.0:
                raise ValueError(f"Impurity fraction for {species} must be finite and non-negative")
            fraction_map[species] = fraction

    if line_radiation_species is None:
        # the species anything is known about: a configured fraction, or an
        # ion density profile in any core_profiles slice
        species_list = list(fraction_map)
        for j in range(len(profiles)):
            cp = profiles[j]
            if "ion" not in cp:
                continue
            for k in range(len(cp["ion"])):
                ion = cp["ion"][k]
                symbol = normalize_atomic_symbol(ion["label"] if "label" in ion else None)
                if symbol is not None and symbol not in species_list and _ATOMIC_NUMBERS.get(symbol, 0) > 1:
                    species_list.append(symbol)
    else:
        species_list = []
        for label in line_radiation_species:
            species = normalize_atomic_symbol(label)
            if species is None:
                raise ValueError(f"Invalid line-radiation species: {label!r}")
            if species not in species_list:
                species_list.append(species)
    assumptions["species"] = list(species_list)
    for species in species_list:
        assumptions["impurity_source"][species] = "none"
    if not species_list:
        logger.warning("no line-radiation species: no impurity fraction configured and no ion profile present; line radiation set to zero.")
        return _done()

    warned_zero_fraction: set[str] = set()
    unavailable_species: set[str] = set()
    for output_index, eq_index in enumerate(eq_indices):
        profile_index = find_time_match_index(profile_times, float(eq_times[output_index]))
        if profile_index is None:
            continue
        cp_slice = profiles[profile_index]
        eq_slice = ods["equilibrium.time_slice"][eq_index]
        grid = cp_slice["grid"] if "grid" in cp_slice else (
            ods["core_profiles.grid"] if "core_profiles.grid" in ods else ODS()
        )
        if (
            "rho_tor_norm" not in grid
            or "electrons.density" not in cp_slice
            or "electrons.temperature" not in cp_slice
        ):
            continue
        rho_cp = np.asarray(grid["rho_tor_norm"], dtype=float)
        ne_cp = np.asarray(cp_slice["electrons.density"], dtype=float)
        te_cp = np.asarray(cp_slice["electrons.temperature"], dtype=float)
        eq_profiles = eq_slice["profiles_1d"] if "profiles_1d" in eq_slice else ODS()
        rho_eq = _sanitize_rho_grid(eq_profiles["rho_tor_norm"]) if "rho_tor_norm" in eq_profiles else None
        rho_target = rho_eq if rho_eq is not None else _sanitize_rho_grid(rho_cp)
        if rho_target is None:
            continue
        ne = _interp_profile_to_target(rho_cp, ne_cp, rho_target)
        te = _interp_profile_to_target(rho_cp, te_cp, rho_target)
        if ne is None or te is None:
            continue
        finite = np.isfinite(ne) & np.isfinite(te) & (ne > 0.0) & (te > 0.0)
        if not np.any(finite):
            continue

        volume_profile = None
        if "volume" in eq_profiles and "rho_tor_norm" in eq_profiles:
            volume_profile = _interp_profile_to_target(
                np.asarray(eq_profiles["rho_tor_norm"], dtype=float),
                np.asarray(eq_profiles["volume"], dtype=float),
                rho_target,
            )
        total_volume = float(volume_series[output_index])
        slice_power = 0.0
        for species in species_list:
            if species in unavailable_species:
                continue
            fraction_profile = _impurity_fraction_profile(cp_slice, rho_cp, rho_target, ne, species)
            if fraction_profile is not None:
                assumptions["impurity_source"][species] = "profile"
            else:
                fraction = fraction_map.get(species)
                source = "configured" if fraction is not None else "none"
                if fraction is None and len(species_list) == 1:
                    fraction = _infer_impurity_fraction_from_zeff(Z_eff, species)
                    if fraction is not None:
                        source = "inferred_from_zeff"
                if assumptions["impurity_source"][species] != "profile":
                    assumptions["impurity_source"][species] = source
                if fraction is not None:
                    assumptions["impurity_fraction"][species] = float(fraction)
                fraction_profile = np.full_like(ne, 0.0 if fraction is None else fraction, dtype=float)
            fraction_profile = np.where(
                np.isfinite(fraction_profile), np.clip(fraction_profile, 0.0, None), 0.0
            )
            if not np.any(fraction_profile > 0.0):
                if species not in warned_zero_fraction:
                    logger.warning("No impurity fraction available for %s; its line radiation is zero.", species)
                    warned_zero_fraction.add(species)
                continue

            coefficients = np.zeros_like(ne, dtype=float)
            try:
                coefficients[finite] = np.asarray(
                    line_cooling_coefficient(species, ne[finite], te[finite]), dtype=float
                ).reshape(-1)
            except ADASDataError as exc:
                logger.warning(
                    "OPEN-ADAS data unavailable for %s; its line radiation is zero: %s",
                    species,
                    exc,
                )
                unavailable_species.add(species)
                continue
            impurity_density = fraction_profile * ne
            emissivity = np.where(
                finite,
                coefficients * np.clip(ne, 0.0, None) * np.clip(impurity_density, 0.0, None),
                np.nan,
            )
            slice_power += integrate_emissivity_profile(emissivity, volume_profile, total_volume)
        result[output_index] = max(float(slice_power), 0.0)
    return _done()


__all__ = [
    "IMPURITY_SOURCES",
    "compute_line_radiation_power_series",
    "compute_time_match_atol",
    "find_time_match_index",
    "integrate_emissivity_profile",
    "normalize_atomic_symbol",
]
