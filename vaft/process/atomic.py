"""High-level atomic-radiation processing for OMAS data structures.

This module owns profile selection, time alignment, impurity-density fallback,
and volume integration. Numerical atomic physics remains in
``vaft.formula.atomic`` and ADF11 file access remains in
``vaft.data.open_adas``.
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
from vaft.formula.atomic import line_cooling_coefficient


logger = logging.getLogger(__name__)

DEFAULT_LINE_RADIATION_SPECIES = ("C", "O")
DEFAULT_IMPURITY_FRACTIONS = {"C": 1.0e-2, "O": 1.0e-2}
_DEFAULT_TIME_MATCH_ATOL = 1.0e-6
_ATOMIC_NUMBERS = {
    "H": 1, "D": 1, "T": 1, "He": 2, "Li": 3, "Be": 4, "B": 5,
    "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Al": 13,
    "Si": 14, "S": 16, "Cl": 17, "Ar": 18, "Ca": 20, "Ti": 22,
    "Fe": 26, "Ni": 28, "Kr": 36, "Mo": 42, "Xe": 54, "W": 74,
}
_ELEMENT_NAMES = {
    "hydrogen": "H", "deuterium": "D", "tritium": "T", "helium": "He",
    "lithium": "Li", "beryllium": "Be", "boron": "B", "carbon": "C",
    "nitrogen": "N", "oxygen": "O", "fluorine": "F", "neon": "Ne",
    "aluminium": "Al", "aluminum": "Al", "silicon": "Si", "sulfur": "S",
    "sulphur": "S", "chlorine": "Cl", "argon": "Ar", "calcium": "Ca",
    "titanium": "Ti", "iron": "Fe", "nickel": "Ni", "krypton": "Kr",
    "molybdenum": "Mo", "xenon": "Xe", "tungsten": "W",
}


def compute_time_match_atol(time_array: ndarray, base_atol: float = _DEFAULT_TIME_MATCH_ATOL) -> float:
    """Return an absolute matching tolerance adapted to the native time spacing.

    ``max(base_atol, 0.25 * min(diff(unique_finite_times)))`` in seconds, which
    absorbs floating-point drift in a time coordinate without ever matching an
    adjacent physical slice. Falls back to ``base_atol`` when fewer than two
    distinct finite times are available.
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
    """Find the index of the time closest to *target_time*, both in seconds.

    Returns ``None`` for an empty array, a non-finite target, or when no
    candidate falls within :func:`compute_time_match_atol`.
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

    Full English names and charge-decorated labels are accepted, e.g.
    ``carbon6+`` -> ``C`` and ``neon`` -> ``Ne``. Invalid labels return
    ``None`` so callers can choose whether absence is fatal.
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
    r"""Integrate an emissivity profile to total radiated power.

    When cumulative enclosed volume :math:`V(\rho)` is available and matches
    the emissivity shape, VAFT evaluates

    .. math:: P=\int \epsilon(\rho)\,dV

    by trapezoidal integration in volume coordinates. Otherwise it falls back to
    ``nanmean(emissivity) * total_volume``. No finite emissivity or no positive
    usable volume produces zero.

    Units: ``emissivity_profile`` in W/m^3, ``volume_profile`` (cumulative
    enclosed volume) and ``total_volume`` in m^3, result in W.
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
) -> ndarray:
    r"""Compute line-radiation power for matched equilibrium slices.

    For each requested species and time slice, the local emissivity is

    .. math::

        \epsilon_{\mathrm{line}}(\rho)
        = n_e(\rho)n_\mathrm{imp}(\rho)
          L_{Z,\mathrm{line}}(n_e(\rho),T_e(\rho)),

    and total power is :math:`P_{\mathrm{line}}=\int\epsilon\,dV`.

    Impurity density follows this precedence:

    1. matching ``core_profiles.profiles_1d.ion`` density profile;
    2. scalar ``impurity_fractions[species]``;
    3. single-species inference from ``Z_eff``;
    4. zero, with one warning per species.

    Parameters
    ----------
    ods : ODS
        OMAS data containing equilibrium and core-profile time slices.
    eq_indices : list of int
        Equilibrium slice indices to process.
    eq_times : array-like
        Time in seconds for each equilibrium index.
    volume_series : array-like
        Total plasma volume in m^3 for each output slice.
    line_radiation_species : list of str, optional
        Species to include. Defaults to carbon and oxygen.
    impurity_fractions : dict, optional
        Scalar ``n_imp / n_e`` values. ``None`` uses the default C/O fractions;
        an empty dictionary disables those defaults.
    Z_eff : float, optional
        Effective charge used only for the single-species fallback.

    Returns
    -------
    numpy.ndarray
        Non-negative line-radiation power in W, one value per ``eq_indices``.

    Raises
    ------
    ValueError
        If array lengths, species labels, or impurity fractions are invalid.
    ADASDataError
        Propagated when atomic data cannot be resolved or evaluated.

    Notes
    -----
    Missing kinetic profiles or unmatched time slices produce zero for those
    slices. Atomic-data failures are deliberately not converted to zero.
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
    if "core_profiles.profiles_1d" not in ods:
        logger.warning("core_profiles missing; line radiation set to zero.")
        return result

    profiles = ods["core_profiles.profiles_1d"]
    profile_times = np.asarray(
        [float(profiles[j]["time"]) if "time" in profiles[j] else float(j) for j in range(len(profiles))],
        dtype=float,
    )
    fraction_map: Dict[str, float] = {}
    if impurity_fractions is None:
        fraction_map = dict(DEFAULT_IMPURITY_FRACTIONS)
    elif impurity_fractions:
        for label, value in impurity_fractions.items():
            species = normalize_atomic_symbol(label)
            if species is None:
                raise ValueError(f"Invalid impurity species label: {label!r}")
            fraction = float(value)
            if not np.isfinite(fraction) or fraction < 0.0:
                raise ValueError(f"Impurity fraction for {species} must be finite and non-negative")
            fraction_map[species] = fraction

    if line_radiation_species is None:
        species_list = list(DEFAULT_LINE_RADIATION_SPECIES)
    else:
        species_list = []
        for label in line_radiation_species:
            species = normalize_atomic_symbol(label)
            if species is None:
                raise ValueError(f"Invalid line-radiation species: {label!r}")
            if species not in species_list:
                species_list.append(species)
    if not species_list:
        return result

    warned_zero_fraction: set[str] = set()
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
            fraction_profile = _impurity_fraction_profile(cp_slice, rho_cp, rho_target, ne, species)
            if fraction_profile is None:
                fraction = fraction_map.get(species)
                if fraction is None and len(species_list) == 1:
                    fraction = _infer_impurity_fraction_from_zeff(Z_eff, species)
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
            coefficients[finite] = np.asarray(
                line_cooling_coefficient(species, ne[finite], te[finite]), dtype=float
            ).reshape(-1)
            impurity_density = fraction_profile * ne
            emissivity = np.where(
                finite,
                coefficients * np.clip(ne, 0.0, None) * np.clip(impurity_density, 0.0, None),
                np.nan,
            )
            slice_power += integrate_emissivity_profile(emissivity, volume_profile, total_volume)
        result[output_index] = max(float(slice_power), 0.0)
    return result


__all__ = [
    "DEFAULT_IMPURITY_FRACTIONS",
    "DEFAULT_LINE_RADIATION_SPECIES",
    "compute_line_radiation_power_series",
    "compute_time_match_atol",
    "find_time_match_index",
    "integrate_emissivity_profile",
    "normalize_atomic_symbol",
]
