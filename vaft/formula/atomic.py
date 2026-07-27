"""Atomic equilibrium and line-radiation formulae backed by OPEN-ADAS ADF11.

The equilibrium construction follows MIT-licensed work, Copyright (c) 2021
Francesco Sciortino. See the third-party notices in the project README.

This module is the numerical layer of VAFT's atomic package.  It accepts SI
electron densities, evaluates native ADF11 tables in log10 space, and returns
charge-state fractions or cooling coefficients without depending on OMAS.
"""

from __future__ import annotations

from os import PathLike
from typing import TypeAlias

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from vaft.data.open_adas import (
    ADF11Data,
    default_adf11_files,
    get_adf11_path,
    read_adf11,
)


ADF11Source: TypeAlias = ADF11Data | str | PathLike[str]


def _validated_profiles(ne_m3, te_eV) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Broadcast and validate electron density and temperature profiles.

    Both profiles must be non-empty, finite, and strictly positive.  The
    returned arrays share a common shape according to NumPy broadcasting.
    """
    ne, te = np.broadcast_arrays(np.asarray(ne_m3, dtype=float), np.asarray(te_eV, dtype=float))
    if ne.size == 0:
        raise ValueError("Electron density and temperature must not be empty")
    if not np.all(np.isfinite(ne)) or np.any(ne <= 0.0):
        raise ValueError("Electron density must contain only finite positive values in m^-3")
    if not np.all(np.isfinite(te)) or np.any(te <= 0.0):
        raise ValueError("Electron temperature must contain only finite positive values in eV")
    return ne, te, ne.shape


def _as_adf11(source: ADF11Source, expected_type: str) -> ADF11Data:
    """Resolve an ADF11 source and enforce its coefficient class."""
    table = source if isinstance(source, ADF11Data) else read_adf11(source)
    if table.file_type != expected_type:
        raise ValueError(f"Expected a {expected_type!r} ADF11 table, got {table.file_type!r}")
    return table


def interpolate_adf11(
    table: ADF11Data,
    ne_m3,
    te_eV,
    *,
    multiply_density: bool = False,
) -> np.ndarray:
    r"""Interpolate every charge-state block of an ADF11 table.

    ADF11 stores :math:`\log_{10} C_z` on rectangular grids of
    :math:`\log_{10} n_e[\mathrm{cm}^{-3}]` and
    :math:`\log_{10} T_e[\mathrm{eV}]`.  VAFT evaluates

    .. math::

        C_z(n_e,T_e) = 10^{\mathcal{I}_2[\log_{10} C_z]},

    where :math:`\mathcal{I}_2` is bilinear interpolation in the two log
    coordinates.  Linear extrapolation is retained outside the tabulated grid
    to match the established ADF11 calculation path.

    ``ne_m3`` (m^-3) and ``te_eV`` (eV) are broadcast together; the result has
    shape ``broadcast(ne_m3, te_eV).shape + (n_blocks,)``.  ``multiply_density``
    scales by :math:`n_e[\mathrm{cm}^{-3}]`, converting ACD/SCD coefficients
    from cm^3/s to transition rates in s^-1.  Empty, non-finite or non-positive
    profiles raise ``ValueError``.
    """

    ne, te, shape = _validated_profiles(ne_m3, te_eV)
    log_ne_cm3 = np.log10(ne * 1.0e-6)
    log_te_eV = np.log10(te)
    points = np.column_stack((log_te_eV.reshape(-1), log_ne_cm3.reshape(-1)))

    values = np.empty((points.shape[0], table.n_charge_states), dtype=float)
    for index, block in enumerate(table.log_coefficients):
        interpolator = RegularGridInterpolator(
            (table.log_temperature_eV, table.log_density_cm3),
            block,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        values[:, index] = np.power(10.0, interpolator(points))
    if multiply_density:
        values *= (ne.reshape(-1) * 1.0e-6)[:, None]
    return values.reshape(shape + (table.n_charge_states,))


def fractional_abundances(
    ne_m3,
    te_eV,
    acd: ADF11Source,
    scd: ADF11Source,
) -> np.ndarray:
    r"""Calculate ionization-equilibrium fractional abundances.

    For adjacent charge states, steady-state balance gives

    .. math::

        \frac{f_{z+1}}{f_z} = \frac{S_z(n_e,T_e)}{\alpha_{z+1}(n_e,T_e)},

    where :math:`S_z` is the SCD effective ionization coefficient and
    :math:`\alpha_{z+1}` is the ACD effective recombination coefficient.  The
    relative populations are constructed as

    .. math::

        \tilde f_0=1,\qquad
        \tilde f_z=\prod_{j=0}^{z-1}\frac{S_j}{\alpha_{j+1}},\qquad
        f_z=\frac{\tilde f_z}{\sum_k\tilde f_k}.

    ``ne_m3`` is in m^-3 and ``te_eV`` in eV; ``acd``/``scd`` are effective
    recombination and ionization tables (``ADF11Data`` or path-like).  The
    returned final axis has ``n_rate_blocks + 1`` entries, running from neutral
    to fully stripped and summing to one.  Incompatible charge-state counts or
    invalid interpolated rates raise ``ValueError``.
    """

    acd_table = _as_adf11(acd, "acd")
    scd_table = _as_adf11(scd, "scd")
    recombination = interpolate_adf11(acd_table, ne_m3, te_eV, multiply_density=True)
    ionization = interpolate_adf11(scd_table, ne_m3, te_eV, multiply_density=True)
    if recombination.shape != ionization.shape:
        raise ValueError(
            "ACD and SCD tables produced incompatible charge-state shapes: "
            f"{recombination.shape} != {ionization.shape}"
        )
    if np.any(recombination <= 0.0) or not np.all(np.isfinite(recombination)):
        raise ValueError("Interpolated recombination rates must be finite and positive")
    if np.any(ionization < 0.0) or not np.all(np.isfinite(ionization)):
        raise ValueError("Interpolated ionization rates must be finite and non-negative")

    ratio = ionization / recombination
    leading = np.ones(ratio.shape[:-1] + (1,), dtype=float)
    relative = np.cumprod(np.concatenate((leading, ratio), axis=-1), axis=-1)
    normalizer = np.sum(relative, axis=-1, keepdims=True)
    if np.any(normalizer <= 0.0) or not np.all(np.isfinite(normalizer)):
        raise ValueError("Could not normalize atomic fractional abundances")
    return relative / normalizer


def _interpolate_temperature_only(table: ADF11Data, te_eV) -> np.ndarray:
    r"""Interpolate a density-independent ADF11 table in log-temperature.

    Unresolved PLT tables are evaluated on their first density column using
    :math:`C_z(T_e)=10^{\mathcal{I}_1[\log_{10}C_z]}`.
    """

    te = np.asarray(te_eV, dtype=float)
    if te.size == 0 or not np.all(np.isfinite(te)) or np.any(te <= 0.0):
        raise ValueError("Electron temperature must contain only finite positive values in eV")
    log_te = np.log10(te).reshape(-1)
    values = np.empty((log_te.size, table.n_charge_states), dtype=float)
    for index, block in enumerate(table.log_coefficients):
        interpolation = RegularGridInterpolator(
            (table.log_temperature_eV,),
            block[:, 0],
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        values[:, index] = np.power(10.0, interpolation(log_te[:, None]))
    return values.reshape(te.shape + (table.n_charge_states,))


def _resolve_source(
    source: ADF11Source | None,
    file_type: str,
    filenames: dict[str, str],
    cache_dir: str | PathLike[str] | None,
) -> ADF11Data:
    """Use an explicit ADF11 source or resolve the configured OPEN-ADAS file."""
    if source is None:
        source = get_adf11_path(filenames[file_type], cache_dir=cache_dir)
    return _as_adf11(source, file_type)


def line_cooling_coefficient(
    species: str,
    ne_m3,
    te_eV,
    *,
    acd: ADF11Source | None = None,
    scd: ADF11Source | None = None,
    plt: ADF11Source | None = None,
    cache_dir: str | PathLike[str] | None = None,
) -> np.ndarray:
    r"""Calculate the equilibrium line-radiation cooling coefficient.

    The charge-state-resolved PLT coefficients are weighted by the equilibrium
    abundance calculated from ACD and SCD data:

    .. math::

        L_{z,\mathrm{line}}(n_e,T_e)
        = 10^{-6}\sum_{q=0}^{Z-1} f_q(n_e,T_e)P^{\mathrm{PLT}}_q(T_e).

    ADF11 PLT coefficients are stored in W cm^3, hence the factor
    :math:`10^{-6}` converts the result to W m^3. The fully stripped state has
    no line-radiation block and is omitted from the sum.

    ``species`` is an atomic symbol with configured default ADF11 files (for
    example ``C``), ``ne_m3`` is in m^-3 and ``te_eV`` in eV.  Any of ``acd``,
    ``scd`` and ``plt`` left as ``None`` is resolved through the VAFT OPEN-ADAS
    cache (``cache_dir``, downloading on a cache miss).  The result is a
    non-negative coefficient in W m^3 with the broadcast input shape.

    Raises ``KeyError`` for an unconfigured species, ``ADASDataError`` when
    lookup, download or parsing fails, and ``ValueError`` for invalid inputs or
    mismatched charge-state dimensions.
    """

    filenames = default_adf11_files(species)
    acd_table = _resolve_source(acd, "acd", filenames, cache_dir)
    scd_table = _resolve_source(scd, "scd", filenames, cache_dir)
    plt_table = _resolve_source(plt, "plt", filenames, cache_dir)

    fractions = fractional_abundances(ne_m3, te_eV, acd_table, scd_table)
    # Unresolved PLT data are treated as density independent, matching the
    # established OPEN-ADAS cooling-factor calculation.
    _, te, _ = _validated_profiles(ne_m3, te_eV)
    line_power_cm3 = _interpolate_temperature_only(plt_table, te)
    if line_power_cm3.shape[-1] + 1 != fractions.shape[-1]:
        raise ValueError(
            "PLT and equilibrium tables contain incompatible charge-state counts: "
            f"{line_power_cm3.shape[-1]} line blocks vs {fractions.shape[-1]} fractions"
        )
    # PLT has no line radiation block for the fully stripped state. ADF11 PLT
    # values are W cm^3; convert the weighted total to W m^3.
    coefficient = np.sum(line_power_cm3 * fractions[..., :-1], axis=-1) * 1.0e-6
    if not np.all(np.isfinite(coefficient)) or np.any(coefficient < 0.0):
        raise ValueError("Line cooling coefficients must be finite and non-negative")
    return coefficient


__all__ = [
    "fractional_abundances",
    "interpolate_adf11",
    "line_cooling_coefficient",
]
