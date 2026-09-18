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
    r"""Interpolate every charge-state block of an ADF11 table at given $(n_e, T_e)$.

    $$C_z(n_e, T_e) = 10^{\,\mathcal{I}_2[\log_{10}C_z]}$$

    where $\mathcal{I}_2$ is bilinear interpolation on the table's rectangular
    grid of $\log_{10}n_e\,[\mathrm{cm^{-3}}]$ and $\log_{10}T_e\,[\mathrm{eV}]$.

    Parameters
    ----------
    table : ADF11Data
        Parsed ADF11 table (``acd``, ``scd``, ``plt``, ...) [n/a].
    ne_m3 : array-like
        Electron density, finite and positive [m^-3].
    te_eV : array-like
        Electron temperature, finite and positive [eV].
    multiply_density : bool, optional
        Multiply by $n_e$ in cm^-3 to turn coefficients into rates; default ``False`` [bool].

    Returns
    -------
    np.ndarray
        Coefficients in the table's own unit, or rates with ``multiply_density`` [any].
        Shape ``broadcast(ne_m3, te_eV).shape + (n_blocks,)``; cm^3/s for
        ACD/SCD and W cm^3 for PLT, s^-1 when multiplied by density.

    Raises
    ------
    ValueError
        Empty, non-finite or non-positive profiles.

    Convention
    ----------
    ADF11 stores densities in cm^-3 and rate coefficients in cm^3 s^-1; the SI
    input is converted to cm^-3 internally and the output is *not* converted
    back, so callers multiply by $10^{-6}$ for m^3 s^-1 (as
    :func:`line_cooling_coefficient` does for W m^3).

    Validity
    --------
    Inside the tabulated grid (typically $10^{8}$-$10^{15}$ cm^-3 and
    1 eV-10 keV for iso-nuclear ADF11 files).

    Limitations
    -----------
    Linear extrapolation in log space is retained outside the grid to match the
    established ADF11 calculation path; extrapolated coefficients are not
    physical and are not flagged.

    Numerical notes
    ---------------
    ``scipy.interpolate.RegularGridInterpolator`` with ``bounds_error=False``,
    ``fill_value=None``, one interpolator per charge-state block.

    References
    ----------
    .. [1] H. P. Summers, *The ADAS User Manual*, version 2.6 (2004),
           https://www.adas.ac.uk/manual.php, ADF11 format.
    .. [2] F. Sciortino et al., Plasma Phys. Control. Fusion 63 (2021) 112001
           (Aurora; the calculation path this follows).
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
    r"""Ionisation-equilibrium (coronal) fractional abundances from ACD and SCD tables.

    $$\frac{f_{z+1}}{f_z} = \frac{S_z(n_e, T_e)}{\alpha_{z+1}(n_e, T_e)}, \qquad
      \tilde f_0 = 1, \quad \tilde f_z = \prod_{j=0}^{z-1}\frac{S_j}{\alpha_{j+1}}, \quad
      f_z = \frac{\tilde f_z}{\sum_k\tilde f_k}$$

    with $S_z$ the SCD effective ionisation and $\alpha_{z+1}$ the ACD effective
    recombination coefficients.

    Parameters
    ----------
    ne_m3 : array-like
        Electron density, finite and positive [m^-3].
    te_eV : array-like
        Electron temperature, finite and positive [eV].
    acd : ADF11Data or path-like
        Effective recombination table [n/a].
    scd : ADF11Data or path-like
        Effective ionisation table [n/a].

    Returns
    -------
    np.ndarray
        Fractional abundances, summing to one over the last axis [-].
        That axis has ``n_rate_blocks + 1`` charge states, neutral to fully
        stripped.

    Raises
    ------
    ValueError
        Mismatched charge-state counts, wrong table class, or non-finite rates.

    Convention
    ----------
    Effective (collisional-radiative, density-dependent) coefficients in the
    ADAS sense, evaluated in cm^-3 internally; the density cancels in the
    ratio, so the result is dimensionless.

    Assumptions
    -----------
    Steady state, no transport: the local balance of ionisation and
    recombination alone sets the charge-state distribution.

    Limitations
    -----------
    Transport in a real edge or start-up plasma shifts the distribution toward
    lower charge states than coronal equilibrium predicts (the effect Aurora
    models with an impurity transport solve).

    References
    ----------
    .. [1] H. P. Summers, *The ADAS User Manual*, version 2.6 (2004), Sec. on
           ADF11 (ionisation balance).
    .. [2] F. Sciortino et al., Plasma Phys. Control. Fusion 63 (2021) 112001,
           Sec. 2 (equilibrium construction; MIT-licensed, see the module docstring).
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
    r"""Equilibrium line-radiation cooling coefficient $L_{line}(n_e, T_e)$ of a species.

    $$L_{\mathrm{line}}(n_e, T_e) = 10^{-6}\sum_{q=0}^{Z-1} f_q(n_e, T_e)\,
      P^{\mathrm{PLT}}_q(n_e, T_e)$$

    the PLT coefficients weighted by the coronal abundances of
    :func:`fractional_abundances`; the fully stripped state has no line
    radiation and is omitted.

    Parameters
    ----------
    species : str
        Atomic symbol with configured default ADF11 files, e.g. ``"C"`` [str].
    ne_m3 : array-like
        Electron density, finite and positive [m^-3].
    te_eV : array-like
        Electron temperature, finite and positive [eV].
    acd : ADF11Data or path-like or None, optional
        Recombination table; ``None`` resolves the configured file [n/a].
    scd : ADF11Data or path-like or None, optional
        Ionisation table; ``None`` resolves the configured file [n/a].
    plt : ADF11Data or path-like or None, optional
        Line-power table; ``None`` resolves the configured file [n/a].
    cache_dir : str or path-like or None, optional
        OPEN-ADAS cache directory; downloads on a miss [n/a].

    Returns
    -------
    np.ndarray
        Cooling coefficient with the broadcast input shape [W m^3].
        Multiply by $n_e n_Z$ for a power density.

    Raises
    ------
    KeyError
        Unconfigured species.
    ADASDataError
        Lookup, download or parsing failure.
    ValueError
        Invalid inputs or mismatched charge-state dimensions.

    Convention
    ----------
    ADF11 PLT is stored in W cm^3; the factor $10^{-6}$ converts to W m^3.
    Coronal equilibrium, no transport, no recombination/bremsstrahlung
    continuum (that is the separate PRB class).

    References
    ----------
    .. [1] H. P. Summers, *The ADAS User Manual*, version 2.6 (2004), ADF11
           PLT class.
    .. [2] D. E. Post et al., At. Data Nucl. Data Tables 20 (1977) 397
           (coronal cooling curves).
    """

    filenames = default_adf11_files(species)
    acd_table = _resolve_source(acd, "acd", filenames, cache_dir)
    scd_table = _resolve_source(scd, "scd", filenames, cache_dir)
    plt_table = _resolve_source(plt, "plt", filenames, cache_dir)

    fractions = fractional_abundances(ne_m3, te_eV, acd_table, scd_table)
    line_power_cm3 = interpolate_adf11(plt_table, ne_m3, te_eV)
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


def mean_charge_from_charge_state_densities(n_z, axis=-1):
    r"""Mean charge of one element over its charge states.

    $$\langle Z\rangle = \sum_j j\,f_j = \frac{\sum_j j\,n_j}{\sum_j n_j},
      \qquad f_j = \frac{n_j}{\sum_k n_k}$$

    Parameters
    ----------
    n_z : array-like
        Density of each charge state along ``axis``, ordered neutral first, so
        that index $j$ is charge $j$; finite and non-negative [m^-3].
    axis : int, optional
        Axis that runs over the charge states [-].

    Returns
    -------
    float or np.ndarray
        Mean charge, with ``axis`` removed [-].

    Raises
    ------
    ValueError
        A non-finite or negative density, or a slice whose densities sum to
        zero, which has no charge distribution to average.

    Convention
    ----------
    **Index is charge, neutral first**, the layout
    :func:`fractional_abundances` returns -- so its output can be passed
    straight in, and densities need not be normalised first because the
    fractions $f_j$ are formed here.  Spectroscopic notation counts from one
    (C I is neutral carbon); pass charge, not ionisation stage, or every value
    comes out one high.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 4.25 (impurity charge states).

    See Also
    --------
    fractional_abundances
    z_eff_from_n_s_Z_s
    """
    density = np.asarray(n_z, dtype=float)
    if density.ndim == 0 or density.shape[axis] == 0:
        raise ValueError("n_z needs at least one charge state along axis")
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("n_z must be finite and non-negative")
    total = np.sum(density, axis=axis)
    if np.any(total <= 0.0):
        raise ValueError("a charge-state distribution with zero total density has no mean charge")
    shape = [1] * density.ndim
    shape[axis] = density.shape[axis]
    charge = np.arange(density.shape[axis], dtype=float).reshape(shape)
    mean = np.sum(charge * density, axis=axis) / total
    return float(mean) if np.ndim(mean) == 0 else mean


def z_eff_from_n_s_Z_s(n_s, Z_s, n_e=None):
    r"""Effective charge of a mixture of ion species.

    $$Z_{\mathrm{eff}} = \frac{\sum_s n_s Z_s^2}{n_e}, \qquad
      n_e = \sum_s n_s Z_s\ \text{when not given}$$

    Parameters
    ----------
    n_s : array-like
        Density of each ion species or charge state, finite and non-negative;
        the last axis runs over species [m^-3].
    Z_s : array-like
        Charge of each, broadcastable against ``n_s``, finite and non-negative
        [-].
    n_e : float or array-like, optional
        Electron density, finite and positive; default the quasi-neutral
        $\sum_s n_s Z_s$ [m^-3].

    Returns
    -------
    float or np.ndarray
        Effective charge, with the species axis removed [-].

    Raises
    ------
    ValueError
        A non-finite or negative density or charge, a non-positive electron
        density, or a quasi-neutral electron density of zero.

    Convention
    ----------
    **Pass the electron density only when it is measured independently.**  The
    default makes the plasma quasi-neutral by construction, and then a pure
    hydrogenic plasma is exactly 1 and every impurity raises it.  A measured
    $n_e$ that disagrees with $\sum_s n_s Z_s$ is taken as given and not
    reconciled, so a density error lands directly in $Z_{\mathrm{eff}}$.

    A charge-resolved impurity is several species here, one per charge state:
    $Z_{\mathrm{eff}}$ weights $Z^2$, so collapsing an element of density $n_I$
    onto its mean charge first underestimates it by exactly
    $n_I\,\mathrm{Var}(Z)/n_e$ -- the quasi-neutral $n_e$ is unchanged, since
    it weights $Z$ only linearly.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 4.25.

    See Also
    --------
    mean_charge_from_charge_state_densities
    vaft.formula.equilibrium.spitzer_resistivity_from_T_e_Z_eff_ln_Lambda
    """
    density = np.asarray(n_s, dtype=float)
    charge = np.asarray(Z_s, dtype=float)
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("n_s must be finite and non-negative")
    if not np.all(np.isfinite(charge)) or np.any(charge < 0.0):
        raise ValueError("Z_s must be finite and non-negative")
    density, charge = np.broadcast_arrays(density, charge)
    weighted = np.sum(density * charge**2, axis=-1)
    if n_e is None:
        electrons = np.sum(density * charge, axis=-1)
        if np.any(electrons <= 0.0):
            raise ValueError(
                "the quasi-neutral electron density is zero; pass n_e or a charged species"
            )
    else:
        electrons = np.asarray(n_e, dtype=float)
        if not np.all(np.isfinite(electrons)) or np.any(electrons <= 0.0):
            raise ValueError("n_e must be finite and positive")
    z_eff = weighted / electrons
    return float(z_eff) if np.ndim(z_eff) == 0 else z_eff


__all__ = [
    "fractional_abundances",
    "interpolate_adf11",
    "line_cooling_coefficient",
    "mean_charge_from_charge_state_densities",
    "z_eff_from_n_s_Z_s",
]
