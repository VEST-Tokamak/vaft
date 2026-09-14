"""Evaluate the analytic neoclassical models on an ODS.

The ODS-facing counterpart of :mod:`vaft.formula.neoclassical`, in the sense
``vaft.omas.process_wrapper.compute_parallel_current_from_toroidal`` describes for
itself: the kernels stay array-in/array-out and know no IMAS paths, and this module
pulls what they need off ``equilibrium`` and ``core_profiles`` and hands it over.

It computes and returns; it does not write. Writing derived leaves is
:mod:`vaft.omas.update`'s job, and a caller who wants the result in an IDS has
:mod:`vaft.machine_mapping.neoclassical` for the solver side.

**The result is directly comparable with ``core_profiles.j_bootstrap``.** IMAS defines
that leaf as ``average(J.B)/B0`` with ``B0`` the vacuum toroidal field, while
:func:`vaft.formula.neoclassical.sauter_bootstrap_current` returns ``<j_par B>`` in
A T m^-2. The division by ``B0`` happens here, using the same per-slice
``vacuum_toroidal_field.b0`` the NEO mapper records -- it writes it per slice precisely
so that this reconstruction is possible.

Two inputs are resolved rather than assumed, and the result says which way each went:

*The trapped fraction.* ``equilibrium.profiles_1d.trapped_fraction`` is the value the
equilibrium integrated over its own field-strength distribution, and is preferred
whenever it is present. :func:`vaft.formula.neoclassical.trapped_particle_fraction` is
the circular approximation and is used only as a fallback; on NEO's reg18 case the two
differ by about one percent, and by more at strong shaping.

*The effective charge.* ``core_profiles.profiles_1d.zeff`` when the ODS carries it, else
the caller's ``z_eff``. There is no machine default here: a VEST-specific assumption
belongs in ``vest.yaml`` with a status, per CONTRIBUTING.md, not in a generic routine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "BootstrapCurrentResult",
    "MODELS",
    "compute_bootstrap_current",
]

#: The analytic formulations this module can evaluate.
MODELS = ("sauter", "redl")

_ELEMENTARY_CHARGE = 1.602176634e-19

#: Below this normalised flux the packaged VEST reconstruction is not trustworthy
#: (issue #317: an unphysical near-axis dvolume_dpsi ramp from a q[0] outlier). It is
#: not applied automatically -- the caller decides with ``rho_range`` -- but it is the
#: documented reason such a cut is usually wanted.
UNTRUSTWORTHY_PSI_NORM = 0.05


@dataclass(frozen=True)
class BootstrapCurrentResult:
    """One analytic bootstrap-current profile, with the inputs that produced it.

    Attributes
    ----------
    j_bootstrap
        ``<J.B>/B0`` [A m^-2], on ``rho_tor_norm``, directly comparable with
        ``core_profiles.profiles_1d.j_bootstrap``.
    parallel_current
        The kernel's own ``<j_par B>`` [A T m^-2], before the division by ``B0``.
    coefficients
        ``L31``, ``L32``, ``L34`` and ``alpha``, each on the same grid.
    provenance
        Which trapped fraction and which Z_eff were used, the slices paired, and the
        flux convention applied.
    """

    model: str
    rho_tor_norm: np.ndarray
    j_bootstrap: np.ndarray
    parallel_current: np.ndarray
    trapped_fraction: np.ndarray
    nu_e_star: np.ndarray
    nu_i_star: np.ndarray
    z_eff: np.ndarray
    b0: float
    time: float
    coefficients: Mapping[str, np.ndarray] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def integrated_current(self, area: np.ndarray) -> float:
        """``I_bs`` [A] by integrating over the poloidal cross-section.

        An approximation, and named as one: it integrates ``<J.B>/B0`` over ``area``,
        which is the toroidal current only to the extent that ``<J.B>/B0`` stands in
        for ``<J_tor/R>/<1/R>``. Useful as a scale, not as a current-balance term.
        """
        values = np.asarray(self.j_bootstrap, dtype=float)
        usable = np.isfinite(values)
        if usable.sum() < 2:
            return float("nan")
        return float(np.trapezoid(values[usable], np.asarray(area, dtype=float)[usable]))


def _array(ods: Any, path: str) -> Optional[np.ndarray]:
    """Read a path without materialising it when absent (issue #118)."""
    try:
        if path not in ods:
            return None
        values = np.atleast_1d(np.asarray(ods[path], dtype=float))
    except (KeyError, ValueError, IndexError, TypeError):
        return None
    return values if values.size else None


def _radial_extent(ods: Any, index: int, size: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """``(r_inboard, r_outboard)`` of one slice, stored else derived privately.

    ``update_equilibrium_profiles_1d_radial_coordinates`` writes them from the 2-D flux
    map; it runs on an isolated copy so the caller's ODS is left as it was, which is the
    pattern ``vaft/plot/backend/recipes.py`` already established.
    """
    base = f"equilibrium.time_slice.{index}.profiles_1d"
    inboard, outboard = _array(ods, f"{base}.r_inboard"), _array(ods, f"{base}.r_outboard")
    if inboard is not None and outboard is not None and inboard.size == outboard.size == size:
        return inboard, outboard

    try:
        import copy

        from omas import ODS

        from vaft.omas import update_equilibrium_profiles_1d_radial_coordinates

        private = ODS(consistency_check=False)
        private["equilibrium"] = copy.deepcopy(ods["equilibrium"])
        update_equilibrium_profiles_1d_radial_coordinates(private, time_slice=index)
    except Exception:  # noqa: BLE001 - an underivable slice is reported, not raised
        return None
    inboard = _array(private, f"{base}.r_inboard")
    outboard = _array(private, f"{base}.r_outboard")
    if inboard is None or outboard is None or inboard.size != size or outboard.size != size:
        return None
    return inboard, outboard


def _kinetic_on(ods: Any, cp_index: int, path: str, grid: np.ndarray) -> Optional[np.ndarray]:
    """One core_profiles quantity, interpolated onto the equilibrium's grid.

    Refuses to extrapolate: ``np.interp`` clamps, which would carry an edge value across
    radii the profile never covered.
    """
    base = f"core_profiles.profiles_1d.{cp_index}"
    values = _array(ods, f"{base}.{path}")
    source = _array(ods, f"{base}.grid.rho_tor_norm")
    if values is None or source is None or values.size != source.size:
        return None
    order = np.argsort(source)
    # anti-alias: spatial. Both axes are rho_tor_norm, a flux coordinate, so there is no
    # sampling rate to reduce and nothing to fold (#425).
    placed = np.interp(grid, source[order], values[order])
    placed[(grid < source.min()) | (grid > source.max())] = np.nan
    return placed


def compute_bootstrap_current(
    ods: Any,
    *,
    model: str = "sauter",
    time_slice: Optional[int] = None,
    z_eff: Optional[float] = None,
    ion_index: int = 0,
    rho_range: Optional[Sequence[float]] = None,
    b0: Optional[float] = None,
) -> BootstrapCurrentResult:
    """Evaluate an analytic bootstrap-current profile from an ODS.

    Parameters
    ----------
    ods : ODS
        Must carry an ``equilibrium`` slice with ``rho_tor_norm``, ``psi``, ``q`` and
        ``f``, and a ``core_profiles`` slice with electron density and temperature and
        one ion temperature. ``r_inboard``/``r_outboard`` are derived privately when
        absent, and ``trapped_fraction`` is used when present.
    model : {'sauter', 'redl'}
        Which formulation. Sauter is Phys. Plasmas 6 (1999) 2834 with the 2002 erratum;
        Redl is Phys. Plasmas 28 (2021) 022502, refitted against NEO across tight
        aspect ratio and the one to prefer for a spherical tokamak.
    time_slice : int, optional
        The ``core_profiles`` slice; the equilibrium slice is matched to its time.
    z_eff : float, optional
        Effective charge, when the ODS carries no ``zeff`` profile.
    ion_index : int, optional
        Which ion species supplies the temperature.
    rho_range : (float, float), optional
        Restrict the evaluation to this ``rho_tor_norm`` band. The near-axis region of a
        reconstruction is often untrustworthy (#317), and cutting it is the caller's
        decision, recorded in the result.
    b0 : float, optional
        Vacuum toroidal field to normalise by. Read at the matched slice when omitted.

    Returns
    -------
    BootstrapCurrentResult

    Raises
    ------
    ValueError
        When a required leaf is missing, naming it and which layer writes it, rather
        than substituting anything; or when *model* is not one of :data:`MODELS`.
    """
    from vaft.formula.neoclassical import (
        electron_collisionality_sauter,
        ion_collisionality_sauter,
        redl_bootstrap_coefficients,
        redl_bootstrap_current,
        sauter_bootstrap_coefficients,
        sauter_bootstrap_current,
        trapped_particle_fraction,
    )
    from vaft.omas.general import find_matching_time_indices

    name = str(model).strip().lower()
    if name not in MODELS:
        raise ValueError(f"model must be one of {MODELS}; got {model!r}")

    cp_index, eq_index, time = find_matching_time_indices(ods, time_slice=time_slice)
    equilibrium = f"equilibrium.time_slice.{eq_index}.profiles_1d"

    required = ("rho_tor_norm", "psi", "q", "f")
    missing = [leaf for leaf in required if _array(ods, f"{equilibrium}.{leaf}") is None]
    if missing:
        raise ValueError(
            f"equilibrium slice {eq_index} is missing {', '.join(missing)}; a bootstrap "
            "current needs the flux coordinate, the flux map, q and F = R*B_phi. "
            "vaft.omas.update writes the derived ones from a 2-D map."
        )

    grid = _array(ods, f"{equilibrium}.rho_tor_norm")
    psi = _array(ods, f"{equilibrium}.psi")
    q_profile = _array(ods, f"{equilibrium}.q")
    f_profile = _array(ods, f"{equilibrium}.f")

    keep = np.ones(grid.size, dtype=bool)
    if rho_range is not None:
        low, high = (float(value) for value in rho_range)
        keep = (grid >= low) & (grid <= high)
        if int(np.count_nonzero(keep)) < 2:
            raise ValueError(f"rho_range {rho_range} leaves fewer than two grid points")

    extent = _radial_extent(ods, eq_index, grid.size)
    if extent is None:
        raise ValueError(
            f"equilibrium slice {eq_index} has no r_inboard/r_outboard and none could be "
            "derived from its 2-D flux map; the inverse aspect ratio cannot be formed"
        )
    inboard, outboard = extent
    minor = 0.5 * (outboard - inboard)
    major = 0.5 * (outboard + inboard)
    epsilon = np.divide(minor, major, out=np.zeros_like(minor), where=major > 0.0)

    density = _kinetic_on(ods, cp_index, "electrons.density_thermal", grid)
    if density is None:
        density = _kinetic_on(ods, cp_index, "electrons.density", grid)
    temperature = _kinetic_on(ods, cp_index, "electrons.temperature", grid)
    ion_temperature = _kinetic_on(ods, cp_index, f"ion.{ion_index}.temperature", grid)
    absent = [
        label
        for label, values in (
            ("electron density", density),
            ("electron temperature", temperature),
            (f"ion {ion_index} temperature", ion_temperature),
        )
        if values is None
    ]
    if absent:
        raise ValueError(
            f"core_profiles slice {cp_index} is missing {', '.join(absent)}; the "
            "neoclassical coefficients are functions of them and nothing is substituted"
        )

    stored = _array(ods, f"{equilibrium}.trapped_fraction")
    if stored is not None and stored.size == grid.size:
        trapped, trapped_source = stored, "equilibrium.profiles_1d.trapped_fraction"
    else:
        trapped = np.asarray(trapped_particle_fraction(np.clip(epsilon, 0.0, 0.999)))
        trapped_source = "circular approximation from r_inboard/r_outboard"

    charge = _kinetic_on(ods, cp_index, "zeff", grid)
    if charge is not None:
        z_source = f"core_profiles.profiles_1d.{cp_index}.zeff"
    elif z_eff is not None:
        charge = np.full(grid.size, float(z_eff))
        z_source = "caller"
    else:
        raise ValueError(
            f"core_profiles slice {cp_index} carries no zeff and no z_eff was supplied; "
            "the coefficients depend on it and this layer does not hold a machine "
            "default (CONTRIBUTING.md sends that to vest.yaml with a status)"
        )

    field_strength = b0 if b0 is not None else _vacuum_field(ods, eq_index, cp_index)
    if field_strength is None:
        raise ValueError(
            "no vacuum_toroidal_field.b0 for this slice and none supplied; IMAS defines "
            "j_bootstrap as <J.B>/B0, so it cannot be formed without one"
        )

    # The kernels take derivatives with respect to poloidal flux *per radian*; the DD
    # stores psi in full weber. Scale psi by the factor, as compute_grad_shafranov_residual
    # does -- getting this wrong moves the result by 2*pi.
    from vaft.data.eqdsk import ods_psi_to_wb_per_radian_factor

    factor = ods_psi_to_wb_per_radian_factor(ods, time_index=eq_index)
    psi_radian = psi * factor

    # Two masks, deliberately separate. `physical` is where the state is usable at all;
    # `keep` is the band the caller asked to see. Gradients are taken over `physical`
    # only -- np.gradient is a centred difference, so including a zero-density boundary
    # point corrupts its neighbour, which is a point the mask kept. But they are *not*
    # taken over `keep`: a radius excluded only by rho_range is real data, and dropping
    # it would corrupt the gradient at the band edge instead.
    physical = np.isfinite(density) & np.isfinite(temperature) & np.isfinite(ion_temperature)
    physical &= (density > 0.0) & (temperature > 0.0) & (ion_temperature > 0.0)
    physical &= (epsilon > 0.0) & np.isfinite(q_profile) & np.isfinite(f_profile)
    usable = keep & physical
    if int(np.count_nonzero(physical)) < 3:
        raise ValueError(
            "fewer than three grid points carry a positive density and temperature; a "
            "centred gradient cannot be formed"
        )
    if int(np.count_nonzero(usable)) < 2:
        raise ValueError(
            "fewer than two grid points carry a positive density and temperature inside "
            "the requested rho_range; the collisionalities are undefined there"
        )

    pressure_e = density * temperature * _ELEMENTARY_CHARGE
    pressure_i = density * ion_temperature * _ELEMENTARY_CHARGE

    def gradient(values: np.ndarray) -> np.ndarray:
        """d/dpsi over the physical points, placed back on the full grid."""
        full = np.full(values.size, np.nan)
        full[physical] = np.gradient(values[physical], psi_radian[physical])
        return full

    def log_gradient(values: np.ndarray) -> np.ndarray:
        return np.divide(
            gradient(values), values, out=np.full(values.size, np.nan), where=physical
        )

    nu_e = np.asarray(
        electron_collisionality_sauter(
            density[usable], temperature[usable], q_profile[usable],
            major[usable], epsilon[usable], charge[usable],
        )
    )
    nu_i = np.asarray(
        ion_collisionality_sauter(
            density[usable], ion_temperature[usable], q_profile[usable],
            major[usable], epsilon[usable], 1.0,
        )
    )

    current_fn = sauter_bootstrap_current if name == "sauter" else redl_bootstrap_current
    coefficient_fn = (
        sauter_bootstrap_coefficients if name == "sauter" else redl_bootstrap_coefficients
    )
    arguments = dict(
        f_trap=trapped[usable], nu_e_star=nu_e, nu_i_star=nu_i, Z_eff=charge[usable],
        I_psi=f_profile[usable], p_e=pressure_e[usable], p_i=pressure_i[usable],
        dp_dpsi=gradient(pressure_e + pressure_i)[usable],
        dln_Te_dpsi=log_gradient(temperature)[usable],
        dln_Ti_dpsi=log_gradient(ion_temperature)[usable],
    )
    parallel = np.full(grid.size, np.nan)
    parallel[usable] = np.asarray(current_fn(**arguments))
    coefficients = coefficient_fn(trapped[usable], nu_e, nu_i, charge[usable])

    def spread(values: np.ndarray) -> np.ndarray:
        full = np.full(grid.size, np.nan)
        full[usable] = np.asarray(values, dtype=float)
        return full

    return BootstrapCurrentResult(
        model=name,
        rho_tor_norm=grid,
        j_bootstrap=parallel / field_strength,
        parallel_current=parallel,
        trapped_fraction=trapped,
        nu_e_star=spread(nu_e),
        nu_i_star=spread(nu_i),
        z_eff=charge,
        b0=float(field_strength),
        time=float(time),
        coefficients={
            "L31": spread(coefficients.L31), "L32": spread(coefficients.L32),
            "L34": spread(coefficients.L34), "alpha": spread(coefficients.alpha),
        },
        provenance={
            "equilibrium_index": int(eq_index),
            "core_profiles_index": int(cp_index),
            "time": float(time),
            "trapped_fraction": trapped_source,
            "z_eff": z_source,
            "wb_per_radian_factor": float(factor),
            "rho_range": None if rho_range is None else tuple(float(v) for v in rho_range),
            "evaluated_points": int(np.count_nonzero(usable)),
            "grid_points": int(grid.size),
        },
    )


def _vacuum_field(ods: Any, eq_index: int, cp_index: int) -> Optional[float]:
    """The vacuum field of the matched slice, core_profiles' own preferred."""
    for path, index in (
        ("core_profiles.vacuum_toroidal_field.b0", cp_index),
        ("equilibrium.vacuum_toroidal_field.b0", eq_index),
    ):
        values = _array(ods, path)
        if values is None or index >= values.size:
            continue
        if np.isfinite(values[index]) and values[index] != 0.0:
            return float(values[index])
    return None
