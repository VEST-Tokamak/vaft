"""Convert an IMAS/OMAS scientific state into a GACODE profile set.

The direction is one-way and deliberate::

    equilibrium + core_profiles   canonical, IMAS
              |
              v
        GACODEProfile             neutral, typed, provenance-bearing
              |
              v
        input.gacode              one external suite's format

The canonical state is never replaced by the GACODE one.  What this module owns
is the conversion, and three decisions it refuses to make silently:

**The radial coordinate.**  GACODE's ``rho`` is ``sqrt(Phi/Phi_boundary)``.
Several packaged VAFT equilibria store ``sqrt(psi_N)`` under the name
``rho_tor_norm`` (issues #276, #420), and the two agree only for a flat-q
cylinder.  The grid is checked with :func:`vaft.data._derived.is_rho_pol_proxy`,
re-derived from ``q`` and ``psi`` when it is a proxy, and the conversion is
refused when it cannot be re-derived.  A GACODE file written on
``sqrt(psi_N)`` and labelled ``rho`` is the exact defect those issues exist to
prevent.

**Time alignment.**  The requested time, the equilibrium slice actually used and
the ``core_profiles`` slice actually used are resolved separately, all three are
recorded, and a pairing outside the tolerance is refused rather than made.

**Missing kinetic information.**  Nothing is fabricated to fill a GACODE field.
A required quantity that is absent raises; an optional one that is absent stays
``None`` and is recorded as ``unavailable``; a value that comes from machine
policy rather than measurement is recorded as ``policy_assumption``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..base import CodeInputs
from ._input_gacode import write_input_gacode
from ._profiles import GACODEProfile

#: Smallest tolerance used when pairing an equilibrium slice with a
#: core_profiles slice, matching `vaft.validation.equilibrium`.
MINIMUM_TIME_TOLERANCE = 1.0e-3

#: GACODE stores densities in 10^19 m^-3 and temperatures in keV.
DENSITY_SCALE = 1.0e19
TEMPERATURE_SCALE = 1.0e3


class ProfileConversionError(ValueError):
    """The canonical state cannot be projected onto a GACODE profile set.

    Distinct from a plain ``ValueError`` so a caller can tell "this shot cannot
    be modelled as it stands" from a programming mistake, and act on it -- by
    truncating the grid, supplying a species assumption, or choosing another
    time -- rather than by working around the adapter.
    """


@dataclass
class GACODEInputs(CodeInputs):
    """A staged GACODE case: the typed profile and the file written from it."""

    profile: Optional[GACODEProfile] = None
    input_gacode: Optional[Path] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


def _array(ods: Any, path: str) -> Optional[np.ndarray]:
    """Read a path without materialising it when it is absent.

    ``ods["missing.path"]`` *creates* the path in OMAS, so every read here is
    guarded by a membership test first.
    """
    try:
        if path not in ods:
            return None
        value = ods[path]
    except (KeyError, ValueError, IndexError, TypeError):
        return None
    array = np.asarray(value, dtype=float)
    return array if array.size else None


def _scalar(ods: Any, path: str) -> Optional[float]:
    array = _array(ods, path)
    if array is None:
        return None
    return float(np.atleast_1d(array)[0])


def _time_tolerance(times: np.ndarray) -> float:
    """Half the median sampling interval, floored.

    The same rule `vaft.validation.equilibrium` uses, so that "these two slices
    describe the same instant" means one thing across VAFT.
    """
    if times is None or times.size < 2:
        return MINIMUM_TIME_TOLERANCE
    return max(0.5 * float(np.median(np.diff(np.sort(times)))), MINIMUM_TIME_TOLERANCE)


def _nearest(times: np.ndarray, target: float) -> tuple[int, float]:
    offsets = np.abs(times - target)
    index = int(np.argmin(offsets))
    return index, float(offsets[index])


def _resolve_times(
    ods: Any,
    *,
    time: Optional[float],
    time_index: Optional[int],
    tolerance: Optional[float],
) -> dict[str, Any]:
    """Resolve, and record, which slices this conversion actually used."""
    equilibrium_times = _array(ods, "equilibrium.time")
    if equilibrium_times is None:
        raise ProfileConversionError("the ODS has no equilibrium.time")
    profile_times = _array(ods, "core_profiles.time")
    if profile_times is None:
        raise ProfileConversionError("the ODS has no core_profiles.time")

    if time_index is not None:
        if not 0 <= int(time_index) < equilibrium_times.size:
            raise ProfileConversionError(
                f"time_index {time_index} is outside the {equilibrium_times.size} "
                "equilibrium slices"
            )
        equilibrium_index = int(time_index)
        requested = float(equilibrium_times[equilibrium_index])
    elif time is not None:
        requested = float(time)
        equilibrium_index, offset = _nearest(equilibrium_times, requested)
        limit = tolerance if tolerance is not None else _time_tolerance(equilibrium_times)
        if offset > limit:
            raise ProfileConversionError(
                f"the nearest equilibrium slice is {offset:.4g} s from the requested "
                f"{requested:.4g} s, beyond the {limit:.4g} s tolerance"
            )
    elif equilibrium_times.size == 1:
        equilibrium_index = 0
        requested = float(equilibrium_times[0])
    else:
        raise ProfileConversionError(
            f"the ODS has {equilibrium_times.size} equilibrium slices; pass time= or "
            "time_index= rather than letting the conversion choose one"
        )

    equilibrium_time = float(equilibrium_times[equilibrium_index])
    limit = tolerance if tolerance is not None else _time_tolerance(profile_times)
    profile_index, offset = _nearest(profile_times, equilibrium_time)
    if offset > limit:
        raise ProfileConversionError(
            f"the nearest core_profiles slice is {offset:.4g} s from the equilibrium "
            f"slice at {equilibrium_time:.4g} s, beyond the {limit:.4g} s tolerance. "
            "Profile and equilibrium slices are not combined outside it."
        )
    return {
        "requested_time": requested,
        "equilibrium_index": equilibrium_index,
        "equilibrium_time": equilibrium_time,
        "core_profiles_index": profile_index,
        "core_profiles_time": float(profile_times[profile_index]),
        "tolerance": float(limit),
    }


def _resolve_rho(ods: Any, prefix: str) -> tuple[np.ndarray, str]:
    """Return the GACODE radial coordinate and how it was obtained.

    Refuses rather than substituting: a `sqrt(psi_N)` proxy that cannot be
    re-derived into a real toroidal coordinate stops the conversion.
    """
    from vaft.data._derived import is_rho_pol_proxy, rho_tor_profile
    from vaft.data.eqdsk import ods_psi_to_wb_per_radian_factor

    rho = _array(ods, f"{prefix}.rho_tor_norm")
    psi = _array(ods, f"{prefix}.psi")
    q = _array(ods, f"{prefix}.q")

    psi_norm = None
    if psi is not None and psi.size > 1 and psi[-1] != psi[0]:
        psi_norm = (psi - psi[0]) / (psi[-1] - psi[0])

    if rho is not None and not is_rho_pol_proxy(rho, psi_norm):
        return rho, "equilibrium.profiles_1d.rho_tor_norm"

    phi = _array(ods, f"{prefix}.phi")
    if phi is not None and phi.size > 1 and float(phi[-1]) != 0.0:
        return np.sqrt(np.abs(phi / phi[-1])), "derived from equilibrium phi"

    if q is not None and psi is not None:
        factor = ods_psi_to_wb_per_radian_factor(ods)
        derived = rho_tor_profile(q, psi / factor)
        if derived is not None:
            return np.asarray(derived.rho_tor_norm, dtype=float), "derived from q and psi"

    if rho is not None:
        raise ProfileConversionError(
            "equilibrium rho_tor_norm is the sqrt(psi_N) proxy (issues #276, #420) and "
            "no toroidal flux is available to re-derive it. GACODE's rho is "
            "sqrt(Phi/Phi_boundary); writing sqrt(psi_N) under that name would be wrong, "
            "so the conversion stops here."
        )
    raise ProfileConversionError(
        "the equilibrium carries no rho_tor_norm, phi, or usable q and psi, so the "
        "GACODE radial coordinate cannot be established"
    )


def _interpolate(source_rho: np.ndarray, values: np.ndarray, target_rho: np.ndarray) -> np.ndarray:
    """Map a profile onto the GACODE grid, refusing to extrapolate.

    ``np.interp`` clamps outside the source range, which silently invents an
    edge value. The range is checked first so that a profile that does not cover
    the equilibrium grid is reported instead.
    """
    if source_rho.size != values.size:
        raise ProfileConversionError(
            f"a core_profiles quantity has {values.size} points against a "
            f"{source_rho.size}-point grid"
        )
    if source_rho.size == target_rho.size and np.allclose(source_rho, target_rho):
        return np.asarray(values, dtype=float)
    lower, upper = float(np.min(source_rho)), float(np.max(source_rho))
    if float(np.min(target_rho)) < lower - 1e-9 or float(np.max(target_rho)) > upper + 1e-9:
        raise ProfileConversionError(
            f"the kinetic profiles span rho [{lower:.4g}, {upper:.4g}] but the "
            f"equilibrium grid spans [{float(np.min(target_rho)):.4g}, "
            f"{float(np.max(target_rho)):.4g}]; extrapolating kinetic data onto an "
            "equilibrium grid it does not cover is not done here"
        )
    order = np.argsort(source_rho)
    return np.interp(target_rho, np.asarray(source_rho)[order], np.asarray(values)[order])


def _ion_species(ods: Any, index: int) -> list[dict[str, Any]]:
    """Read the ion species table from core_profiles."""
    prefix = f"core_profiles.profiles_1d.{index}.ion"
    species: list[dict[str, Any]] = []
    position = 0
    while True:
        base = f"{prefix}.{position}"
        try:
            present = f"{base}.label" in ods or f"{base}.z_ion" in ods
        except (KeyError, ValueError, TypeError):
            present = False
        if not present:
            break
        label = str(ods[f"{base}.label"]) if f"{base}.label" in ods else f"ion{position}"
        charge = _scalar(ods, f"{base}.z_ion")
        mass = _scalar(ods, f"{base}.element.0.a")
        density = _array(ods, f"{base}.density_thermal")
        if density is None:
            density = _array(ods, f"{base}.density")
        species.append(
            {
                "label": label,
                "z": 1.0 if charge is None else float(charge),
                "mass": mass,
                "density": density,
                "temperature": _array(ods, f"{base}.temperature"),
                "velocity_toroidal": _array(ods, f"{base}.velocity.toroidal"),
            }
        )
        position += 1
    return species


def _require_positive(name: str, values: np.ndarray) -> None:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ProfileConversionError(f"{name} contains non-finite values")
    if np.any(array <= 0.0):
        bad = int(np.argmax(array <= 0.0))
        raise ProfileConversionError(
            f"{name} is not positive at grid point {bad} (value {array[bad]:.6g}). "
            "GACODE takes logarithmic gradients of densities and temperatures, so a "
            "zero or negative value is not usable. Truncate the grid with rho_max=, or "
            "supply a fit that stays positive; nothing is clipped here."
        )


def prepare_gacode_profile(
    ods: Any,
    *,
    time: Optional[float] = None,
    time_index: Optional[int] = None,
    tolerance: Optional[float] = None,
    rho_max: Optional[float] = None,
    z_eff: Optional[float] = None,
    shot: Optional[int] = None,
) -> GACODEProfile:
    """Project an ODS equilibrium and core_profiles onto a GACODE profile set.

    Parameters
    ----------
    ods
        An OMAS ODS carrying ``equilibrium`` and ``core_profiles``.
    time, time_index
        Which slice to convert. ``time_index`` indexes the equilibrium
        directly; ``time`` snaps to the nearest slice within the tolerance. With
        a single equilibrium slice, neither is needed.
    tolerance
        Seconds within which an equilibrium slice and a core_profiles slice are
        taken to describe the same instant. Defaults to half the median
        sampling interval, floored at one millisecond.
    rho_max
        Truncate the GACODE grid at this normalised radius. This is the
        caller's explicit decision about an edge region the profiles do not
        support, and it is recorded in provenance.
    z_eff
        A single effective charge to use when the ODS carries no Z_eff and no
        impurity species. Recorded as ``caller_supplied``.
    shot
        Shot number for the file header; read from ``dataset_description`` when
        omitted.

    Raises
    ------
    ProfileConversionError
        The state cannot be projected: an unusable radial coordinate, slices
        that cannot be paired, kinetic profiles that do not cover the
        equilibrium grid, or a non-positive density or temperature.
    """
    times = _resolve_times(ods, time=time, time_index=time_index, tolerance=tolerance)
    equilibrium_prefix = (
        f"equilibrium.time_slice.{times['equilibrium_index']}.profiles_1d"
    )
    global_prefix = (
        f"equilibrium.time_slice.{times['equilibrium_index']}.global_quantities"
    )
    profile_index = times["core_profiles_index"]
    profile_prefix = f"core_profiles.profiles_1d.{profile_index}"

    rho, rho_source = _resolve_rho(ods, equilibrium_prefix)
    provenance: dict[str, dict[str, Any]] = {
        "rho": {"kind": "derived", "source": rho_source},
        "time": {"kind": "derived", "source": "slice resolution", **times},
    }

    keep = np.ones(rho.size, dtype=bool)
    if rho_max is not None:
        keep = rho <= float(rho_max) + 1e-12
        if int(np.count_nonzero(keep)) < 2:
            raise ProfileConversionError(
                f"rho_max={rho_max} leaves fewer than two grid points"
            )
        provenance["rho_max"] = {
            "kind": "caller_supplied",
            "value": float(rho_max),
            "points_dropped": int(rho.size - np.count_nonzero(keep)),
        }
    rho = rho[keep]

    def equilibrium_profile(name: str) -> Optional[np.ndarray]:
        values = _array(ods, f"{equilibrium_prefix}.{name}")
        return None if values is None else values[keep]

    r_inboard = equilibrium_profile("r_inboard")
    r_outboard = equilibrium_profile("r_outboard")
    if r_inboard is not None and r_outboard is not None:
        rmin = 0.5 * (r_outboard - r_inboard)
        rmaj = 0.5 * (r_outboard + r_inboard)
        provenance["rmin"] = {"kind": "derived", "source": "r_outboard, r_inboard"}
        provenance["rmaj"] = {"kind": "derived", "source": "r_outboard, r_inboard"}
    else:
        rmin = rmaj = None
        provenance["rmin"] = provenance["rmaj"] = {
            "kind": "unavailable",
            "reason": "the equilibrium carries no r_inboard/r_outboard",
        }

    upper = equilibrium_profile("triangularity_upper")
    lower = equilibrium_profile("triangularity_lower")
    if upper is not None and lower is not None:
        delta = 0.5 * (upper + lower)
        provenance["delta"] = {
            "kind": "derived",
            "source": "mean of triangularity_upper and triangularity_lower",
        }
    else:
        delta = equilibrium_profile("triangularity")
        provenance["delta"] = (
            {"kind": "derived", "source": "equilibrium triangularity"}
            if delta is not None
            else {"kind": "unavailable", "reason": "no triangularity on the equilibrium"}
        )

    # zeta is GACODE's squareness and the IMAS squareness_* family is defined
    # per quadrant with a different sign convention; they are not the same
    # number, so nothing is written rather than something close.
    provenance["zeta"] = {
        "kind": "unavailable",
        "reason": "IMAS squareness_* is per-quadrant and is not GACODE's zeta",
    }

    from vaft.data.eqdsk import ods_psi_to_wb_per_radian_factor

    psi_factor = ods_psi_to_wb_per_radian_factor(ods)
    psi = equilibrium_profile("psi")
    polflux = None
    if psi is not None:
        polflux = (psi - psi[0]) * psi_factor
        provenance["polflux"] = {
            "kind": "derived",
            "source": "equilibrium psi, referenced to the axis",
            "wb_per_radian_factor": float(psi_factor),
        }

    # Read untruncated: rho stays normalised to the *plasma boundary* even when
    # the grid is cut short, so Phi(rho) = torfluxa * rho^2 only holds if
    # torfluxa is the boundary value. Taking phi[-1] after truncation would
    # rescale the whole radial coordinate silently.
    phi_full = _array(ods, f"{equilibrium_prefix}.phi")
    torfluxa = None
    if phi_full is not None:
        torfluxa = float(phi_full[-1]) * psi_factor
        provenance["torfluxa"] = {
            "kind": "derived",
            "source": "equilibrium phi at the plasma boundary, before any rho_max cut",
        }

    # Kinetic profiles, mapped onto the equilibrium grid.
    profile_rho = _array(ods, f"{profile_prefix}.grid.rho_tor_norm")
    if profile_rho is None:
        raise ProfileConversionError(
            "core_profiles has no grid.rho_tor_norm, so its profiles cannot be placed "
            "on the equilibrium's radial grid"
        )
    n_e = _array(ods, f"{profile_prefix}.electrons.density_thermal")
    if n_e is None:
        n_e = _array(ods, f"{profile_prefix}.electrons.density")
    t_e = _array(ods, f"{profile_prefix}.electrons.temperature")
    if n_e is None or t_e is None:
        raise ProfileConversionError(
            "core_profiles carries no electron density or temperature; GACODE cannot be "
            "run without them and nothing is substituted"
        )
    ne = _interpolate(profile_rho, n_e, rho)
    te = _interpolate(profile_rho, t_e, rho)
    _require_positive("electron density", ne)
    _require_positive("electron temperature", te)
    provenance["ne"] = {"kind": "measured", "source": f"{profile_prefix}.electrons"}
    provenance["te"] = {"kind": "measured", "source": f"{profile_prefix}.electrons"}

    species = _ion_species(ods, profile_index)
    if not species:
        raise ProfileConversionError(
            "core_profiles carries no ion species; GACODE needs at least one and the "
            "adapter does not invent a main ion"
        )
    densities, temperatures, charges, masses, labels, kinds = [], [], [], [], [], []
    for entry in species:
        if entry["density"] is None or entry["temperature"] is None:
            raise ProfileConversionError(
                f"ion species {entry['label']!r} has no density or no temperature"
            )
        density = _interpolate(profile_rho, entry["density"], rho) / DENSITY_SCALE
        temperature = _interpolate(profile_rho, entry["temperature"], rho) / TEMPERATURE_SCALE
        _require_positive(f"{entry['label']} density", density)
        _require_positive(f"{entry['label']} temperature", temperature)
        densities.append(density)
        temperatures.append(temperature)
        charges.append(entry["z"])
        masses.append(entry["mass"] if entry["mass"] is not None else entry["z"] * 2.0)
        labels.append(entry["label"].replace(" ", ""))
        kinds.append("[therm]")
    provenance["ni"] = {
        "kind": "measured",
        "source": f"{profile_prefix}.ion",
        "species": labels,
    }

    # Toroidal rotation is optional to GACODE and is only written when every
    # species has it: a per-ion array with one species silently zeroed would
    # claim a stationary impurity rather than an unmeasured one.
    rotation = [entry["velocity_toroidal"] for entry in species]
    if all(values is not None for values in rotation):
        vtor = np.vstack([_interpolate(profile_rho, values, rho) for values in rotation])
        provenance["vtor"] = {
            "kind": "measured",
            "source": f"{profile_prefix}.ion.:.velocity.toroidal",
        }
    else:
        vtor = None
        provenance["vtor"] = {
            "kind": "unavailable",
            "reason": "not every ion species carries velocity.toroidal",
        }
    if any(entry["mass"] is None for entry in species):
        provenance["mass"] = {
            "kind": "policy_assumption",
            "reason": "an ion element mass was absent; 2Z amu assumed for it",
        }

    effective_charge = _array(ods, f"{profile_prefix}.zeff")
    if effective_charge is not None:
        z_eff_profile = _interpolate(profile_rho, effective_charge, rho)
        provenance["z_eff"] = {"kind": "measured", "source": f"{profile_prefix}.zeff"}
    elif z_eff is not None:
        z_eff_profile = np.full(rho.size, float(z_eff))
        provenance["z_eff"] = {"kind": "caller_supplied", "value": float(z_eff)}
    elif len(charges) > 1:
        stacked = np.vstack(densities)
        z_eff_profile = (
            np.sum(stacked * np.asarray(charges)[:, None] ** 2, axis=0)
            / (ne / DENSITY_SCALE)
        )
        provenance["z_eff"] = {
            "kind": "derived",
            "source": "quasi-neutral sum over the ion species present",
        }
    else:
        z_eff_profile = None
        provenance["z_eff"] = {
            "kind": "unavailable",
            "reason": "one ion species, no zeff profile, and no z_eff= supplied",
        }

    current = _scalar(ods, f"{global_prefix}.ip")
    profile = GACODEProfile(
        rho=rho,
        z=np.asarray(charges, dtype=float),
        mass=np.asarray(masses, dtype=float),
        name=tuple(labels),
        type=tuple(kinds),
        rmin=rmin,
        rmaj=rmaj,
        zmag=_array(ods, f"{equilibrium_prefix}.geometric_axis.z"),
        kappa=equilibrium_profile("elongation"),
        delta=delta,
        polflux=polflux,
        q=equilibrium_profile("q"),
        ptot=equilibrium_profile("pressure"),
        fpol=equilibrium_profile("f"),
        torfluxa=torfluxa,
        rcentr=_scalar(ods, "equilibrium.vacuum_toroidal_field.r0"),
        bcentr=_scalar(ods, "equilibrium.vacuum_toroidal_field.b0"),
        current=None if current is None else current / 1.0e6,
        ne=ne / DENSITY_SCALE,
        te=te / TEMPERATURE_SCALE,
        ni=np.vstack(densities),
        ti=np.vstack(temperatures),
        z_eff=z_eff_profile,
        vtor=vtor,
        shot=shot if shot is not None else _shot_number(ods),
        time=int(round(times["equilibrium_time"] * 1.0e3)),
        header={
            "original": "vaft.code.gacode.inputs.prepare_gacode_profile",
            "statefile": "IMAS core_profiles",
            "gfile": "IMAS equilibrium",
        },
    )
    if profile.zmag is not None:
        profile.zmag = np.asarray(profile.zmag, dtype=float)[keep]
    profile.provenance = provenance
    return profile


def _shot_number(ods: Any) -> Optional[int]:
    value = _scalar(ods, "dataset_description.data_entry.pulse")
    return None if value is None else int(value)


def prepare_gacode_inputs(
    ods: Any,
    workdir: str | Path,
    **kwargs: Any,
) -> GACODEInputs:
    """Convert an ODS and stage ``input.gacode`` in *workdir*.

    Keyword arguments are those of :func:`prepare_gacode_profile`.  The working
    directory is the caller's: nothing here uses a temporary directory, so a run
    stays inspectable after it finishes.
    """
    directory = Path(workdir)
    directory.mkdir(parents=True, exist_ok=True)
    profile = prepare_gacode_profile(ods, **kwargs)
    written = write_input_gacode(profile, directory / "input.gacode")
    return GACODEInputs(
        workdir=directory,
        files=(written,),
        ods=ods,
        profile=profile,
        input_gacode=written,
        provenance=dict(profile.provenance),
    )
