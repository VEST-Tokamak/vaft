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

**The sign convention.**  VAFT holds COCOS 11 internally; ``input.gacode`` is
COCOS 2 in its signs (see the ``gacode`` entry in :mod:`vaft.data.cocos`), so
the toroidal field, the current, ``fpol``, the toroidal flux, the toroidal
velocity and the poloidal flux all change sign on the way out, and ``q`` does
not.  The factors come from :func:`omas.omas_physics.cocos_transform` rather
than being written out here, and the conversion is recorded in provenance.
Skipping it does not change the bootstrap current -- flipping the field and the
current together preserves the helicity -- but NEO reads the field directions
from these signs, so every lab-frame direction it reports would be mirrored.

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

from vaft.data.cocos import VAFT_INTERNAL_COCOS

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
        # rho_tor_profile wants psi in full weber: stored -> Wb/rad -> Wb.
        factor = ods_psi_to_wb_per_radian_factor(ods)
        derived = rho_tor_profile(q, psi * factor * 2.0 * np.pi)
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


def _resolve_profile_rho(
    ods: Any,
    profile_prefix: str,
    equilibrium_prefix: str,
    profile_rho: np.ndarray,
    equilibrium_rho: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The core_profiles grid as a real ``rho_tor_norm``, and how that was settled.

    The same ``sqrt(psi_N)`` proxy :func:`_resolve_rho` guards against on the
    equilibrium is stored under ``core_profiles...grid.rho_tor_norm`` by every
    writer that copied the equilibrium's array verbatim (issues #276, #420, #574).
    Interpolating it against the re-derived equilibrium coordinate puts each
    kinetic point at the wrong radius, so a detected proxy is carried through
    ``psi_N`` onto the equilibrium's own ``psi_N -> rho`` table instead.

    ``equilibrium_rho`` is the untruncated coordinate :func:`_resolve_rho` returned.
    """
    from vaft.data._derived import is_rho_pol_proxy

    equilibrium_psi = _array(ods, f"{equilibrium_prefix}.psi")
    equilibrium_psi_norm = None
    if (
        equilibrium_psi is not None
        and equilibrium_psi.size == equilibrium_rho.size
        and equilibrium_psi.size > 1
        and equilibrium_psi[-1] != equilibrium_psi[0]
    ):
        equilibrium_psi_norm = (equilibrium_psi - equilibrium_psi[0]) / (
            equilibrium_psi[-1] - equilibrium_psi[0]
        )

    # psi_N at the profile points, from whatever the grid says about itself.
    psi_norm, psi_source = None, None
    rho_pol = _array(ods, f"{profile_prefix}.grid.rho_pol_norm")
    grid_psi = _array(ods, f"{profile_prefix}.grid.psi")
    stored = _array(ods, f"{equilibrium_prefix}.rho_tor_norm")
    if rho_pol is not None and rho_pol.size == profile_rho.size:
        psi_norm, psi_source = rho_pol**2, "grid.rho_pol_norm"
    elif (
        grid_psi is not None
        and grid_psi.size == profile_rho.size
        and equilibrium_psi_norm is not None
    ):
        # Normalised with the equilibrium's axis and boundary, not the grid's own
        # end points: a kinetic grid need not reach either.
        psi_norm = (grid_psi - equilibrium_psi[0]) / (
            equilibrium_psi[-1] - equilibrium_psi[0]
        )
        psi_source = "grid.psi"
    elif (
        stored is not None
        and equilibrium_psi_norm is not None
        and stored.size == profile_rho.size
        and np.allclose(stored, profile_rho)
    ):
        psi_norm, psi_source = equilibrium_psi_norm, "the equilibrium grid it copies"

    if not is_rho_pol_proxy(profile_rho, psi_norm):
        if psi_norm is None:
            return profile_rho, {
                "kind": "unverified",
                "source": f"{profile_prefix}.grid.rho_tor_norm",
                "reason": (
                    "the grid carries no psi or rho_pol_norm to test it against, so "
                    "it is taken to be the toroidal coordinate its name says"
                ),
            }
        return profile_rho, {
            "kind": "measured",
            "source": f"{profile_prefix}.grid.rho_tor_norm",
            "checked_against": psi_source,
        }

    if psi_norm is None:
        # Detected against the uniform-psi_N default, which is the proxy's own
        # definition: psi_N is its square.
        psi_norm, psi_source = profile_rho**2, "the proxy itself"
    if equilibrium_psi_norm is None:
        raise ProfileConversionError(
            "core_profiles grid.rho_tor_norm is the sqrt(psi_N) proxy (issues #276, "
            "#420) and the equilibrium carries no psi to carry it onto the toroidal "
            "coordinate. Interpolating it as rho_tor_norm would misplace every "
            "kinetic point, so the conversion stops here."
        )
    order = np.argsort(equilibrium_psi_norm)
    converted = np.interp(
        np.clip(psi_norm, 0.0, 1.0), equilibrium_psi_norm[order], equilibrium_rho[order]
    )
    return converted, {
        "kind": "derived",
        "source": (
            f"{profile_prefix}.grid.rho_tor_norm is the sqrt(psi_N) proxy; mapped "
            f"through psi_N ({psi_source}) onto the equilibrium's rho_tor_norm"
        ),
    }


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


#: Impurities a caller can ask to be modelled explicitly, with the charge and mass
#: GACODE needs for each. Fully stripped: at VEST's temperatures carbon is not, but
#: NEO takes one charge state per species, and the fully-stripped value is the
#: convention every Z_eff-consistent impurity model in the field uses.
IMPURITIES: Mapping[str, tuple[float, float, str]] = {
    "C": (6.0, 12.011, "C6+"),
    "O": (8.0, 15.999, "O8+"),
    "N": (7.0, 14.007, "N7+"),
    "He": (2.0, 4.0026, "He2+"),
}


#: Standard atomic masses of the hydrogen isotopes [amu], by the leading letter
#: of the species label.
HYDROGEN_ISOTOPE_MASSES: Mapping[str, float] = {"H": 1.00784, "D": 2.01410, "T": 3.01605}


def _assumed_mass(label: str, charge: float) -> float:
    """Mass [amu] for an ion whose ``element.0.a`` is absent.

    ``2 Z`` is right to a percent for the fully-stripped light impurities, but
    for ``Z = 1`` it asserts deuterium, and VEST runs hydrogen.  A hydrogenic
    ion therefore takes its isotope from its label, and one whose label does
    not say which isotope it is is refused rather than guessed.
    """
    if float(charge) != 1.0:
        return float(charge) * 2.0
    key = str(label).strip()[:1].upper()
    if key in HYDROGEN_ISOTOPE_MASSES:
        return HYDROGEN_ISOTOPE_MASSES[key]
    raise ProfileConversionError(
        f"ion species {label!r} has Z = 1 and no element.0.a, and its label does not "
        "say whether it is H, D or T; the mass differs by a factor of two or three, "
        "so it is not assumed"
    )


def impurity_fractions(z_eff: float, z_impurity: float) -> tuple[float, float]:
    r"""``(n_main/n_e, n_imp/n_e)`` for one hydrogenic ion and one impurity.

    The two conditions that fix them are quasi-neutrality and the definition of the
    effective charge, with the main ion hydrogenic (``Z = 1``):

    .. math::

        n_e = n_H + Z_I n_I, \qquad Z_{\mathrm{eff}} n_e = n_H + Z_I^2 n_I

    which give ``n_I/n_e = (Z_eff - 1)/(Z_I(Z_I - 1))`` and
    ``n_H/n_e = 1 - Z_I n_I/n_e``. For carbon at ``Z_eff = 2`` that is
    ``n_C = n_e/30`` and ``n_H = 0.8 n_e`` -- the standard construction, and the
    reason a plasma cannot be both ``Z_eff = 2`` and purely hydrogenic.

    Raises
    ------
    ValueError
        When ``z_eff`` is outside ``[1, Z_I]``, where no non-negative pair exists.
    """
    charge = float(z_impurity)
    target = float(z_eff)
    if charge <= 1.0:
        raise ValueError(f"an impurity needs Z > 1; got {z_impurity!r}")
    if not 1.0 <= target <= charge:
        raise ValueError(
            f"Z_eff = {target} is unreachable with a Z = {charge:g} impurity in "
            f"hydrogen: it must lie between 1 (no impurity) and {charge:g} (no main ion)"
        )
    impurity = (target - 1.0) / (charge * (charge - 1.0))
    return 1.0 - charge * impurity, impurity


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
        charge_assumed = False
        if charge is None:
            # No charge state stored. The nuclear charge is the fully-stripped
            # value, which is the convention of this adapter (see IMPURITIES);
            # without even that, Z = 1 would turn a carbon entry into hydrogen.
            charge = _scalar(ods, f"{base}.element.0.z_n")
            charge_assumed = True
            if charge is None:
                raise ProfileConversionError(
                    f"ion species {label!r} carries neither z_ion nor element.0.z_n; "
                    "its charge is not something the adapter will assume"
                )
        mass = _scalar(ods, f"{base}.element.0.a")
        density = _array(ods, f"{base}.density_thermal")
        if density is None:
            density = _array(ods, f"{base}.density")
        species.append(
            {
                "label": label,
                "z": float(charge),
                "z_assumed": charge_assumed,
                "mass": mass,
                "density": density,
                "temperature": _array(ods, f"{base}.temperature"),
                "velocity_toroidal": _array(ods, f"{base}.velocity.toroidal"),
            }
        )
        position += 1
    return species


def _impurity_entry(name: str) -> tuple[float, float, str]:
    """The charge, mass and GACODE label of a named impurity."""
    key = str(name).strip()
    for candidate in (key, key.capitalize(), key.upper()):
        if candidate in IMPURITIES:
            return IMPURITIES[candidate]
    raise ProfileConversionError(
        f"unknown impurity {name!r}; known: {', '.join(sorted(IMPURITIES))}. The table "
        "is vaft.code.gacode.inputs.IMPURITIES and takes a charge and a mass."
    )


def _resolve_target_charge(z_eff: Optional[float]) -> float:
    """The effective charge an impurity is being asked to realize.

    Deliberately the caller's number alone. A measured ``zeff`` profile cannot be
    the target: one impurity fraction cannot follow a radially varying charge and
    stay a single species with one density profile. Where a state carries one,
    ``prepare_gacode_profile`` refuses the disagreement rather than resolving it
    here, so that the refusal names both values.
    """
    if z_eff is not None:
        return float(z_eff)
    raise ProfileConversionError(
        "impurity= needs a z_eff= to aim at; it is the target the impurity fraction "
        "is derived from, and this layer holds no machine default"
    )


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
    impurity: Optional[str] = None,
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

        Note what this alone does *not* do: it fills a column NEO ignores. NEO builds
        its collision operator from the species list, so a file with ``z_eff = 2`` and
        one hydrogenic ion describes a plasma at ``Z_eff = 1`` to every solver that
        reads it (issue #803). Pass ``impurity`` to make the two agree.
    impurity
        Model the requested ``z_eff`` as a named impurity -- ``"C"``, ``"O"``, ``"N"``
        or ``"He"`` -- rather than as a column. The impurity and the main ion are
        given the densities that satisfy quasi-neutrality and the requested effective
        charge together (:func:`impurity_fractions`), so the species list NEO reads
        carries the charge the caller asked for. Requires a ``z_eff``, and the state
        must have exactly one ion species for the pair to be determined.
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

    rho_full = rho
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

    sign = _cocos_factors()
    provenance["cocos"] = {
        "kind": "derived",
        "from": VAFT_INTERNAL_COCOS,
        "to": _gacode_cocos(),
        "factors": {key: float(sign[key]) for key in ("PSI", "TOR", "BT", "IP", "F", "Q")},
        "confirmed": False,
        "source": "vaft.data.cocos.convention_for('gacode')",
    }

    psi_factor = ods_psi_to_wb_per_radian_factor(ods)
    psi = equilibrium_profile("psi")
    polflux = None
    if psi is not None:
        # stored -> full weber, then COCOS 11 -> 2, which also divides by 2*pi.
        polflux = (psi - psi[0]) * psi_factor * 2.0 * np.pi * sign["PSI"]
        provenance["polflux"] = {
            "kind": "derived",
            "source": "equilibrium psi, referenced to the axis",
            "wb_per_radian_factor": float(psi_factor),
        }

    # profiles_1d.phi is the full toroidal flux in weber whichever way psi is
    # stored (vaft/data/eqdsk.py), so the psi storage factor must not touch it:
    # applying it would write torfluxa 2*pi too large for a per-radian ODS.
    # GACODE's torfluxa is per radian by its own definition -- expro derives
    # B_unit as d(torfluxa rho^2)/d(r^2/2) -- which is where the 2*pi comes from.
    #
    # Read untruncated: rho stays normalised to the *plasma boundary* even when
    # the grid is cut short, so Phi(rho) = torfluxa * rho^2 only holds if
    # torfluxa is the boundary value.
    phi_full = _array(ods, f"{equilibrium_prefix}.phi")
    torfluxa = None
    if phi_full is not None:
        torfluxa = float(phi_full[-1]) / (2.0 * np.pi) * sign["TOR"]
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
    profile_rho, provenance["profile_grid"] = _resolve_profile_rho(
        ods, profile_prefix, equilibrium_prefix, profile_rho, rho_full
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
    if provenance["profile_grid"]["kind"] == "derived":
        # The values are the measurement; the radius they sit at is not what the
        # file said, and a reader of ne/te should not have to look elsewhere.
        for key in ("ne", "te"):
            provenance[key]["grid"] = provenance["profile_grid"]["source"]

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
        masses.append(
            entry["mass"] if entry["mass"] is not None
            else _assumed_mass(entry["label"], entry["z"])
        )
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
        vtor = sign["TOR"] * np.vstack(
            [_interpolate(profile_rho, values, rho) for values in rotation]
        )
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
            "reason": (
                "an ion element mass was absent; a hydrogenic ion took its "
                "isotope's standard mass from its label, any other 2Z amu"
            ),
        }
    if any(entry["z_assumed"] for entry in species):
        provenance["z"] = {
            "kind": "policy_assumption",
            "reason": (
                "an ion carried no z_ion; its nuclear charge element.0.z_n was "
                "taken as the charge state (fully stripped)"
            ),
            "species": [
                entry["label"].replace(" ", "") for entry in species if entry["z_assumed"]
            ],
        }

    if impurity is not None:
        target = _resolve_target_charge(z_eff)
        charge, mass, label = _impurity_entry(impurity)
        if len(species) != 1:
            raise ProfileConversionError(
                f"impurity={impurity!r} needs exactly one ion species to pair with; "
                f"this slice carries {len(species)} ({', '.join(labels)}). With more "
                "than one the split that realizes a given Z_eff is not determined, so "
                "the adapter will not choose one."
            )
        main_fraction, impurity_fraction = impurity_fractions(target, charge)
        if impurity_fraction <= 0.0:
            raise ProfileConversionError(
                f"z_eff={target:g} needs no {label} at all, so impurity={impurity!r} "
                "would write a species with zero density everywhere -- which GACODE "
                "takes a logarithmic gradient of, and which this converter refuses "
                "for a measured profile. Drop impurity= for a hydrogenic plasma."
            )
        electron_density = ne / DENSITY_SCALE
        # Both ion densities come from n_e, not from the measured main-ion profile:
        # a plasma cannot be quasi-neutral, carry this impurity, and keep n_H = n_e.
        # Scaling the measurement is the assumption being made, and it is recorded.
        densities = [electron_density * main_fraction, electron_density * impurity_fraction]
        temperatures = [temperatures[0], temperatures[0]]
        charges = [charges[0], charge]
        masses = [masses[0], mass]
        labels = [labels[0], label]
        kinds = [kinds[0], "[therm]"]
        if vtor is not None:
            vtor = np.vstack([vtor[0], vtor[0]])
        provenance["ni"] = {
            "kind": "policy_assumption",
            "source": f"{profile_prefix}.electrons.density_thermal",
            "species": labels,
            "reason": (
                f"{label} added so the species list realizes Z_eff = {target:g}; both "
                "ion densities are derived from n_e by quasi-neutrality, which "
                f"overrides the measured main-ion profile ({main_fraction:.4g} n_e "
                f"and {impurity_fraction:.4g} n_e)"
            ),
        }
        provenance["ti"] = {
            "kind": "policy_assumption",
            "reason": f"{label} is given the main ion's temperature; none is measured",
        }

    effective_charge = _array(ods, f"{profile_prefix}.zeff")
    if impurity is not None:
        # The species list is what NEO reads, so once it realizes a charge the column
        # must agree with it or the file contradicts itself -- which is the defect
        # #803 is about, and it would come back here for any state carrying a zeff.
        stacked = np.vstack(densities)
        z_eff_profile = (
            np.sum(stacked * np.asarray(charges)[:, None] ** 2, axis=0)
            / (ne / DENSITY_SCALE)
        )
        provenance["z_eff"] = {
            "kind": "derived",
            "source": f"the {', '.join(labels)} species list this conversion built",
            "value": float(np.mean(z_eff_profile)),
        }
        if effective_charge is not None:
            measured = float(np.mean(_interpolate(profile_rho, effective_charge, rho)))
            provenance["z_eff"]["overrode_measured"] = measured
            if abs(measured - float(np.mean(z_eff_profile))) > 1e-3:
                raise ProfileConversionError(
                    f"{profile_prefix}.zeff measures {measured:.4g} but impurity="
                    f"{impurity!r} with z_eff={float(z_eff):.4g} builds a species list "
                    f"at {float(np.mean(z_eff_profile)):.4g}. Writing both would put a "
                    "column in input.gacode that contradicts the species NEO actually "
                    "reads. Pass z_eff= matching the measurement, or drop impurity= to "
                    "keep the measured profile as a column."
                )
    elif effective_charge is not None:
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
    # b0 is sampled on the equilibrium time base, and VEST's drifts by up to a
    # factor of two within a shot (#325), so it is read at the converted slice
    # rather than at index 0.
    field = _array(ods, "equilibrium.vacuum_toroidal_field.b0")
    b0 = None
    if field is not None:
        field = np.atleast_1d(field)
        b0 = float(field[min(times["equilibrium_index"], field.size - 1)])
    q_profile = equilibrium_profile("q")
    f_profile = equilibrium_profile("f")
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
        q=None if q_profile is None else q_profile * sign["Q"],
        ptot=equilibrium_profile("pressure"),
        fpol=None if f_profile is None else f_profile * sign["F"],
        torfluxa=torfluxa,
        rcentr=_scalar(ods, "equilibrium.vacuum_toroidal_field.r0"),
        bcentr=None if b0 is None else b0 * sign["BT"],
        current=None if current is None else current / 1.0e6 * sign["IP"],
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


def _gacode_cocos() -> int:
    from vaft.data.cocos import convention_for

    return int(convention_for("gacode").cocos)


def _cocos_factors() -> Mapping[str, float]:
    """Multipliers taking VAFT's COCOS 11 quantities into input.gacode's convention."""
    from omas.omas_physics import cocos_transform

    return cocos_transform(VAFT_INTERNAL_COCOS, _gacode_cocos())


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
