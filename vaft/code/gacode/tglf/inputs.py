"""Project a GACODE profile onto TGLF's local input, and write ``input.tglf``.

**TGLF never reads ``input.gacode``.** NEO is a profile code, so #550 could hand it a
whole :class:`~vaft.code.gacode._profiles.GACODEProfile` and let ``expro`` do the rest.
TGLF is local: it takes dimensionless quantities at one flux surface -- shaping and its
radial derivatives, normalised logarithmic gradients, species ratios, and four
normalisations -- and everything ``expro`` did for NEO has to be done here instead.

That makes this module the place where GACODE's conventions have to be reproduced rather
than deferred to, which is the failure mode this milestone has paid for three times (the
``b_unit`` sign in #661, the ``torfluxa`` factor in #661, the ``z_eff`` column in #803).
So every quantity below is held against GACODE's own answer:
``$GACODEHOME/profiles_gen/locpargen/locpargen`` writes ``input.tglf.locpargen`` from any
``input.gacode``, and ``test_tglf_input.py`` compares all 34 physics keys against it at
three radii. The agreement is exact to the printed precision.

Three conventions that agreement pinned down, none of which a self-consistent check would
have caught:

*Differentiate on the grid, then interpolate.* ``expro`` builds ``skappa``, ``sdelta``,
``drmaj`` and the logarithmic gradients over the whole profile with a three-point
Lagrange derivative (``bound_deriv``) and interpolates *those*. Interpolating first and
differentiating at the target radius leaves ``S_KAPPA_LOC`` 87 percent out on VEST, where
the value is ~6e-3 and small absolute errors are enormous relative ones.

*The sound speed is normalised to deuterium.* ``expro``'s ``mp`` is already
``mass_deuterium/2``, so its ``2.0*mp`` is the deuterium mass. Using ``mp`` gives a
``c_s`` too large by sqrt(2) -- which for a while cancelled a second error and made
``XNUE`` look right.

*The electron mass is carried in two different units.* ``expro``'s ``masse`` is in proton
units (5.4489e-4); TGLF's ``MASS_1`` is deuterium-normalised (``masse/2``). The collision
rate takes ``sqrt(masse/2)``, so feeding it TGLF's already-halved value is sqrt(2) wrong,
and ``XNUE`` is the only output that shows it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ...base import CodeInputs
from .._profiles import GACODEProfile
from ._types import (
    MAX_SPECIES,
    TGLF_DEFAULT_SCALARS,
    TGLF_DEFAULT_SPECIES,
    TGLFConfig,
)

__all__ = [
    "TGLFInput",
    "TGLFInputs",
    "LocalConversionError",
    "bound_deriv",
    "prepare_tglf_case",
    "prepare_tglf_input",
    "tglf_parameters",
    "write_input_tglf",
]

# GACODE's own constants, in the cgs units its expressions are written in
# (`f2py/expro/expro_util.f90:13-19`, `f2py/expro/expro.f90:7`). Restated rather than
# taken from scipy: these are the values GACODE computed with, and a comparison against
# its output has to use them.
_K_ERG_PER_EV = 1.6022e-12
_E_STATCOUL = 4.8032e-10
_C_CM_S = 2.9979e10
_MASS_DEUTERIUM_G = 3.34358e-24
_TEMP_NORM_FAC = 1602.2
_CHARGE_NORM_FAC = 1.6022


class LocalConversionError(ValueError):
    """The profile cannot be projected onto a local TGLF input.

    Distinct from a plain ``ValueError`` so a caller can tell "this state cannot be
    modelled at this radius" from a programming mistake, matching
    :class:`~vaft.code.gacode.inputs.ProfileConversionError`.
    """


@dataclass(frozen=True)
class TGLFInput:
    """TGLF's local input at one flux surface.

    Every field is dimensionless in TGLF's own normalisation: lengths by the minor
    radius ``a``, temperatures and densities by their electron values, the field by
    ``B_unit``. Species are indexed as TGLF indexes them, **electrons first**.

    Attributes
    ----------
    provenance
        Per-quantity ``{"kind": ...}`` records in the convention
        :class:`~vaft.code.gacode._profiles.GACODEProfile` uses. A quantity the state
        cannot supply is ``{"kind": "unavailable", "reason": ...}``.

        Note what that does *not* mean: the key is still written to ``input.tglf`` with
        TGLF's own default, because a file that omits a setting cannot afterwards be
        told apart from one that chose the default deliberately. The provenance is what
        separates them, and it is the only thing that does -- a reader of the file alone
        sees ``VEXB_SHEAR=0.0`` and cannot know nobody measured it.
    """

    rho: float
    rmin_loc: float
    rmaj_loc: float
    drmajdx_loc: float
    zmaj_loc: float
    dzmajdx_loc: float
    q_loc: float
    q_prime_loc: float
    p_prime_loc: float
    kappa_loc: float
    s_kappa_loc: float
    delta_loc: float
    s_delta_loc: float
    zeta_loc: float
    s_zeta_loc: float

    #: Species arrays, electrons at index 0.
    zs: np.ndarray
    mass: np.ndarray
    as_: np.ndarray
    taus: np.ndarray
    rlns: np.ndarray
    rlts: np.ndarray

    betae: float
    xnue: float
    zeff: float
    debye: float
    sign_bt: float
    sign_it: float

    vexb_shear: Optional[float] = None
    names: Sequence[str] = ()
    provenance: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def n_species(self) -> int:
        return int(np.size(self.zs))

    def missing(self) -> tuple[str, ...]:
        """Names this input records as unavailable, in provenance order."""
        return tuple(
            name
            for name, record in self.provenance.items()
            if record.get("kind") == "unavailable"
        )

    def check_tglf_requirements(self) -> tuple[str, ...]:
        """Quantities TGLF needs that this input does not carry.

        The analogue of ``GACODEProfile.check_neo_requirements``: a precondition
        report, not a verdict. ``vexb_shear`` is *not* required -- TGLF defaults it to
        zero, and a state with no rotation legitimately has none to give.
        """
        required = (
            "rmin_loc", "rmaj_loc", "q_loc", "kappa_loc", "betae", "xnue", "zeff",
        )
        absent = [name for name in required if not np.isfinite(getattr(self, name))]
        for name in ("zs", "mass", "as_", "taus", "rlns", "rlts"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.size == 0 or not np.all(np.isfinite(values)):
                absent.append(name)
        return tuple(absent)


@dataclass
class TGLFInputs(CodeInputs):
    """A staged TGLF case: the directory, the file in it, and what made it."""

    local: Optional[TGLFInput] = None
    input_tglf: Optional[Path] = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)


def bound_deriv(values: Any, radius: Any) -> np.ndarray:
    """GACODE's three-point Lagrange derivative on a possibly unequal grid.

    ``le3/profiles_3d/bound_deriv.f90``, transcribed because the *order* of operations
    matters more than the scheme: ``expro`` differentiates the whole profile with this
    and interpolates the result, and doing it the other way round is wrong by far more
    than the difference between two reasonable derivative estimates.

    Parameters
    ----------
    values, radius
        The function and the grid it is sampled on, same length, at least 3 points.

    Returns
    -------
    np.ndarray
        ``d(values)/d(radius)`` at each grid point.
    """
    f = np.asarray(values, dtype=float)
    r = np.asarray(radius, dtype=float)
    if f.shape != r.shape or f.ndim != 1:
        raise ValueError("values and radius must be one-dimensional and the same length")
    count = r.size
    if count < 3:
        raise ValueError("a three-point derivative needs at least three points")
    index = np.arange(count)
    left = np.where(index == 0, 0, np.where(index == count - 1, count - 3, index - 1))
    first, second, third = left, left + 1, left + 2
    here = r[index]
    r1, r2, r3 = r[first], r[second], r[third]
    f1, f2, f3 = f[first], f[second], f[third]
    return (
        ((here - r1) + (here - r2)) / (r3 - r1) / (r3 - r2) * f3
        + ((here - r1) + (here - r3)) / (r2 - r1) / (r2 - r3) * f2
        + ((here - r2) + (here - r3)) / (r1 - r2) / (r1 - r3) * f1
    )


def _at(grid: np.ndarray, values: np.ndarray, target: float) -> float:
    """Interpolate to one radius with a cubic spline, as ``cub_spline1`` does."""
    from scipy.interpolate import CubicSpline

    return float(CubicSpline(grid, np.asarray(values, dtype=float))(target))


def prepare_tglf_input(
    profile: GACODEProfile,
    rho: float,
    *,
    config: Optional[TGLFConfig] = None,
) -> TGLFInput:
    """Project *profile* onto TGLF's local input at ``r/a = rho``.

    Parameters
    ----------
    profile
        A :class:`~vaft.code.gacode._profiles.GACODEProfile`, normally from
        :func:`~vaft.code.gacode.inputs.prepare_gacode_profile`. It must carry the
        geometry and kinetic profiles GACODE needs; what is absent is reported rather
        than filled.
    rho
        The flux surface, as ``r/a`` -- the same coordinate ``locpargen`` takes, not
        ``rho_tor_norm``.
    config
        Optional. Only ``n_species`` is read here, and only to *check* that the profile
        carries the species the caller expects; the physics settings reach the file
        through :func:`tglf_parameters`.

    Returns
    -------
    TGLFInput

    Raises
    ------
    LocalConversionError
        The profile lacks something the projection needs, or *rho* is outside it.
    """
    target = float(rho)
    if not 0.0 < target < 1.0:
        raise LocalConversionError(
            f"rho must be a normalised minor radius strictly inside the plasma; "
            f"got {target}"
        )

    required = ("rmin", "rmaj", "q", "kappa", "ne", "te", "ni", "ti", "torfluxa")
    absent = [
        name for name in required if getattr(profile, name, None) is None
    ]
    if absent:
        raise LocalConversionError(
            f"the profile is missing {', '.join(absent)}; TGLF's local input is derived "
            "from them and nothing is substituted. "
            "vaft.code.gacode.inputs.prepare_gacode_profile records what it could not "
            "supply in `provenance`."
        )

    rmin = np.asarray(profile.rmin, dtype=float)
    minor_radius = float(rmin[-1])
    grid = rmin / minor_radius
    if not grid[0] <= target <= grid[-1]:
        raise LocalConversionError(
            f"r/a = {target} is outside the converted profile, which spans "
            f"{grid[0]:.4g} to {grid[-1]:.4g}. A rho_max cut moves the outer edge."
        )

    provenance: dict[str, Mapping[str, Any]] = {
        "rho": {"kind": "caller_supplied", "value": target},
        "source": {
            "kind": "derived",
            "reason": "local projection of a GACODEProfile, verified against locpargen",
        },
    }

    electrons = np.asarray(profile.ne, dtype=float)
    electron_temperature = np.asarray(profile.te, dtype=float)
    ion_density = np.atleast_2d(np.asarray(profile.ni, dtype=float))
    ion_temperature = np.atleast_2d(np.asarray(profile.ti, dtype=float))
    charge = np.asarray(profile.z, dtype=float)
    ion_mass = np.asarray(profile.mass, dtype=float)

    count = charge.size + 1
    if count > MAX_SPECIES:
        raise LocalConversionError(
            f"TGLF takes at most {MAX_SPECIES} species including electrons; this "
            f"profile has {count}. Drop an impurity or model it as part of Z_eff."
        )
    requested = None if config is None else config.n_species
    if requested is not None and int(requested) != count:
        # Refused rather than silently truncated: dropping a species changes the
        # plasma, and which one to drop is not this layer's decision. #803 is the
        # precedent -- the impurity model belongs to whoever builds the profile.
        raise LocalConversionError(
            f"the configuration asks for {int(requested)} species and this profile "
            f"carries {count} ({', '.join(('e',) + tuple(profile.name))}). Build the "
            "profile with the species you want -- prepare_gacode_profile's impurity= "
            "decides them -- rather than having them dropped here."
        )

    electron_density_at = _at(grid, electrons, target)
    electron_temperature_at = _at(grid, electron_temperature, target)

    def normalised_gradient(values: np.ndarray) -> float:
        """``a * d(-ln X)/dr``, on the grid then interpolated -- expro's order."""
        derivative = bound_deriv(-np.log(np.asarray(values, dtype=float)), rmin)
        return minor_radius * _at(grid, derivative, target)

    # Species, electrons first, exactly as TGLF indexes them.
    zs = np.concatenate(([-1.0], charge))
    mass = np.concatenate(([profile.masse / 2.0], ion_mass / 2.0))
    fractions = [1.0] + [
        _at(grid, ion_density[i], target) / electron_density_at
        for i in range(charge.size)
    ]
    ratios = [1.0] + [
        _at(grid, ion_temperature[i], target) / electron_temperature_at
        for i in range(charge.size)
    ]
    density_gradients = [normalised_gradient(electrons)] + [
        normalised_gradient(ion_density[i]) for i in range(charge.size)
    ]
    temperature_gradients = [normalised_gradient(electron_temperature)] + [
        normalised_gradient(ion_temperature[i]) for i in range(charge.size)
    ]

    # Geometry. The shaping shears are built on the grid and interpolated.
    major = np.asarray(profile.rmaj, dtype=float)
    q_profile = np.abs(np.asarray(profile.q, dtype=float))
    kappa = np.asarray(profile.kappa, dtype=float)
    elevation = (
        np.zeros_like(rmin) if profile.zmag is None
        else np.asarray(profile.zmag, dtype=float)
    )
    if profile.zmag is None:
        provenance["zmaj_loc"] = {
            "kind": "unavailable",
            "reason": "the profile carries no zmag; the surface is taken as centred",
        }
    triangularity, squareness = _shaping(profile, rmin, provenance)

    q_at = _at(grid, q_profile, target)
    shear = _at(grid, rmin * bound_deriv(np.log(q_profile), rmin), target)

    # Normalisations, in GACODE's cgs constants.
    toroidal_flux = float(profile.torfluxa) * np.asarray(profile.rho, dtype=float) ** 2
    b_unit = bound_deriv(toroidal_flux, 0.5 * rmin**2)
    sound_speed = np.sqrt(_K_ERG_PER_EV * (1e3 * electron_temperature) / _MASS_DEUTERIUM_G)
    gyroradius = sound_speed / (
        _E_STATCOUL * (1e4 * b_unit) / (_MASS_DEUTERIUM_G * _C_CM_S)
    )
    b_unit_at = _at(grid, b_unit, target)
    sound_speed_at = _at(grid, sound_speed / 1e2, target)
    gyroradius_at = _at(grid, gyroradius / 1e2, target)

    betae = 4.027e-3 * electron_density_at * electron_temperature_at / b_unit_at**2

    collision_constant = (
        np.sqrt(2.0) * np.pi * _CHARGE_NORM_FAC**4 / (4.0 * np.pi * 8.8542) ** 2
        * 1e9 / (np.sqrt(_MASS_DEUTERIUM_G * 1e24) * _TEMP_NORM_FAC**1.5)
    )
    coulomb_log = 24.0 - np.log(
        np.sqrt(electron_density_at * 1e13) / (electron_temperature_at * 1e3)
    )
    collision_rate = (
        collision_constant * coulomb_log * electron_density_at
        / (np.sqrt(profile.masse / 2.0) * electron_temperature_at**1.5)
    )

    effective_charge = (
        2.0 if profile.z_eff is None
        else _at(grid, np.asarray(profile.z_eff, dtype=float), target)
    )
    if profile.z_eff is None:
        provenance["zeff"] = {
            "kind": "policy_assumption",
            "reason": "the profile carries no z_eff column; TGLF's own default is used",
        }

    # beta_star sums every species' pressure-gradient drive, with a-normalised
    # gradients -- the one place the normalised form is the one that enters.
    drive = electron_density_at * electron_temperature_at * (
        density_gradients[0] + temperature_gradients[0]
    )
    for i in range(charge.size):
        drive += (
            _at(grid, ion_density[i], target)
            * _at(grid, ion_temperature[i], target)
            * (density_gradients[i + 1] + temperature_gradients[i + 1])
        )
    beta_star = drive * betae / (electron_density_at * electron_temperature_at)

    rotation = _rotation(profile, provenance)

    return TGLFInput(
        rho=target,
        rmin_loc=target,
        rmaj_loc=_at(grid, major, target) / minor_radius,
        drmajdx_loc=_at(grid, bound_deriv(major, rmin), target),
        zmaj_loc=_at(grid, elevation, target) / minor_radius,
        dzmajdx_loc=_at(grid, bound_deriv(elevation, rmin), target),
        q_loc=q_at,
        q_prime_loc=(q_at / target) ** 2 * shear,
        p_prime_loc=(abs(q_at) / target) * (-beta_star / (8.0 * np.pi)),
        kappa_loc=_at(grid, kappa, target),
        s_kappa_loc=_at(grid, rmin / kappa * bound_deriv(kappa, rmin), target),
        delta_loc=_at(grid, triangularity, target),
        s_delta_loc=_at(grid, rmin * bound_deriv(triangularity, rmin), target),
        zeta_loc=_at(grid, squareness, target),
        s_zeta_loc=_at(grid, rmin * bound_deriv(squareness, rmin), target),
        zs=zs,
        mass=mass,
        as_=np.asarray(fractions, dtype=float),
        taus=np.asarray(ratios, dtype=float),
        rlns=np.asarray(density_gradients, dtype=float),
        rlts=np.asarray(temperature_gradients, dtype=float),
        betae=betae,
        xnue=collision_rate * minor_radius / sound_speed_at,
        zeff=effective_charge,
        debye=7.43 * np.sqrt(
            1e3 * electron_temperature_at / (1e13 * electron_density_at)
        ) / abs(gyroradius_at),
        sign_bt=1.0,
        sign_it=1.0,
        vexb_shear=rotation,
        names=("e",) + tuple(profile.name),
        provenance=provenance,
    )


def _shaping(
    profile: GACODEProfile, rmin: np.ndarray, provenance: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Triangularity and squareness, zero-filled only where the profile says absent."""
    zeros = np.zeros_like(rmin)
    triangularity = zeros if profile.delta is None else np.asarray(profile.delta, float)
    if profile.delta is None:
        absent_delta = {
            "kind": "unavailable",
            "reason": "no triangularity on the profile; the surface is taken as elliptic",
        }
        # The shear is a radial derivative of the same absent quantity, so it is no
        # better known than the value; recording only the value would let a consumer
        # read S_DELTA_LOC as measured.
        provenance["delta_loc"] = absent_delta
        provenance["s_delta_loc"] = absent_delta
    squareness = zeros if profile.zeta is None else np.asarray(profile.zeta, float)
    if profile.zeta is None:
        absent_zeta = {
            "kind": "unavailable",
            "reason": (
                "no squareness on the profile; TGLF's own default of zero is used, "
                "which is a claim about the surface rather than a missing value"
            ),
        }
        provenance["zeta_loc"] = absent_zeta
        provenance["s_zeta_loc"] = absent_zeta
    return triangularity, squareness


def _rotation(profile: GACODEProfile, provenance: dict) -> Optional[float]:
    """The ExB shear, when the profile carries the rotation to derive it from.

    ``prepare_gacode_profile`` does not populate ``w0`` today, so this is normally
    ``None``. TGLF's default of zero then reaches the file -- a zero ExB shear
    suppresses nothing and is a physical statement, so the provenance records that
    nobody chose it. Suppressing the key instead would make the file unreproducible
    without making the assumption any more visible.
    """
    if getattr(profile, "w0", None) is None:
        provenance["vexb_shear"] = {
            "kind": "unavailable",
            "reason": (
                "the profile carries no w0, so no ExB shear can be derived; TGLF's "
                "default of zero applies and is recorded as the assumption it is"
            ),
        }
        return None
    # `w0` is present but the derivation is not written yet, and saying "derived" here
    # would be a provenance that asserts a measurement behind a value that is not
    # there -- the exact distinction this module relies on provenance to carry. It is
    # recorded as unavailable, with the reason naming what is missing.
    provenance["vexb_shear"] = {
        "kind": "unavailable",
        "reason": (
            "the profile carries w0, but the ExB shear derivation from it is not "
            "implemented yet (#553 increment 2); TGLF's default of zero applies and "
            "no rotation information reaches the run"
        ),
    }
    return None


def tglf_parameters(
    local: TGLFInput, config: Optional[TGLFConfig] = None
) -> dict[str, Any]:
    """The ``KEY=VALUE`` settings this local input and configuration mean.

    Every key TGLF knows is written, defaults included, for the reason
    ``neo_parameters`` writes them: a file that omits a setting cannot be told apart
    later from one that chose the default deliberately.
    """
    configuration = config or TGLFConfig()
    parameters: dict[str, Any] = dict(TGLF_DEFAULT_SCALARS)

    count = local.n_species
    parameters["NS"] = count
    parameters["SAT_RULE"] = int(configuration.sat_rule)
    parameters["USE_TRANSPORT_MODEL"] = bool(configuration.use_transport_model)
    parameters["GEOMETRY_FLAG"] = int(configuration.geometry_flag)
    parameters["USE_BPER"] = bool(configuration.use_bper)
    parameters["USE_BPAR"] = bool(configuration.use_bpar)

    parameters.update(
        {
            "SIGN_BT": float(local.sign_bt),
            "SIGN_IT": float(local.sign_it),
            "RMIN_LOC": float(local.rmin_loc),
            "RMAJ_LOC": float(local.rmaj_loc),
            "DRMAJDX_LOC": float(local.drmajdx_loc),
            "ZMAJ_LOC": float(local.zmaj_loc),
            "DZMAJDX_LOC": float(local.dzmajdx_loc),
            "Q_LOC": float(local.q_loc),
            "Q_PRIME_LOC": float(local.q_prime_loc),
            "P_PRIME_LOC": float(local.p_prime_loc),
            "KAPPA_LOC": float(local.kappa_loc),
            "S_KAPPA_LOC": float(local.s_kappa_loc),
            "DELTA_LOC": float(local.delta_loc),
            "S_DELTA_LOC": float(local.s_delta_loc),
            "ZETA_LOC": float(local.zeta_loc),
            "S_ZETA_LOC": float(local.s_zeta_loc),
            "BETAE": float(local.betae),
            "XNUE": float(local.xnue),
            "ZEFF": float(local.zeff),
            "DEBYE": float(local.debye),
        }
    )
    if local.vexb_shear is not None:
        parameters["VEXB_SHEAR"] = float(local.vexb_shear)

    for index in range(count):
        tag = index + 1
        for key, values in (
            ("ZS", local.zs), ("MASS", local.mass), ("AS", local.as_),
            ("TAUS", local.taus), ("RLNS", local.rlns), ("RLTS", local.rlts),
        ):
            parameters[f"{key}_{tag}"] = float(np.asarray(values)[index])
        for key, default in TGLF_DEFAULT_SPECIES.items():
            parameters.setdefault(f"{key}_{tag}", default)

    parameters.update(
        {str(key).upper(): value for key, value in configuration.extra_parameters.items()}
    )
    return parameters


def write_input_tglf(parameters: Mapping[str, Any], path: str | Path) -> Path:
    """Write an ``input.tglf``, one ``KEY=VALUE`` per line."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={_render(value)}" for key, value in parameters.items()]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _render(value: Any) -> str:
    """TGLF reads Fortran logicals; NEO's ``1``/``0`` would be a parse error."""
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def prepare_tglf_case(
    profile: GACODEProfile,
    rho: float,
    workdir: str | Path,
    config: Optional[TGLFConfig] = None,
) -> TGLFInputs:
    """Stage ``input.tglf`` for one surface in *workdir*.

    Unlike NEO, no ``input.gacode`` is written: TGLF does not read one, and leaving a
    copy in the directory would suggest the run depended on it.
    """
    directory = Path(workdir)
    directory.mkdir(parents=True, exist_ok=True)

    local = prepare_tglf_input(profile, rho, config=config)
    absent = local.check_tglf_requirements()
    if absent:
        raise LocalConversionError(
            f"the local input is missing {', '.join(absent)}; TGLF cannot be run on it "
            "and nothing is substituted."
        )

    parameters = tglf_parameters(local, config)
    written = write_input_tglf(parameters, directory / "input.tglf")
    return TGLFInputs(
        workdir=directory,
        files=(written,),
        local=local,
        input_tglf=written,
        parameters=parameters,
        provenance={
            "rho": float(rho),
            "local": dict(local.provenance),
            "profile": dict(profile.provenance),
            "species": tuple(local.names),
        },
    )
