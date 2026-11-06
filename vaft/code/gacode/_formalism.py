"""The plasma formalism of each GACODE-suite run.

NEO, TGLF, TGLF-NN and CGYRO share VAFT's GACODE plumbing -- ``input.gacode``, the
profile projection, the launcher -- but not their physics. Each run is given a
:class:`vaft.code.formalism.PlasmaFormalism`, derived from the ``KEY=VALUE``
input the run was actually written with (upstream defaults where a key is absent),
never entered by hand. The classification follows the plasma-model audit in
``docs/_guide/Plasma_models.md`` section 4:

* **NEO** solves the steady first-order drift-kinetic equation for every species:
  ``kinetic`` / ``drift_kinetic``, ``neoclassical_transport``, ``steady_state``.
  ``SIM_MODEL`` 0, 3 and 5 return only the analytic theory (Hinton-Hazeltine,
  Sauter, NCLASS) and 4 a neural network: those are ``reduced`` models
  ``derived_from="drift_kinetic"``, not drift-kinetic solves (``neo/src/neo_check.f90``).
  ``AE_FLAG=1`` (adiabatic electrons) removes the electrons from the kinetic
  population. The geometry is read the way NEO reads it: ``EQUILIBRIUM_MODEL``
  (0 s-alpha, 1 large aspect ratio, 2 Miller) only with ``PROFILE_MODEL=1``; with
  ``PROFILE_MODEL=2`` the shape comes from ``input.gacode`` (Phase A §10's
  ``equilibrium_model=0`` trap).
* **TGLF** solves no kinetic equation: a gyro-Landau-fluid moment system derived from
  gyrokinetics. ``reduced`` / ``derived_from="gyrokinetic"``; ``turbulent_transport``
  and ``quasilinear`` with ``USE_TRANSPORT_MODEL`` (the saturated quasilinear fluxes),
  ``microstability`` and ``linear`` without it (eigenmodes at one ky). Its
  model-reduction choices -- the saturation rule and the linear controls TGLF's
  start-up presets impose with it (``tglf/src/tglf_startup.f90``: SAT2/3 force
  ``XNU_MODEL=3``, ``WDIA_TRAPPED=1``, ``UNITS=CGYRO``; SAT1 ``XNU_MODEL=2``; SAT0
  ``UNITS=GYRO``, ``XNU_MODEL=2``) -- stay in ``extensions``, as the effective values.
* **TGLF-NN** is a learned surrogate *of* a TGLF variant. Its physical record is that
  variant's TGLF record; the realization (``learned_surrogate``) and the model it was
  trained to reproduce are ``extensions``. No kinetic equation is "neural network".
* **CGYRO** is the first-principles reference; its record lives in
  :func:`vaft.code.gacode.cgyro.plasma_formalism`, which now reads ``AE_FLAG``,
  ``EQUILIBRIUM_MODEL``, ``GLOBAL_FLAG``, the collision model and the rotation inputs
  from the written input through :func:`cgyro_input_physics` here.

A shared input format is not a shared formalism: the same local state run through
NEO, TGLF and CGYRO gives three different records, and nothing here collapses them
into "GACODE".
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..formalism import UNKNOWN, PlasmaFormalism

__all__ = [
    "cgyro_input_physics",
    "neo_plasma_formalism",
    "tglf_effective_controls",
    "tglf_plasma_formalism",
    "tglf_surrogate_plasma_formalism",
]

#: NEO's own defaults for the keys read here (``neo/src/neo_interface.f90``).
_NEO_DEFAULTS = {
    "SIM_MODEL": 2, "AE_FLAG": 0, "PROFILE_MODEL": 1, "EQUILIBRIUM_MODEL": 0,
    "COLLISION_MODEL": 4, "ROTATION_MODEL": 1, "SPITZER_MODEL": 0, "THREED_MODEL": 0,
}
_NEO_SIM_MODELS = {0: "theory", 1: "numerical_with_theory_and_nclass", 2: "numerical_with_theory",
                   3: "theory_with_nclass", 4: "neural_network", 5: "theory_modified_sauter"}
#: SIM_MODEL values that solve the drift-kinetic equation numerically.
_NEO_NUMERICAL = (1, 2)
_NEO_COLLISIONS = {1: "connor", 2: "reduced_hirshman_sigmar", 3: "full_hirshman_sigmar",
                   4: "full_linearized_fokker_planck",
                   5: "full_linearized_fokker_planck_adhoc_field_particle"}
_NEO_GEOMETRY = {0: "s_alpha", 1: "large_aspect_ratio", 2: "miller"}

#: TGLF's own defaults for the keys read here (``tglf/src/tglf_modules.f90``).
_TGLF_DEFAULTS = {
    "SAT_RULE": 0, "USE_TRANSPORT_MODEL": True, "GEOMETRY_FLAG": 1, "USE_BPER": False,
    "USE_BPAR": False, "UNITS": "GYRO", "NMODES": 2, "ADIABATIC_ELEC": False, "XNU_MODEL": 2,
}

#: CGYRO's own defaults (``cgyro/bin/cgyro_parse.py``).
_CGYRO_DEFAULTS = {"AE_FLAG": 0, "EQUILIBRIUM_MODEL": 2, "GLOBAL_FLAG": 0, "COLLISION_MODEL": 4,
                   "ROTATION_MODEL": 1, "GAMMA_E": 0.0, "GAMMA_P": 0.0, "MACH": 0.0,
                   "N_FIELD": 1, "NONLINEAR_FLAG": 0, "PROFILE_MODEL": 1}
#: N_FIELD -> VAFT's CGYRO field-model name (cgyro_check.f90: 1 phi, 2 + A_par, 3 + B_par).
_CGYRO_FIELDS = {1: "es", 2: "em-aperp", 3: "em-aperp-bpar"}
#: ``cgyro/src/cgyro_check.f90``. Model 2 is upstream's Miller extended harmonic (MXH);
#: VAFT calls it ``miller``, and the gyrokinetic formalism record keeps that name.
_CGYRO_GEOMETRY = {1: "s_alpha", 2: "miller"}
_CGYRO_COLLISIONS = {1: "lorentz", 2: "connor", 4: "sugama", 5: "simple_lorentz", 6: "landau",
                     7: "galerkin_sugama"}


def _upper(parameters: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return {str(key).upper(): value for key, value in dict(parameters or {}).items()}


def _value(parameters: Mapping[str, Any], defaults: Mapping[str, Any], key: str) -> Any:
    return parameters[key] if key in parameters else defaults[key]


def _integer(parameters: Mapping[str, Any], defaults: Mapping[str, Any], key: str) -> int:
    value = _value(parameters, defaults, key)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be an integer, got {value!r}") from None


def _flag(parameters: Mapping[str, Any], defaults: Mapping[str, Any], key: str) -> bool:
    """A Fortran logical as written: True/False, .true./.false., T/F, 1/0."""
    value = _value(parameters, defaults, key)
    if isinstance(value, bool):
        return value
    text = str(value).strip().strip(".").lower()
    if text in ("true", "t", "1"):
        return True
    if text in ("false", "f", "0"):
        return False
    raise ValueError(f"{key} must be a logical, got {value!r}")


def neo_plasma_formalism(parameters: Optional[Mapping[str, Any]] = None) -> PlasmaFormalism:
    """The record of one NEO run, from the ``input.neo`` it was written with.

    ``parameters`` are the written ``KEY=VALUE`` pairs
    (:func:`vaft.code.gacode.neo.inputs.neo_parameters`); absent keys take NEO's own
    defaults, so ``None`` is NEO run with nothing set.
    """
    p = _upper(parameters)
    sim = _integer(p, _NEO_DEFAULTS, "SIM_MODEL")
    if sim not in _NEO_SIM_MODELS:
        raise ValueError(f"SIM_MODEL={sim} is not one NEO accepts (0-5)")
    adiabatic = _integer(p, _NEO_DEFAULTS, "AE_FLAG") == 1
    profile_model = _integer(p, _NEO_DEFAULTS, "PROFILE_MODEL")
    equilibrium_model = _integer(p, _NEO_DEFAULTS, "EQUILIBRIUM_MODEL")
    collision = _integer(p, _NEO_DEFAULTS, "COLLISION_MODEL")
    geometry = (_NEO_GEOMETRY.get(equilibrium_model, UNKNOWN) if profile_model == 1
                else "input_gacode_shape" if profile_model == 2 else UNKNOWN)
    extensions = {
        "sim_model": sim,
        "sim_model_name": _NEO_SIM_MODELS[sim],
        "collision_model": collision,
        "collision_model_name": _NEO_COLLISIONS.get(collision, UNKNOWN),
        "profile_model": profile_model,
        "equilibrium_model": equilibrium_model,
        "geometry_model": geometry,
        "rotation_model": _integer(p, _NEO_DEFAULTS, "ROTATION_MODEL"),
        "spitzer_model": _integer(p, _NEO_DEFAULTS, "SPITZER_MODEL"),
        "threed_model": _integer(p, _NEO_DEFAULTS, "THREED_MODEL"),
        "species_model": "adiabatic_electrons" if adiabatic else "kinetic_electrons",
    }
    population = ("ions",) if adiabatic else ("all",)
    common = dict(scientific_operation="neoclassical_transport", spatial_domain="local",
                  topology_domain="closed_flux_surface", regime="steady_state", solver="neo",
                  extensions=extensions)
    if sim not in _NEO_NUMERICAL:
        # Analytic theory or the NEO neural network: a reduced model of drift-kinetic
        # neoclassical theory, with no distribution solved for.
        return PlasmaFormalism(bulk_description="reduced", derived_from="drift_kinetic", **common)
    return PlasmaFormalism(bulk_description="kinetic", kinetic_equation="drift_kinetic",
                           kinetic_population=population, distribution_formulation="delta_f",
                           orbit_representation="guiding_center", **common)


def tglf_effective_controls(parameters: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """SAT rule, units, collision model and trapped diamagnetic switch TGLF actually runs.

    ``tglf/src/tglf_startup.f90`` has ``USE_PRESETS = .TRUE.`` hard-coded: it zeroes
    WDIA_TRAPPED, then SAT2/3 set ``XNU_MODEL=3``, ``WDIA_TRAPPED=1`` and GYRO units to
    CGYRO; SAT1 sets ``XNU_MODEL=2``; SAT0 sets GYRO units and ``XNU_MODEL=2`` and rounds
    ``NMODES > 2`` up to 4. Whatever the input said for those keys is overwritten.
    """
    p = _upper(parameters)
    sat = _integer(p, _TGLF_DEFAULTS, "SAT_RULE")
    if not 0 <= sat <= 3:
        raise ValueError(f"SAT_RULE={sat} is not 0, 1, 2 or 3")
    # Compared exactly as tglf_startup.f90 does (units_in.eq."GYRO"), not normalised.
    units = _value(p, _TGLF_DEFAULTS, "UNITS")
    nmodes = _value(p, _TGLF_DEFAULTS, "NMODES")
    if sat in (2, 3):
        xnu, wdia = 3, 1.0
        units = "CGYRO" if units == "GYRO" else units
    elif sat == 1:
        xnu, wdia = 2, 0.0
    else:
        xnu, wdia, units = 2, 0.0, "GYRO"
        nmodes = _integer(p, _TGLF_DEFAULTS, "NMODES")
        nmodes = 4 if nmodes > 2 else nmodes
    return {"sat_rule": sat, "units": units, "xnu_model": xnu, "wdia_trapped": wdia, "nmodes": nmodes}


def _tglf_physics(p: Mapping[str, Any]) -> dict[str, Any]:
    transport = _flag(p, _TGLF_DEFAULTS, "USE_TRANSPORT_MODEL")
    bper, bpar = _flag(p, _TGLF_DEFAULTS, "USE_BPER"), _flag(p, _TGLF_DEFAULTS, "USE_BPAR")
    geometry_flag = _integer(p, _TGLF_DEFAULTS, "GEOMETRY_FLAG")
    extensions = {
        **tglf_effective_controls(p),
        "use_transport_model": transport,
        "use_bper": bper,
        "use_bpar": bpar,
        "geometry_model": {0: "s_alpha", 1: "miller"}.get(geometry_flag, UNKNOWN),
        "species_model": ("adiabatic_electrons" if _flag(p, _TGLF_DEFAULTS, "ADIABATIC_ELEC")
                          else "kinetic_electrons"),
        "closure": "gyro_landau_fluid",
    }
    return dict(scientific_operation="turbulent_transport" if transport else "microstability",
                bulk_description="reduced", derived_from="gyrokinetic", spatial_domain="local",
                topology_domain="closed_flux_surface", regime="quasilinear" if transport else "linear",
                field_model="electromagnetic" if (bper or bpar) else "electrostatic",
                extensions=extensions)


def tglf_plasma_formalism(parameters: Optional[Mapping[str, Any]] = None) -> PlasmaFormalism:
    """The record of one TGLF run, from the ``input.tglf`` it was written with."""
    return PlasmaFormalism(solver="tglf", **_tglf_physics(_upper(parameters)))


def tglf_surrogate_plasma_formalism(
    *, sat_rule: Optional[int], electromagnetic: Optional[bool], model: str,
    parameters: Optional[Mapping[str, Any]] = None,
) -> PlasmaFormalism:
    """The record of a TGLF-NN prediction: the TGLF variant it reproduces, realized by a network.

    The physical fields are those of TGLF in the configuration the model was trained on:
    its SAT rule and field model come from the model's identity, never from the caller's
    request (a mismatch is the caller's business, recorded beside it). Where the model
    does not say, the field is :data:`~vaft.code.formalism.UNKNOWN`. ``parameters`` --
    the request -- supply only what the model identity cannot (geometry, species model).
    """
    # Only what the model identity cannot say comes from the request: geometry and the
    # electron model. A malformed request key is not this record's business.
    requested = {key: value for key, value in _upper(parameters).items()
                 if key in ("GEOMETRY_FLAG", "ADIABATIC_ELEC")}
    base = _tglf_physics({**requested, "USE_TRANSPORT_MODEL": True})
    extensions = dict(base["extensions"])
    for key in ("sat_rule", "units", "xnu_model", "wdia_trapped", "nmodes", "use_bper", "use_bpar"):
        extensions.pop(key, None)       # these belong to the trained variant, not the request
    # The SAT rule implies XNU_MODEL and WDIA_TRAPPED unconditionally; the variant's
    # UNITS and NMODES depend on its own training inputs, which the identity does not state.
    extensions.update({"sat_rule": UNKNOWN if sat_rule is None else int(sat_rule),
                       "units": UNKNOWN, "nmodes": UNKNOWN})
    if sat_rule is not None:
        implied = tglf_effective_controls({"SAT_RULE": int(sat_rule)})
        extensions.update(xnu_model=implied["xnu_model"], wdia_trapped=implied["wdia_trapped"])
    extensions.update({"realization": "learned_surrogate", "surrogate_of": "tglf", "model": str(model)})
    base["extensions"] = extensions
    base["field_model"] = (UNKNOWN if electromagnetic is None
                           else "electromagnetic" if electromagnetic else "electrostatic")
    return PlasmaFormalism(solver="tglf-nn", **base)


def cgyro_input_physics(parameters: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """What CGYRO's written input fixes about its physics, beyond the configuration object.

    Returned as ``species_model``, ``geometry_model``, ``spatial_domain`` (``local``,
    or :data:`~vaft.code.formalism.UNKNOWN` with ``GLOBAL_FLAG=1``, upstream's
    "not ready for production" global-spectral mode), the collision model and the
    rotation inputs, so :func:`vaft.code.gacode.cgyro.plasma_formalism` reads them from
    the input instead of assuming them.
    """
    p = _upper(parameters)
    adiabatic = _integer(p, _CGYRO_DEFAULTS, "AE_FLAG") == 1
    equilibrium = _integer(p, _CGYRO_DEFAULTS, "EQUILIBRIUM_MODEL")
    global_flag = _integer(p, _CGYRO_DEFAULTS, "GLOBAL_FLAG")
    collision = _integer(p, _CGYRO_DEFAULTS, "COLLISION_MODEL")
    shear = {key.lower(): float(_value(p, _CGYRO_DEFAULTS, key)) for key in ("GAMMA_E", "GAMMA_P", "MACH")}
    n_field = _integer(p, _CGYRO_DEFAULTS, "N_FIELD")
    profile_model = _integer(p, _CGYRO_DEFAULTS, "PROFILE_MODEL")
    if profile_model != 1:
        # PROFILE_MODEL=2 takes the shear and Mach number from input.gacode
        # (cgyro_make_profiles.f90), not from the zeros VAFT writes in input.cgyro.
        rotation: Any = UNKNOWN
    else:
        rotation = "none" if all(value == 0.0 for value in shear.values()) else shear
    return {
        "field_model": _CGYRO_FIELDS.get(n_field, UNKNOWN),
        "nonlinear": _integer(p, _CGYRO_DEFAULTS, "NONLINEAR_FLAG") == 1,
        "profile_model": profile_model,
        "species_model": "adiabatic_electrons" if adiabatic else "kinetic_electrons",
        "kinetic_population": ("ions",) if adiabatic else ("all",),
        "geometry_model": _CGYRO_GEOMETRY.get(equilibrium, UNKNOWN),
        "spatial_domain": "local" if global_flag == 0 else UNKNOWN,
        "global_flag": global_flag,
        "collision_model": collision,
        "collision_model_name": _CGYRO_COLLISIONS.get(collision, UNKNOWN),
        "rotation_model": _integer(p, _CGYRO_DEFAULTS, "ROTATION_MODEL"),
        "rotation": rotation,
    }
