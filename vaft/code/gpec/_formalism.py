"""The plasma formalism of each GPEC-suite run (#1734, Phase C1 of #1723).

Every scientifically distinct GPEC-suite run is given a
:class:`vaft.code.formalism.PlasmaFormalism` (#1727), derived from what the run
was actually prepared with -- the namelists in its own directory -- never entered
by hand.  The classification follows the Phase A audit
(``docs/_guide/Plasma_models.md`` section 3, #1725):

* **DCON**, ``kin_flag=f``: ideal-MHD energy principle -- ``fluid`` /
  ``ideal_mhd``, ``ideal_stability``, ``static``.
* **DCON**, ``kin_flag=t``: the same Euler-Lagrange problem with PENTRC's
  drift-kinetic delta W_k added to its coefficients -- ``hybrid`` /
  ``drift_kinetic`` / coupling ``energy``, bounce-averaged, delta-f.  The
  populations follow ``ion_flag`` / ``electron_flag``; the pitch-angle domain
  (``passing_flag`` / ``trapped_flag``) and PENTRC's ``nutype`` / ``f0type``
  stay in ``extensions``.  No named limit (Kruskal-Oberman, ...) is recorded:
  ``kin_flag`` alone does not identify one.
* **RDCON**: ``resistive_stability``.  ``resistive_mhd`` only when RMATCH
  actually solved the inner layer (it wrote ``globalsol.bin``); without it the
  run is the ideal outer region with the GGJ criteria as diagnostics.  Te/ne
  supplied for the Spitzer resistivity are *profile-informed resistive
  parameters*, recorded in ``extensions`` -- never a kinetic equation.
* **STRIDE**: the ideal outer region and its Delta-prime -- ``ideal_mhd``,
  ``resistive_stability``, no resistivity.
* **GPEC**: ``perturbed_equilibrium``.  Its kinetic content is inherited from
  the DCON run it reads (``gpec/idcon.f:62`` reads ``kin_flag`` back from
  ``euler.bin``), so the DCON cell's ``dcon.in`` decides ideal vs hybrid.  The
  ``singthresh_*`` layer models are auxiliary and stay in ``extensions``.
* **PENTRC**: a drift-kinetic *closure* on a given GPEC displacement --
  ``hybrid`` / coupling ``closure``, ``toroidal_torque``, one species per run.

MATCH and RMATCH are not given records of their own: they are numerical stages
of the DCON and RDCON calculations they complete (MATCH reconstructs DCON's
ideal eigenfunction; RMATCH solves RDCON's inner layer), and are recorded in
those runs' ``extensions``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ._dcon_output import _namelist_bool, _namelist_values

__all__ = ["resolve_plasma_formalism"]

#: DCON's own defaults (``dcon/dcon_mod.f:108-114``), used when ``dcon.in`` omits a key.
_DCON_DEFAULTS = {
    "kin_flag": False,
    "con_flag": False,
    "passing_flag": False,
    "trapped_flag": True,
    "ion_flag": True,
    "electron_flag": False,
    "vac_flag": False,
}

#: PENTRC's own defaults (``pentrc/pentrc_interface.f90:63-147``).
_PENTRC_METHOD_DEFAULTS = {
    "fgar_flag": True,
    "tgar_flag": False,
    "pgar_flag": False,
    "rlar_flag": False,
    "clar_flag": False,
    "fcgl_flag": False,
}
_PENTRC_DEFAULTS = {"electron": False, "nutype": "harmonic", "f0type": "maxwellian"}

_COMMON = {
    "field_model": "mhd_displacement",
    "spatial_domain": "whole_volume",
    "topology_domain": "closed_flux_surface",
}


def _flag(values: Mapping[str, str], key: str, defaults: Mapping[str, bool]) -> bool:
    value = _namelist_bool(dict(values), key)
    return defaults[key] if value is None else value


def _text(values: Mapping[str, str], key: str, default: str) -> str:
    raw = values.get(key, "").strip().strip("'\"")
    return raw.lower() if raw else default


def _populations(ion: bool, electron: bool) -> tuple[str, ...]:
    populations = (("thermal_ions",) if ion else ()) + (("electrons",) if electron else ())
    return populations or ("none",)


def _dcon_kinetic(dcon: Mapping[str, str], pentrc: Mapping[str, str], source: str) -> dict[str, Any]:
    """Fields shared by kinetic DCON and the GPEC run that inherits it."""
    ion = _flag(dcon, "ion_flag", _DCON_DEFAULTS)
    electron = _flag(dcon, "electron_flag", _DCON_DEFAULTS)
    return {
        "bulk_description": "hybrid",
        "fluid_model": "ideal_mhd",
        "kinetic_equation": "drift_kinetic",
        "kinetic_coupling": "energy",
        "kinetic_population": _populations(ion, electron),
        "distribution_formulation": "delta_f",
        "orbit_representation": "bounce_averaged",
        "kinetic_extensions": {
            "kinetic_source": source,
            "passing_particles": _flag(dcon, "passing_flag", _DCON_DEFAULTS),
            "trapped_particles": _flag(dcon, "trapped_flag", _DCON_DEFAULTS),
            "ion_contribution": ion,
            "electron_contribution": electron,
            "collision_operator": _text(pentrc, "nutype", _PENTRC_DEFAULTS["nutype"]),
            "equilibrium_distribution": _text(pentrc, "f0type", _PENTRC_DEFAULTS["f0type"]),
            "finite_larmor_radius": False,
        },
    }


def _dcon(run_dir: Path, pentrc_dir: Optional[Path] = None) -> dict[str, Any]:
    dcon = _namelist_values(run_dir / "dcon.in")
    pentrc = _namelist_values((pentrc_dir or run_dir) / "pentrc.in")
    extensions = {
        "free_boundary": _flag(dcon, "vac_flag", _DCON_DEFAULTS),
        "integrate_through_layers": _flag(dcon, "con_flag", _DCON_DEFAULTS),
        "namelist": "dcon.in" if dcon else "missing (upstream defaults)",
        "eigenfunction_reconstruction": "match (numerical stage, no separate record)",
    }
    if _flag(dcon, "kin_flag", _DCON_DEFAULTS):
        fields = _dcon_kinetic(dcon, pentrc, "dcon kin_flag=t (PENTRC operator inside the Euler-Lagrange equation)")
        extensions.update(fields.pop("kinetic_extensions"))
        return {"scientific_operation": "ideal_stability", **fields, "extensions": extensions}
    return {
        "scientific_operation": "ideal_stability",
        "bulk_description": "fluid",
        "fluid_model": "ideal_mhd",
        "extensions": extensions,
    }


def _rdcon(run_dir: Path, config: Any) -> dict[str, Any]:
    inner = (run_dir / "globalsol.bin").is_file()
    profiles = getattr(getattr(config, "rdcon", None), "t_e", None) is not None
    return {
        "scientific_operation": "resistive_stability",
        "bulk_description": "fluid",
        "fluid_model": "resistive_mhd" if inner else "ideal_mhd",
        "extensions": {
            "outer_region": "ideal_mhd",
            "inner_layer": "rmatch: linear resistive MHD (GGJ)" if inner else "not solved (rmatch wrote no globalsol.bin)",
            "local_criteria": "GGJ D_I, D_R, H as diagnostics",
            # Te/ne -> Spitzer eta is a profile-informed coefficient of a fluid model,
            # not a kinetic equation (Phase A section 3.4).
            "resistivity": "profile_informed_spitzer" if profiles else "template_scalar",
        },
    }


def _stride(run_dir: Path) -> dict[str, Any]:
    return {
        "scientific_operation": "resistive_stability",
        "bulk_description": "fluid",
        "fluid_model": "ideal_mhd",
        "extensions": {"outer_region": "ideal_mhd", "inner_layer": "not solved", "resistivity": "none"},
    }


def _gpec(run_dir: Path, dcon_dir: Optional[Path]) -> dict[str, Any]:
    gpec = _namelist_values(run_dir / "gpec.in")
    thresholds = sorted(key for key in gpec if key.startswith("singthresh") and _namelist_bool(gpec, key))
    extensions: dict[str, Any] = {
        "kinetic_inherited_from": "dcon euler.bin (gpec/idcon.f:62)",
        "auxiliary_threshold_models": thresholds,
    }
    dcon = _namelist_values(dcon_dir / "dcon.in") if dcon_dir is not None else {}
    if dcon_dir is None or not dcon:
        extensions["dcon_namelist"] = "not found; ideal assumed from DCON's default kin_flag=f"
    if _flag(dcon, "kin_flag", _DCON_DEFAULTS):
        pentrc = _namelist_values(dcon_dir / "pentrc.in") if dcon_dir is not None else {}
        fields = _dcon_kinetic(dcon, pentrc, "inherited from the kinetic DCON run")
        extensions.update(fields.pop("kinetic_extensions"))
        return {"scientific_operation": "perturbed_equilibrium", **fields, "extensions": extensions}
    return {
        "scientific_operation": "perturbed_equilibrium",
        "bulk_description": "fluid",
        "fluid_model": "ideal_mhd",
        "extensions": extensions,
    }


def _pentrc(run_dir: Path) -> dict[str, Any]:
    pentrc = _namelist_values(run_dir / "pentrc.in")
    methods = sorted(
        key[: -len("_flag")] for key in _PENTRC_METHOD_DEFAULTS
        if _flag(pentrc, key, _PENTRC_METHOD_DEFAULTS)
    )
    electron = _flag(pentrc, "electron", _PENTRC_DEFAULTS)
    return {
        "scientific_operation": "toroidal_torque",
        "bulk_description": "hybrid",
        # The displacement is a given ideal(-kinetic) GPEC xi; the closure is not fed back.
        "fluid_model": "ideal_mhd",
        "kinetic_equation": "drift_kinetic",
        "kinetic_coupling": "closure",
        "kinetic_population": ("electrons",) if electron else ("thermal_ions",),
        "distribution_formulation": "delta_f",
        "orbit_representation": "bounce_averaged",
        "extensions": {
            "methods": methods,
            "collision_operator": _text(pentrc, "nutype", _PENTRC_DEFAULTS["nutype"]),
            "equilibrium_distribution": _text(pentrc, "f0type", _PENTRC_DEFAULTS["f0type"]),
            "displacement_source": "gpec xi (not self-consistent)",
            "finite_larmor_radius": False,
        },
    }


def resolve_plasma_formalism(module: str, run_dir: Path, *, config: Any = None, dcon_dir: Optional[Path] = None):
    """The :class:`vaft.code.formalism.PlasmaFormalism` of one prepared GPEC-suite cell.

    ``run_dir`` is the cell's own directory; ``config`` the
    :class:`GPECSuiteConfig` it ran with (only RDCON's resistivity source is read
    from it); ``dcon_dir`` the DCON cell an ideal-GPEC run read, whose
    ``dcon.in`` decides whether GPEC inherited a kinetic response.  Returns
    ``None`` for a module with no scientific formalism of its own.
    """
    from ..formalism import PlasmaFormalism

    run_dir = Path(run_dir)
    if module == "dcon":
        fields = _dcon(run_dir)
        regime = "static"
    elif module == "rdcon":
        fields = _rdcon(run_dir, config)
        regime = "linear"
    elif module == "stride":
        fields = _stride(run_dir)
        regime = "linear"
    elif module == "gpec":
        fields = _gpec(run_dir, None if dcon_dir is None else Path(dcon_dir))
        regime = "static"
    elif module == "pentrc":
        fields = _pentrc(run_dir)
        regime = "static"
    else:
        return None
    return PlasmaFormalism(regime=regime, solver=module, **_COMMON, **fields)
