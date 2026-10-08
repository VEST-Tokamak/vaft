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
  actually solved the inner layer: on its stability path (``match_flag``)
  ``match_solution`` writes ``delta.out`` (``rmatch/match.f:814``).  Without
  it the run is the ideal outer region with the GGJ criteria as diagnostics.
  A ``globalsol.bin`` comes only from the RPEC path (``coil%rpec_flag``,
  ``match.f:1463``), a resistive *perturbed equilibrium* that stops there, and
  is classified so.  A per-surface ``eta`` array in ``rmatch.in`` (VAFT writes
  one from Te/ne for the Spitzer resistivity) is a *profile-informed resistive
  parameter*, recorded in ``extensions`` -- never a kinetic equation.
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


class _Unresolved(Exception):
    """The prepared input does not say what the run was: no record is better than a guess."""

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
    """The flag as the run read it: the namelist's value, else the upstream default.

    A key that is present but unreadable stops the classification rather than
    falling back to the default, which could turn a kinetic run into an ideal one.
    """
    if key not in values:
        return defaults[key]
    value = _namelist_bool(dict(values), key)
    if value is None:
        raise _Unresolved(f"{key}={values[key]!r} is not a namelist logical")
    return value


def _required(path: Path) -> dict[str, str]:
    """The governing namelist of a run; absent means the run cannot be classified."""
    values = _namelist_values(path)
    if not values:
        raise _Unresolved(f"{path.name} not found in {path.parent}")
    return values


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


def _kinetic_requested(dcon: Mapping[str, str]) -> bool:
    """``kin_flag`` with at least one species: with neither ``ion_flag`` nor
    ``electron_flag`` DCON adds a zero delta W_k and the run is ideal."""
    return _flag(dcon, "kin_flag", _DCON_DEFAULTS) and (
        _flag(dcon, "ion_flag", _DCON_DEFAULTS) or _flag(dcon, "electron_flag", _DCON_DEFAULTS)
    )


def _dcon(run_dir: Path, pentrc_dir: Optional[Path] = None) -> dict[str, Any]:
    dcon = _required(run_dir / "dcon.in")
    pentrc = _namelist_values((pentrc_dir or run_dir) / "pentrc.in")
    extensions = {
        "free_boundary": _flag(dcon, "vac_flag", _DCON_DEFAULTS),
        "integrate_through_layers": _flag(dcon, "con_flag", _DCON_DEFAULTS),
        "eigenfunction_reconstruction": "match (numerical stage, no separate record)",
    }
    if _flag(dcon, "kin_flag", _DCON_DEFAULTS) and not _kinetic_requested(dcon):
        extensions["kin_flag_without_species"] = True
    if _kinetic_requested(dcon):
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
    _required(run_dir / "rdcon.in")
    rmatch = _namelist_values(run_dir / "rmatch.in")
    eta = [token for token in rmatch.get("eta", "").replace(",", " ").split() if token]
    if config is not None and getattr(getattr(config, "rdcon", None), "has_kinetic_profiles", False):
        resistivity = "profile_informed_spitzer"
    elif len(eta) > 1:
        resistivity = "per_surface_array"
    elif eta:
        resistivity = "template_scalar"
    else:
        resistivity = "unknown"
    extensions = {
        "outer_region": "ideal_mhd",
        "local_criteria": "GGJ D_I, D_R, H as diagnostics",
        # A resistivity from Te/ne is a profile-informed coefficient of a fluid
        # model, not a kinetic equation (Phase A section 3.4).
        "resistivity": resistivity,
    }
    if (run_dir / "globalsol.bin").is_file():
        # RPEC: resistive perturbed equilibrium under an applied field (match.f:266-272).
        extensions["inner_layer"] = "rmatch RPEC: linear resistive MHD (GGJ) under an applied field"
        return {"scientific_operation": "perturbed_equilibrium", "bulk_description": "fluid",
                "fluid_model": "resistive_mhd", "extensions": extensions}
    solved = (run_dir / "delta.out").is_file()
    extensions["inner_layer"] = (
        "rmatch match_flag: linear resistive MHD (GGJ)" if solved else "not solved (rmatch wrote no delta.out)"
    )
    return {
        "scientific_operation": "resistive_stability",
        "bulk_description": "fluid",
        "fluid_model": "resistive_mhd" if solved else "ideal_mhd",
        "extensions": extensions,
    }


def _stride(run_dir: Path) -> dict[str, Any]:
    _required(run_dir / "stride.in")
    return {
        "scientific_operation": "resistive_stability",
        "bulk_description": "fluid",
        "fluid_model": "ideal_mhd",
        "extensions": {"outer_region": "ideal_mhd", "inner_layer": "not solved", "resistivity": "none"},
    }


def _gpec(run_dir: Path, dcon_dir: Optional[Path]) -> dict[str, Any]:
    gpec = _required(run_dir / "gpec.in")
    thresholds = sorted(key for key in gpec if key.startswith("singthresh") and _namelist_bool(gpec, key))
    extensions: dict[str, Any] = {
        "kinetic_inherited_from": "dcon euler.bin (gpec/idcon.f:62)",
        "auxiliary_threshold_models": thresholds,
    }
    if dcon_dir is None:
        raise _Unresolved("ideal GPEC inherits kin_flag from its DCON cell, which was not given")
    dcon = _required(Path(dcon_dir) / "dcon.in")
    if _kinetic_requested(dcon):
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
    pentrc = _required(run_dir / "pentrc.in")
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
    ``None`` for a module with no scientific formalism of its own, and when the
    prepared input cannot settle it -- the governing namelist is absent, or a
    physics flag in it is unreadable -- rather than claiming the upstream
    defaults for a run nobody can show used them.
    """
    from ..formalism import PlasmaFormalism

    run_dir = Path(run_dir)
    builders = {
        "dcon": (lambda: _dcon(run_dir), "static"),
        "rdcon": (lambda: _rdcon(run_dir, config), "linear"),
        "stride": (lambda: _stride(run_dir), "linear"),
        "gpec": (lambda: _gpec(run_dir, None if dcon_dir is None else Path(dcon_dir)), "static"),
        "pentrc": (lambda: _pentrc(run_dir), "static"),
    }
    if module not in builders:
        return None
    build, regime = builders[module]
    try:
        fields = build()
    except _Unresolved:
        return None
    if fields["scientific_operation"] == "perturbed_equilibrium" and module == "rdcon":
        regime = "static"
    return PlasmaFormalism(regime=regime, solver=module, **_COMMON, **fields)
