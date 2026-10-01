"""Project a GACODE profile onto CGYRO's local input, and write ``input.cgyro``.

**CGYRO is built from TGLF's local input, not beside it.** The purpose of running CGYRO
here is to judge TGLF (#1482: the saturation rule alone moves the VEST flux ~5x), and a
comparison is only a comparison when both codes see the same surface. So the projection
is done once, by :func:`~vaft.code.gacode.tglf.inputs.prepare_tglf_input` -- which is
already held against GACODE's ``locpargen`` key by key -- and this module only renames
and reorders it into CGYRO's ``PROFILE_MODEL=1`` keys. Nothing physical is recomputed.

CGYRO could instead read ``input.gacode`` itself (``PROFILE_MODEL=2``) and project the
surface internally with ``expro_locsim``. That path is the *oracle*: run in test mode
(``cgyro -t``) it writes the local parameters it resolved to ``out.cgyro.equilibrium``,
and :func:`compare_with_oracle` holds this module's translation against it, the way
``test_tglf_input.py`` holds the TGLF projection against ``locpargen``.

The translation, and why each key is what it is (``cgyro_make_profiles.F90``):

=====================  ===================================  ===========================
CGYRO                  from TGLF                             note
=====================  ===================================  ===========================
``RMIN``               ``RMIN_LOC``                          r/a
``RMAJ``               ``RMAJ_LOC``                          R0/a
``SHIFT``              ``DRMAJDX_LOC``                       dR0/dr
``ZMAG``, ``DZMAG``    ``ZMAJ_LOC``, ``DZMAJDX_LOC``
``Q``                  ``Q_LOC``                             positive; sign from IPCCW*BTCCW
``S``                  ``Q_PRIME_LOC * (RMIN/Q)**2``         ``Q_PRIME_LOC = (q/r)^2 s``
``KAPPA`` ... ``S_ZETA``  the same shaping keys
``BETAE_UNIT``         ``BETAE``                             both use B_unit
``NU_EE``              ``XNUE``                              same formula, both D-normalised
``LAMBDA_STAR``        ``DEBYE``                             lambda_D / rho_s
``IPCCW``, ``BTCCW``   ``SIGN_IT``, ``SIGN_BT``              GACODE's ipccw/btccw
``Z/MASS/DENS/TEMP``   ``ZS/MASS/AS/TAUS``                   masses already deuterium units
``DLNNDR/DLNTDR``      ``RLNS/RLTS``                         a/L, electrons moved last
=====================  ===================================  ===========================

``BETA_STAR`` is not an input: CGYRO rebuilds it from ``BETAE_UNIT`` and the species
gradients (``set_betastar``) with the formula TGLF's ``P_PRIME_LOC`` came from. ``Z_EFF``
is not written either: with CGYRO's default ``Z_EFF_METHOD=2`` it is recomputed from the
species list, and writing a value CGYRO ignores would make the file contradict itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ...base import CodeInputs
from .._profiles import GACODEProfile
from ..tglf.inputs import LocalConversionError, TGLFInput, prepare_tglf_input
from ._types import MAX_SPECIES, CGYROConfig

__all__ = [
    "CGYROInput",
    "CGYROInputs",
    "LocalConversionError",
    "cgyro_input_from_tglf",
    "cgyro_parameters",
    "compare_with_oracle",
    "input_sha256",
    "prepare_cgyro_case",
    "prepare_cgyro_input",
    "write_input_cgyro",
]

#: The geometry keys, in the order CGYRO's `cgyro_parse.py` declares them.
GEOMETRY_KEYS = (
    "RMIN", "RMAJ", "Q", "S", "SHIFT", "KAPPA", "S_KAPPA", "DELTA", "S_DELTA",
    "ZETA", "S_ZETA", "ZMAG", "DZMAG",
)
SPECIES_KEYS = ("Z", "MASS", "DENS", "TEMP", "DLNNDR", "DLNTDR")


@dataclass(frozen=True)
class CGYROInput:
    """CGYRO's local input at one flux surface, in CGYRO's ``PROFILE_MODEL=1`` terms.

    Species are in CGYRO's order, **electrons last**. :attr:`tglf` is the TGLF local
    input this was translated from, kept so that a comparison can point at the exact
    numbers both codes were given, and so the gyro-Bohm normalisation
    (:attr:`TGLFInput.normalisation`) is shared rather than derived twice.
    """

    r_over_a: float
    geometry: Mapping[str, float]
    species: Mapping[str, np.ndarray]
    names: Sequence[str]
    betae_unit: float
    nu_ee: float
    lambda_star: float
    ipccw: float
    btccw: float
    tglf: TGLFInput
    provenance: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def n_species(self) -> int:
        return int(np.size(self.species["Z"]))

    @property
    def normalisation(self):
        return self.tglf.normalisation

    @property
    def z_eff(self) -> float:
        """``sum(n_i Z_i^2)/n_e`` over the species list -- what CGYRO computes."""
        z = np.asarray(self.species["Z"], dtype=float)
        dens = np.asarray(self.species["DENS"], dtype=float)
        electron = dens[z < 0][0]
        return float(np.sum(dens[z > 0] * z[z > 0] ** 2) / electron)

    def missing(self) -> tuple[str, ...]:
        """Names recorded as unavailable, carried over from the TGLF projection."""
        return tuple(
            name for name, record in self.provenance.items()
            if record.get("kind") == "unavailable"
        )


@dataclass
class CGYROInputs(CodeInputs):
    """A staged CGYRO case: the directory, ``input.cgyro``, and what made it."""

    local: Optional[CGYROInput] = None
    input_cgyro: Optional[Path] = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)


def cgyro_input_from_tglf(local: TGLFInput) -> CGYROInput:
    """Translate a TGLF local input into CGYRO's keys. Renaming only, no new physics.

    Raises
    ------
    LocalConversionError
        More species than CGYRO accepts, or no single electron species.
    """
    zs = np.asarray(local.zs, dtype=float)
    count = zs.size
    if count > MAX_SPECIES:
        raise LocalConversionError(
            f"CGYRO takes at most {MAX_SPECIES} species; this input has {count}"
        )
    electrons = np.flatnonzero(zs < 0)
    if electrons.size != 1:
        # cgyro_make_profiles refuses both cases; refusing here names the cause.
        raise LocalConversionError(
            f"CGYRO needs exactly one kinetic electron species; got {electrons.size}"
        )
    order = [i for i in range(count) if i != electrons[0]] + [int(electrons[0])]

    q = abs(float(local.q_loc))
    if q == 0.0:
        raise LocalConversionError("q is zero at this surface; the shear is undefined")
    rmin = float(local.rmin_loc)
    shear = float(local.q_prime_loc) * (rmin / q) ** 2

    geometry = {
        "RMIN": rmin,
        "RMAJ": float(local.rmaj_loc),
        "Q": q,
        "S": shear,
        "SHIFT": float(local.drmajdx_loc),
        "KAPPA": float(local.kappa_loc),
        "S_KAPPA": float(local.s_kappa_loc),
        "DELTA": float(local.delta_loc),
        "S_DELTA": float(local.s_delta_loc),
        "ZETA": float(local.zeta_loc),
        "S_ZETA": float(local.s_zeta_loc),
        "ZMAG": float(local.zmaj_loc),
        "DZMAG": float(local.dzmajdx_loc),
    }
    species = {
        "Z": zs[order],
        "MASS": np.asarray(local.mass, dtype=float)[order],
        "DENS": np.asarray(local.as_, dtype=float)[order],
        "TEMP": np.asarray(local.taus, dtype=float)[order],
        "DLNNDR": np.asarray(local.rlns, dtype=float)[order],
        "DLNTDR": np.asarray(local.rlts, dtype=float)[order],
    }
    names = tuple(np.asarray(tuple(local.names) or [""] * count, dtype=object)[order])

    provenance: dict[str, Mapping[str, Any]] = {
        name: dict(record) for name, record in local.provenance.items()
    }
    provenance["source"] = {
        "kind": "derived",
        "reason": (
            "renamed from the TGLF local input (prepare_tglf_input, locpargen-verified); "
            "species reordered electrons-last"
        ),
    }
    provenance["s"] = {"kind": "derived", "reason": "Q_PRIME_LOC * (RMIN/Q)**2"}
    provenance["z_eff"] = {
        "kind": "derived",
        "reason": (
            "not written: CGYRO's Z_EFF_METHOD=2 recomputes it from the species list"
        ),
        "species_value": None,
        "tglf_value": float(local.zeff),
    }
    if local.vexb_shear is None:
        provenance["gamma_e"] = {
            "kind": "unavailable",
            "reason": "no ExB shear on the state; CGYRO's GAMMA_E=0 applies",
        }

    result = CGYROInput(
        r_over_a=float(local.rho),
        geometry=geometry,
        species=species,
        names=names,
        betae_unit=float(local.betae),
        nu_ee=float(local.xnue),
        lambda_star=float(local.debye),
        ipccw=float(local.sign_it),
        btccw=float(local.sign_bt),
        tglf=local,
        provenance=provenance,
    )
    provenance["z_eff"]["species_value"] = result.z_eff
    return result


def prepare_cgyro_input(
    profile: GACODEProfile,
    r_over_a: float,
    *,
    config: Optional[CGYROConfig] = None,
) -> CGYROInput:
    """Project *profile* onto CGYRO's local input at ``r/a``.

    ``config`` is accepted for symmetry with the other backends and is not read: the
    physics comes from the profile, the numerics are written by :func:`cgyro_parameters`.
    """
    del config
    return cgyro_input_from_tglf(prepare_tglf_input(profile, r_over_a))


def cgyro_parameters(
    local: CGYROInput, config: Optional[CGYROConfig] = None
) -> dict[str, Any]:
    """The ``KEY=VALUE`` settings this local input and configuration mean.

    Numerics first, then geometry, then species -- the order a reader checks them in.
    """
    configuration = config or CGYROConfig()
    parameters: dict[str, Any] = {
        "PROFILE_MODEL": 1,
        "EQUILIBRIUM_MODEL": 2,
        "NONLINEAR_FLAG": int(bool(configuration.nonlinear)),
        "N_FIELD": int(configuration.n_field),
        "N_ENERGY": int(configuration.n_energy),
        "N_XI": int(configuration.n_xi),
        "N_THETA": int(configuration.n_theta),
        "N_RADIAL": int(configuration.n_radial),
        "N_TOROIDAL": int(configuration.n_toroidal),
        "KY": float(configuration.ky),
        "BOX_SIZE": int(configuration.box_size),
        "DELTA_T": float(configuration.delta_t),
        "DELTA_T_METHOD": int(configuration.delta_t_method),
        "MAX_TIME": float(configuration.max_time),
        "FREQ_TOL": float(configuration.freq_tol),
        "PRINT_STEP": int(configuration.print_step),
        "COLLISION_MODEL": int(configuration.collision_model),
        "IPCCW": float(local.ipccw),
        "BTCCW": float(local.btccw),
    }
    parameters.update({key: float(local.geometry[key]) for key in GEOMETRY_KEYS})
    parameters.update(
        {
            "BETAE_UNIT": float(local.betae_unit),
            "NU_EE": float(local.nu_ee),
            "LAMBDA_STAR": float(local.lambda_star),
            "GAMMA_E": 0.0,
            "GAMMA_P": 0.0,
            "MACH": 0.0,
            "N_SPECIES": int(local.n_species),
        }
    )
    for index in range(local.n_species):
        for key in SPECIES_KEYS:
            parameters[f"{key}_{index + 1}"] = float(np.asarray(local.species[key])[index])
    parameters.update(
        {str(key).upper(): value for key, value in configuration.extra_parameters.items()}
    )
    return parameters


def write_input_cgyro(parameters: Mapping[str, Any], path: str | Path) -> Path:
    """Write an ``input.cgyro``, one ``KEY=VALUE`` per line."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={_render(value)}" for key, value in parameters.items()]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _render(value: Any) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def input_sha256(path: str | Path) -> str:
    """The hash a run is identified by: of the file CGYRO reads, not of its inputs."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_cgyro_case(
    profile: GACODEProfile,
    r_over_a: float,
    workdir: str | Path,
    config: Optional[CGYROConfig] = None,
    *,
    state_key: Optional[Mapping[str, Any]] = None,
) -> CGYROInputs:
    """Stage ``input.cgyro`` for one surface in *workdir*.

    ``state_key`` is recorded verbatim in the provenance -- Lane T's provisional
    ``(shot, time_efit_s, efit_lineage)`` today -- so a run directory says which state
    it belongs to without a side table.
    """
    local = prepare_cgyro_input(profile, r_over_a, config=config)
    absent = local.tglf.check_tglf_requirements()
    if absent:
        raise LocalConversionError(
            f"the local input is missing {', '.join(absent)}; CGYRO cannot be run on it "
            "and nothing is substituted."
        )
    return stage_cgyro_case(
        local, workdir, config, state_key=state_key, profile=profile
    )


def stage_cgyro_case(
    local: CGYROInput,
    workdir: str | Path,
    config: Optional[CGYROConfig] = None,
    *,
    state_key: Optional[Mapping[str, Any]] = None,
    profile: Optional[GACODEProfile] = None,
) -> CGYROInputs:
    """Write ``input.cgyro`` for an already-projected local input.

    The split from :func:`prepare_cgyro_case` lets a ky or resolution scan project the
    surface once and stage it many times.
    """
    directory = Path(workdir)
    directory.mkdir(parents=True, exist_ok=True)
    configuration = config or CGYROConfig()
    parameters = cgyro_parameters(local, configuration)
    written = write_input_cgyro(parameters, directory / "input.cgyro")
    return CGYROInputs(
        workdir=directory,
        files=(written,),
        local=local,
        input_cgyro=written,
        parameters=parameters,
        provenance={
            "r_over_a": float(local.r_over_a),
            "input_sha256": input_sha256(written),
            "state_key": None if state_key is None else dict(state_key),
            "resolution": configuration.resolution(),
            "field_model": configuration.field_model,
            "local": {k: dict(v) for k, v in local.provenance.items()},
            "profile": {} if profile is None else dict(profile.provenance),
            "species": tuple(local.names),
        },
    )


#: Keys compared against CGYRO's own projection. ``out.cgyro.equilibrium`` prints five
#: significant digits (``1pe12.4``), so agreement is judged relative to that precision,
#: with an absolute floor for values that are near zero on VEST (S_KAPPA ~ 6e-3, ZETA = 0).
#: ``lambda_star`` is deliberately absent: CGYRO's ``PROFILE_MODEL=2`` multiplies it by
#: ``LAMBDA_STAR_SCALE``, whose default is 0 (Debye shielding off), while TGLF keeps
#: ``DEBYE_FACTOR=1``. The translation carries TGLF's value -- the comparison is with
#: TGLF -- so the two are expected to differ and the difference is not an error.
ORACLE_KEYS = (
    "rmin", "rmaj", "q", "shear", "shift", "kappa", "s_kappa", "delta", "s_delta",
    "zeta", "s_zeta", "zmag", "dzmag", "betae_unit",
)
ORACLE_RELATIVE = 2.0e-4
ORACLE_ABSOLUTE = 1.0e-6


def _agreement(ours: float, theirs: float) -> dict[str, Any]:
    tolerance = max(ORACLE_RELATIVE * abs(theirs), ORACLE_ABSOLUTE)
    difference = float(ours) - float(theirs)
    return {
        "vaft": float(ours), "cgyro": float(theirs), "difference": difference,
        "tolerance": tolerance, "ok": bool(abs(difference) <= tolerance),
    }


def compare_with_oracle(
    local: CGYROInput, equilibrium: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Compare a translation with CGYRO's own ``PROFILE_MODEL=2`` projection.

    Parameters
    ----------
    local
        This module's translation.
    equilibrium
        :attr:`CgyroOutputs.equilibrium` (or
        :func:`~vaft.code.gacode.cgyro.outputs.parse_equilibrium`) from a ``cgyro -t``
        run on the same ``input.gacode`` at the same ``RMIN``.

    Returns
    -------
    dict
        ``{key: {"vaft", "cgyro", "difference", "tolerance", "ok"}}`` for every geometry
        key, ``betae_unit``, ``nu_ee`` and every species' density, temperature, mass and
        gradients. Species are matched by charge and mass, not position.
    """
    ours = {
        "rmin": local.geometry["RMIN"], "rmaj": local.geometry["RMAJ"],
        "q": local.geometry["Q"], "shear": local.geometry["S"],
        "shift": local.geometry["SHIFT"], "kappa": local.geometry["KAPPA"],
        "s_kappa": local.geometry["S_KAPPA"], "delta": local.geometry["DELTA"],
        "s_delta": local.geometry["S_DELTA"], "zeta": local.geometry["ZETA"],
        "s_zeta": local.geometry["S_ZETA"], "zmag": local.geometry["ZMAG"],
        "dzmag": local.geometry["DZMAG"], "betae_unit": local.betae_unit,
    }
    report: dict[str, dict[str, Any]] = {}
    for key in ORACLE_KEYS:
        theirs = equilibrium.get(key)
        if theirs is None:
            continue
        if key == "q":
            theirs = abs(theirs)  # CGYRO stores q signed by IPCCW*BTCCW
        report[key] = _agreement(ours[key], theirs)

    their_species = equilibrium.get("species") or {}
    if not their_species:
        return report
    z_theirs = np.asarray(their_species["z"], dtype=float)
    m_theirs = np.asarray(their_species["mass"], dtype=float)
    for index in range(local.n_species):
        z = float(local.species["Z"][index])
        m = float(local.species["MASS"][index])
        match = np.flatnonzero(
            (np.abs(z_theirs - z) < 1e-6) & (np.abs(m_theirs - m) <= 1e-3 * abs(m))
        )
        if match.size != 1:
            report[f"species_{index + 1}"] = {
                "vaft": z, "cgyro": float("nan"), "difference": float("nan"),
                "tolerance": 0.0, "ok": False,
            }
            continue
        j = int(match[0])
        for ours_key, theirs_key in (
            ("DENS", "dens"), ("TEMP", "temp"), ("MASS", "mass"),
            ("DLNNDR", "dlnndr"), ("DLNTDR", "dlntdr"),
        ):
            report[f"{ours_key}_{index + 1}"] = _agreement(
                local.species[ours_key][index], np.asarray(their_species[theirs_key])[j]
            )
        if z < 0:
            # nu of the electron species is nu_ee itself (cgyro_make_profiles).
            report["nu_ee"] = _agreement(local.nu_ee, np.asarray(their_species["nu"])[j])
    return report
