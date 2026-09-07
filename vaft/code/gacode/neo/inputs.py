"""Generate ``input.neo`` and stage a NEO case.

Staging is not a user-facing step.  ``prepare_neo_case`` takes the profile and
the configuration and leaves a directory NEO can be pointed at; the caller never
copies files by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from ...base import CodeInputs
from .._input_gacode import write_input_gacode
from .._profiles import GACODEProfile
from ._types import NEOConfig, PROFILE_MODEL_EXPERIMENTAL


@dataclass
class NEOInputs(CodeInputs):
    """A staged NEO case: the directory, the files in it, and what made them."""

    profile: Optional[GACODEProfile] = None
    input_neo: Optional[Path] = None
    input_gacode: Optional[Path] = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)


def neo_parameters(
    config: NEOConfig, profile: Optional[GACODEProfile] = None
) -> dict[str, Any]:
    """The ``KEY=VALUE`` settings this configuration means.

    Written out in full rather than relying on NEO's defaults: `input.neo`
    records only what it is given, so a file that omits a setting cannot be told
    apart later from one that chose NEO's default deliberately.
    """
    species = config.n_species
    if species is None:
        if profile is None:
            raise ValueError(
                "n_species is not set and no profile was given, so the species count "
                "cannot be established; NEO counts electrons too"
            )
        species = profile.n_ion + 1

    parameters: dict[str, Any] = {
        "N_ENERGY": int(config.n_energy),
        "N_XI": int(config.n_xi),
        "N_THETA": int(config.n_theta),
        "N_RADIAL": int(config.n_radial),
        "RMIN_OVER_A": float(config.rmin_over_a),
        "SILENT_FLAG": 0,
        "EQUILIBRIUM_MODEL": int(config.equilibrium_model),
        "COLLISION_MODEL": int(config.collision_model),
        "PROFILE_MODEL": int(config.profile_model),
        "PROFILE_ERAD0_MODEL": int(config.profile_erad0_model),
        "ROTATION_MODEL": int(config.rotation_model),
        "SPITZER_MODEL": int(config.spitzer_model),
        "IPCCW": int(config.ipccw),
        "BTCCW": int(config.btccw),
        "N_SPECIES": int(species),
    }
    parameters.update({str(k).upper(): v for k, v in config.extra_parameters.items()})
    return parameters


def write_input_neo(parameters: Mapping[str, Any], path: str | Path) -> Path:
    """Write an ``input.neo`` file, one ``KEY=VALUE`` per line."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={_render(value)}" for key, value in parameters.items()]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _render(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def prepare_neo_case(
    profile: GACODEProfile,
    workdir: str | Path,
    config: Optional[NEOConfig] = None,
) -> NEOInputs:
    """Stage ``input.gacode`` and ``input.neo`` in *workdir*.

    The directory is the caller's and is not a temporary one, so a run stays
    inspectable afterwards.

    Raises
    ------
    ValueError
        The profile lacks something NEO needs for the configured
        ``PROFILE_MODEL``, or the configuration is internally inconsistent.
    """
    configuration = config or NEOConfig()
    directory = Path(workdir)
    directory.mkdir(parents=True, exist_ok=True)

    if int(configuration.profile_model) >= PROFILE_MODEL_EXPERIMENTAL:
        absent = profile.check_neo_requirements()
        if absent:
            raise ValueError(
                f"PROFILE_MODEL={configuration.profile_model} reads input.gacode, but "
                f"the profile is missing {', '.join(absent)}. Nothing is substituted; "
                "supply them or use PROFILE_MODEL=1 with local parameters."
            )

    parameters = neo_parameters(configuration, profile)
    input_gacode = write_input_gacode(profile, directory / "input.gacode")
    input_neo = write_input_neo(parameters, directory / "input.neo")
    return NEOInputs(
        workdir=directory,
        files=(input_gacode, input_neo),
        profile=profile,
        input_neo=input_neo,
        input_gacode=input_gacode,
        parameters=parameters,
        provenance={
            "profile": dict(profile.provenance),
            "n_exp": profile.n_exp,
            "n_ion": profile.n_ion,
            "species": tuple(profile.name),
        },
    )
