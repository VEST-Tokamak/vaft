"""NUBEAM neutral-beam Monte Carlo adapter.

Runs NUBEAM and parses its native output. This layer stops at NUBEAM's own
terms; :mod:`vaft.machine_mapping.core_sources` writes the plasma-side source
term and :mod:`vaft.machine_mapping.distributions` the fast-ion population.
The birth markers and lost-particle records have no IDS home yet -- that is the
remainder of issue #490 section 6.

The installation is external -- NTCC requires each user to accept its licence
before downloading the source -- so VAFT owns the build recipe
(``install/nubeam/``) and this adapter contract, not the source itself. Point
``$NUBEAMHOME`` at the installation root.
"""

from __future__ import annotations

from .config import (
    NUBEAM_GENERATOR_EXECUTABLE,
    NUBEAM_HOME_ENV,
    NUBEAM_HOME_EXECUTABLE,
    NUBEAM_LONGEST_OUTPUT_SUFFIX,
    NUBEAM_PATH_BUFFER_CHARS,
    NUBEAM_UPDATE_STATE_EXECUTABLE,
    NUBEAMConfig,
    workdir_budget,
)
from .inputs import (
    PACKAGED_VEST_CASE_DIR,
    PACKAGED_VEST_CASE_GFILE,
    NUBEAMCase,
    NUBEAMInputError,
    NUBEAMInputs,
    check_workdir_length,
    inputf_runid,
    NUBEAM_NAMELISTS,
    StagedNamelists,
    input_plasma_state_name,
    inputf_state_filename,
    packaged_vest_case,
    stage_nubeam_namelists,
    prepare_nubeam_inputs,
    rewrite_inputf_equilibrium,
)
from .outputs import (
    LOST_PARTICLE_FIELDS,
    NUBEAMBirthMarkers,
    NUBEAMFluxSurfaceAverages,
    NUBEAMLostParticles,
    NUBEAMPowerBalance,
    NUBEAMOutputs,
    NUBEAMRadialGrid,
    NUBEAMResult,
    collect_nubeam_outputs,
    parse_power_balance,
)
from .runner import (
    NUBEAMExecutionError,
    find_nubeam_executable,
    find_plasma_state_generator,
    find_update_state_executable,
    build_plasma_state,
    generate_plasma_state,
    run_nubeam,
    run_nubeam_case,
    run_nubeam_ods,
)
from .plasma_state import (
    PLASMA_STATE_NAMELIST,
    PlasmaStateInputError,
    PlasmaStateProfiles,
    PlasmaStateSpec,
    ion_densities_from_zeff,
    legacy_case_spec,
    profiles_from_core_profiles,
    read_legacy_profiles,
    read_shot_configuration,
    render_plasma_state_namelist,
    spec_from_ods,
)

__all__ = [
    "LOST_PARTICLE_FIELDS",
    "NUBEAMBirthMarkers",
    "NUBEAMCase",
    "NUBEAMConfig",
    "NUBEAMExecutionError",
    "NUBEAMFluxSurfaceAverages",
    "NUBEAMInputError",
    "NUBEAMInputs",
    "NUBEAMLostParticles",
    "NUBEAMOutputs",
    "NUBEAMPowerBalance",
    "NUBEAMRadialGrid",
    "NUBEAMResult",
    "NUBEAM_GENERATOR_EXECUTABLE",
    "NUBEAM_HOME_ENV",
    "NUBEAM_HOME_EXECUTABLE",
    "NUBEAM_LONGEST_OUTPUT_SUFFIX",
    "NUBEAM_NAMELISTS",
    "NUBEAM_PATH_BUFFER_CHARS",
    "NUBEAM_UPDATE_STATE_EXECUTABLE",
    "PACKAGED_VEST_CASE_DIR",
    "PACKAGED_VEST_CASE_GFILE",
    "PLASMA_STATE_NAMELIST",
    "PlasmaStateInputError",
    "PlasmaStateProfiles",
    "PlasmaStateSpec",
    "StagedNamelists",
    "build_plasma_state",
    "check_workdir_length",
    "collect_nubeam_outputs",
    "find_nubeam_executable",
    "find_plasma_state_generator",
    "find_update_state_executable",
    "generate_plasma_state",
    "input_plasma_state_name",
    "inputf_runid",
    "inputf_state_filename",
    "ion_densities_from_zeff",
    "legacy_case_spec",
    "packaged_vest_case",
    "parse_power_balance",
    "prepare_nubeam_inputs",
    "profiles_from_core_profiles",
    "read_legacy_profiles",
    "read_shot_configuration",
    "render_plasma_state_namelist",
    "rewrite_inputf_equilibrium",
    "run_nubeam",
    "run_nubeam_case",
    "run_nubeam_ods",
    "spec_from_ods",
    "stage_nubeam_namelists",
    "workdir_budget",
]
