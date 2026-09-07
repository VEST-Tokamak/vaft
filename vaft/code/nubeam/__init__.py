"""NUBEAM neutral-beam Monte Carlo adapter.

Runs NUBEAM and parses its native output. This layer stops at NUBEAM's own
terms; :mod:`vaft.machine_mapping.core_sources` writes the plasma-side source
term and :mod:`vaft.machine_mapping.distributions` the fast-ion population.
The birth markers and lost-particle records have no IDS home yet -- that is the
remainder of issue #490 section 6.

The installation is external -- NTCC requires each user to accept its licence
before downloading the source -- so VAFT owns the build recipe
(``external/nubeam/``) and this adapter contract, not the source itself. Point
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
    inputf_state_filename,
    packaged_vest_case,
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
    generate_plasma_state,
    run_nubeam,
    run_nubeam_case,
)

__all__ = [
    "NUBEAM_GENERATOR_EXECUTABLE",
    "NUBEAM_HOME_ENV",
    "NUBEAM_HOME_EXECUTABLE",
    "NUBEAM_LONGEST_OUTPUT_SUFFIX",
    "NUBEAM_PATH_BUFFER_CHARS",
    "NUBEAM_UPDATE_STATE_EXECUTABLE",
    "PACKAGED_VEST_CASE_DIR",
    "PACKAGED_VEST_CASE_GFILE",
    "LOST_PARTICLE_FIELDS",
    "NUBEAMBirthMarkers",
    "NUBEAMCase",
    "NUBEAMFluxSurfaceAverages",
    "NUBEAMLostParticles",
    "NUBEAMPowerBalance",
    "NUBEAMConfig",
    "NUBEAMExecutionError",
    "NUBEAMInputError",
    "NUBEAMInputs",
    "NUBEAMOutputs",
    "NUBEAMRadialGrid",
    "NUBEAMResult",
    "check_workdir_length",
    "collect_nubeam_outputs",
    "find_nubeam_executable",
    "find_plasma_state_generator",
    "find_update_state_executable",
    "generate_plasma_state",
    "inputf_runid",
    "inputf_state_filename",
    "packaged_vest_case",
    "parse_power_balance",
    "prepare_nubeam_inputs",
    "rewrite_inputf_equilibrium",
    "run_nubeam",
    "run_nubeam_case",
    "workdir_budget",
]
