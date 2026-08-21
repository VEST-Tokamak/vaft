"""Magnetic- and kinetic-constrained EFIT adapter.

A Python-first wrapper around the VEST EFIT binary that follows the common
``vaft.code`` adapter convention (``prepare`` / ``run`` / ``collect``), plus
legacy VEST/OMFIT signal-processing helpers, k-file/constraints generation,
and a kinetic-pressure constraint mode layered on top of the magnetic
reconstruction.

Submodules:

* ``config``   -- typed scientific k-file configuration (``EFITScientificConfig`` etc.)
* ``status``   -- EFIT slice-status/validation (``EFITSliceStatus``, ``validate_efit_slice``)
* ``legacy``   -- legacy VEST/OMFIT signal-processing helpers
* ``kfile``    -- constraints-ODS and k-file generation
* ``magnetic`` -- the core magnetic EFIT adapter (``EFITConfig``, ``run_efit``, ...)
* ``kinetic``  -- kinetic-pressure constraint mode (``KineticEFITConfig``, ``run_kinetic_efit``, ...)

Kinetic pressure is an EFIT constraint mode, not a separate code adapter, so
its API is re-exported here alongside the magnetic adapter to present one
canonical public namespace: ``vaft.code.efit``.
"""

from .config import (
    EFITConstraintConfig,
    EFITInitializationConfig,
    EFITNumericsConfig,
    EFITProfileConfig,
    EFITScientificConfig,
    efit_parameter_grid,
)
from .status import (
    EFIT_FAILURE_CODES,
    EFITSliceStatus,
    EFITValidationConfig,
    apply_temporal_continuity,
    validate_efit_slice,
)
from .legacy import (
    gauss_fit4,
    min_gauss_fit4,
    vfit_equilibrium_form_constraints,
    vfit_pf_active_efit26,
    correct_flux_loop,
    vfit_signal_startend,
    smooth,
    vest_rspv1,
    calculate_md_by_ods,
    brokenFinder,
    vest_signal_onoffsetpeak,
    vest_Halpha_tstart_tend,
    set_discharge_index,
)
from .magnetic import (
    EFIT_HOME_ENV,
    EFIT_HOME_EXECUTABLE,
    EFIT_EXEC_ENV,
    EFITConfig,
    EFITInputs,
    EFITResult,
    _efit_unconfigured_reason,
    resolved_efit_configuration,
    find_efit_executable,
    prepare_efit_inputs,
    run_efit,
    collect_efit_outputs,
    gfile_to_omas,
)
from .kfile import generate_constraints_ods, generate_kfile
from .kinetic import (
    EQE,
    SPLINE_SIG_FRAC,
    KineticEFITConfig,
    KineticEFITInputs,
    KineticEFITResult,
    PressurePoints,
    _raw_ne_te_ti,
    _resolve_ti_te_ratio,
    build_kinetic_core_profiles,
    inject_pressure_constraint,
    kinetic_pressure_points,
    prepare_kinetic_efit_inputs,
    run_kinetic_chain,
    run_kinetic_efit,
    scale_plasma,
)

__all__ = [
    "EFITConstraintConfig",
    "EFITInitializationConfig",
    "EFITNumericsConfig",
    "EFITProfileConfig",
    "EFITScientificConfig",
    "efit_parameter_grid",
    "EFIT_FAILURE_CODES",
    "EFITSliceStatus",
    "EFITValidationConfig",
    "apply_temporal_continuity",
    "validate_efit_slice",
    "EFITConfig",
    "EFITInputs",
    "EFITResult",
    "resolved_efit_configuration",
    "find_efit_executable",
    "prepare_efit_inputs",
    "run_efit",
    "collect_efit_outputs",
    "gfile_to_omas",
    "generate_constraints_ods",
    "generate_kfile",
    "EQE",
    "KineticEFITConfig",
    "KineticEFITInputs",
    "KineticEFITResult",
    "PressurePoints",
    "build_kinetic_core_profiles",
    "kinetic_pressure_points",
    "inject_pressure_constraint",
    "scale_plasma",
    "prepare_kinetic_efit_inputs",
    "run_kinetic_efit",
    "run_kinetic_chain",
]
