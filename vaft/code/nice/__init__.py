"""NICE free-boundary magnetic-reconstruction adapter.

Preparation and collection are pure Python and do not require NICE.  Only
``run_nice`` resolves and executes the standalone ``nice_recon`` program.
"""

from .config import (
    NiceConfig,
    NiceDiagnostic,
    NiceInputs,
    NiceResult,
    vest_reference_parameter_file,
)
from .diagnostics import diagnostics_from_ods
from .geometry import geometry_hash, nice_geometry_from_ods
from .inputs import prepare_nice_inputs
from .outputs import collect_nice_outputs
from .runner import resolve_nice_executable, run_nice
from .study import (
    assert_same_physical_channels,
    compare_diagnostic_residuals,
    compare_equilibria,
    constraint_family_configs,
    lcfs_rms_displacement,
    physical_channel_signature,
    run_nice_window,
    summarize_window,
    write_study_report,
    write_window_report,
)

__all__ = [
    "NiceConfig",
    "NiceDiagnostic",
    "NiceInputs",
    "NiceResult",
    "collect_nice_outputs",
    "diagnostics_from_ods",
    "geometry_hash",
    "nice_geometry_from_ods",
    "prepare_nice_inputs",
    "resolve_nice_executable",
    "run_nice",
    "vest_reference_parameter_file",
    "assert_same_physical_channels",
    "compare_diagnostic_residuals",
    "compare_equilibria",
    "constraint_family_configs",
    "lcfs_rms_displacement",
    "physical_channel_signature",
    "run_nice_window",
    "summarize_window",
    "write_study_report",
    "write_window_report",
]
