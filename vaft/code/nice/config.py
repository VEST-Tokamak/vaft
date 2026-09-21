"""Typed configuration and result objects for the NICE reconstruction adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


def vest_reference_parameter_file() -> Path:
    """Packaged, revision-pinned VEST magnetic-reconstruction parameters."""
    return Path(__file__).with_name("vest_reference_param.xml")


@dataclass(frozen=True)
class NiceConfig:
    """Configuration for one standalone NICE magnetic reconstruction.

    NICE reads a directory of text files.  ``parameter_file`` deliberately has
    no implicit machine-specific default: a scientifically reviewed NICE XML
    file must be selected for real runs, while preparation remains NICE-free.
    """

    executable: Optional[str] = None
    nice_home: Optional[Path | str] = None
    source_revision: Optional[str] = None
    vaft_revision: Optional[str] = None
    input_snapshot_hash: Optional[str] = None
    build_options: Mapping[str, Any] = field(default_factory=dict)
    profile_basis: Mapping[str, Any] = field(default_factory=dict)
    solver_tolerances: Mapping[str, Any] = field(default_factory=dict)
    initialization_method: str = "from_scratch"
    workdir: Path | str = Path(".")
    parameter_file: Optional[Path | str] = None
    env: Mapping[str, str] = field(default_factory=dict)
    timeout: Optional[float] = None
    arguments: tuple[str, ...] = ()

    shot: Optional[int] = None
    time: Optional[float] = None
    time_index: Optional[int] = None
    cocos_in: int = 11
    cocos_out: int = 11
    major_radius: float = 0.4

    include_plasma_current: bool = True
    include_flux_loops: bool = True
    include_bpol_probes: bool = True
    include_diamagnetic_flux: bool = False
    disabled_channels: tuple[str, ...] = ()
    diagnostic_source: str = "magnetics"
    correct_active_response: bool = False
    # Winding polarity of physical flux measurements relative to NICE's
    # COCOS flux input. VEST's calibrated +Green-flux convention requires -1.
    flux_loop_input_sign: float = 1.0
    default_bpol_uncertainty: float = 1.0e-3
    default_flux_uncertainty: float = 1.0e-4
    default_ip_uncertainty: float = 1.0e3
    default_diamagnetic_uncertainty: float = 1.0e-4

    # Fixed current per pf_passive loop [A].  None reads pf_passive.loop[:].current.
    passive_currents: Optional[tuple[float, ...]] = None
    passive_current_mode: str = "diagnostic_subtraction"


@dataclass(frozen=True)
class NiceDiagnostic:
    family: str
    ods_path: str
    identifier: str
    geometry: Mapping[str, Any]
    value: float
    uncertainty: float
    normalization: float
    enabled: bool
    reason: str = ""
    original_value: Optional[float] = None
    conditioned_value: Optional[float] = None
    weight: float = 1.0
    passive_response: float = 0.0
    active_response_correction: float = 0.0
    original_uncertainty: Optional[float] = None
    original_enabled: Optional[bool] = None


@dataclass
class NiceInputs:
    workdir: Path
    input_dir: Path
    output_dir: Path
    shot: int
    time: float
    geometry: Mapping[str, Any]
    diagnostics: tuple[NiceDiagnostic, ...]
    passive_currents: tuple[float, ...]
    manifest: Mapping[str, Any]
    manifest_file: Path
    files: tuple[Path, ...] = ()
    ods: Any = None


@dataclass
class NiceResult:
    returncode: Optional[int]
    workdir: Path
    process_succeeded: bool = False
    converged: Optional[bool] = None
    # Issue #666 stage success: a valid numerically converged equilibrium.
    # Auxiliary diagnostic/physics-quality warnings do not gate this flag.
    scientifically_usable: Optional[bool] = None
    termination_reason: str = ""
    nonlinear_iterations: Optional[int] = None
    objective: Mapping[str, Any] = field(default_factory=dict)
    iteration_history: tuple[Mapping[str, Any], ...] = ()
    diagnostic_residuals: tuple[Mapping[str, Any], ...] = ()
    output_files: tuple[Path, ...] = ()
    logs: tuple[Path, ...] = ()
    stdout: str = ""
    stderr: str = ""
    parsing_errors: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    ods: Any = None

    @property
    def ok(self) -> bool:
        return bool(
            self.process_succeeded and self.converged and self.scientifically_usable
        )
